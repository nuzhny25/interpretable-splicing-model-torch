import numpy as np
import os
import sys
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # repo root: model, utils
sys.path.insert(
    0, os.path.join(_HERE, "..", "dataset_preparations")
)  # maf_processing, chunking

from maf_processing import SPECIES
from chunking import chunk_sequence
from model import PNASModel
from utils import add_flanking, one_hot_batch

OUTPUT_DATA_DIR = "../data/aligned_permutations/"
N_OF_PERMUTATIONS = 1000
MATRIX_PATH = "../data/multiz100/alignment_matrix.npy"
WINDOW_SIZE = 70
STEP_SIZE = 10


def extract_seq(row):
    return "".join(row).replace("N", "").replace("-", "")


def alignment_mapping(matrix_np):
    mapping = [[] for _ in range(matrix_np.shape[1])]
    for aligned_idx, row in enumerate(matrix_np):
        for species_idx, char in enumerate(row):
            if char != "-" and char != "N":
                mapping[species_idx].append(aligned_idx)

    return mapping


def load_aligned_sr(data, mapping, suffix="sr"):
    sr = {}
    for idx, name in enumerate(SPECIES):
        if f"{name}_{suffix}" not in data:
            continue
        vals_sr = data[f"{name}_{suffix}"]
        nuc_map = mapping[idx]
        aligned, sr_vals = [], []
        for i in range(len(vals_sr)):
            center_nuc = i * STEP_SIZE + WINDOW_SIZE // 2
            if 0 <= center_nuc < len(nuc_map):
                aligned.append(nuc_map[center_nuc])
                sr_vals.append(vals_sr[i])
        if aligned:
            order = np.argsort(aligned)
            sr[name] = (np.array(aligned)[order], np.array(sr_vals)[order])
    return sr


def sd_on_grid(species_dict, grid, step=STEP_SIZE):
    """Cross-species SD at each grid column, nearest-neighbor within step/2."""
    tol = step / 2
    stacked = []
    for pos, vals in species_dict.values():
        vals_s = vals.astype(float)
        ins = np.searchsorted(pos, grid)
        left = np.clip(ins - 1, 0, len(pos) - 1)
        right = np.clip(ins, 0, len(pos) - 1)
        dl = np.abs(grid - pos[left])
        dr = np.abs(grid - pos[right])
        use_right = dr < dl
        nearest = np.where(use_right, right, left)
        dist = np.where(use_right, dr, dl)
        col = np.where(dist <= tol, vals_s[nearest], np.nan)
        stacked.append(col)
    stacked = np.array(stacked)
    sd = np.nanstd(stacked, axis=0, ddof=1)
    valid = np.sum(~np.isnan(stacked), axis=0) >= 10
    sd[~valid] = np.nan
    return sd


def permute_filters(model, orig_skip, orig_incl, rng, scheme="indepndent"):
    """Permute the 6 kernel-position columns of the sequence conv filters.

    Rebuilds the weights from the pristine ``orig_skip`` / ``orig_incl`` each
    call, so permutations don't compound across iterations. Different column
    permutations are applied to ``conv_skip`` and ``conv_incl``.

    Args:
        scheme: ``"shared"`` uses one column order for every filter;
            ``"independent"`` draws a fresh column order per filter.
        rng: A ``np.random.Generator`` for reproducibility.
    """
    num_filters, _, kernel_size = orig_skip.shape  # (20, 4, 6)
    with torch.no_grad():
        if scheme == "shared":
            perm_skip = torch.from_numpy(rng.permutation(kernel_size))
            perm_incl = torch.from_numpy(rng.permutation(kernel_size))

            model.conv_skip.weight.copy_(orig_skip[:, :, perm_skip])
            model.conv_incl.weight.copy_(orig_incl[:, :, perm_incl])
        elif scheme == "independent":
            for k in range(num_filters):
                perm_skip = torch.from_numpy(rng.permutation(kernel_size))
                perm_incl = torch.from_numpy(rng.permutation(kernel_size))

                model.conv_skip.weight[k] = orig_skip[k][:, perm_skip]
                model.conv_incl.weight[k] = orig_incl[k][:, perm_incl]
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")


def main():
    matrix = np.load(MATRIX_PATH)

    state_dict = torch.load("../model_weights.pt", map_location="cpu")
    species_names = list(SPECIES.keys())

    model = PNASModel(input_length=WINDOW_SIZE + 20)
    model.load_state_dict(state_dict)
    model.eval()

    # Keep a pristine copy of the sequence-filter weights to permute from.
    orig_skip = model.conv_skip.weight.detach().clone()
    orig_incl = model.conv_incl.weight.detach().clone()
    rng = np.random.default_rng(0)

    # baseline on the un-permuted matrix
    print("original matrix")
    original_results = {}
    original = matrix.T
    for j, row in enumerate(original):
        sequence = extract_seq(row)
        chunks, _ = chunk_sequence(sequence, WINDOW_SIZE, STEP_SIZE)
        if not chunks:
            continue

        seq_oh = one_hot_batch(add_flanking(chunks))
        x_seq = torch.tensor(seq_oh, dtype=torch.float32)

        with torch.no_grad():
            a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")
            sr_balance = a_incl.sum(dim=1) - a_skip.sum(dim=1)

        original_results[f"{species_names[j]}_sr"] = sr_balance.numpy()
        original_results[f"{species_names[j]}_incl"] = a_incl.sum(dim=1).numpy()
        original_results[f"{species_names[j]}_skip"] = a_skip.sum(dim=1).numpy()

    mapping = alignment_mapping(matrix)
    original_aligned_sr = load_aligned_sr(original_results, mapping, "sr")
    original_aligned_incl = load_aligned_sr(original_results, mapping, "incl")
    original_aligned_skip = load_aligned_sr(original_results, mapping, "skip")
    original_positions = np.concatenate(
        [pos for pos, _ in original_aligned_sr.values()]
    )
    original_grid = np.arange(
        original_positions.min(), original_positions.max() + 1, STEP_SIZE
    )
    original_std_sr = sd_on_grid(original_aligned_sr, original_grid)
    original_std_incl = sd_on_grid(original_aligned_incl, original_grid)
    original_std_skip = sd_on_grid(original_aligned_skip, original_grid)
    original_avg_std_sr = np.nanmean(original_std_sr)
    original_avg_std_incl = np.nanmean(original_std_incl)
    original_avg_std_skip = np.nanmean(original_std_skip)

    per_column_avg = {}  # per_column_avg[scheme][track] -> per-grid-column mean std

    for scheme in ["independent"]:
        print(f"=== filter-permutation scheme: {scheme} ===")

        standard_deviation_sr = {}
        standard_deviation_incl = {}
        standard_deviation_skip = {}

        # Per-grid-column std for each permutation; averaged over perms below.
        percol_sr = []
        percol_incl = []
        percol_skip = []

        for i in range(N_OF_PERMUTATIONS):

            resulting = {}

            # Permute the conv filter columns (rebuilt from pristine weights).
            permute_filters(model, orig_skip, orig_incl, rng, scheme)

            print(f"permutation {i}")

            for j, row in enumerate(original):
                # For each row:
                # extract the sequence
                sequence = extract_seq(row)

                # chunk the sequence
                chunks, _ = chunk_sequence(sequence, WINDOW_SIZE, STEP_SIZE)
                if not chunks:
                    continue

                seq_oh = one_hot_batch(add_flanking(chunks))
                x_seq = torch.tensor(seq_oh, dtype=torch.float32)

                with torch.no_grad():
                    a_incl, a_skip = model.compute_sequence_activations(
                        x_seq, agg="mean"
                    )
                    sr_balance = a_incl.sum(dim=1) - a_skip.sum(dim=1)

                # save the model activations
                resulting[f"{species_names[j]}_sr"] = sr_balance.numpy()
                resulting[f"{species_names[j]}_incl"] = a_incl.sum(dim=1).numpy()
                resulting[f"{species_names[j]}_skip"] = a_skip.sum(dim=1).numpy()

            # put the model activations onto the shared grid

            aligned_sr_track = load_aligned_sr(resulting, mapping, "sr")
            aligned_incl_track = load_aligned_sr(resulting, mapping, "incl")
            aligned_skip_track = load_aligned_sr(resulting, mapping, "skip")

            positions = np.concatenate([pos for pos, _ in aligned_sr_track.values()])
            grid = np.arange(positions.min(), positions.max() + 1, STEP_SIZE)

            std_sr = sd_on_grid(aligned_sr_track, grid)
            std_incl = sd_on_grid(aligned_incl_track, grid)
            std_skip = sd_on_grid(aligned_skip_track, grid)

            # record per-grid-column std for this permutation
            percol_sr.append(std_sr)
            percol_incl.append(std_incl)
            percol_skip.append(std_skip)

            # calculate the average standard deviation

            standard_deviation_sr[f"perm{i}"] = np.nanmean(std_sr)
            standard_deviation_incl[f"perm{i}"] = np.nanmean(std_incl)
            standard_deviation_skip[f"perm{i}"] = np.nanmean(std_skip)

            # save the standard deviation

        # average the per-grid-column std across permutations (one curve/scheme)
        per_column_avg[scheme] = {
            "sr": np.nanmean(np.vstack(percol_sr), axis=0),
            "incl": np.nanmean(np.vstack(percol_incl), axis=0),
            "skip": np.nanmean(np.vstack(percol_skip), axis=0),
        }

        for track_name, std_dict, orig_avg in [
            ("sr", standard_deviation_sr, original_avg_std_sr),
            ("incl", standard_deviation_incl, original_avg_std_incl),
            ("skip", standard_deviation_skip, original_avg_std_skip),
        ]:
            perm_values = np.array(list(std_dict.values()))
            perm_mean = float(np.nanmean(perm_values))
            perm_std = float(np.nanstd(perm_values, ddof=1))
            z_score = (orig_avg - perm_mean) / perm_std

            print(
                f"[{scheme}/{track_name}] Average std of {N_OF_PERMUTATIONS} permuted filters : {perm_mean:.4f}\n",
                f"                vs original (un-permuted) filters std : {orig_avg:.4f}\n",
                f"   original is {z_score:.4f} standard deviations from the permuted mean\n",
                f"  the standard deviation of the standard deviation of permuted values is {perm_std:.4f}",
            )

        np.savez(
            f"{OUTPUT_DATA_DIR}/average_filter_permutations_stds_{scheme}.npz",
            grid=original_grid,
            sr_percolumn=per_column_avg[scheme]["sr"],
            incl_percolumn=per_column_avg[scheme]["incl"],
            skip_percolumn=per_column_avg[scheme]["skip"],
            **{f"sr_{k}": v for k, v in standard_deviation_sr.items()},
            **{f"incl_{k}": v for k, v in standard_deviation_incl.items()},
            **{f"skip_{k}": v for k, v in standard_deviation_skip.items()},
        )

    # --- plot per-grid-column std: original vs avg independent vs avg shared ---
    PLOT_DIR = "../plots/filter_permutations"
    os.makedirs(PLOT_DIR, exist_ok=True)

    tracks = [
        ("sr", original_std_sr),
        ("incl", original_std_incl),
        ("skip", original_std_skip),
    ]
    fig, axes = plt.subplots(len(tracks), 1, figsize=(11, 9), sharex=True)
    for ax, (track_name, orig_percol) in zip(axes, tracks):
        ax.plot(original_grid, orig_percol, color="black", lw=1.8, label="original")
        ax.plot(
            original_grid,
            per_column_avg["independent"][track_name],
            color="tab:blue",
            lw=1.2,
            label="avg independent perm",
        )
        ax.plot(
            original_grid,
            per_column_avg["shared"][track_name],
            color="tab:orange",
            lw=1.2,
            label="avg shared perm",
        )
        ax.set_ylabel(f"{track_name}\ncross-species std")
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("aligned position (grid column)")
    fig.suptitle(f"Per-position cross-species std (N={N_OF_PERMUTATIONS} permutations)")
    fig.tight_layout()
    out_png = f"{PLOT_DIR}/per_column_std.png"
    fig.savefig(out_png, dpi=150)
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    main()
