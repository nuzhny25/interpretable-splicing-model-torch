from maf_processing import SPECIES
from chunking import chunk_sequence
import numpy as np
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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


def main():
    matrix = np.load(MATRIX_PATH)

    state_dict = torch.load("../model_weights.pt", map_location="cpu")
    species_names = list(SPECIES.keys())
    standard_deviation_sr = {}
    standard_deviation_incl = {}
    standard_deviation_skip = {}

    model = PNASModel(input_length=WINDOW_SIZE + 20)
    model.load_state_dict(state_dict)
    model.eval()

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

    original_mapping = alignment_mapping(matrix)
    original_aligned_sr = load_aligned_sr(original_results, original_mapping, "sr")
    original_aligned_incl = load_aligned_sr(original_results, original_mapping, "incl")
    original_aligned_skip = load_aligned_sr(original_results, original_mapping, "skip")
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

    for i in range(N_OF_PERMUTATIONS):

        resulting = {}

        # Get a permutaiton of the columns of the aligned matrix
        permutation = matrix.copy()
        np.random.shuffle(permutation)
        permutation = permutation.T

        print(f"permutation {i}")

        for j, row in enumerate(permutation):
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
                a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")
                sr_balance = a_incl.sum(dim=1) - a_skip.sum(dim=1)

            # save the model activations
            resulting[f"{species_names[j]}_sr"] = sr_balance.numpy()
            resulting[f"{species_names[j]}_incl"] = a_incl.sum(dim=1).numpy()
            resulting[f"{species_names[j]}_skip"] = a_skip.sum(dim=1).numpy()

        # produce the mapping for the permuted aligned matrix

        mapping = alignment_mapping(permutation.T)

        # put the model activations onto the shared grid

        aligned_sr_track = load_aligned_sr(resulting, mapping, "sr")
        aligned_incl_track = load_aligned_sr(resulting, mapping, "incl")
        aligned_skip_track = load_aligned_sr(resulting, mapping, "skip")

        positions = np.concatenate([pos for pos, _ in aligned_sr_track.values()])
        grid = np.arange(positions.min(), positions.max() + 1, STEP_SIZE)

        std_sr = sd_on_grid(aligned_sr_track, grid)
        std_incl = sd_on_grid(aligned_incl_track, grid)
        std_skip = sd_on_grid(aligned_skip_track, grid)

        # calculate the average standard deviation

        standard_deviation_sr[f"perm{i}"] = np.nanmean(std_sr)
        standard_deviation_incl[f"perm{i}"] = np.nanmean(std_incl)
        standard_deviation_skip[f"perm{i}"] = np.nanmean(std_skip)

        # save the standard deviation

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
            f"[{track_name}] Average std of {N_OF_PERMUTATIONS} permuted matrices : {perm_mean:.4f}\n",
            f"                vs original (un-permuted) matrix std : {orig_avg:.4f}\n",
            f"   original is {z_score:.4f} standard deviations from the permuted mean\n",
            f"  the standard deviation of the standard deviation of permuted values is {perm_std:.4f}",
        )

    np.savez(
        f"{OUTPUT_DATA_DIR}/average_stds.npz",
        **{f"sr_{k}": v for k, v in standard_deviation_sr.items()},
        **{f"incl_{k}": v for k, v in standard_deviation_incl.items()},
        **{f"skip_{k}": v for k, v in standard_deviation_skip.items()},
    )


if __name__ == "__main__":
    main()
