"""Control test for the real-data cross-species SR-deviation model.

Loads the model trained to minimize the cross-species absolute deviation of its
SR-balance signal on the real MALAT1 multiz alignment
(``custom_model/weights/real_weights_aad.pt``) and runs it on inputs where the
real signal is destroyed or transformed, recording the *same* quantities the
training loss uses:

  * ``col_abs_dev``           — per aligned-column absolute deviation of SR across
                                species (the loss numerator), NaN at invalid cols.
  * ``within_species_scale``  — the loss denominator (each species' own mean
                                absolute deviation across positions, averaged).
  * ``mean_col_abs_dev``      — mean of ``col_abs_dev`` over valid columns.
  * ``norm_loss``             — ``mean(col_abs_dev[valid] / within_species_scale)``,
                                i.e. the exact training ``sd_loss`` for that matrix.

Evaluated matrices (204 total):
  1. 100 gap-preserving random matrices  — real gap pattern kept, every nucleotide
     cell replaced with a uniform-random A/C/G/T.
  2. 100 shuffled matrices               — real matrix with its aligned-position
     axis permuted (matches dataset_preparations/aligned_permutations.py).
  3. 4 orientations of the real matrix   — original, reversed, complement,
     reverse_complement.

For the random and shuffled controls only the average over all runs is recorded
(the individual per-matrix values are nearly identical): a single scalar for
``{prefix}_mean_col_abs_dev`` / ``_norm_loss`` / ``_within_species_scale`` and the
per-column ``{prefix}_col_abs_dev`` averaged across the runs. Each control also
records, for ``mean_col_abs_dev`` / ``within_species_scale`` / ``norm_loss``, the
run-to-run standard deviation (``{prefix}_{metric}_std``) and how many of those
standard deviations the control average lies from the original value
(``{prefix}_{metric}_n_std_from_original``).
The four orientations keep their individual per-matrix values.

If the model learned real conservation, the original matrix should show markedly
lower per-column deviation than the random/shuffled controls, and generally lower
than the reversed/complement orientations. This is the custom_model analogue of
the original-PNAS control in dataset_preparations/aligned_permutations.py.

The metric is inlined here (a deterministic all-species pass) rather than imported
from the trainer, but computes exactly the loss's numerator/denominator, including
the same Gaussian smoothing of the SR track.
"""

import math
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Import PNASModel / str_to_vector the same way custom_model/plot/plotting.py does:
# put the custom_model/ directory first on sys.path so the regular module
# custom_model/custom_model.py wins over the custom_model/ namespace package.
CUSTOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CUSTOM_DIR)
from custom_model import PNASModel  # noqa: E402
from custom_utils import str_to_vector  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
MATRIX_PATH = os.path.join(
    CUSTOM_DIR, "..", "data", "multiz100", "alignment_matrix.npy"
)
WEIGHTS_PATH = os.path.join(CUSTOM_DIR, "weights", "real_weights_aad.pt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "control_results.npz")

N_RANDOM = 100  # number of gap-preserving random matrices
N_SHUFFLE = 100  # number of aligned-row-shuffled matrices
MIN_SPECIES = 5  # a column counts only if >= this many species present (matches loss)
SMOOTH_SIGMA = 1.5  # Gaussian smoothing of the SR profile along position (matches loss)
SEED = 0

NUCLEOTIDES = np.array(["A", "C", "G", "T"])
GAP_CHARS = ("-", "N")
# Complement map (A<->T, C<->G, incl. soft-masked lowercase); gaps unchanged.
COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
    "-": "-",
    "N": "N",
}
_complement_vec = np.vectorize(COMPLEMENT.get)


def build_smoothing_kernel(sigma, device):
    """Fixed sum-normalized Gaussian kernel + half-width, as in the trainer.

    Returns ``(None, 0)`` when smoothing is disabled.
    """
    if not sigma:
        return None, 0
    radius = math.ceil(3 * sigma)  # kernel half-width (~3 sigma)
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)  # (1, 1, K)
    return kernel, radius


def complement_matrix(matrix):
    """Per-character A<->T / C<->G complement of the alignment matrix."""
    return _complement_vec(matrix).astype("<U1")


def compute_metric(matrix, model, gauss_kernel, smooth_radius, device):
    """Per-column absolute deviation of SR + within-species scale for one matrix.

    Replicates the training loss over *all* species in ``matrix``: score each
    species' gap-removed sequence with ``compute_sr_profile``, Gaussian-smooth the
    SR track, scatter each window to the aligned column of its first nucleotide,
    then take the NaN-aware cross-species mean absolute deviation down each column
    (numerator) divided by the within-species dynamic range (denominator).

    Args:
        matrix: ``(n_aligned, n_species)`` array of single characters; gaps are
            ``-`` / ``N``, everything else (incl. lowercase) is a nucleotide.

    Returns:
        ``(col_abs_dev, within_species_scale, mean_col_abs_dev, norm_loss)`` where
        ``col_abs_dev`` is a length-``n_aligned`` array with NaN at invalid columns.
    """
    n_aligned, n_species = matrix.shape
    values = torch.zeros(n_species, n_aligned, device=device)
    mask = torch.zeros(n_species, n_aligned, device=device)

    for j in range(n_species):
        row = matrix[:, j]
        is_nuc = (row != GAP_CHARS[0]) & (row != GAP_CHARS[1])
        nuc_aligned = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()
        if len(seq) < model.seq_kernel_size:
            continue  # too short to form even one window

        x_seq = torch.tensor(str_to_vector(seq), dtype=torch.float32, device=device)
        with torch.no_grad():
            sr_j = model.compute_sr_profile(x_seq.unsqueeze(0)).squeeze(0)  # (Lj-5,)

            # Gaussian-smooth the dense (gap-free) SR track along position — same as
            # the loss (reflect padding so the position-0 block isn't damped).
            if gauss_kernel is not None:
                sr_j = F.conv1d(
                    F.pad(
                        sr_j.view(1, 1, -1),
                        (smooth_radius, smooth_radius),
                        mode="reflect",
                    ),
                    gauss_kernel,
                ).view(-1)

        # Place each window's value at the aligned column of its first nucleotide.
        cols = torch.tensor(
            nuc_aligned[: sr_j.shape[0]], dtype=torch.long, device=device
        )
        values[j, cols] = sr_j
        mask[j, cols] = 1.0

    # Numerator: NaN-aware cross-species mean absolute deviation per column.
    count = mask.sum(0)  # (n_aligned,)
    col_mean = (values * mask).sum(0) / count.clamp(min=1)
    col_abs_dev = ((values - col_mean).abs() * mask).sum(0) / count.clamp(min=1)
    valid = count >= MIN_SPECIES

    # Denominator: within-species dynamic range (each species' own mean absolute
    # deviation across positions, averaged over the present species).
    rc = mask.sum(1)  # (n_species,)
    present = rc > 0
    row_mean = (values * mask).sum(1) / rc.clamp(min=1)
    row_abs_dev = ((values - row_mean.unsqueeze(1)).abs() * mask).sum(1) / rc.clamp(
        min=1
    )
    within_species_scale = row_abs_dev[present].mean()

    mean_col_abs_dev = col_abs_dev[valid].mean()
    norm_loss = (col_abs_dev[valid] / within_species_scale).mean()

    # NaN-mask invalid columns for the saved per-column array.
    col_abs_dev_out = col_abs_dev.clone()
    col_abs_dev_out[~valid] = float("nan")

    return (
        col_abs_dev_out.cpu().numpy(),
        float(within_species_scale.item()),
        float(mean_col_abs_dev.item()),
        float(norm_loss.item()),
    )


def gap_preserving_random(matrix, rng):
    """Copy ``matrix`` keeping its gap pattern, randomizing every nucleotide cell."""
    out = matrix.copy()
    is_nuc = (matrix != GAP_CHARS[0]) & (matrix != GAP_CHARS[1])
    out[is_nuc] = rng.choice(NUCLEOTIDES, size=int(is_nuc.sum()))
    return out


def shuffle_aligned_rows(matrix, rng):
    """Permute the aligned-position (row) axis — matches aligned_permutations.py."""
    return matrix[rng.permutation(matrix.shape[0])]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    matrix = np.load(MATRIX_PATH)
    n_aligned, n_species = matrix.shape
    print(
        f"Loaded alignment matrix {matrix.shape} (aligned x species) from "
        f"{os.path.normpath(MATRIX_PATH)} on {device}"
    )

    model = PNASModel().to(device)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    # Checkpoints saved before metadata stamping are a bare state_dict.
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print(f"Loaded model weights from {os.path.normpath(WEIGHTS_PATH)}")

    gauss_kernel, smooth_radius = build_smoothing_kernel(SMOOTH_SIGMA, device)
    if gauss_kernel is not None:
        print(
            f"SR-profile smoothing: Gaussian sigma={SMOOTH_SIGMA} nt, "
            f"kernel size={gauss_kernel.shape[-1]}"
        )

    def evaluate(mat):
        return compute_metric(mat, model, gauss_kernel, smooth_radius, device)

    results = {}

    # ── 1. Four orientations of the real matrix ──────────────────────────────
    complement = complement_matrix(matrix)
    orientations = {
        "original": matrix,
        "reversed": matrix[::-1],
        "complement": complement,
        "reverse_complement": complement[::-1],
    }
    for name, mat in orientations.items():
        col_ad, wss, mean_ad, nloss = evaluate(mat)
        results[f"{name}_col_abs_dev"] = col_ad
        results[f"{name}_within_species_scale"] = np.float64(wss)
        results[f"{name}_mean_col_abs_dev"] = np.float64(mean_ad)
        results[f"{name}_norm_loss"] = np.float64(nloss)
        print(
            f"[orientation] {name:20s} mean_col_abs_dev={mean_ad:.6f}  norm_loss={nloss:.6f}"
        )

    # ── 2. 100 gap-preserving random matrices ────────────────────────────────
    # ── 3. 100 aligned-row-shuffled matrices ─────────────────────────────────
    control_specs = [
        ("random", N_RANDOM, gap_preserving_random),
        ("shuffled", N_SHUFFLE, shuffle_aligned_rows),
    ]
    for prefix, n_iter, make_matrix in control_specs:
        col_ad = np.empty((n_iter, n_aligned), dtype=np.float64)
        wss = np.empty(n_iter, dtype=np.float64)
        mean_ad = np.empty(n_iter, dtype=np.float64)
        nloss = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            col_ad[i], wss[i], mean_ad[i], nloss[i] = evaluate(make_matrix(matrix, rng))
            if (i + 1) % 25 == 0 or i == 0:
                print(f"[{prefix}] {i + 1}/{n_iter}  mean_col_abs_dev={mean_ad[i]:.6f}")
        # Collapse the n_iter runs to a single average — the individual per-matrix
        # values are nearly identical, so only the average over all runs is recorded:
        # scalar summaries plus the per-column deviation averaged across the runs.
        with warnings.catch_warnings():
            # A column absent from every run averages to NaN; that's expected.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            results[f"{prefix}_col_abs_dev"] = np.nanmean(
                col_ad, axis=0
            )  # (n_aligned,)
        results[f"{prefix}_within_species_scale"] = np.float64(wss.mean())
        results[f"{prefix}_mean_col_abs_dev"] = np.float64(mean_ad.mean())
        results[f"{prefix}_norm_loss"] = np.float64(nloss.mean())
        # Standard deviation of each scalar metric across the n_iter runs, and how
        # many of those standard deviations the control average lies from the
        # original value (the control-test z-statistic). The original scalars were
        # filled by the orientations loop above, so they are available here.
        for metric, dist in (
            ("mean_col_abs_dev", mean_ad),
            ("within_species_scale", wss),
            ("norm_loss", nloss),
        ):
            std = float(dist.std(ddof=1))
            orig = float(results[f"original_{metric}"])
            n_std = (
                (float(results[f"{prefix}_{metric}"]) - orig) / std
                if std
                else float("nan")
            )
            results[f"{prefix}_{metric}_std"] = np.float64(std)
            results[f"{prefix}_{metric}_n_std_from_original"] = np.float64(n_std)
        print(
            f"[{prefix}] average over {n_iter}: "
            f"mean_col_abs_dev={float(results[f'{prefix}_mean_col_abs_dev']):.6f}  "
            f"norm_loss={float(results[f'{prefix}_norm_loss']):.6f}"
        )

    # ── Save everything ──────────────────────────────────────────────────────
    np.savez(OUTPUT_PATH, **results)
    print(f"\nSaved control results to {os.path.normpath(OUTPUT_PATH)}")

    # ── Console summary: original vs the random & shuffled averages over 100 ──
    counts = {"random": N_RANDOM, "shuffled": N_SHUFFLE}
    print("\n=== Control summary (original vs. control averages) ===")
    for metric in ("mean_col_abs_dev", "within_species_scale", "norm_loss"):
        orig = float(results[f"original_{metric}"])
        print(f"\n{metric}: original = {orig:.6f}")
        for prefix in ("random", "shuffled"):
            avg = float(results[f"{prefix}_{metric}"])
            std = float(results[f"{prefix}_{metric}_std"])
            n_std = float(results[f"{prefix}_{metric}_n_std_from_original"])
            ratio = avg / orig if orig else float("nan")
            print(
                f"  {prefix:8s} average over {counts[prefix]:3d} = {avg:.6f}  "
                f"({ratio:.2f}x the original; run-to-run std={std:.6f}, "
                f"{n_std:+.1f} std from the original)"
            )

    print("\n=== Orientations ===")
    print(
        f"  {'orientation':20s} {'mean_col_abs_dev':>18s} "
        f"{'within_species_scale':>22s} {'norm_loss':>14s}"
    )
    for name in orientations:
        print(
            f"  {name:20s} "
            f"{float(results[f'{name}_mean_col_abs_dev']):>18.6f} "
            f"{float(results[f'{name}_within_species_scale']):>22.6f} "
            f"{float(results[f'{name}_norm_loss']):>14.6f}"
        )


if __name__ == "__main__":
    main()
