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
  * ``mean_activation``       — mean summed softplus over every (species, window),
                                the lambda-free quantity the L1 penalty scales.
  * ``l1``                    — ``L1_LAMBDA * mean_activation``, the training L1 term.
  * ``mean_abs_r``            — mean |Pearson r| over the 780 pairs of the 40 filter
                                channels, scored on every ``POOL_LENGTH``-mer of the
                                evaluated matrix (the lambda-free orthogonality term).
  * ``ortho``                 — ``ORTHO_LAMBDA * mean_abs_r``.
  * ``total_loss``            — ``norm_loss + ortho + l1``, the full training
                                objective of custom_real_train_orthogonality.py.

Every one of these is recomputed *on the matrix being evaluated*, so the random and
shuffled controls test all three terms rather than only the conservation numerator.
``mean_abs_r`` is the term that answers "is the low filter redundancy a property of
the data, or just of the penalty?": the penalty is applied identically whatever the
input, so a control matrix that scores an equally low mean |r| means the ortho term
learned nothing alignment-specific.

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

Everything is saved to ``control_results.npz`` and then rendered to
``control_summary.png`` by plot_control_results.py, which is where the comparisons
are read: at eight metrics across six conditions they are a figure, not a console
table. The npz carries a ``meta_json`` stamp of the run so that figure can identify
itself, and the plotter can be re-run against the npz alone to retune the figure
without repeating the ~2.6-minute scan.
"""

import json
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
WEIGHTS_PATH = os.path.join(CUSTOM_DIR, "weights", "real_weights_ortho.pt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "control_results.npz")
# Figure rendered from OUTPUT_PATH at the end of the run. The comparisons this
# script used to print as two console tables live here instead; plot_control_results.py
# can re-render it from the npz alone, without repeating the scan.
PLOT_PATH = os.path.join(os.path.dirname(__file__), "control_summary.png")

N_RANDOM = 100  # number of gap-preserving random matrices
N_SHUFFLE = 100  # number of aligned-row-shuffled matrices
MIN_SPECIES = 5  # a column counts only if >= this many species present (matches loss)
SMOOTH_SIGMA = 1.5  # Gaussian smoothing of the SR profile along position (matches loss)
SEED = 0

# ── Penalty terms (mirroring custom_real_train_orthogonality.py) ─────────────
# Defaults only: main() overrides both lambdas from the checkpoint's own metadata
# when it carries them, so the reported total_loss is the loss that checkpoint was
# actually trained under rather than whatever is hardcoded here.
L1_LAMBDA = 0.02  # strength of the L1 penalty on the summed softplus activations
ORTHO_LAMBDA = 0.5  # strength of the mean-|r| orthogonality penalty
POOL_LENGTH = 20  # nt each filter is scored on for the correlation (--pool-length 20)
POOL_WINDOWS = POOL_LENGTH - 5  # width of the box sum along the activation track
ORTHO_EPS = 1e-12  # variance floor, so a filter gone flat can't produce 0/0 in corr
NUM_FILTERS = 20  # per bank; incl + skip -> 40 correlated channels

# Every scalar recorded per matrix. Driving the results dict, the run-to-run std /
# n-std-from-original block and the console summary off one list keeps them in step.
SCALAR_METRICS = (
    "mean_col_abs_dev",
    "within_species_scale",
    "norm_loss",
    "mean_activation",
    "l1",
    "mean_abs_r",
    "ortho",
    "total_loss",
)

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


def compute_metric(
    matrix,
    model,
    gauss_kernel,
    smooth_radius,
    device,
    box_kernel=None,
    triu=None,
    l1_lambda=L1_LAMBDA,
    ortho_lambda=ORTHO_LAMBDA,
):
    """Every term of the training loss, evaluated on one matrix.

    Replicates the training loss over *all* species in ``matrix``: score each
    species' gap-removed sequence with ``compute_sr_profile``, Gaussian-smooth the
    SR track, scatter each window to the aligned column of its first nucleotide,
    then take the NaN-aware cross-species mean absolute deviation down each column
    (numerator) divided by the within-species dynamic range (denominator).

    The ``compute_sr_profile`` call is written out inline as its two softplus
    convolutions because the orthogonality term needs the 40 *per-filter* tracks,
    and the method returns only their sum over the filter axis. Summing the two
    expansions reproduces it exactly, so the SR numbers are unchanged.

    Both penalty terms are then accumulated over the same pass: the mean summed
    softplus (L1), and the sufficient statistics of the 40 pooled filter scores
    (orthogonality). Following the trainer, the correlation is taken over the whole
    matrix — every ``POOL_LENGTH``-mer of every species is one observation — so its
    running sum and Gram matrix are reduced once, after the species loop.

    Args:
        matrix: ``(n_aligned, n_species)`` array of single characters; gaps are
            ``-`` / ``N``, everything else (incl. lowercase) is a nucleotide.
        box_kernel: Precomputed ``(40, 1, POOL_WINDOWS)`` grouped-conv box kernel
            for the pooled filter scores. Built here when None.
        triu: Precomputed ``(2, 780)`` upper-triangle index pair for the 40x40
            correlation matrix. Built here when None.
        l1_lambda: Scales ``mean_activation`` into the reported ``l1``.
        ortho_lambda: Scales ``mean_abs_r`` into the reported ``ortho``.

    Returns:
        Dict with the length-``n_aligned`` ``"col_abs_dev"`` array (NaN at invalid
        columns) plus every scalar named in ``SCALAR_METRICS``.
    """
    # Same objects the trainer builds once outside its epoch loop. The box kernel is
    # deliberately *not* sum-normalized: the pooled score sums the softplus over the
    # windows inside a POOL_LENGTH-mer rather than averaging them.
    if box_kernel is None:
        box_kernel = torch.ones(2 * NUM_FILTERS, 1, POOL_WINDOWS, device=device)
    if triu is None:
        triu = torch.triu_indices(
            2 * NUM_FILTERS, 2 * NUM_FILTERS, offset=1, device=device
        )
    n_aligned, n_species = matrix.shape
    values = torch.zeros(n_species, n_aligned, device=device)
    mask = torch.zeros(n_species, n_aligned, device=device)

    # L1 accumulators: sequences differ in length, so track the running activation
    # sum and element count and divide once at the end (matches the trainer).
    act_sum = torch.zeros((), device=device)
    act_count = 0

    # Orthogonality accumulators, in float64: a pooled score is a sum of POOL_WINDOWS
    # non-negative softplus values, so its mean dwarfs its spread and
    # E[p^2] - E[p]^2 is a difference of large near-equal numbers. Only 40 and 40x40
    # wide, so double costs nothing.
    ortho_sum = torch.zeros(2 * NUM_FILTERS, dtype=torch.float64, device=device)
    ortho_gram = torch.zeros(
        2 * NUM_FILTERS, 2 * NUM_FILTERS, dtype=torch.float64, device=device
    )
    ortho_n = 0

    for j in range(n_species):
        row = matrix[:, j]
        is_nuc = (row != GAP_CHARS[0]) & (row != GAP_CHARS[1])
        nuc_aligned = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()
        if len(seq) < model.seq_kernel_size:
            continue  # too short to form even one window

        x_seq = torch.tensor(str_to_vector(seq), dtype=torch.float32, device=device)
        with torch.no_grad():
            # compute_sr_profile written out, so the 40 per-filter tracks survive for
            # the orthogonality term; summing over dim 1 is that method exactly.
            f_incl = F.softplus(model.conv_incl(x_seq.unsqueeze(0)))  # (1, 20, Lj-5)
            f_skip = F.softplus(model.conv_skip(x_seq.unsqueeze(0)))  # (1, 20, Lj-5)
            a_incl, a_skip = f_incl.sum(1), f_skip.sum(1)  # both (1, Lj-5), >= 0
            sr_j = (a_incl - a_skip).squeeze(0)  # (Lj-5,)

            # L1 term, over every (species, window) activation of this matrix.
            act_sum = act_sum + (a_incl + a_skip).sum()
            act_count += a_incl.numel()

            # Pooled POOL_LENGTH-mer scores for the orthogonality term, taken from the
            # *raw* (unsmoothed) activations — the pooled mode has no position axis
            # left to smooth. Summing softplus over the POOL_WINDOWS windows inside a
            # POOL_LENGTH-mer is a box sum along the activation track, so one grouped
            # conv scores every POOL_LENGTH-mer of this species without ever
            # materializing them.
            act = torch.cat([f_incl, f_skip], dim=1)  # (1, 40, Lj-5)
            if act.shape[-1] >= POOL_WINDOWS:  # else too short for one POOL_LENGTH-mer
                # -> (40, Lj-19): one score per filter per POOL_LENGTH-mer offset.
                pooled = F.conv1d(act, box_kernel, groups=2 * NUM_FILTERS).squeeze(0)
                p = pooled.double()
                ortho_sum = ortho_sum + p.sum(1)
                ortho_gram = ortho_gram + p @ p.T
                ortho_n += pooled.shape[-1]

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

    # L1 term: mean summed softplus over every (species, window) activation.
    mean_activation = float((act_sum / act_count).item()) if act_count else float("nan")

    # Orthogonality term: the accumulators hold sum(p) and sum(p p^T) over every
    # POOL_LENGTH-mer of the matrix, so the population covariance of the 40 pooled
    # filter scores follows in closed form. ddof is irrelevant — correlation
    # normalizes it away — which is why this matches the ddof=1 np.cov path in
    # filter_coactivation_covariance.py exactly rather than approximately.
    if ortho_n:
        mean_p = ortho_sum / ortho_n
        cov_p = ortho_gram / ortho_n - torch.outer(mean_p, mean_p)  # (40, 40)
        std_p = cov_p.diagonal().clamp(min=ORTHO_EPS).sqrt()  # floor: dead filters
        corr_p = cov_p / torch.outer(std_p, std_p)
        # abs(), not r^2 or signed r: a signed mean is minimized by driving
        # correlations negative, which is anti-correlation, not orthogonality.
        mean_abs_r = float(corr_p[triu[0], triu[1]].abs().mean().item())
    else:
        # No species long enough for even one POOL_LENGTH-mer.
        mean_abs_r = float("nan")

    # NaN-mask invalid columns for the saved per-column array.
    col_abs_dev_out = col_abs_dev.clone()
    col_abs_dev_out[~valid] = float("nan")

    l1 = l1_lambda * mean_activation
    ortho = ortho_lambda * mean_abs_r
    return {
        "col_abs_dev": col_abs_dev_out.cpu().numpy(),
        "within_species_scale": float(within_species_scale.item()),
        "mean_col_abs_dev": float(mean_col_abs_dev.item()),
        "norm_loss": float(norm_loss.item()),
        "mean_activation": mean_activation,
        "l1": l1,
        "mean_abs_r": mean_abs_r,
        "ortho": ortho,
        "total_loss": float(norm_loss.item()) + ortho + l1,
    }


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

    # Penalty strengths: prefer the ones this checkpoint was actually trained under,
    # so total_loss is its real objective. A checkpoint from a trainer without an
    # orthogonality term (e.g. real_weights_aad.pt) carries no "ortho_lambda", and
    # falls back to the module default — its mean_abs_r is still meaningful, but its
    # total_loss is then a loss that run never optimized.
    ckpt_hparams = (
        (ckpt.get("metadata") or {}).get("hparams", {})
        if isinstance(ckpt, dict)
        else {}
    )
    l1_lambda = float(ckpt_hparams.get("l1_lambda", L1_LAMBDA))
    ortho_lambda = float(ckpt_hparams.get("ortho_lambda", ORTHO_LAMBDA))
    print(
        f"Penalty strengths: l1_lambda={l1_lambda:g}, ortho_lambda={ortho_lambda:g} "
        f"({'from the checkpoint metadata' if ckpt_hparams else 'module defaults — checkpoint carries no hparams'})"
    )
    if "ortho_lambda" not in ckpt_hparams:
        print(
            "  NOTE: this checkpoint records no ortho_lambda, so total_loss below is "
            "not an objective it was trained under; read mean_abs_r on its own."
        )

    gauss_kernel, smooth_radius = build_smoothing_kernel(SMOOTH_SIGMA, device)
    if gauss_kernel is not None:
        print(
            f"SR-profile smoothing: Gaussian sigma={SMOOTH_SIGMA} nt, "
            f"kernel size={gauss_kernel.shape[-1]}"
        )

    # Built once and reused across all 204 evaluations, as in the trainer: the
    # grouped box kernel that turns the 40 activation tracks into pooled
    # POOL_LENGTH-mer scores, and the 780 unique off-diagonal pairs of the 40x40
    # correlation matrix (incl-incl, skip-skip and incl-skip alike).
    box_kernel = torch.ones(2 * NUM_FILTERS, 1, POOL_WINDOWS, device=device)
    triu = torch.triu_indices(2 * NUM_FILTERS, 2 * NUM_FILTERS, offset=1, device=device)
    print(
        f"Orthogonality term: mean |r| over the {triu.shape[1]} pairs of "
        f"{2 * NUM_FILTERS} filter channels, on every {POOL_LENGTH}-mer of the "
        f"evaluated matrix ({POOL_WINDOWS} windows summed per {POOL_LENGTH}-mer)"
    )

    def evaluate(mat):
        return compute_metric(
            mat,
            model,
            gauss_kernel,
            smooth_radius,
            device,
            box_kernel=box_kernel,
            triu=triu,
            l1_lambda=l1_lambda,
            ortho_lambda=ortho_lambda,
        )

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
        m = evaluate(mat)
        results[f"{name}_col_abs_dev"] = m["col_abs_dev"]
        for metric in SCALAR_METRICS:
            results[f"{name}_{metric}"] = np.float64(m[metric])
        print(
            f"[orientation] {name:20s} mean_col_abs_dev={m['mean_col_abs_dev']:.6f}  "
            f"norm_loss={m['norm_loss']:.6f}  mean|r|={m['mean_abs_r']:.4f}  "
            f"l1={m['l1']:.6f}  total_loss={m['total_loss']:.6f}"
        )

    # ── 2. 100 gap-preserving random matrices ────────────────────────────────
    # ── 3. 100 aligned-row-shuffled matrices ─────────────────────────────────
    control_specs = [
        ("random", N_RANDOM, gap_preserving_random),
        ("shuffled", N_SHUFFLE, shuffle_aligned_rows),
    ]
    for prefix, n_iter, make_matrix in control_specs:
        col_ad = np.empty((n_iter, n_aligned), dtype=np.float64)
        scalars = {
            metric: np.empty(n_iter, dtype=np.float64) for metric in SCALAR_METRICS
        }
        for i in range(n_iter):
            m = evaluate(make_matrix(matrix, rng))
            col_ad[i] = m["col_abs_dev"]
            for metric in SCALAR_METRICS:
                scalars[metric][i] = m[metric]
            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"[{prefix}] {i + 1}/{n_iter}  "
                    f"mean_col_abs_dev={scalars['mean_col_abs_dev'][i]:.6f}  "
                    f"mean|r|={scalars['mean_abs_r'][i]:.4f}"
                )
        # Collapse the n_iter runs to a single average — the individual per-matrix
        # values are nearly identical, so only the average over all runs is recorded:
        # scalar summaries plus the per-column deviation averaged across the runs.
        with warnings.catch_warnings():
            # A column absent from every run averages to NaN; that's expected.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            results[f"{prefix}_col_abs_dev"] = np.nanmean(
                col_ad, axis=0
            )  # (n_aligned,)
        # Standard deviation of each scalar metric across the n_iter runs, and how
        # many of those standard deviations the control average lies from the
        # original value (the control-test z-statistic). The original scalars were
        # filled by the orientations loop above, so they are available here.
        for metric in SCALAR_METRICS:
            dist = scalars[metric]
            results[f"{prefix}_{metric}"] = np.float64(dist.mean())
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
            f"norm_loss={float(results[f'{prefix}_norm_loss']):.6f}  "
            f"mean|r|={float(results[f'{prefix}_mean_abs_r']):.4f}  "
            f"total_loss={float(results[f'{prefix}_total_loss']):.6f}"
        )

    # ── Save everything ──────────────────────────────────────────────────────
    # Run identity, stamped into the npz so the figure rendered from it can say
    # which checkpoint and settings produced these numbers. Mirrors the
    # "hparams_json" the trainer writes into its own history npz.
    results["meta_json"] = json.dumps(
        {
            "weights_path": WEIGHTS_PATH,
            "min_species": MIN_SPECIES,
            "smooth_sigma": SMOOTH_SIGMA,
            "n_random": N_RANDOM,
            "n_shuffle": N_SHUFFLE,
            "l1_lambda": l1_lambda,
            "ortho_lambda": ortho_lambda,
            "pool_length": POOL_LENGTH,
        }
    )
    np.savez(OUTPUT_PATH, **results)
    print(f"\nSaved control results to {os.path.normpath(OUTPUT_PATH)}")

    # ── Render the comparisons ───────────────────────────────────────────────
    # Imported here rather than at module scope: cross_model_control.py imports this
    # module for its metric, and should not pick up matplotlib's Agg backend as a
    # side effect of that import.
    from plot_control_results import render  # noqa: E402

    render(results, PLOT_PATH)
    print(f"Saved control summary figure to {os.path.normpath(PLOT_PATH)}")


if __name__ == "__main__":
    main()
