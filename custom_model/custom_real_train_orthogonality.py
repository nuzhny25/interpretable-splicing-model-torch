"""Real cross-species training for the normalized abs-dev-minimization loss.

Loads the real MALAT1 multiz alignment matrix (``data/multiz100/alignment_matrix.npy``,
shape ``(n_aligned, n_species)`` of single characters with ``A/C/G/T`` plus
lowercase soft-masked ``a/c/g/t`` and the gap symbols ``-`` / ``N``). Each epoch
trains on a fresh random combination of 10 species drawn from the alignment.

For every sampled species the SR-balance profile is computed by the simplified
20+20-filter model on that species' gap-removed sequence (gaps are ``-`` and
``N``; lowercase letters are nucleotides). Each 6-nt sliding-window value is then
scattered back into alignment coordinates at the aligned column of the window's
*first* nucleotide. Aligned columns that are gaps in a species — and the trailing
nucleotides that never start a full window — are left absent (conceptually NaN).

This produces a ``(10, n_aligned)`` matrix with NaN at gaps; the loss is the
NaN-aware cross-species mean absolute deviation taken down each column, restricted
to columns where at least ``MIN_SPECIES`` of the 10 sampled species are present.

The per-column cross-species mean absolute deviation is divided by the
within-species dynamic range (mean over rows of each row's mean absolute deviation
across positions), mirroring the SD normalization in
filter_permutations/filter_permutations.py. This makes the loss scale-free:
shrinking all activations toward zero shrinks the numerator and denominator
equally, so the trivial "constant SR" collapse no longer lowers the loss.

The NaN matrix is implemented as a ``values`` + ``mask`` pair rather than literal
NaN, so gradients stay finite: masked entries contribute 0 and receive no
gradient, exactly matching a NaN-aware mean absolute deviation over the non-NaN
entries.

The second loss term is an **orthogonality penalty** on the filter bank, replacing
the activation L1 penalty this script's sibling
(custom_real_train_absolute_deviation.py) uses. The model combines its 20+20
filters purely additively, so nothing in the architecture stops several of them
from converging on the same motif; L1 buys activation sparsity but says nothing
about that redundancy. The penalty is the mean absolute Pearson correlation over
the 780 pairs of the 40 filter channels, measured exactly as
``filter_coactivation_covariance.py --pooled-space --pool-length 20 --weighting
malat1`` measures it: every POOL_LENGTH-nt window of the alignment is one
observation, and a filter's score on it is its softplus summed over the
POOL_LENGTH-5 sliding windows inside. Position-summing is translation-invariant
within the window, so two filters that learn the same motif at different registers
inside the 6-nt kernel are scored as the near-duplicates they are instead of as
unrelated (the 6-mer-window view scores that pair near zero).

That pooled score is just a width-(POOL_LENGTH-5) box sum along the per-window
activation track, so it is accumulated from the same convolution the SR profile
already needs — no 20-mer is ever materialized or re-convolved. Because the
correlation is over the whole alignment rather than one species, its running
sufficient statistics (sum and Gram matrix of the pooled scores) are accumulated
across the species loop and reduced once, after it.
"""

import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from custom_model import PNASModel
from custom_utils import str_to_vector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)

# ── Input / output paths ─────────────────────────────────────────────────────
# The three lines that change when this script is pointed at a different gene. They are
# grouped so a swap is one edit: the outputs must move with the input, or a control run
# silently overwrites the checkpoint and curves of the run it is meant to be compared
# against. The orthogonality penalty follows GENE_DIR too — it is always measured on the
# same alignment the model is being trained on. For the NEAT1 control:
#     GENE_DIR     = "neat1"
#     WEIGHTS_PATH = .../"real_weights_ortho_neat1.pt"
#     HISTORY_PATH = .../"real_weights_ortho_history_neat1.npz"
GENE_DIR = "multiz100"  # subdirectory of data/ holding alignment_matrix.npy (MALAT1)
MATRIX_PATH = os.path.join(_HERE, "..", "data", GENE_DIR, "alignment_matrix.npy")

# Trained checkpoint. Format matches train.py (weights nested under "model_state_dict")
# so it can be reloaded by plot_filters.py and PNASModel.load_partial_state_dict.
# Named for the orthogonality loss so this run cannot overwrite the L1-penalized
# real_weights_aad.pt it is meant to be compared against.
WEIGHTS_PATH = os.path.join(_HERE, "weights", "real_weights_ortho.pt")

# Per-epoch history of the loss components, written next to the weights so the curves
# travel with the checkpoint that produced them. plot/plot_training_curves.py reads this
# npz; without it a finished run keeps only its first-10-epoch average and final epoch,
# and nothing in between can be plotted after the fact.
HISTORY_PATH = os.path.join(_HERE, "weights", "real_weights_ortho_history.npz")

# ── Training hyperparameters ─────────────────────────────────────────────────
# N_SAMPLE is the full species count of the MALAT1 alignment (62): using every species
# every epoch keeps the all_species precompute path below, and makes the orthogonality
# penalty the whole-alignment correlation matrix that filter_coactivation_covariance.py
# reports rather than a per-epoch random subset of it.
N_SAMPLE = 62  # number of species sampled per epoch
MIN_SPECIES = 31  # a column counts toward the loss only if >= this many species present
NUM_EPOCHS = 500
LR = 1e-2
L1_LAMBDA = 0  # strength of L1 penalty on the softplus activations; 0 disables it
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)
SEED = 0

# ── Orthogonality penalty ────────────────────────────────────────────────────
# ORTHO_LAMBDA scales the mean |r| over the 780 filter pairs into the loss. Pooled mean
# |r| sits around 0.3 at initialization and sd_loss lands near 0.2-0.3, so 0.5 starts the
# two terms on comparable footing; retune by watching the logs.
ORTHO_LAMBDA = 0.5
# Sequence length each filter is scored on, matching --pool-length 20. Each such window
# contributes POOL_LENGTH - 5 sliding 6-nt activations, summed into one score per filter.
POOL_LENGTH = 20
POOL_WINDOWS = POOL_LENGTH - 5  # width of the box sum along the activation track
# Variance floor for the correlation denominator. A filter whose softplus has gone flat
# has zero variance, where both the 0/0 correlation and sqrt'(0) = inf would poison the
# gradient of every pair it appears in.
ORTHO_EPS = 1e-12
NUM_FILTERS = 20  # per bank; incl + skip -> 40 correlated channels

HISTORY_EVERY = 10  # recording cadence; also gates the log line below
FLUSH_EVERY = 500  # rewrite the npz this often so a killed run still has its curves

NUCLEOTIDES = ["A", "C", "G", "T"]


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load the real alignment matrix and precompute per-species inputs ──
    # matrix: (n_aligned, n_species) of single characters. Gaps are "-" and "N";
    # everything else (including lowercase soft-masked a/c/g/t) is a nucleotide.
    matrix = np.load(MATRIX_PATH)
    n_aligned, n_total = matrix.shape

    # For each species: the gap-removed one-hot sequence and, for every nucleotide,
    # the aligned column it occupies. species_cols[j][k] is the aligned column of
    # the k-th nucleotide = the first nucleotide of sliding window k.
    species_oh, species_cols = [], []
    for j in range(n_total):
        row = matrix[:, j]
        is_nuc = (row != "-") & (row != "N")
        nuc_aligned = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()
        oh = torch.tensor(
            str_to_vector(seq), dtype=torch.float32, device=device
        )  # (4, Lj)
        species_oh.append(oh.unsqueeze(0))  # (1, 4, Lj)
        species_cols.append(torch.tensor(nuc_aligned, dtype=torch.long, device=device))

    logger.info(
        f"Real alignment matrix: {matrix.shape} (aligned positions x species) "
        f"from {MATRIX_PATH} on {device} — sampling {N_SAMPLE} species/epoch, "
        f"column valid when >= {MIN_SPECIES} species present"
    )

    model = PNASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once.
    # Sum-normalized so it's a weighted average (preserves overall scale). It carries
    # no learnable parameters and is applied with reflect padding at the use site so
    # the conserved block at position 0 isn't damped by zero edges. Smoothing denoises
    # single-position jitter so the conserved-motif signal dominates the column SD.
    # It is applied to each species' dense (gap-free) SR track before scattering, so
    # no NaNs ever enter the convolution.
    if SMOOTH_SIGMA:
        smooth_radius = math.ceil(3 * SMOOTH_SIGMA)  # kernel half-width (~3 sigma)
        offsets = torch.arange(
            -smooth_radius, smooth_radius + 1, device=device, dtype=torch.float32
        )
        gauss_kernel = torch.exp(-(offsets**2) / (2 * SMOOTH_SIGMA**2))
        gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)  # (1, 1, K)
        logger.info(
            f"SR-profile smoothing: Gaussian sigma={SMOOTH_SIGMA} nt, "
            f"kernel size={gauss_kernel.shape[-1]}"
        )

    # ── Orthogonality-penalty scaffolding, built once ────────────────────────
    # Box kernel of width POOL_WINDOWS, one copy per filter channel and applied with
    # groups=, so a single grouped conv1d turns the 40 per-window activation tracks into
    # the 40 pooled POOL_LENGTH-nt scores at every offset. Unlike the Gaussian above it
    # is *not* sum-normalized: filter_coactivation_covariance.py sums the softplus over
    # the windows inside a pooled sequence rather than averaging them. (Correlation is
    # scale-free, so this only matters for reading the raw pooled magnitudes.)
    box_kernel = torch.ones(2 * NUM_FILTERS, 1, POOL_WINDOWS, device=device)
    # The 780 unique off-diagonal pairs of the 40x40 matrix — incl-incl, skip-skip and
    # incl-skip alike, matching the full matrix the analysis script reports. Cross-bank
    # pairs belong in the penalty: the two banks are subtracted from one another, so a
    # correlated incl/skip pair is two filters cancelling instead of dividing the work.
    triu = torch.triu_indices(2 * NUM_FILTERS, 2 * NUM_FILTERS, offset=1, device=device)
    logger.info(
        f"Orthogonality penalty: mean |r| over the {triu.shape[1]} pairs of "
        f"{2 * NUM_FILTERS} filter channels, on every {POOL_LENGTH}-mer of the alignment "
        f"({POOL_WINDOWS} windows summed per {POOL_LENGTH}-mer), lambda={ORTHO_LAMBDA:g}"
    )

    # Running sums of the per-epoch loss statistics over the first 10 epochs, so the
    # metadata can report the early-training average (a snapshot of the from-scratch
    # starting regime) alongside the final-epoch values.
    first10_sums = {
        "loss": 0.0,
        "sd_loss": 0.0,
        "ortho": 0.0,
        "mean_abs_r": 0.0,
        "l1": 0.0,
        "raw_mean_col_sd": 0.0,
        "within_species_scale": 0.0,
    }
    n_first10 = 0

    # Per-epoch history of every scalar the log line below reports, recorded on the same
    # HISTORY_EVERY cadence. All of these are already computed for the log, so keeping
    # them is free; retaining them is what lets the component curves be plotted after the
    # run instead of being inferred from the two endpoints in the metadata.
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    history = {
        k: []
        for k in (
            "epoch",
            "loss",
            "sd_loss",
            "ortho",
            "mean_abs_r",
            "l1",
            "raw_mean_col_sd",
            "within_species_scale",
            "mean_abs_w",
            "valid_cols",
        )
    }

    # ── Constant per-epoch structure when every species is used ──
    # With N_SAMPLE == n_total the per-epoch randperm only permutes the rows of the
    # full species set, and every term of the loss is invariant to row order. So the
    # presence mask — and everything derived from it: the per-column species counts,
    # the valid-column set, the per-row window counts, and the total number of
    # (species, window) activations — is identical in every epoch. Build it once here
    # so the loop below stops re-scattering the mask N_SAMPLE rows at a time and
    # re-reducing it over 15k columns. With N_SAMPLE < n_total the sampled set really
    # does change every epoch, so the loop keeps its original per-epoch path.
    all_species = N_SAMPLE == n_total
    if all_species:
        fixed_sel = list(range(n_total))
        # Columns that receive a value for each species: a length-Lj sequence has
        # Lj-5 six-nt windows, so the trailing 5 nucleotides start no window and are
        # absent from the mask (same truncation the scatter below applies).
        fixed_win_cols = [
            species_cols[j][: species_oh[j].shape[-1] - 5] for j in fixed_sel
        ]
        fixed_mask = torch.zeros(N_SAMPLE, n_aligned, device=device)
        for r, cols_r in enumerate(fixed_win_cols):
            fixed_mask[r, cols_r] = 1.0
        fixed_count = fixed_mask.sum(0)  # (n_aligned,) species present per column
        fixed_valid = fixed_count >= MIN_SPECIES
        fixed_rc = fixed_mask.sum(1)  # (N_SAMPLE,) windows per species
        fixed_act_count = sum(c.numel() for c in fixed_win_cols)
        logger.info(
            f"All {n_total} species used every epoch — mask, column counts and valid "
            f"columns precomputed once: {int(fixed_valid.sum().item())} valid columns, "
            f"{fixed_act_count} (species, window) activations"
        )

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()

        # Fresh random combination of N_SAMPLE species for this epoch (the full set,
        # in fixed order, when all_species — the draw would be a no-op permutation).
        sel = fixed_sel if all_species else torch.randperm(n_total)[:N_SAMPLE].tolist()

        # Build the aligned (N_SAMPLE, n_aligned) matrix of SR values. We never put
        # literal NaN into the differentiable tensor; `mask` marks present entries
        # (== non-NaN). Masked-out entries stay 0 and receive no gradient.
        values = torch.zeros(N_SAMPLE, n_aligned, device=device)
        mask = (
            fixed_mask
            if all_species
            else torch.zeros(N_SAMPLE, n_aligned, device=device)
        )

        # Accumulate the summed softplus activations over all sampled species so the
        # L1 penalty below matches custom_synthetic_train.py: a mean over every
        # (species, window) activation. Sequences differ in length, so we track the
        # running activation sum and element count and divide once at the end.
        act_sum = torch.zeros((), device=device)
        act_count = fixed_act_count if all_species else 0

        # Sufficient statistics for the orthogonality penalty's correlation matrix,
        # accumulated over every species so the correlation is over the whole alignment
        # (one observation per POOL_LENGTH-mer) rather than per species. float64: the
        # pooled scores are sums of POOL_WINDOWS non-negative softplus values, so their
        # mean dwarfs their spread and E[p^2] - E[p]^2 is a difference of large near-equal
        # numbers — float32 would spend most of its precision in that cancellation. The
        # accumulators are only 40 and 40x40 wide, so double costs nothing here.
        ortho_sum = torch.zeros(2 * NUM_FILTERS, dtype=torch.float64, device=device)
        ortho_gram = torch.zeros(
            2 * NUM_FILTERS, 2 * NUM_FILTERS, dtype=torch.float64, device=device
        )
        ortho_n = 0
        for r, j in enumerate(sel):
            # Per-filter softplus activations, kept un-summed so the orthogonality
            # penalty below can correlate the 40 channels against each other; summing
            # over the filter axis reproduces PNASModel.compute_sr_profile exactly (it
            # returns only that sum, so the breakdown is not recoverable from it).
            # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
            # both (1, Lj-5) and >= 0; used by the activation L1 penalty below.
            f_incl = F.softplus(model.conv_incl(species_oh[j]))  # (1, 20, Lj-5), >= 0
            f_skip = F.softplus(model.conv_skip(species_oh[j]))  # (1, 20, Lj-5), >= 0
            a_incl, a_skip = f_incl.sum(1), f_skip.sum(1)  # both (1, Lj-5)
            sr_j = (a_incl - a_skip).squeeze(0)  # (Lj-5,)
            act_sum = act_sum + (a_incl + a_skip).sum()
            if not all_species:
                act_count += a_incl.numel()

            # Pooled POOL_LENGTH-mer scores for the orthogonality penalty, taken from the
            # *raw* activations (the pooled mode of filter_coactivation_covariance.py has
            # no position axis left to smooth, so it disables smoothing). Summing softplus
            # over the POOL_WINDOWS windows inside a POOL_LENGTH-mer is a box sum along
            # the activation track, so every POOL_LENGTH-mer of this species is scored by
            # one grouped conv over the track we already have — the 20-mers themselves are
            # never materialized or re-convolved.
            act = torch.cat([f_incl, f_skip], dim=1)  # (1, 40, Lj-5)
            if act.shape[-1] >= POOL_WINDOWS:  # else too short for one POOL_LENGTH-mer
                # -> (40, Lj-19): one score per filter per POOL_LENGTH-mer offset.
                pooled = F.conv1d(act, box_kernel, groups=2 * NUM_FILTERS).squeeze(0)
                p = pooled.double()
                ortho_sum = ortho_sum + p.sum(1)
                ortho_gram = ortho_gram + p @ p.T
                ortho_n += pooled.shape[-1]

            # Gaussian-smooth the dense SR track along position (gap-free, no NaN).
            # Reflect-pad by the kernel half-width, then conv1d with the single-channel
            # kernel. Differentiable, so gradients flow back to the conv filters.
            if SMOOTH_SIGMA:
                sr_j = F.conv1d(
                    F.pad(
                        sr_j.view(1, 1, -1),
                        (smooth_radius, smooth_radius),
                        mode="reflect",
                    ),
                    gauss_kernel,
                ).view(-1)

            # Place each window's value at the aligned column of its first nucleotide.
            cols = species_cols[j][: sr_j.shape[0]]
            values[r, cols] = sr_j
            if not all_species:
                mask[r, cols] = 1.0

        # Numerator: NaN-aware cross-species mean absolute deviation per column
        # (about each column's cross-species mean), valid only where at least
        # MIN_SPECIES of the sampled species are present. Masked-out entries are
        # zeroed *after* the abs() so they contribute 0 and receive no gradient.
        count = fixed_count if all_species else mask.sum(0)  # (n_aligned,)
        col_mean = (values * mask).sum(0) / count.clamp(min=1)
        col_std = ((values - col_mean).abs() * mask).sum(0) / count.clamp(min=1)
        valid = fixed_valid if all_species else count >= MIN_SPECIES

        # Denominator: within-species dynamic range = each row's mean absolute
        # deviation across positions (NaN-aware, about that row's own mean), averaged
        # over rows. Dividing by it makes the loss scale-free (matches
        # filter_permutations/filter_permutations.py), so the model can't cheat to a
        # low abs-dev by shrinking all activations toward a constant.
        rc = fixed_rc if all_species else mask.sum(1)  # (N_SAMPLE,)
        row_mean = (values * mask).sum(1) / rc.clamp(min=1)
        within_species_scale = (
            ((values - row_mean.unsqueeze(1)).abs() * mask).sum(1) / rc.clamp(min=1)
        ).mean()  # scalar

        sd_loss = (
            (col_std[valid]) / within_species_scale
        ).mean()  # normalized average column abs-dev

        # ── Orthogonality penalty: mean |r| over the 780 filter pairs ────────
        # The Gram accumulators above hold sum(p) and sum(p p^T) over every
        # POOL_LENGTH-mer of the alignment, so the population (ddof=0) covariance of the
        # 40 pooled filter scores follows in closed form. ddof does not matter here:
        # correlation normalizes it away, which is why this matches the analysis script's
        # np.cov (ddof=1) path exactly rather than approximately.
        mean_p = ortho_sum / ortho_n
        cov_p = ortho_gram / ortho_n - torch.outer(mean_p, mean_p)  # (40, 40)
        std_p = cov_p.diagonal().clamp(min=ORTHO_EPS).sqrt()  # floor: dead filters
        corr_p = cov_p / torch.outer(std_p, std_p)
        # abs(), not r^2 or signed r: a signed mean is minimized by driving correlations
        # negative rather than to zero, which is anti-correlation, not orthogonality.
        mean_abs_r = corr_p[triu[0], triu[1]].abs().mean()
        ortho_penalty = ORTHO_LAMBDA * mean_abs_r.to(sd_loss.dtype)

        # L1 penalty on the summed softplus activations (activity sparsity -> peaked,
        # data-driven SR profiles). a_incl/a_skip are already >= 0, so no abs() is
        # needed; dividing by the total element count keeps the penalty independent of
        # seq length / number of species (matches custom_synthetic_train.py). Disabled
        # by default here (L1_LAMBDA = 0) in favour of the orthogonality penalty; keep an
        # eye on mean|conv_w| in the log, since both remaining terms are scale-free and
        # nothing else anchors the overall activation magnitude.
        l1_penalty = L1_LAMBDA * (act_sum / act_count)
        loss = sd_loss + ortho_penalty + l1_penalty
        loss.backward()
        optimizer.step()

        # Accumulate the first-10-epoch statistics for the metadata average below.
        if epoch <= 10:
            first10_sums["loss"] += loss.item()
            first10_sums["sd_loss"] += sd_loss.item()
            first10_sums["ortho"] += ortho_penalty.item()
            first10_sums["mean_abs_r"] += mean_abs_r.item()
            first10_sums["l1"] += l1_penalty.item()
            first10_sums["raw_mean_col_sd"] += col_std[valid].mean().item()
            first10_sums["within_species_scale"] += within_species_scale.item()
            n_first10 += 1

        if epoch == 1 or epoch % HISTORY_EVERY == 0:
            mean_abs_w = (
                model.conv_incl.weight.abs().mean().item()
                + model.conv_skip.weight.abs().mean().item()
            ) / 2
            logger.info(
                f"epoch {epoch:4d} | loss = {loss.item():.6f} "
                f"| sd loss = {sd_loss.item():.6f} "
                f"| ortho = {ortho_penalty.item():.6f} "
                f"| mean|r| = {mean_abs_r.item():.4f} "
                f"| l1 = {l1_penalty.item():.6f} "
                f"| raw mean col abs-dev = {col_std[valid].mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| valid cols = {int(valid.sum().item())} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

            # Same numbers as the log line, retained for plot_training_curves.py.
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["sd_loss"].append(sd_loss.item())
            history["ortho"].append(ortho_penalty.item())
            history["mean_abs_r"].append(mean_abs_r.item())
            # "l1" is the other term of the loss sum, still recorded under its own name:
            # plot/plot_training_curves.py hardcodes this key, so dropping it would break
            # the plotter on this run's npz. At L1_LAMBDA = 0 it is a flat zero band.
            history["l1"].append(l1_penalty.item())
            history["raw_mean_col_sd"].append(col_std[valid].mean().item())
            history["within_species_scale"].append(within_species_scale.item())
            history["mean_abs_w"].append(mean_abs_w)
            history["valid_cols"].append(int(valid.sum().item()))

            # Periodic flush: a run killed part-way still leaves plottable curves.
            if epoch % FLUSH_EVERY == 0:
                np.savez(HISTORY_PATH, **{k: np.asarray(v) for k, v in history.items()})

    # ── Run metadata for the plotter/sidecar: dataset identity (real alignment, so
    # no planted motif), the source matrix, the hyperparameters, and the final-epoch
    # loss values. The "final" numbers are read from the loop variables still in scope
    # after the loop (their last-iteration values), so nothing in the training loop
    # above changes; raw_mean_col_sd is over valid columns to match the training log. ──
    # Average of each statistic over the first min(10, NUM_EPOCHS) epochs.
    first_10_epochs_avg = {k: v / n_first10 for k, v in first10_sums.items()}
    metadata = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "motif": None,
        "matrix_path": MATRIX_PATH,
        "hparams": {
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "ortho_lambda": ORTHO_LAMBDA,
            "pool_length": POOL_LENGTH,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "seed": SEED,
            "n_sample": N_SAMPLE,
            "min_species": MIN_SPECIES,
        },
        "final": {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "ortho": float(ortho_penalty.item()),
            "mean_abs_r": float(mean_abs_r.item()),
            "l1": float(l1_penalty.item()),
            "raw_mean_col_sd": float(col_std[valid].mean().item()),
            "within_species_scale": float(within_species_scale.item()),
        },
        "first_10_epochs_avg": first_10_epochs_avg,
        # Where the per-epoch component curves for this run live, and at what cadence
        # they were sampled, so a checkpoint points at its own training history.
        "history_path": HISTORY_PATH,
        "history_every": HISTORY_EVERY,
    }

    # ── Final history write: the same arrays as the periodic flush plus the run's
    # hyperparameters, so plot_training_curves.py can stamp the figure from the npz
    # alone without also loading the checkpoint. ──
    np.savez(
        HISTORY_PATH,
        hparams_json=json.dumps(metadata["hparams"]),
        **{k: np.asarray(v) for k, v in history.items()},
    )
    logger.info(
        f"Saved {len(history['epoch'])} recorded epochs of loss components to {HISTORY_PATH}"
    )

    # ── Save the trained weights to WEIGHTS_PATH ──
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        WEIGHTS_PATH,
    )
    logger.info(f"Saved model weights to {WEIGHTS_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
