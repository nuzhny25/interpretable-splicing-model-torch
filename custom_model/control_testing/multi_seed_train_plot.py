"""Multi-seed cross-species orthogonality training + combined filter readout.

Combines the real cross-species orthogonality trainer
(``custom_model/custom_real_train_orthogonality.py``) with the filter plotter
(``custom_model/plot_filters.py``): it trains a *fresh* ``PNASModel`` from
scratch on each of several random seeds (default 0,1,2,3 — seed 0 is the
original run) and renders **one combined figure per readout** so all runs can be
compared side by side:

  * ``combined_filters_heatmap.png`` — every INCL/SKIP filter of every seed as a
    4x6 weight heat map, seeds stacked vertically (one labelled block per seed,
    shared symmetric color scale so panels are comparable across seeds).
  * ``combined_filter_logos.png``    — each seed's *significant* filters as
    information-content (bits) sequence logos, one labelled row-block per seed.

The training math (all-species SR scatter into alignment coordinates, NaN-aware
cross-species mean-absolute-deviation numerator normalized by the within-species
dynamic range, the pooled-20-mer orthogonality penalty on the 780 filter pairs,
L1 activation penalty, Gaussian SR smoothing) is identical to
``custom_real_train_orthogonality.py``; only the seed set, epoch count, and the
two logo thresholds are exposed as CLI args. Each seed's weights are also saved
to a seed-suffixed checkpoint next to the figures, with the per-epoch component
curves in a matching ``_history_seed{S}.npz``, without touching the canonical
``weights/real_weights_ortho.pt``.

Note that the orthogonality trainer uses every species of the alignment every
epoch (``N_SAMPLE = 62``) rather than a 10-species sample, so a seed is
substantially more expensive per epoch than the old absolute-deviation run —
budget accordingly before raising ``--num-epochs`` or the seed count.

Run (defaults: seeds 0-3, 1000 epochs each — this trains four models, so it is
~4x the single-run cost):

    python multi_seed_train_plot.py
    python multi_seed_train_plot.py --seeds 0 1 2 3 --num-epochs 1000
"""

import argparse
import itertools
import json
import logging
import math
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import PNASModel / utils the same way control_sr_deviation.py does: put the
# custom_model/ directory first on sys.path so the regular module
# custom_model/custom_model.py wins over the custom_model/ namespace package.
CUSTOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CUSTOM_DIR)
from custom_model import PNASModel  # noqa: E402
from custom_utils import DNA_ALPHABET, one_hot_batch, str_to_vector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)

# ── Training hyperparameters (identical to custom_real_train_orthogonality.py
# so each seed's run is faithful) ─────────────────────────────────────────────
# GENE_DIR is the one line that changes when this is pointed at a different gene; the
# orthogonality penalty follows it, since it is always measured on the same alignment
# the model is being trained on.
GENE_DIR = "multiz100"  # subdirectory of data/ holding alignment_matrix.npy (MALAT1)
MATRIX_PATH = os.path.join(CUSTOM_DIR, "..", "data", GENE_DIR, "alignment_matrix.npy")
# N_SAMPLE is the full species count of the MALAT1 alignment (62): using every species
# every epoch keeps the all-species precompute path below, and makes the orthogonality
# penalty the whole-alignment correlation matrix that filter_coactivation_covariance.py
# reports rather than a per-epoch random subset of it.
N_SAMPLE = 62  # number of species sampled per epoch
MIN_SPECIES = 31  # a column counts toward the loss only if >= this many species present
LR = 1e-2
L1_LAMBDA = 0.02  # strength of L1 penalty on the softplus activations; 0 disables it
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)

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

HISTORY_EVERY = 10  # per-seed curve recording cadence (epochs), as in the trainer
FLUSH_EVERY = 500  # rewrite a seed's npz this often so a killed run keeps its curves
LOG_EVERY = 100  # console log cadence (epochs); larger than the trainer's 10 to keep
# the combined multi-seed log readable

NUCLEOTIDES = ["A", "C", "G", "T"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Random seeds to train (one independent model each). Default 0 1 2 3 "
        "(seed 0 is the original run).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Training epochs per seed (default 1000, matching the trainer).",
    )
    parser.add_argument(
        "--out-heatmaps",
        default=os.path.join(_HERE, "combined_filters_heatmap.png"),
        help="Output path for the combined (seed-tiled) filter heat-map figure.",
    )
    parser.add_argument(
        "--out-logos",
        default=os.path.join(_HERE, "combined_filter_logos.png"),
        help="Output path for the combined (seed-tiled) information-content logo figure.",
    )
    parser.add_argument(
        "--min-contribution-frac",
        type=float,
        default=0.2,
        help="Per seed, only render logos for filters whose peak softplus activation is "
        "at least this fraction of that seed's strongest filter's peak (default 0.2). "
        "Use 0 to show every filter.",
    )
    parser.add_argument(
        "--logo-activation-frac",
        type=float,
        default=0.5,
        help="Per filter, build its logo from every 6-mer whose softplus activation is at "
        "least this fraction of that filter's own peak activation (default 0.75).",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Output path for the run-metadata sidecar JSON. Defaults to "
        "combined_plot_metadata.json next to --out-logos.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── One-time, seed-independent setup ─────────────────────────────────────
    # The alignment and every species' gap-removed input encoding are the same
    # for all seeds, so build them once and reuse across the seed loop.
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
        f"from {os.path.normpath(MATRIX_PATH)} on {device} — sampling {N_SAMPLE} "
        f"species/epoch, column valid when >= {MIN_SPECIES} species present"
    )
    logger.info(
        f"Training {len(args.seeds)} seed(s) {args.seeds} for {args.num_epochs} epochs each"
    )

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once
    # (it carries no learnable parameters and is identical across seeds). Sum-
    # normalized so it is a weighted average; applied with reflect padding at the
    # use site so the conserved block at position 0 isn't damped by zero edges.
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

    # ── Orthogonality-penalty scaffolding, built once (seed-independent) ─────
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

    # ── Constant per-epoch structure when every species is used ──
    # With N_SAMPLE == n_total the per-epoch randperm only permutes the rows of the
    # full species set, and every term of the loss is invariant to row order. So the
    # presence mask — and everything derived from it: the per-column species counts,
    # the valid-column set, the per-row window counts, and the total number of
    # (species, window) activations — is identical in every epoch *and in every seed*.
    # Build it once here so the loop below stops re-scattering the mask N_SAMPLE rows
    # at a time and re-reducing it over 15k columns. With N_SAMPLE < n_total the
    # sampled set really does change every epoch, so the loop keeps its original
    # per-epoch path.
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

    # ── Seed loop: train one fresh model per seed (training logic kept inline) ──
    trained = []  # list of {"seed", "model", "final"} for the plotting passes below
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = PNASModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Running sums of the per-epoch loss statistics over the first 10 epochs, so
        # the metadata can report the early-training average (a stable snapshot of the
        # from-scratch starting regime) alongside the final-epoch values.
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

        # Per-epoch history of every scalar the trainer's log line reports, recorded on
        # the HISTORY_EVERY cadence and written to a seed-suffixed npz, so each seed's
        # component curves can be plotted after the fact by plot/plot_training_curves.py
        # instead of being inferred from the two endpoints in the metadata.
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
        history_path = os.path.join(_HERE, f"real_weights_aad_history_seed{seed}.npz")

        for epoch in range(1, args.num_epochs + 1):
            optimizer.zero_grad()

            # Fresh random combination of N_SAMPLE species for this epoch (the full
            # set, in fixed order, when all_species — the draw would be a no-op
            # permutation).
            sel = (
                fixed_sel
                if all_species
                else torch.randperm(n_total)[:N_SAMPLE].tolist()
            )

            # Build the aligned (N_SAMPLE, n_aligned) matrix of SR values. We never
            # put literal NaN into the differentiable tensor; `mask` marks present
            # entries. Masked-out entries stay 0 and receive no gradient.
            values = torch.zeros(N_SAMPLE, n_aligned, device=device)
            mask = (
                fixed_mask
                if all_species
                else torch.zeros(N_SAMPLE, n_aligned, device=device)
            )

            # Accumulate the summed softplus activations over all sampled species so
            # the L1 penalty is a mean over every (species, window) activation.
            act_sum = torch.zeros((), device=device)
            act_count = fixed_act_count if all_species else 0

            # Sufficient statistics for the orthogonality penalty's correlation matrix,
            # accumulated over every species so the correlation is over the whole
            # alignment (one observation per POOL_LENGTH-mer) rather than per species.
            # float64: the pooled scores are sums of POOL_WINDOWS non-negative softplus
            # values, so their mean dwarfs their spread and E[p^2] - E[p]^2 is a
            # difference of large near-equal numbers — float32 would spend most of its
            # precision in that cancellation. The accumulators are only 40 and 40x40
            # wide, so double costs nothing here.
            ortho_sum = torch.zeros(2 * NUM_FILTERS, dtype=torch.float64, device=device)
            ortho_gram = torch.zeros(
                2 * NUM_FILTERS, 2 * NUM_FILTERS, dtype=torch.float64, device=device
            )
            ortho_n = 0
            for r, j in enumerate(sel):
                # Per-filter softplus activations, kept un-summed so the orthogonality
                # penalty below can correlate the 40 channels against each other;
                # summing over the filter axis reproduces PNASModel.compute_sr_profile
                # exactly (it returns only that sum, so the breakdown is not recoverable
                # from it). a_incl / a_skip are the raw (unsmoothed) summed softplus
                # activations, both (1, Lj-5) and >= 0; used by the L1 penalty below.
                f_incl = F.softplus(model.conv_incl(species_oh[j]))  # (1,20,Lj-5), >= 0
                f_skip = F.softplus(model.conv_skip(species_oh[j]))  # (1,20,Lj-5), >= 0
                a_incl, a_skip = f_incl.sum(1), f_skip.sum(1)  # both (1, Lj-5)
                sr_j = (a_incl - a_skip).squeeze(0)  # (Lj-5,)
                act_sum = act_sum + (a_incl + a_skip).sum()
                if not all_species:
                    act_count += a_incl.numel()

                # Pooled POOL_LENGTH-mer scores for the orthogonality penalty, taken
                # from the *raw* activations (the pooled mode of
                # filter_coactivation_covariance.py has no position axis left to smooth,
                # so it disables smoothing). Summing softplus over the POOL_WINDOWS
                # windows inside a POOL_LENGTH-mer is a box sum along the activation
                # track, so every POOL_LENGTH-mer of this species is scored by one
                # grouped conv over the track we already have — the 20-mers themselves
                # are never materialized or re-convolved.
                act = torch.cat([f_incl, f_skip], dim=1)  # (1, 40, Lj-5)
                if act.shape[-1] >= POOL_WINDOWS:  # else too short for one 20-mer
                    # -> (40, Lj-19): one score per filter per POOL_LENGTH-mer offset.
                    pooled = F.conv1d(act, box_kernel, groups=2 * NUM_FILTERS).squeeze(
                        0
                    )
                    p = pooled.double()
                    ortho_sum = ortho_sum + p.sum(1)
                    ortho_gram = ortho_gram + p @ p.T
                    ortho_n += pooled.shape[-1]

                # Gaussian-smooth the dense SR track along position (gap-free, no NaN).
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
            # MIN_SPECIES of the sampled species are present.
            count = fixed_count if all_species else mask.sum(0)  # (n_aligned,)
            col_mean = (values * mask).sum(0) / count.clamp(min=1)
            col_std = ((values - col_mean).abs() * mask).sum(0) / count.clamp(min=1)
            valid = fixed_valid if all_species else count >= MIN_SPECIES

            # Denominator: within-species dynamic range = each row's mean absolute
            # deviation across positions (NaN-aware), averaged over rows. Dividing by
            # it makes the loss scale-free so the model can't cheat to a low abs-dev
            # by shrinking all activations toward a constant.
            rc = fixed_rc if all_species else mask.sum(1)  # (N_SAMPLE,)
            row_mean = (values * mask).sum(1) / rc.clamp(min=1)
            within_species_scale = (
                ((values - row_mean.unsqueeze(1)).abs() * mask).sum(1) / rc.clamp(min=1)
            ).mean()  # scalar

            sd_loss = (
                (col_std[valid]) / within_species_scale
            ).mean()  # normalized average column abs-dev

            # ── Orthogonality penalty: mean |r| over the 780 filter pairs ────
            # The Gram accumulators above hold sum(p) and sum(p p^T) over every
            # POOL_LENGTH-mer of the alignment, so the population (ddof=0) covariance of
            # the 40 pooled filter scores follows in closed form. ddof does not matter
            # here: correlation normalizes it away, which is why this matches the
            # analysis script's np.cov (ddof=1) path exactly rather than approximately.
            mean_p = ortho_sum / ortho_n
            cov_p = ortho_gram / ortho_n - torch.outer(mean_p, mean_p)  # (40, 40)
            std_p = cov_p.diagonal().clamp(min=ORTHO_EPS).sqrt()  # floor: dead filters
            corr_p = cov_p / torch.outer(std_p, std_p)
            # abs(), not r^2 or signed r: a signed mean is minimized by driving
            # correlations negative rather than to zero, which is anti-correlation, not
            # orthogonality.
            mean_abs_r = corr_p[triu[0], triu[1]].abs().mean()
            ortho_penalty = ORTHO_LAMBDA * mean_abs_r.to(sd_loss.dtype)

            # L1 penalty on the summed softplus activations (activity sparsity ->
            # peaked, data-driven SR profiles). a_incl/a_skip are already >= 0.
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

            # Curves are recorded on the trainer's HISTORY_EVERY cadence; the console
            # line is throttled to the coarser LOG_EVERY so the multi-seed log stays
            # readable. mean_abs_w is computed once for whichever of the two fires.
            log_now = epoch == 1 or epoch % LOG_EVERY == 0
            record_now = epoch == 1 or epoch % HISTORY_EVERY == 0
            if log_now or record_now:
                mean_abs_w = (
                    model.conv_incl.weight.abs().mean().item()
                    + model.conv_skip.weight.abs().mean().item()
                ) / 2

            if log_now:
                logger.info(
                    f"[seed {seed}] epoch {epoch:5d} | loss = {loss.item():.6f} "
                    f"| sd loss = {sd_loss.item():.6f} "
                    f"| ortho = {ortho_penalty.item():.6f} "
                    f"| mean|r| = {mean_abs_r.item():.4f} "
                    f"| l1 = {l1_penalty.item():.6f} "
                    f"| raw mean col abs-dev = {col_std[valid].mean().item():.6f} "
                    f"| within-species scale = {within_species_scale.item():.6f} "
                    f"| valid cols = {int(valid.sum().item())} "
                    f"| mean|conv_w| = {mean_abs_w:.6f}"
                )

            if record_now:
                # Same numbers as the trainer's log line, retained for
                # plot/plot_training_curves.py.
                history["epoch"].append(epoch)
                history["loss"].append(loss.item())
                history["sd_loss"].append(sd_loss.item())
                history["ortho"].append(ortho_penalty.item())
                history["mean_abs_r"].append(mean_abs_r.item())
                # "l1" is the other term of the loss sum, still recorded under its own
                # name: plot/plot_training_curves.py hardcodes this key, so dropping it
                # would break the plotter on this run's npz.
                history["l1"].append(l1_penalty.item())
                history["raw_mean_col_sd"].append(col_std[valid].mean().item())
                history["within_species_scale"].append(within_species_scale.item())
                history["mean_abs_w"].append(mean_abs_w)
                history["valid_cols"].append(int(valid.sum().item()))

                # Periodic flush: a run killed part-way still leaves plottable curves
                # for the seeds it got through, and for the seed it died in.
                if epoch % FLUSH_EVERY == 0:
                    np.savez(
                        history_path, **{k: np.asarray(v) for k, v in history.items()}
                    )

        # Final-epoch loss values (loop variables still in scope after the loop).
        final = {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "ortho": float(ortho_penalty.item()),
            "mean_abs_r": float(mean_abs_r.item()),
            "l1": float(l1_penalty.item()),
            "raw_mean_col_sd": float(col_std[valid].mean().item()),
            "within_species_scale": float(within_species_scale.item()),
        }
        # Average of each statistic over the first min(10, num_epochs) epochs.
        first_10_epochs_avg = {k: v / n_first10 for k, v in first10_sums.items()}
        model.eval()
        trained.append(
            {
                "seed": seed,
                "model": model,
                "final": final,
                "first_10_epochs_avg": first_10_epochs_avg,
            }
        )

        # Save this seed's weights to a seed-suffixed checkpoint next to the figures
        # (non-destructive: does not touch the canonical weights/real_weights_ortho.pt).
        # Format matches train.py so plot_filters.py / load_partial_state_dict can reload.
        metadata = {
            "dataset": "real",
            "script": os.path.basename(__file__),
            "motif": None,
            "matrix_path": MATRIX_PATH,
            "hparams": {
                "num_epochs": args.num_epochs,
                "lr": LR,
                "ortho_lambda": ORTHO_LAMBDA,
                "pool_length": POOL_LENGTH,
                "l1_lambda": L1_LAMBDA,
                "smooth_sigma": SMOOTH_SIGMA,
                "seed": seed,
                "n_sample": N_SAMPLE,
                "min_species": MIN_SPECIES,
            },
            "final": final,
            "first_10_epochs_avg": first_10_epochs_avg,
            # Where this seed's per-epoch component curves live, and at what cadence
            # they were sampled, so a checkpoint points at its own training history.
            "history_path": history_path,
            "history_every": HISTORY_EVERY,
        }

        # Final history write: the same arrays as the periodic flush plus this seed's
        # hyperparameters, so plot_training_curves.py can stamp the figure from the npz
        # alone without also loading the checkpoint.
        np.savez(
            history_path,
            hparams_json=json.dumps(metadata["hparams"]),
            **{k: np.asarray(v) for k, v in history.items()},
        )

        weights_path = os.path.join(_HERE, f"real_weights_aad_seed{seed}.pt")
        torch.save(
            {
                "epoch": args.num_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": metadata,
            },
            weights_path,
        )
        trained[-1]["weights"] = weights_path
        trained[-1]["history"] = history_path
        logger.info(
            f"[seed {seed}] done — final loss {final['loss']:.6f} "
            f"(mean|r| {final['mean_abs_r']:.4f}; first-{n_first10}-epoch avg loss "
            f"{first_10_epochs_avg['loss']:.6f}); saved weights to "
            f"{os.path.normpath(weights_path)} and {len(history['epoch'])} recorded "
            f"epochs to {os.path.normpath(history_path)}"
        )

    num_filters, channels, kernel_size = model.conv_incl.weight.shape
    n_cols = 5

    # Shared run-metadata footer used on both figures.
    caption = (
        f"real MALAT1 cross-species AAD + orthogonality loss | seeds {args.seeds} | "
        f"epochs {args.num_epochs} | lr {LR}, orthoλ {ORTHO_LAMBDA} "
        f"(pool {POOL_LENGTH}), l1λ {L1_LAMBDA}, σ {SMOOTH_SIGMA} | "
        + "  ".join(
            f"s{t['seed']}: loss={t['final']['loss']:.3f}, "
            f"mean|r|={t['final']['mean_abs_r']:.2f}"
            for t in trained
        )
    )

    # ── Combined heat-map figure: seeds stacked vertically, one block per seed ──
    # Shared symmetric color scale (diverging around 0) over ALL seeds' both banks
    # so every panel is directly comparable across seeds.
    all_w = torch.cat(
        [
            torch.cat(
                [
                    t["model"].conv_incl.weight.detach().cpu().reshape(-1),
                    t["model"].conv_skip.weight.detach().cpu().reshape(-1),
                ]
            )
            for t in trained
        ]
    )
    vmax = all_w.abs().max().item()

    rows_per_conv = math.ceil(num_filters / n_cols)  # rows for one 20-filter bank
    rows_per_seed = 2 * rows_per_conv  # INCL bank on top, SKIP bank below
    n_rows = len(trained) * rows_per_seed
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, n_rows * 1.9),
        squeeze=False,
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )
    for ax in axes.ravel():
        ax.axis("off")  # hide unused cells

    im = None
    for s, t in enumerate(trained):
        seed_row0 = s * rows_per_seed
        convs = [("INCL", t["model"].conv_incl), ("SKIP", t["model"].conv_skip)]
        for block, (name, conv) in enumerate(convs):
            weight = conv.weight.detach().cpu()
            bias = conv.bias.detach().cpu()
            for filter_idx in range(num_filters):
                ax = axes[seed_row0 + block * rows_per_conv + filter_idx // n_cols][
                    filter_idx % n_cols
                ]
                ax.axis("on")
                im = ax.imshow(
                    weight[filter_idx], cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto"
                )
                ax.set_title(
                    f"{name} #{filter_idx}  b={bias[filter_idx]:+.2f}", fontsize=8
                )
                ax.set_yticks(range(len(NUCLEOTIDES)))
                ax.set_yticklabels(NUCLEOTIDES, fontsize=6)
                ax.set_xticks(range(kernel_size))
                ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)

    fig.colorbar(im, ax=axes, shrink=0.6, label="filter weight")
    # Rotated seed labels on the far left of each seed's block (computed from the
    # block's top/bottom axis positions, so they never collide with panel titles).
    for s, t in enumerate(trained):
        seed_row0 = s * rows_per_seed
        y_top = axes[seed_row0][0].get_position().y1
        y_bot = axes[seed_row0 + rows_per_seed - 1][0].get_position().y0
        fig.text(
            0.01,
            (y_top + y_bot) / 2,
            f"seed {t['seed']}",
            rotation=90,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
    # Horizontal divider lines separating consecutive seed blocks, drawn in the gap
    # between the last row of one seed and the first row of the next (figure coords,
    # like plot_filters.py's INCL/SKIP divider).
    for s in range(1, len(trained)):
        y_above = (
            axes[s * rows_per_seed - 1][0].get_position().y0
        )  # bottom of prev seed
        y_below = axes[s * rows_per_seed][0].get_position().y1  # top of next seed
        y_line = (y_above + y_below) / 2
        x_right = axes[s * rows_per_seed][n_cols - 1].get_position().x1
        fig.add_artist(
            Line2D(
                [0.01, x_right],
                [y_line, y_line],
                transform=fig.transFigure,
                color="0.3",
                lw=1.6,
            )
        )
    fig.suptitle(
        "Multi-seed conv_incl / conv_skip filters "
        "(rows A/C/G/T, cols = kernel position)"
    )
    fig.text(0.5, 0.004, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.savefig(args.out_heatmaps, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Wrote {len(trained) * 2 * num_filters} filter heat maps "
        f"({len(trained)} seeds) to {os.path.normpath(args.out_heatmaps)}"
    )

    # ── Combined logos figure: one row-block per seed ────────────────────────
    # Because the kernel (6) spans the whole input window, a filter's response over
    # every possible input is an exhaustive scan of the 4**6 = 4096 six-mers. The
    # all-6-mers one-hot encoding is identical for every filter/seed, so build once.
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    # Per seed: keep its significant filters (peak softplus >= min_contribution_frac
    # of that seed's own strongest filter), INCL bank first then SKIP, each sorted
    # strongest-first — matching plot_filters.py's within-model semantics.
    seed_blocks = []
    for t in trained:
        convs = [("INCL", t["model"].conv_incl), ("SKIP", t["model"].conv_skip)]
        filter_logos = {name: [] for name, _ in convs}
        for name, conv in convs:
            weight = conv.weight.detach().cpu().numpy()
            bias = conv.bias.detach().cpu().numpy()
            for filter_idx in range(num_filters):
                # For one-hot inputs the conv is the sum of the weights under the
                # active base at each position, plus bias; softplus matches the model.
                raw = (
                    np.einsum("ncp,cp->n", onehot, weight[filter_idx])
                    + bias[filter_idx]
                )
                outputs = np.logaddexp(0.0, raw)  # softplus, numerically stable
                peak = float(outputs.max())
                thresh = args.logo_activation_frac * peak
                # Every 6-mer reaching the fraction of this filter's peak.
                sel_idx = [i for i in np.argsort(-outputs) if outputs[i] >= thresh]
                sel_seqs = [all_seqs[i] for i in sel_idx]
                n_sel = len(sel_seqs)

                # Per-position base frequencies of the above-threshold 6-mers. No
                # pseudocount, so nucleotides absent from every selected 6-mer stay 0.
                counts = one_hot_batch(sel_seqs).sum(axis=0)  # (4, kernel_size)
                freq = counts / n_sel
                # Shannon information content per position (bits): 2 + Σ f·log2 f.
                info = 2.0 + np.sum(
                    np.where(freq > 0, freq * np.log2(freq, where=freq > 0), 0.0),
                    axis=0,
                )
                heights = freq * info[None, :]  # (4, kernel_size), all >= 0
                filter_logos[name].append(
                    {
                        "name": name,
                        "filter_idx": filter_idx,
                        "bias": float(bias[filter_idx]),
                        "peak": peak,
                        "n_sel": n_sel,
                        "heights": heights,
                    }
                )

        max_peak = max(d["peak"] for name in filter_logos for d in filter_logos[name])
        threshold = args.min_contribution_frac * max_peak
        kept = []
        for name, _ in convs:
            kept.extend(
                sorted(
                    (d for d in filter_logos[name] if d["peak"] >= threshold),
                    key=lambda d: d["peak"],
                    reverse=True,
                )
            )
        seed_blocks.append(
            {
                "seed": t["seed"],
                "kept": kept,
                "rows": max(1, math.ceil(len(kept) / n_cols)),
            }
        )

    num_total = len(trained) * 2 * num_filters
    num_kept = sum(len(b["kept"]) for b in seed_blocks)
    logger.info(
        f"Rendering {num_kept}/{num_total} filters across {len(trained)} seeds with "
        f"peak softplus >= {args.min_contribution_frac:g}x each seed's strongest"
    )

    # Grid sized to the kept filters: one row-block per seed, each packed row-major
    # into n_cols columns (wrapping to extra rows when a seed keeps > n_cols filters).
    logo_n_rows = sum(b["rows"] for b in seed_blocks)
    row_h, header_in, footer_in = 1.9, 1.1, 0.6
    fig_h = logo_n_rows * row_h + header_in + footer_in
    fig, axes = plt.subplots(
        logo_n_rows,
        n_cols,
        figsize=(n_cols * 2.2, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.7, "wspace": 0.3},
    )
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=footer_in / fig_h,
        left=0.07,
        right=0.97,
    )
    for ax in axes.ravel():
        ax.axis("off")  # hide unused cells

    row_offset = 0
    for bi, b in enumerate(seed_blocks):
        # Horizontal divider line above every seed block after the first, drawn in the
        # gap between the previous block's last row and this block's first row.
        if bi > 0:
            y_above = axes[row_offset - 1][0].get_position().y0  # bottom of prev seed
            y_below = axes[row_offset][0].get_position().y1  # top of this seed
            y_line = (y_above + y_below) / 2
            x_right = axes[row_offset][n_cols - 1].get_position().x1
            fig.add_artist(
                Line2D(
                    [0.01, x_right],
                    [y_line, y_line],
                    transform=fig.transFigure,
                    color="0.3",
                    lw=1.6,
                )
            )
        for i, d in enumerate(b["kept"]):
            ax = axes[row_offset + i // n_cols][i % n_cols]
            ax.axis("on")
            df = pd.DataFrame(d["heights"].T, columns=NUCLEOTIDES)
            logomaker.Logo(
                df,
                ax=ax,
                color_scheme="classic",  # heights are all >= 0: nothing below the axis
            )
            ax.set_ylim(0, 2)  # standard information-content scale (bits)
            if i % n_cols == 0:
                ax.set_ylabel("bits", fontsize=6)
            ax.set_title(
                f"{d['name']} #{d['filter_idx']}  b={d['bias']:+.2f}  "
                f"max={d['peak']:.2f}  n={d['n_sel']}",
                fontsize=8,
            )
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
        # Rotated seed label on the far left of this seed's block.
        y_top = axes[row_offset][0].get_position().y1
        y_bot = axes[row_offset + b["rows"] - 1][0].get_position().y0
        fig.text(
            0.01,
            (y_top + y_bot) / 2,
            f"seed {b['seed']}",
            rotation=90,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
        row_offset += b["rows"]

    fig.suptitle(
        "Multi-seed conv_incl / conv_skip information-content (bits) logos\n"
        f"({num_kept}/{num_total} filters across {len(trained)} seeds; peak softplus "
        f">= {args.min_contribution_frac:g}x each seed's strongest; from 6-mers >= "
        f"{args.logo_activation_frac:g}x each filter's peak activation; "
        f"present nucleotides only)",
        y=1 - 0.12 * header_in / fig_h,
        va="top",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.45 * footer_in / fig_h,
        caption,
        ha="center",
        va="center",
        fontsize=7,
        wrap=True,
    )
    fig.savefig(args.out_logos, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {num_kept} filter logos to {os.path.normpath(args.out_logos)}")

    # ── Sidecar JSON: shared hparams, per-seed final losses + checkpoints, and the
    # figures this run produced, written next to the figures. ──
    meta_out = args.meta_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out_logos)), "combined_plot_metadata.json"
    )
    sidecar = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "matrix_path": os.path.abspath(MATRIX_PATH),
        "hparams": {
            "seeds": args.seeds,
            "num_epochs": args.num_epochs,
            "lr": LR,
            "ortho_lambda": ORTHO_LAMBDA,
            "pool_length": POOL_LENGTH,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "n_sample": N_SAMPLE,
            "min_species": MIN_SPECIES,
            "min_contribution_frac": args.min_contribution_frac,
            "logo_activation_frac": args.logo_activation_frac,
        },
        "per_seed": [
            {
                "seed": t["seed"],
                "final": t["final"],
                "first_10_epochs_avg": t["first_10_epochs_avg"],
                "weights": os.path.abspath(t["weights"]),
                "history": os.path.abspath(t["history"]),
            }
            for t in trained
        ],
        "provenance": {
            "heatmap_png": os.path.abspath(args.out_heatmaps),
            "logos_png": os.path.abspath(args.out_logos),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(meta_out, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    logger.info(f"Wrote run metadata to {os.path.normpath(meta_out)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
