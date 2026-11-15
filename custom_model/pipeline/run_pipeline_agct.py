"""C/G-swapped encoding control for the purine/pyrimidine finding.

The channel-order counterpart of run_pipeline.py. Everything is the same — same genes,
same architectures, same loss, same readouts, same 4 channels of base identity — except
the order of the one-hot channels, which is A, G, C, T instead of A, C, G, T:

    run_pipeline.py        channel 0=A  1=C  2=G  3=T
    run_pipeline_agct.py   channel 0=A  1=G  2=C  3=T

**Why.** pipeline/filter_substitution_classes.py found that the A/C/G/T banks put ~0.6 of
their gauge-fixed energy on the purine/pyrimidine axis ({A,G} vs {C,T}) against a 1/3
null — the filters are blind to transitions. That axis is channels {0,2} vs {1,3} in the
A/C/G/T order. This script asks whether the preference follows the *letters* or the
*channel slots*. Under the A/G/C/T order the purines sit in channels {0,1}, so:

    follows letters: slot-order shares show K/M ~0.6 (slots {0,1} vs {2,3});
                     relabelled to true letters, R/Y ~0.6 as in the original runs
    follows slots:   slot-order shares still show R/Y ~0.6, which in true letters is
                     K/M ({G,T} vs {A,C}) — an encoding or analysis artifact

filter_substitution_classes.py reads the stamped alphabet and reports both views; point
its ``--runs-dir`` at ``pipeline/runs_agct`` for the side-by-side verdict.

**What it can and cannot show.** PNASModel treats its input channels symmetrically: every
conv weight is drawn i.i.d. and nothing downstream is tied to a slot. Reordering the
channels is therefore mathematically the original model started from a rearranged
initialization, and the expected outcome is "follows letters". The control's value is
ruling out an encoding error, a labeling error in the analysis, or a hidden slot bias. It
does *not* separate the alignment's substitution frequencies from other properties of
the data that also group A with G (R/Y-rich tracts, for instance). ``--seed`` gives the
identical initial weight tensor as run_pipeline.py at the same architecture — the slots
are only assigned to different letters — so each run here pairs seed-for-seed with its
A/C/G/T counterpart.

    python pipeline/run_pipeline_agct.py --gene malat1 --num-filters 20 --kernel-size 6
    python pipeline/run_pipeline_agct.py --batch pipeline/agct_control.txt

Three things are chosen per run rather than edited into module constants:

  * ``--gene``        which alignment matrix to train on (``data/<dir>/alignment_matrix.npy``)
  * ``--num-filters`` filters per bank, so ``10`` means 10 inclusion + 10 skipping
  * ``--kernel-size`` filter width in nucleotides

Everything downstream follows from those: the model is built at that size, the
orthogonality penalty covers the C(2F, 2) pairs of the 2F resulting channels, and each
plot reads the architecture *and the alphabet* back off the checkpoint, so titles,
captions, axis labels and logo letters name the true base in each channel — a logo from
this script reads A/C/G/T exactly like one from run_pipeline.py.

The alignment matrices are shared with the A/C/G/T pipeline and are not rebuilt: the
channel order is applied at encode time, after the same gap/``N`` strip, so a gene needs
no new preprocessing to be trained here.

Outputs land in ``pipeline/runs_agct/<run_name>/`` (``<gene>_f<filters>_k<kernel>_s<seed>``
unless ``--run-name`` overrides it) — a separate tree, not a tagged name inside
``runs/`` as run_pipeline_ry.py does. These banks are shape-identical to the A/C/G/T ones,
and filter_bank_distance.py and compare_runs.py take the gene from ``run_config.json``,
so a swapped MALAT1 run inside ``runs/`` would silently count as a second MALAT1 run of
the same seed. Keeping the untagged name means each run pairs by name with its
counterpart in ``runs/`` and every ``<gene>_f<F>_k<K>_s<S>`` tool reads the tree as-is:

    runs_agct/<run_name>/
      run_config.json   resolved settings + every output path + timestamp
      weights.pt        checkpoint (model_state_dict / optimizer_state_dict / metadata)
      history.npz       per-epoch loss components
      plots/
        filters.png             per-filter weight heat maps
        filter_logos.png        information-content logos from the exhaustive k-mer scan
        filter_logos_real.png   the same logos weighted by real alignment windows
        model_logos.png         whole-model logos: highest / lowest net-SR k-mers
        real_filters.txt        raw weights + consensus motifs + real-data fire counts
        plot_metadata.json      machine-readable sidecar for the two logo figures
        training_curves.png     loss components and their share of the total
        covariance/             filter co-activation correlation matrix + heatmap
        sr_heatmap.html         interactive per-species, per-window SR profile over the
                                whole alignment (self-contained page, a few MB)

``--skip-train`` re-plots an existing run from its saved checkpoint, and ``--only``
narrows that to particular stages, so a new figure can be backfilled into runs that
were trained before it existed:

    python pipeline/run_pipeline_agct.py --skip-train --only sr-heatmap \
        --gene malat1 --num-filters 20 --kernel-size 6

The training loss is the one from custom_real_train_orthogonality.py, unchanged in
substance: a scale-free cross-species mean-absolute-deviation term plus an orthogonality
penalty on the filter bank. The trainer is reproduced here rather than imported because
that script is constant-driven and is deliberately left untouched, so it keeps working
standalone exactly as before. The plotting stages call the existing scripts, which read
the alphabet this script stamps into the checkpoint and so serve every pipeline.

Because the loss lives in this file as well as in run_pipeline.py, run_pipeline_ry.py and
custom_real_train_orthogonality.py, a change to it has to be mirrored in all four or the
swapped runs stop being a control for the A/C/G/T ones.

Unlike the R/Y pipeline, every number here transfers to run_pipeline.py, ``mean_abs_r``
included: the channel and parameter counts are identical, and any bank expressible in
one channel order is expressible in the other with the same activation on every
sequence — the reorder moves weights between slots, never changes what a bank can do.

The orthogonality penalty and the covariance readout share one ``--pool-length``
(default ``3 * kernel_size``), so the run's ``mean_abs_r`` curve and its correlation
heatmap measure the same quantity on the same window width.

Several runs can be queued in a text file, one run per line written exactly as the
flags would be typed, and executed back to back:

    python pipeline/run_pipeline_agct.py --batch pipeline/agct_control.txt
    python pipeline/run_pipeline_agct.py --batch pipeline/agct_control.txt --dry-run
    python pipeline/run_pipeline_agct.py --batch pipeline/agct_control.txt --epochs 50

Flags given alongside ``--batch`` are defaults every queued run inherits unless its own
line overrides them (the third form above re-runs the whole sweep at 50 epochs). Each
run executes in its own process, so one failure does not take the queue down —
``--stop-on-error`` opts into abandoning the rest. Runs land in their own directories
as usual, and ``pipeline/runs_agct/_batch_<timestamp>.json`` records what ran, what
failed and where each one landed. See pipeline/runs.txt for the file format.
"""

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
ROOT = os.path.abspath(os.path.join(CUSTOM_DIR, ".."))
# Ahead of ROOT: "custom_model" names both ROOT/custom_model/ (a namespace portion) and
# custom_model/custom_model.py (the module we want). The regular module only wins if its
# directory comes first on sys.path.
sys.path.insert(0, ROOT)
sys.path.insert(0, CUSTOM_DIR)

from custom_model import PNASModel  # noqa: E402
from custom_utils import ALPHABETS, encode_sequence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The one thing that makes this script different from run_pipeline.py. Fixed rather than
# a flag, for the reason run_pipeline_ry.py gives: a channel order behind an optional flag
# is how a checkpoint ends up mislabelled. ALPHABETS["agct"] is "AGCT", so N_CHANNELS is
# 4 and only the order differs; the name is stamped into every checkpoint, which is what
# lets the plotters and filter_substitution_classes.py relabel the slots to true letters.
ALPHABET_NAME = "agct"
ALPHABET = ALPHABETS[ALPHABET_NAME]
N_CHANNELS = len(ALPHABET)
# Separate output tree from runs/ — see the module docstring for why a tagged name
# inside runs/ is not enough for a shape-identical control.
RUNS_ROOT = os.path.join(_HERE, "runs_agct")

# Friendly gene names -> (data/ subdirectory holding alignment_matrix.npy, display label).
# Any name not listed here is used as the directory name directly and labels itself, so a
# newly added alignment works without touching this table.
GENES = {
    "malat1": ("multiz100", "MALAT1"),
    "neat1": ("neat1", "NEAT1"),
    # Built by dataset_preparations/maf_to_alignment_matrix.py; listed here only so the
    # figures say "HOTAIR" rather than "hotair" — the directory name already matches, so
    # each would resolve without an entry.
    "hotair": ("hotair", "HOTAIR"),
    "il6": ("il6", "IL6"),
    "vegfa": ("vegfa", "VEGFA"),
    "xist": ("xist", "XIST"),
}

# The plotting stages, in the order they run. Names are the --only vocabulary.
PLOT_STEPS = ("filters", "curves", "covariance", "sr-heatmap")

# Variance floor for the orthogonality correlation denominator. A filter whose softplus
# has gone flat has zero variance, where both the 0/0 correlation and sqrt'(0) = inf
# would poison the gradient of every pair it appears in.
ORTHO_EPS = 1e-12


def resolve_gene(name):
    """Map a --gene value to (matrix path, data subdirectory, display label)."""
    gene_dir, label = GENES.get(name.lower(), (name, name))
    matrix_path = os.path.join(ROOT, "data", gene_dir, "alignment_matrix.npy")
    if not os.path.exists(matrix_path):
        available = sorted(
            d
            for d in os.listdir(os.path.join(ROOT, "data"))
            if os.path.exists(os.path.join(ROOT, "data", d, "alignment_matrix.npy"))
        )
        raise SystemExit(
            f"No alignment matrix for --gene {name} (looked for {matrix_path}). "
            f"Alignments present under data/: {', '.join(available) or 'none'}. "
            f"Known aliases: {', '.join(sorted(GENES))}."
        )
    return matrix_path, gene_dir, label


def train(args, matrix_path, gene_dir, gene_label, weights_path, history_path):
    """Train the orthogonality model and write the checkpoint + history npz.

    The loss is identical to custom_real_train_orthogonality.py: a scale-free
    cross-species mean-absolute-deviation term over the aligned SR profile, plus an
    orthogonality penalty on the pooled filter activations, plus an optional activation
    L1. What differs is only that the architecture and hyperparameters come from
    ``args`` instead of module constants.
    """
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_filters, kernel_size = args.num_filters, args.kernel_size
    n_channels = 2 * num_filters  # incl + skip, correlated together by the penalty
    # Windows summed into one pooled score. At kernel_size the pool holds exactly one
    # window and the penalty degenerates to single-window correlation.
    pool_windows = args.pool_length - kernel_size + 1

    # ── Load the real alignment matrix and precompute per-species inputs ──
    # matrix: (n_aligned, n_species) of single characters. Gaps are "-" and "N";
    # everything else (including lowercase soft-masked a/c/g/t) is a nucleotide.
    matrix = np.load(matrix_path)
    n_aligned, n_total = matrix.shape

    # Both already resolved against the alignment's species count in main(), so the run
    # manifest and the checkpoint metadata report the same numbers this loop uses.
    n_sample, min_species = args.n_sample, args.min_species

    # For each species: the gap-removed one-hot sequence and, for every nucleotide,
    # the aligned column it occupies. species_cols[j][k] is the aligned column of
    # the k-th nucleotide = the first nucleotide of sliding window k.
    # The one-hot is over A/G/C/T rather than A/C/G/T: the same 4 channels of base
    # identity with C and G exchanged, applied after the gap strip, so the sequence
    # length and the column mapping are identical to the A/C/G/T pipeline's.
    species_oh, species_cols = [], []
    for j in range(n_total):
        row = matrix[:, j]
        is_nuc = (row != "-") & (row != "N")
        nuc_aligned = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()
        oh = torch.tensor(
            encode_sequence(seq, ALPHABET_NAME), dtype=torch.float32, device=device
        )  # (4, Lj)
        species_oh.append(oh.unsqueeze(0))  # (1, 4, Lj)
        species_cols.append(torch.tensor(nuc_aligned, dtype=torch.long, device=device))

    # A species shorter than the kernel holds no window at all and would fail inside
    # conv1d with an opaque shape error; say so here instead.
    shortest = min(oh.shape[-1] for oh in species_oh)
    if shortest < kernel_size:
        raise SystemExit(
            f"--kernel-size {kernel_size} exceeds the shortest gap-removed species "
            f"sequence ({shortest} nt) in {matrix_path}."
        )

    logger.info(
        f"Real alignment matrix: {matrix.shape} (aligned positions x species) "
        f"from {matrix_path} on {device} — sampling {n_sample} species/epoch, "
        f"column valid when >= {min_species} species present"
    )

    # Same (F, 4, K) architecture as run_pipeline.py, built with the same RNG draws in
    # the same order, so at a given --seed the initial weight tensor is bit-identical to
    # the A/C/G/T run's; only the letters its channel slots stand for differ.
    model = PNASModel(
        seq_in_channels=N_CHANNELS, num_filters=num_filters, kernel_size=kernel_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(
        f"Model: {num_filters}+{num_filters} incl/skip filters, "
        f"{kernel_size}-nt kernel over {N_CHANNELS} {'/'.join(ALPHABET)} channels "
        f"(C/G-swapped control) ({gene_label})"
    )

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once.
    # Sum-normalized so it's a weighted average (preserves overall scale). It carries
    # no learnable parameters and is applied with reflect padding at the use site so
    # the conserved block at position 0 isn't damped by zero edges. Smoothing denoises
    # single-position jitter so the conserved-motif signal dominates the column SD.
    # It is applied to each species' dense (gap-free) SR track before scattering, so
    # no NaNs ever enter the convolution.
    if args.smooth_sigma:
        smooth_radius = math.ceil(3 * args.smooth_sigma)  # kernel half-width (~3 sigma)
        offsets = torch.arange(
            -smooth_radius, smooth_radius + 1, device=device, dtype=torch.float32
        )
        gauss_kernel = torch.exp(-(offsets**2) / (2 * args.smooth_sigma**2))
        gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)  # (1, 1, K)
        logger.info(
            f"SR-profile smoothing: Gaussian sigma={args.smooth_sigma} nt, "
            f"kernel size={gauss_kernel.shape[-1]}"
        )

    # ── Orthogonality-penalty scaffolding, built once ────────────────────────
    # Box kernel of width pool_windows, one copy per filter channel and applied with
    # groups=, so a single grouped conv1d turns the per-window activation tracks into
    # the pooled scores at every offset. Unlike the Gaussian above it is *not*
    # sum-normalized: filter_coactivation_covariance.py sums the softplus over the
    # windows inside a pooled sequence rather than averaging them. (Correlation is
    # scale-free, so this only matters for reading the raw pooled magnitudes.)
    box_kernel = torch.ones(n_channels, 1, pool_windows, device=device)
    # The unique off-diagonal pairs of the (2F x 2F) matrix — incl-incl, skip-skip and
    # incl-skip alike, matching the full matrix the analysis script reports. Cross-bank
    # pairs belong in the penalty: the two banks are subtracted from one another, so a
    # correlated incl/skip pair is two filters cancelling instead of dividing the work.
    triu = torch.triu_indices(n_channels, n_channels, offset=1, device=device)
    logger.info(
        f"Orthogonality penalty: mean |r| over the {triu.shape[1]} pairs of "
        f"{n_channels} filter channels, on every {args.pool_length}-mer of the alignment "
        f"({pool_windows} windows summed per {args.pool_length}-mer), "
        f"lambda={args.ortho_lambda:g}"
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
    # history_every cadence. All of these are already computed for the log, so keeping
    # them is free; retaining them is what lets the component curves be plotted after the
    # run instead of being inferred from the two endpoints in the metadata.
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
    # With n_sample == n_total the per-epoch randperm only permutes the rows of the
    # full species set, and every term of the loss is invariant to row order. So the
    # presence mask — and everything derived from it: the per-column species counts,
    # the valid-column set, the per-row window counts, and the total number of
    # (species, window) activations — is identical in every epoch. Build it once here
    # so the loop below stops re-scattering the mask n_sample rows at a time and
    # re-reducing it over every column. With n_sample < n_total the sampled set really
    # does change every epoch, so the loop keeps its original per-epoch path.
    all_species = n_sample == n_total
    if all_species:
        fixed_sel = list(range(n_total))
        # Columns that receive a value for each species: a length-Lj sequence has
        # Lj-K+1 windows, so the trailing K-1 nucleotides start no window and are
        # absent from the mask (same truncation the scatter below applies).
        fixed_win_cols = [
            species_cols[j][: species_oh[j].shape[-1] - (kernel_size - 1)]
            for j in fixed_sel
        ]
        fixed_mask = torch.zeros(n_sample, n_aligned, device=device)
        for r, cols_r in enumerate(fixed_win_cols):
            fixed_mask[r, cols_r] = 1.0
        fixed_count = fixed_mask.sum(0)  # (n_aligned,) species present per column
        fixed_valid = fixed_count >= min_species
        fixed_rc = fixed_mask.sum(1)  # (n_sample,) windows per species
        fixed_act_count = sum(c.numel() for c in fixed_win_cols)
        logger.info(
            f"All {n_total} species used every epoch — mask, column counts and valid "
            f"columns precomputed once: {int(fixed_valid.sum().item())} valid columns, "
            f"{fixed_act_count} (species, window) activations"
        )

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Fresh random combination of n_sample species for this epoch (the full set,
        # in fixed order, when all_species — the draw would be a no-op permutation).
        sel = fixed_sel if all_species else torch.randperm(n_total)[:n_sample].tolist()

        # Build the aligned (n_sample, n_aligned) matrix of SR values. We never put
        # literal NaN into the differentiable tensor; `mask` marks present entries
        # (== non-NaN). Masked-out entries stay 0 and receive no gradient.
        values = torch.zeros(n_sample, n_aligned, device=device)
        mask = (
            fixed_mask
            if all_species
            else torch.zeros(n_sample, n_aligned, device=device)
        )

        # Accumulate the summed softplus activations over all sampled species so the
        # L1 penalty below is a mean over every (species, window) activation.
        # Sequences differ in length, so we track the running activation sum and
        # element count and divide once at the end.
        act_sum = torch.zeros((), device=device)
        act_count = fixed_act_count if all_species else 0

        # Sufficient statistics for the orthogonality penalty's correlation matrix,
        # accumulated over every species so the correlation is over the whole alignment
        # (one observation per pool_length-mer) rather than per species. float64: the
        # pooled scores are sums of pool_windows non-negative softplus values, so their
        # mean dwarfs their spread and E[p^2] - E[p]^2 is a difference of large near-equal
        # numbers — float32 would spend most of its precision in that cancellation. The
        # accumulators are only 2F and 2Fx2F wide, so double costs nothing here.
        ortho_sum = torch.zeros(n_channels, dtype=torch.float64, device=device)
        ortho_gram = torch.zeros(
            n_channels, n_channels, dtype=torch.float64, device=device
        )
        ortho_n = 0
        for r, j in enumerate(sel):
            # Per-filter softplus activations, kept un-summed so the orthogonality
            # penalty below can correlate the channels against each other; summing
            # over the filter axis reproduces PNASModel.compute_sr_profile exactly (it
            # returns only that sum, so the breakdown is not recoverable from it).
            # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
            # both (1, Lj-K+1) and >= 0; used by the activation L1 penalty below.
            f_incl = F.softplus(model.conv_incl(species_oh[j]))  # (1, F, Lj-K+1), >= 0
            f_skip = F.softplus(model.conv_skip(species_oh[j]))  # (1, F, Lj-K+1), >= 0
            a_incl, a_skip = f_incl.sum(1), f_skip.sum(1)  # both (1, Lj-K+1)
            sr_j = (a_incl - a_skip).squeeze(0)  # (Lj-K+1,)
            act_sum = act_sum + (a_incl + a_skip).sum()
            if not all_species:
                act_count += a_incl.numel()

            # Pooled pool_length-mer scores for the orthogonality penalty, taken from the
            # *raw* activations (the pooled mode of filter_coactivation_covariance.py has
            # no position axis left to smooth, so it disables smoothing). Summing softplus
            # over the pool_windows windows inside a pool_length-mer is a box sum along
            # the activation track, so every pool_length-mer of this species is scored by
            # one grouped conv over the track we already have — the pooled sequences
            # themselves are never materialized or re-convolved.
            act = torch.cat([f_incl, f_skip], dim=1)  # (1, 2F, Lj-K+1)
            if act.shape[-1] >= pool_windows:  # else too short for one pool_length-mer
                # -> (2F, ...): one score per filter per pool_length-mer offset.
                pooled = F.conv1d(act, box_kernel, groups=n_channels).squeeze(0)
                p = pooled.double()
                ortho_sum = ortho_sum + p.sum(1)
                ortho_gram = ortho_gram + p @ p.T
                ortho_n += pooled.shape[-1]

            # Gaussian-smooth the dense SR track along position (gap-free, no NaN).
            # Reflect-pad by the kernel half-width, then conv1d with the single-channel
            # kernel. Differentiable, so gradients flow back to the conv filters.
            if args.smooth_sigma:
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

        if ortho_n == 0:
            raise SystemExit(
                f"--pool-length {args.pool_length} needs {pool_windows} windows, but no "
                "species sequence is long enough to supply one. Lower --pool-length."
            )

        # Numerator: NaN-aware cross-species mean absolute deviation per column
        # (about each column's cross-species mean), valid only where at least
        # min_species of the sampled species are present. Masked-out entries are
        # zeroed *after* the abs() so they contribute 0 and receive no gradient.
        count = fixed_count if all_species else mask.sum(0)  # (n_aligned,)
        col_mean = (values * mask).sum(0) / count.clamp(min=1)
        col_std = ((values - col_mean).abs() * mask).sum(0) / count.clamp(min=1)
        valid = fixed_valid if all_species else count >= min_species

        # Denominator: within-species dynamic range = each row's mean absolute
        # deviation across positions (NaN-aware, about that row's own mean), averaged
        # over rows. Dividing by it makes the loss scale-free, so the model can't cheat
        # to a low abs-dev by shrinking all activations toward a constant.
        rc = fixed_rc if all_species else mask.sum(1)  # (n_sample,)
        row_mean = (values * mask).sum(1) / rc.clamp(min=1)
        within_species_scale = (
            ((values - row_mean.unsqueeze(1)).abs() * mask).sum(1) / rc.clamp(min=1)
        ).mean()  # scalar

        sd_loss = (
            (col_std[valid]) / within_species_scale
        ).mean()  # normalized average column abs-dev

        # ── Orthogonality penalty: mean |r| over the filter pairs ────────────
        # The Gram accumulators above hold sum(p) and sum(p p^T) over every
        # pool_length-mer of the alignment, so the population (ddof=0) covariance of the
        # pooled filter scores follows in closed form. ddof does not matter here:
        # correlation normalizes it away, which is why this matches the analysis script's
        # np.cov (ddof=1) path exactly rather than approximately.
        mean_p = ortho_sum / ortho_n
        cov_p = ortho_gram / ortho_n - torch.outer(mean_p, mean_p)  # (2F, 2F)
        std_p = cov_p.diagonal().clamp(min=ORTHO_EPS).sqrt()  # floor: dead filters
        corr_p = cov_p / torch.outer(std_p, std_p)
        # abs(), not r^2 or signed r: a signed mean is minimized by driving correlations
        # negative rather than to zero, which is anti-correlation, not orthogonality.
        mean_abs_r = corr_p[triu[0], triu[1]].abs().mean()
        ortho_penalty = args.ortho_lambda * mean_abs_r.to(sd_loss.dtype)

        # L1 penalty on the summed softplus activations (activity sparsity -> peaked,
        # data-driven SR profiles). a_incl/a_skip are already >= 0, so no abs() is
        # needed; dividing by the total element count keeps the penalty independent of
        # seq length / number of species. Disabled by default (--l1-lambda 0) in favour
        # of the orthogonality penalty; keep an eye on mean|conv_w| in the log, since
        # both remaining terms are scale-free and nothing else anchors the overall
        # activation magnitude.
        l1_penalty = args.l1_lambda * (act_sum / act_count)
        loss = sd_loss + ortho_penalty + l1_penalty
        loss.backward()
        optimizer.step()

        # Accumulate the first-10-epoch statistics for the metadata average below.
        if epoch <= 10:
            first10_sums["loss"] += loss.item()
            first10_sums["sd_loss"] += sd_loss.item()
            first10_sums["ortho"] += float(ortho_penalty)
            first10_sums["mean_abs_r"] += mean_abs_r.item()
            first10_sums["l1"] += float(l1_penalty)
            first10_sums["raw_mean_col_sd"] += col_std[valid].mean().item()
            first10_sums["within_species_scale"] += within_species_scale.item()
            n_first10 += 1

        if epoch == 1 or epoch % args.history_every == 0:
            mean_abs_w = (
                model.conv_incl.weight.abs().mean().item()
                + model.conv_skip.weight.abs().mean().item()
            ) / 2
            logger.info(
                f"epoch {epoch:4d} | loss = {loss.item():.6f} "
                f"| sd loss = {sd_loss.item():.6f} "
                f"| ortho = {float(ortho_penalty):.6f} "
                f"| mean|r| = {mean_abs_r.item():.4f} "
                f"| l1 = {float(l1_penalty):.6f} "
                f"| raw mean col abs-dev = {col_std[valid].mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| valid cols = {int(valid.sum().item())} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

            # Same numbers as the log line, retained for plot_training_curves.py.
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["sd_loss"].append(sd_loss.item())
            history["ortho"].append(float(ortho_penalty))
            history["mean_abs_r"].append(mean_abs_r.item())
            # "l1" is the other term of the loss sum, still recorded under its own name
            # even at --l1-lambda 0, where it is an identically-zero series that
            # plot_training_curves.py then drops from the figure.
            history["l1"].append(float(l1_penalty))
            history["raw_mean_col_sd"].append(col_std[valid].mean().item())
            history["within_species_scale"].append(within_species_scale.item())
            history["mean_abs_w"].append(mean_abs_w)
            history["valid_cols"].append(int(valid.sum().item()))

            # Periodic flush: a run killed part-way still leaves plottable curves.
            if epoch % args.flush_every == 0:
                np.savez(history_path, **{k: np.asarray(v) for k, v in history.items()})

    # ── Run metadata for the plotters/sidecar: dataset identity (real alignment, so
    # no planted motif), the source matrix, the architecture, the hyperparameters, and
    # the final-epoch loss values. The "final" numbers are read from the loop variables
    # still in scope after the loop (their last-iteration values); raw_mean_col_sd is
    # over valid columns to match the training log. ──
    first_10_epochs_avg = {k: v / n_first10 for k, v in first10_sums.items()}
    metadata = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "motif": None,
        "matrix_path": matrix_path,
        "hparams": {
            # The three per-run choices, recorded first so every downstream caption can
            # state what was trained without re-deriving it.
            "gene": gene_dir,
            "gene_label": gene_label,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            # The sequence alphabet, so a plotter names the channels from the checkpoint
            # rather than guessing. Load-bearing here in a way it is not for R/Y: with 4
            # channels, resolve_alphabet() would infer "acgt" from the count alone and
            # every plot would put C's label on G's weights and vice versa.
            "alphabet": ALPHABET_NAME,
            "n_channels": N_CHANNELS,
            "num_epochs": args.epochs,
            "lr": args.lr,
            "ortho_lambda": args.ortho_lambda,
            "pool_length": args.pool_length,
            "l1_lambda": args.l1_lambda,
            "smooth_sigma": args.smooth_sigma,
            "seed": args.seed,
            "n_sample": n_sample,
            "min_species": min_species,
        },
        "final": {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "ortho": float(ortho_penalty),
            "mean_abs_r": float(mean_abs_r.item()),
            "l1": float(l1_penalty),
            "raw_mean_col_sd": float(col_std[valid].mean().item()),
            "within_species_scale": float(within_species_scale.item()),
        },
        "first_10_epochs_avg": first_10_epochs_avg,
        # Where the per-epoch component curves for this run live, and at what cadence
        # they were sampled, so a checkpoint points at its own training history.
        "history_path": history_path,
        "history_every": args.history_every,
    }

    # ── Final history write: the same arrays as the periodic flush plus the run's
    # hyperparameters, so plot_training_curves.py can stamp the figure from the npz
    # alone without also loading the checkpoint. ──
    np.savez(
        history_path,
        hparams_json=json.dumps(metadata["hparams"]),
        **{k: np.asarray(v) for k, v in history.items()},
    )
    logger.info(
        f"Saved {len(history['epoch'])} recorded epochs of loss components to "
        f"{history_path}"
    )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        weights_path,
    )
    logger.info(f"Saved model weights to {weights_path}")
    return metadata


def run_step(title, cmd, required=True):
    """Run one plotting script, echoing the command so a run is reproducible by hand.

    ``required=False`` downgrades a non-zero exit to a warning. Training is the
    expensive part of a run and the manifest is written only after every step, so an
    optional figure must not be able to throw away a finished model.
    """
    logger.info(f"\n=== {title} ===\n$ {' '.join(cmd)}")
    # cwd=CUSTOM_DIR: the top-level plot scripts do a bare `from custom_model import
    # PNASModel`, which resolves to custom_model.py only when that directory leads
    # sys.path. All I/O paths passed below are absolute, so cwd affects imports only.
    proc = subprocess.run(cmd, cwd=CUSTOM_DIR, check=required)
    if proc.returncode:
        logger.warning(f"{title} failed (exit {proc.returncode}); continuing.")
    return proc.returncode == 0


def read_queue(path):
    """Parse a batch file into a list of (line number, argument list) pairs.

    One run per line, written exactly as the flags would be typed:

        --gene malat1 --num-filters 10 --kernel-size 3
        --gene neat1  --num-filters 20 --kernel-size 6 --epochs 1000

    ``#`` starts a comment (whole-line or trailing), blank lines are skipped, and a
    line ending in ``\\`` continues onto the next so a long run can be wrapped. Values
    are split with shell quoting rules, so ``--run-name "my run"`` survives intact.
    """
    with open(path) as fh:
        raw = fh.read()

    runs, buffer, buffer_lineno = [], "", None
    for lineno, line in enumerate(raw.splitlines(), start=1):
        # shlex in POSIX mode drops "#" comments itself, but only when the "#" starts a
        # token; strip here as well so a trailing comment is handled the same way.
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if buffer_lineno is None:
            buffer_lineno = lineno
        if text.endswith("\\"):
            buffer += text[:-1] + " "
            continue
        tokens = shlex.split(buffer + text)
        buffer, start_line = "", buffer_lineno
        buffer_lineno = None
        if tokens:
            runs.append((start_line, tokens))
    if buffer.strip():
        raise SystemExit(
            f"{path}: line {buffer_lineno} ends with a trailing '\\' but the file ends."
        )
    if not runs:
        raise SystemExit(f"{path} contains no runs (only blank lines and comments).")
    return runs


def run_batch(queue_path, shared_args, stop_on_error, dry_run):
    """Execute every run in a queue file, one after another.

    Each run is a fresh subprocess of this same script rather than an in-process call,
    so one run that fails — a bad flag, an out-of-memory kill, a missing alignment —
    leaves the queue intact and the rest still execute. ``shared_args`` are whatever
    flags were typed alongside ``--batch``; they are placed *before* the line's own
    flags so the line always wins, making them per-batch defaults.
    """
    runs = read_queue(queue_path)
    runs_root = RUNS_ROOT
    logger.info(
        f"=== Batch: {len(runs)} run(s) from {queue_path} ==="
        + (
            f"\nShared flags applied to every run: {' '.join(shared_args)}"
            if shared_args
            else ""
        )
    )
    for i, (lineno, tokens) in enumerate(runs, start=1):
        logger.info(f"  [{i}/{len(runs)}] line {lineno}: {' '.join(tokens)}")
    if dry_run:
        logger.info("\n--dry-run: nothing was executed.")
        return 0

    results = []
    batch_started = datetime.now()
    for i, (lineno, tokens) in enumerate(runs, start=1):
        cmd = [sys.executable, os.path.abspath(__file__)] + shared_args + tokens
        logger.info(
            f"\n{'=' * 78}\n=== Batch run {i}/{len(runs)} (line {lineno}) ===\n"
            f"$ {' '.join(cmd)}\n{'=' * 78}"
        )
        started = datetime.now()
        # check=False: a failing run is recorded and the batch continues, unless
        # --stop-on-error. The child streams straight to this terminal, so a long batch
        # stays watchable rather than going silent until the end.
        proc = subprocess.run(cmd, cwd=CUSTOM_DIR)
        elapsed = (datetime.now() - started).total_seconds()

        # Locate the run directory by the manifest the child writes last, rather than by
        # re-deriving the name here — that derivation lives in one place (main) and this
        # keeps it there. A failed run writes none, which is itself the right answer.
        run_dir = None
        if os.path.isdir(runs_root):
            manifests = [
                (os.path.getmtime(m), os.path.dirname(m))
                for d in os.listdir(runs_root)
                for m in [os.path.join(runs_root, d, "run_config.json")]
                if os.path.exists(m) and os.path.getmtime(m) >= started.timestamp()
            ]
            if manifests:
                run_dir = max(manifests)[1]

        results.append(
            {
                "index": i,
                "queue_line": lineno,
                "args": tokens,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "elapsed_seconds": round(elapsed, 1),
                "run_dir": run_dir,
            }
        )
        if proc.returncode != 0:
            logger.error(f"Batch run {i}/{len(runs)} FAILED (exit {proc.returncode}).")
            if stop_on_error:
                logger.error("--stop-on-error: abandoning the remaining runs.")
                break

    # ── Batch summary: what ran, what failed, and where each run landed ──
    n_ok = sum(r["ok"] for r in results)
    logger.info(f"\n{'=' * 78}\n=== Batch summary: {n_ok}/{len(runs)} succeeded ===")
    for r in results:
        status = "ok    " if r["ok"] else "FAILED"
        where = r["run_dir"] or "-"
        logger.info(
            f"  [{r['index']}/{len(runs)}] {status} {r['elapsed_seconds']:8.1f}s  "
            f"{' '.join(r['args'])}\n           -> {where}"
        )
    skipped = len(runs) - len(results)
    if skipped:
        logger.info(f"  {skipped} run(s) never started (--stop-on-error).")

    os.makedirs(runs_root, exist_ok=True)
    summary_path = os.path.join(
        runs_root, f"_batch_{batch_started.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "queue": os.path.abspath(queue_path),
                "shared_args": shared_args,
                "started_at": batch_started.isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "n_queued": len(runs),
                "n_succeeded": n_ok,
                "runs": results,
            },
            fh,
            indent=2,
        )
    logger.info(f"\nWrote batch summary to {summary_path}")
    # Non-zero exit when anything failed, so a shell loop or cron job notices.
    return 0 if n_ok == len(runs) else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gene",
        default="malat1",
        help="Which alignment to train on: a known alias (%s) or the name of any "
        "data/ subdirectory holding an alignment_matrix.npy. Default: malat1."
        % ", ".join(sorted(GENES)),
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=20,
        help="Filters per bank: this many inclusion filters and this many skipping "
        "filters, so 10 gives a 10+10 model. Default: 20.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=6,
        help="Filter width in nucleotides. Default: 6 (the original PNAS kernel).",
    )
    parser.add_argument(
        "--pool-length",
        type=int,
        default=None,
        help="Sequence length the orthogonality penalty and the covariance readout "
        "pool over. Default: 3 x --kernel-size, wide enough to absorb the register "
        "ambiguity of a kernel that is wider than the motifs it learns.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument(
        "--ortho-lambda",
        type=float,
        default=0.5,
        help="Weight of the mean-|r| orthogonality penalty. Default: 0.5.",
    )
    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0,
        help="Weight of the L1 penalty on the softplus activations. Default: 0 "
        "(disabled), which also drops the term from the training-curve figure.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing of the SR profile along position, in nt. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="Species sampled per epoch. Default: all of them, which keeps the "
        "constant-structure fast path.",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=None,
        help="A column counts toward the loss only if at least this many sampled "
        "species are present. Default: half the sampled species.",
    )
    parser.add_argument(
        "--history-every",
        type=int,
        default=10,
        help="Epoch cadence for the history npz.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Rewrite the history npz this often so a killed run keeps its curves.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the output directory under pipeline/runs_agct/. Default: "
        "<gene>_f<num_filters>_k<kernel_size>_s<seed>, the same name as the A/C/G/T "
        "counterpart in pipeline/runs/.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Re-plot an existing run from its saved weights.pt/history.npz instead "
        "of training again.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PLOT_STEPS,
        default=None,
        help="Run only these plotting steps instead of all of them. With "
        "--skip-train this backfills one artifact into an existing run without "
        "redoing the others; the run manifest is then updated in place rather "
        "than rewritten, so the original training metadata survives.",
    )
    parser.add_argument(
        "--batch",
        default=None,
        help="Path to a queue file of runs to execute one after another, one run per "
        "line written exactly as the flags would be typed (see pipeline/runs.txt). "
        "'#' comments and blank lines are ignored, a trailing '\\' continues a line, "
        "and any flags given alongside --batch become defaults every queued run "
        "inherits unless its own line overrides them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="--batch only: abandon the remaining runs as soon as one fails. By "
        "default a failed run is recorded and the queue continues.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="--batch only: list the runs the queue file resolves to and exit without "
        "training anything.",
    )
    parser.add_argument(
        "--logo-activation-frac",
        type=float,
        default=0.5,
        help="Passed to plot_filters.py: per filter, the fraction of its own peak "
        "activation a k-mer or window must reach to enter its logo. Default: 0.5.",
    )
    parser.add_argument(
        "--min-contribution-frac",
        type=float,
        default=0.05,
        help="Passed to plot_filters.py: only render logos for filters whose peak "
        "reaches this fraction of the strongest filter's peak. Default: 0.05.",
    )
    args = parser.parse_args()

    # ── Batch mode: hand off to the queue runner before any of the single-run
    # validation below, which would otherwise reject nothing but also mean nothing —
    # in batch mode the per-run values live on the queue lines, and each queued run
    # re-enters this script and validates itself. ──
    if args.batch:
        queue_path = args.batch
        if not os.path.exists(queue_path):
            # A bare name is looked up in pipeline/ too, so `--batch runs.txt` works
            # from anywhere without spelling out the directory.
            candidate = os.path.join(_HERE, os.path.basename(queue_path))
            if os.path.exists(candidate):
                queue_path = candidate
            else:
                parser.error(f"--batch file not found: {args.batch}")
        # Everything else typed on the command line becomes a per-batch default. Taken
        # from argv rather than from the parsed namespace so only flags actually typed
        # are forwarded — passing the whole namespace would turn every argparse default
        # into an explicit flag and silently override the queue lines it should defer to.
        shared, argv = [], sys.argv[1:]
        i = 0
        while i < len(argv):
            token = argv[i]
            if token == "--batch":
                i += 2  # skip the flag and its value
                continue
            if token.startswith("--batch="):
                i += 1
                continue
            if token in ("--stop-on-error", "--dry-run"):
                i += 1  # batch-runner controls, not per-run flags
                continue
            shared.append(token)
            i += 1
        raise SystemExit(
            run_batch(queue_path, shared, args.stop_on_error, args.dry_run)
        )

    if args.dry_run or args.stop_on_error:
        parser.error("--dry-run and --stop-on-error apply only with --batch.")

    if args.num_filters < 1:
        parser.error("--num-filters must be at least 1.")
    if args.kernel_size < 1:
        parser.error("--kernel-size must be at least 1.")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    # Default the pool width to three kernels: wide enough that two filters encoding the
    # same motif at different registers land in a common window, which is the whole point
    # of pooling, without diluting the score across unrelated sequence.
    if args.pool_length is None:
        args.pool_length = 3 * args.kernel_size
    if args.pool_length < args.kernel_size:
        parser.error(
            f"--pool-length {args.pool_length} is narrower than --kernel-size "
            f"{args.kernel_size}; a pool must hold at least one full window."
        )

    matrix_path, gene_dir, gene_label = resolve_gene(args.gene)

    # Resolve the two alignment-dependent defaults here rather than inside train(), so
    # the run manifest below records the numbers actually used instead of a bare None.
    # mmap: only the header's shape is needed, not the array.
    n_species = int(np.load(matrix_path, mmap_mode="r").shape[1])
    if args.n_sample is None:
        args.n_sample = n_species
    if args.n_sample > n_species:
        parser.error(
            f"--n-sample {args.n_sample} exceeds the {n_species} species in {matrix_path}."
        )
    if args.min_species is None:
        # Half the sampled species, matching the original 31-of-62 rule but derived, so
        # a differently-sized alignment gets a sensible threshold of its own.
        args.min_species = math.ceil(args.n_sample / 2)
    if not 1 <= args.min_species <= args.n_sample:
        parser.error(
            f"--min-species {args.min_species} must be between 1 and the "
            f"{args.n_sample} sampled species."
        )

    # Untagged, and under RUNS_ROOT rather than runs/: the separate tree is what keeps this
    # run from overwriting (or being mistaken for) the A/C/G/T run of the same gene,
    # architecture and seed, and the shared name is what pairs the two for the readout.
    run_name = (
        args.run_name
        or f"{gene_dir}_f{args.num_filters}_k{args.kernel_size}_s{args.seed}"
    )
    run_dir = os.path.join(RUNS_ROOT, run_name)
    plots_dir = os.path.join(run_dir, "plots")
    cov_dir = os.path.join(plots_dir, "covariance")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(cov_dir, exist_ok=True)

    weights_path = os.path.join(run_dir, "weights.pt")
    history_path = os.path.join(run_dir, "history.npz")

    logger.info(
        f"=== Pipeline run '{run_name}' ===\n"
        f"  gene         {gene_label} ({gene_dir})\n"
        f"  alphabet     {'/'.join(ALPHABET)} ({N_CHANNELS} channels; C/G-swapped control)\n"
        f"  filters      {args.num_filters}+{args.num_filters} incl/skip\n"
        f"  kernel       {args.kernel_size} nt\n"
        f"  pool length  {args.pool_length} nt "
        f"({args.pool_length - args.kernel_size + 1} windows/pool)\n"
        f"  output       {run_dir}"
    )

    if args.skip_train:
        for path in (weights_path, history_path):
            if not os.path.exists(path):
                raise SystemExit(
                    f"--skip-train needs an existing run, but {path} is missing."
                )
        logger.info("--skip-train: reusing the existing checkpoint and history.")
    else:
        train(args, matrix_path, gene_dir, gene_label, weights_path, history_path)

    # ── Plotting ──
    # Each script reads the architecture back off the checkpoint, so the figures are
    # labelled with what this run actually trained rather than with any default.
    py = sys.executable

    selected = set(args.only) if args.only else None

    def want(step):
        return selected is None or step in selected

    if want("filters"):
        run_step(
            "Filters, sequence logos and metadata",
            [
                py,
                os.path.join(CUSTOM_DIR, "plot_filters.py"),
                "--weights",
                weights_path,
                "--matrix",
                matrix_path,
                "--gene-label",
                gene_label,
                "--out",
                os.path.join(plots_dir, "filters.png"),
                "--txt",
                os.path.join(plots_dir, "real_filters.txt"),
                "--logos",
                os.path.join(plots_dir, "filter_logos.png"),
                "--real-logos",
                os.path.join(plots_dir, "filter_logos_real.png"),
                "--model-logos",
                os.path.join(plots_dir, "model_logos.png"),
                "--meta-out",
                os.path.join(plots_dir, "plot_metadata.json"),
                "--logo-activation-frac",
                str(args.logo_activation_frac),
                "--min-contribution-frac",
                str(args.min_contribution_frac),
            ],
        )

    if want("curves"):
        run_step(
            "Training curves",
            [
                py,
                os.path.join(CUSTOM_DIR, "plot", "plot_training_curves.py"),
                "--history",
                history_path,
                "--out",
                os.path.join(plots_dir, "training_curves.png"),
            ],
        )

    # Pooled space with real windows from this run's own gene: the same measurement the
    # orthogonality penalty optimizes, at the same pool width, so the heatmap and the
    # run's mean_abs_r curve are directly comparable.
    if want("covariance"):
        run_step(
            "Filter co-activation covariance",
            [
                py,
                os.path.join(CUSTOM_DIR, "filter_coactivation_covariance.py"),
                "--checkpoint",
                weights_path,
                "--matrix",
                matrix_path,
                "--pooled-space",
                "--pool-length",
                str(args.pool_length),
                "--weighting",
                "real",
                "--out-dir",
                cov_dir,
            ],
        )

    # The whole alignment, per species and per window, as one self-contained page: the
    # SR profile the loss is computed from, before any aggregation. required=False
    # because it is the one artifact whose size scales with the alignment (a few MB on
    # XIST/VEGFA) and a failure here must not discard a trained model's manifest.
    if want("sr-heatmap"):
        run_step(
            "Interactive SR heatmap",
            [
                py,
                os.path.join(CUSTOM_DIR, "plot", "interactive_sr_heatmap.py"),
                "--weights",
                weights_path,
                "--matrix",
                matrix_path,
                "--gene-label",
                gene_label,
                "--run-name",
                run_name,
                "--smooth-sigma",
                str(args.smooth_sigma),
                "--out",
                os.path.join(plots_dir, "sr_heatmap.html"),
            ],
            required=False,
        )

    # ── Run manifest: the resolved settings and every artifact this run produced, so
    # the directory is self-describing without opening the checkpoint. ──
    config = {
        "run_name": run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gene": {"arg": args.gene, "dir": gene_dir, "label": gene_label},
        "matrix_path": matrix_path,
        "alphabet": {
            "name": ALPHABET_NAME,
            "letters": ALPHABET,
            "n_channels": N_CHANNELS,
            "mapping": {"channel_order": ALPHABET, "swapped": ["C", "G"]},
        },
        "architecture": {
            "num_filters_per_bank": args.num_filters,
            "kernel_size": args.kernel_size,
            # "channels" is the filter-bank width (incl + skip), not the input width —
            # that is alphabet.n_channels above.
            "channels": 2 * args.num_filters,
            "filter_pairs": args.num_filters * (2 * args.num_filters - 1),
        },
        "hparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "ortho_lambda": args.ortho_lambda,
            "l1_lambda": args.l1_lambda,
            "pool_length": args.pool_length,
            "smooth_sigma": args.smooth_sigma,
            "seed": args.seed,
            "n_sample": args.n_sample,
            "min_species": args.min_species,
            "history_every": args.history_every,
        },
        "outputs": {
            "weights": weights_path,
            "history": history_path,
            "filters_png": os.path.join(plots_dir, "filters.png"),
            "filter_logos_png": os.path.join(plots_dir, "filter_logos.png"),
            "filter_logos_real_png": os.path.join(plots_dir, "filter_logos_real.png"),
            "model_logos_png": os.path.join(plots_dir, "model_logos.png"),
            "filters_txt": os.path.join(plots_dir, "real_filters.txt"),
            "plot_metadata_json": os.path.join(plots_dir, "plot_metadata.json"),
            "training_curves_png": os.path.join(plots_dir, "training_curves.png"),
            "covariance_dir": cov_dir,
            "sr_heatmap_html": os.path.join(plots_dir, "sr_heatmap.html"),
        },
        "skipped_training": args.skip_train,
    }
    config_path = os.path.join(run_dir, "run_config.json")

    # A --only run regenerates part of an existing run, so it must not overwrite the
    # manifest wholesale: generated_at and skipped_training describe the training that
    # actually happened, and rewriting them would relabel a trained run as skipped.
    # Merge the output paths in and record the partial pass separately instead.
    if args.only and os.path.exists(config_path):
        with open(config_path) as fh:
            existing = json.load(fh)
        existing.setdefault("outputs", {}).update(config["outputs"])
        existing["last_partial_run"] = {
            "at": config["generated_at"],
            "steps": sorted(args.only),
        }
        config = existing

    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    logger.info(f"\nWrote run manifest to {config_path}")
    logger.info(f"Pipeline complete — all plots are in {plots_dir}")


if __name__ == "__main__":
    main()
