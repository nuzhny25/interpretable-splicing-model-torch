"""Per-window co-activation covariance between the N+N sequence filters.

The filter count (``F`` below) and kernel width (``k``) are read from the checkpoint,
so any architecture is analysed correctly; the numbers quoted throughout are for the
original 20+20 filters at kernel 6, where ``2F = 40`` channels give 780 pairs.

Because the model combines filters purely additively (the SR profile is
``sum_i softplus(incl_i) - sum_i softplus(skip_i)``), there is no pairwise
filter-interaction term inside the model itself. "How filter i interacts with
filter j" is therefore defined empirically here as how their per-window
activations co-vary across the real MALAT1 alignment.

For every species in ``data/multiz100/alignment_matrix.npy`` the gap-removed
sequence is run through ``conv_incl`` and ``conv_skip``; each k-nt sliding
window produces one non-negative ``softplus`` activation per filter. Stacking
the two banks gives ``2F`` filter channels (``incl_00..``, ``skip_00..``). Every
window across every species is one sample (row) of the data matrix ``X`` of shape
``(n_windows, 2F)``. Before the correlation, each
of the ``2F`` per-window tracks is Gaussian-smoothed along position with the same
operator the training loss uses (sum-normalized kernel, radius ~3 sigma, reflect
padding, applied per species to the dense gap-free track); sigma comes from the
checkpoint's own ``smooth_sigma`` unless ``--smooth-sigma`` overrides it, and
``--smooth-sigma 0`` recovers the raw per-window matrix. Smoothing is linear, so
smoothing the tracks separately is exactly the per-filter decomposition of the
smoothed SR track the loss saw: the correlation then measures co-activation over
denoised neighbourhoods rather than of single-window jitter. The deliverable is the
``2F x 2F`` covariance matrix ``cov(X)`` and its normalized correlation ``corr(X)``:
entry (i, j) says whether filters i and j fire together at the same window.

``--kmer-space`` computes the same matrix exactly, but far more cheaply. The
convolution spans the whole window on one-hot input, so a filter's activation at
a window is a pure function of that window's k-mer: the 421,668-row alignment
matrix ``X`` has at most ``4**k`` distinct rows (4096 at kernel 6). Enumerating
all of them once gives an activation table ``A`` of shape ``(4**k, 2F)``, and the
alignment covariance is recovered exactly as the k-mer-count-weighted covariance of ``A``
(verified to ~6e-08 in correlation; the only gap is ``np.cov``'s ddof=1). This
mode is an exhaustive enumeration rather than a sample, so there is no sampling
error and no phylogenetic pseudo-replication, and it uses the population
(ddof=0) weighted covariance. It has no position axis, so it is necessarily
unsmoothed and reproduces the alignment scan only under ``--smooth-sigma 0``.

``--weighting`` then selects what the rows are weighted by (or, under
``--pooled-space``, what they are sampled from), isolating one factor per rung:
  * ``uniform``      — intrinsic filter-bank geometry, sequence-independent.
  * ``composition``  — k-mer probability under independent bases drawn at the
    alignment's mononucleotide frequencies (adds base composition only).
  * ``real`` (a.k.a. ``malat1``) — observed k-mer counts in the alignment
    (reproduces the default alignment-scan matrix exactly).
Differences between rungs attribute structure to geometry vs. composition vs.
real k-mer usage.

``--pooled-space`` fixes a structural blind spot in both of the above. Both score
filters on a *single* k-nt window, so an observation is "all ``2F`` filters at one
window" and the pairing is locked to window index. But the kernel is wider
than the motifs it learns (most filters use only 3-5 of the 6 slots at kernel 6), so two
filters can learn the *same* motif at *different registers* inside the kernel.
When they do, they fire on disjoint sets of k-mers — their large deviations never
land on the same observation — and the covariance collapses even though the
filters are near-duplicates. ``incl_10`` and ``skip_05`` of ``real_weights_aad.pt``
are exactly this case: same G-rich motif one window apart, top-64 6-mer sets
overlapping 0/64, r = +0.11 (rank 263/780) despite r = +0.78 at lag -1.

This mode instead scores each filter on ``--pool-length`` nt (default 20) of
sequence and *sums* the softplus activations over all ``L - k + 1`` sliding windows
inside it. Summing over position makes the score translation-invariant within the
sequence, so a motif contributes no matter which register it lands in and the
offset is marginalized out rather than needing a lag axis. ``--n-samples``
sequences (default 500,000) are drawn per the ``--weighting`` rung above:
``uniform``/``composition`` generate i.i.d. bases, ``real``/``malat1`` takes windows
from the alignment (all of them when fewer than ``--n-samples`` exist, which for
this alignment is the case at L=20: 420,800 windows). On ``real_weights_aad.pt``
this lifts the ``incl_10``/``skip_05`` pair from +0.11 to +0.69.

Two caveats when reading a pooled matrix:
  * The baseline floats up — mean |r| over the pairs goes 0.165 (6-mer, 20+20) to
    ~0.29-0.34 (pooled), because pooling makes every filter partly a detector for
    "is there a favourable patch anywhere in this window". Values are *not*
    comparable to the 6-mer matrix; recalibrate what counts as high.
  * Register shift and co-occurrence are conflated, and measurably so. Pooling
    cannot distinguish "same motif, different register" from "two motifs that
    co-occur within L nt". Across the full matrix the lift is *not* purely
    register recovery: pairs in which at least one filter uses all six slots —
    and therefore has no slack to be a clean translate of anything — gain more
    on average (+0.19) than pairs where both filters have a dead slot (+0.10).
    So read a high pooled r as "these two respond to overlapping sequence within
    L nt", which is the weaker claim. To assert an actual register shift for a
    specific pair, confirm it against the two filters' slot maps (compare their
    informative slots) or against the lagged cross-correlation on a sequence
    scan. ``incl_10``/``skip_05`` (slots 0-3 vs 2-4, r = +0.78 at lag -1) does
    confirm; not every promoted pair will.

Note that ``--kmer-space`` and the alignment scan measure *same-window*
co-activation only; spacing relationships ("filter A fires 10 nt upstream of B")
are a lagged cross-correlation and are not captured by either. ``--pooled-space``
is insensitive to spacing within its window rather than blind to it.

Outputs (written to ``--out-dir``):
  * ``covariance.npy`` / ``correlation.npy`` — ``2F x 2F`` matrices (filter order).
  * ``correlation.csv`` — labeled correlation matrix for quick inspection.
  * ``filter_order.txt`` — row/column order of the heatmap (clustering leaf
    order, or the fixed filter order under ``--natural-order``).
  * ``coactivation_correlation.png`` — correlation heatmap (clustered by
    default; ``--natural-order`` keeps the fixed incl_00..skip_NN axes).
  * ``mean_activation.csv`` — per-filter mean/std activation (scale reference).
  * ``block_structure.csv`` — mean/sd/range of r within incl-incl, skip-skip and
    across incl-skip; a positive incl-skip mean means the two additive banks
    respond to the same sequence and partially cancel rather than dividing it.
  * ``kmer_activations.npy`` / ``kmer_weights.npy`` / ``kmer_drivers.csv``
    — ``--kmer-space`` only: the ``4**k x 2F`` table, its row weights, and the
    k-mers contributing most of the covariance for the strongest filter pairs.
"""

import argparse
import itertools
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

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "multiz100", "alignment_matrix.npy"
)
# The filter count and kernel width are read from the checkpoint inside main() rather
# than fixed here: they are properties of the model being analysed, and hardcoding 20/6
# silently mis-slices the block structure below for any other architecture.
POOL_BATCH = 20_000  # --pooled-space: sequences scored per forward pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            os.path.dirname(__file__), "weights", "real_weights_aad.pt"
        ),
        help="Model checkpoint (.pt). Default: the MAD/AAD model real_weights_aad.pt.",
    )
    parser.add_argument("--matrix", default=MATRIX_PATH, help="Alignment matrix .npy.")
    parser.add_argument(
        "--kmer-space",
        action="store_true",
        help="Build the matrix from all 4**k k-mers (4096 at the default kernel 6) "
        "instead of scanning the alignment. Exact (window activations depend only on "
        "the window's k-mer), ~100x cheaper, and free of pseudo-replication.",
    )
    parser.add_argument(
        "--pooled-space",
        action="store_true",
        help="Score filters on --pool-length nt of sequence and sum the softplus "
        "activations over all sliding windows inside it, instead of using a "
        "single k-nt window. Position-summing is translation-invariant within the "
        "window, so filters that learn the same motif at different offsets inside "
        "the kernel are no longer scored as unrelated.",
    )
    parser.add_argument(
        "--pool-length",
        type=int,
        default=20,
        help="--pooled-space only: sequence length to pool over; each contributes "
        "pool_length - kernel_size + 1 windows. Default: 20. The gain saturates around "
        "twice the kernel width (the span the register ambiguity can occupy); a "
        "pool_length equal to the kernel degenerates to one window per sequence, i.e. "
        "a sampled version of --kmer-space.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500_000,
        help="--pooled-space only: how many sequences to score. Default: 500000. "
        "Under 'real'/'malat1' weighting this caps the number of real windows used; if "
        "fewer exist, all of them are used. Default: 500000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="--pooled-space only: RNG seed for sequence sampling. Default: 0.",
    )
    parser.add_argument(
        "--weighting",
        choices=("uniform", "composition", "malat1", "real"),
        default="uniform",
        help="--kmer-space / --pooled-space only: the sequence distribution. "
        "'uniform' gives intrinsic filter geometry; 'composition' uses the "
        "alignment's mononucleotide frequencies; 'real' uses the actual sequence of "
        "whichever alignment --matrix points at (observed k-mer counts under "
        "--kmer-space, where it reproduces the alignment-scan matrix exactly; real "
        "sampled windows under --pooled-space). 'malat1' is the original name for "
        "'real' and behaves identically. Default: uniform.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (in windows) applied along position to each "
        "per-window activation track before the correlation, matching the "
        "training loss's smoothing operator. Default: the checkpoint's own "
        "smooth_sigma (1.5 if the checkpoint carries none); 0 disables it. "
        "Alignment scan only — --kmer-space has no position axis.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "filter_covariance"),
        help="Directory for the covariance/correlation matrices and heatmap.",
    )
    parser.add_argument(
        "--natural-order",
        action="store_true",
        help="Plot the heatmap in the fixed incl_00..skip_NN order instead of "
        "the clustered order (fixed axes are comparable across checkpoints).",
    )
    args = parser.parse_args()
    if args.kmer_space and args.pooled_space:
        parser.error("--kmer-space and --pooled-space are mutually exclusive.")
    # --pool-length is validated against the kernel size after the checkpoint is loaded.
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # Size the model to the checkpoint before loading. load_state_dict raises on a
    # shape mismatch even at strict=False, so the architecture has to be read off the
    # weights first; it then drives the labels, the block slicing and the k-mer
    # enumeration below, none of which are correct at a fixed 20/6.
    num_filters, _, kernel_size = sd["conv_incl.weight"].shape
    model = PNASModel(num_filters=num_filters, kernel_size=kernel_size).to(device)
    model.load_partial_state_dict(sd)
    model.eval()
    logger.info(
        f"Checkpoint architecture: {num_filters}+{num_filters} incl/skip filters, "
        f"{kernel_size}-nt kernel -> {2 * num_filters} correlated channels, "
        f"{2 * num_filters * (2 * num_filters - 1) // 2} pairs."
    )

    # Deferred from the parser: the minimum pooling width is the kernel size, which is
    # only known once the checkpoint is loaded. Below it a sequence holds no full window.
    if args.pooled_space and args.pool_length < kernel_size:
        parser.error(
            f"--pool-length must be at least the checkpoint's {kernel_size}-nt kernel "
            f"size (got {args.pool_length})."
        )

    # ── Smoothing sigma: the checkpoint's own, unless overridden ─────────────
    # The alignment scan below smooths each per-window track exactly as the loss
    # that trained this checkpoint did, so the correlation is between the tracks
    # the model was actually optimized on. Reading smooth_sigma from the
    # checkpoint keeps the two in step across checkpoints trained at different
    # sigmas; --smooth-sigma 0 falls back to raw per-window activations.
    smooth_sigma = args.smooth_sigma
    if smooth_sigma is None:
        smooth_sigma = 1.5  # matches SMOOTH_SIGMA in the training scripts
        if isinstance(ckpt, dict):
            hp = ckpt.get("metadata", {}).get("hparams", {})
            if hp.get("smooth_sigma") is not None:
                smooth_sigma = float(hp["smooth_sigma"])
    if args.kmer_space and smooth_sigma:
        logger.warning(
            f"--kmer-space enumerates isolated 6-mers and has no position axis; "
            f"ignoring smoothing (sigma {smooth_sigma:g})."
        )
        smooth_sigma = 0.0
    if args.pooled_space and smooth_sigma:
        logger.warning(
            f"--pooled-space sums activations over position, so its output has no "
            f"position axis; ignoring smoothing (sigma {smooth_sigma:g})."
        )
        smooth_sigma = 0.0

    labels = [f"incl_{i:02d}" for i in range(num_filters)] + [
        f"skip_{i:02d}" for i in range(num_filters)
    ]
    kmers = None  # set under --kmer-space; also flags the weighted-covariance path
    weights = None  # per-row weights over X; None -> unweighted np.cov

    if args.pooled_space:
        # ── Position-pooled scoring over longer sequences ─────────────────────
        # The 6-nt kernel is wider than the motifs it learns (most filters use
        # only 3-5 of the 6 slots), so two filters can encode the same motif at
        # different offsets inside the window and then fire on disjoint 6-mers —
        # their large deviations never share an observation and the covariance
        # collapses. Summing softplus over every window of an L-nt sequence is
        # translation-invariant within that sequence, so the offset is
        # marginalized out instead of being scored as unrelatedness.
        L, n_samples = args.pool_length, args.n_samples
        rng = np.random.default_rng(args.seed)
        eye = np.eye(4, dtype=np.float32)

        if args.weighting == "uniform":
            codes = rng.integers(0, 4, size=(n_samples, L))
            logger.info(f"Sampling {n_samples:,} uniform random {L}-mers on {device}.")
        else:
            # Both remaining rungs need the alignment's own sequence. Reuse
            # str_to_vector for its character validation, then collapse the
            # one-hot back to integer codes (ACGT order, as everywhere else).
            matrix = np.load(args.matrix)
            per_species = []
            for j in range(matrix.shape[1]):
                row = matrix[:, j]
                seq = "".join(row[(row != "-") & (row != "N")]).upper()
                per_species.append(str_to_vector(seq).argmax(0).astype(np.int64))

            if args.weighting == "composition":
                base_counts = np.bincount(
                    np.concatenate(per_species), minlength=4
                ).astype(np.float64)
                p_base = base_counts / base_counts.sum()
                codes = rng.choice(4, size=(n_samples, L), p=p_base)
                logger.info(
                    f"Sampling {n_samples:,} {L}-mers at alignment composition "
                    "A/C/G/T = "
                    + ", ".join(f"{p:.4f}" for p in p_base)
                    + f" on {device}."
                )
            else:  # real/malat1: windows taken from the alignment itself
                codes = np.concatenate(
                    [
                        np.lib.stride_tricks.sliding_window_view(s, L)
                        for s in per_species
                        if len(s) >= L
                    ]
                )
                if len(codes) > n_samples:
                    codes = codes[rng.choice(len(codes), n_samples, replace=False)]
                    logger.info(
                        f"Sampling {n_samples:,} of the real {L}-mer windows on "
                        f"{device}."
                    )
                else:
                    # Fewer real windows than requested (the case at L=20, where
                    # the alignment holds 420,800): use every one rather than
                    # resampling with replacement, so the rung is exhaustive.
                    logger.info(
                        f"Only {len(codes):,} real {L}-mer windows exist "
                        f"(< --n-samples {n_samples:,}); using all of them on "
                        f"{device} — exhaustive, no sampling error."
                    )

        # One score per filter per sequence: softplus summed over all L-K+1
        # windows. Batched because the intermediate (batch, 2F, L-K+1) is the
        # memory peak.
        X = np.empty((len(codes), 2 * num_filters), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(codes), POOL_BATCH):
                chunk = codes[start : start + POOL_BATCH]
                x = torch.from_numpy(eye[chunk].transpose(0, 2, 1).copy()).to(
                    device
                )  # (b, 4, L)
                a_incl = F.softplus(model.conv_incl(x)).sum(-1)  # (b, F), >= 0
                a_skip = F.softplus(model.conv_skip(x)).sum(-1)  # (b, F), >= 0
                X[start : start + len(chunk)] = (
                    torch.cat([a_incl, a_skip], dim=1).cpu().numpy()
                )
        logger.info(
            f"Pooled {L - kernel_size + 1} window(s) per sequence over {len(codes):,} "
            f"{L}-mers -> X {X.shape}. Note the pooled baseline sits well above the "
            f"single-window "
            f"one (mean |r| ~0.3 vs ~0.17): read strength against that floor, not "
            f"against the --kmer-space matrix."
        )
    elif args.kmer_space:
        # ── Exhaustive k-mer enumeration ─────────────────────────────────────
        # The kernel spans the whole window on one-hot input, so a window's
        # activation depends only on that window's k-mer: 4**kernel_size rows
        # cover the whole input space (4096 at the default kernel 6).
        kmers = ["".join(p) for p in itertools.product("ACGT", repeat=kernel_size)]
        logger.info(
            f"Enumerating all {len(kmers)} {kernel_size}-mers on {device}; "
            f"weighting={args.weighting}."
        )
        with torch.no_grad():
            x = torch.tensor(
                np.stack([str_to_vector(k) for k in kmers]),
                dtype=torch.float32,
                device=device,
            )  # (4**K, 4, K)
            a_incl = F.softplus(model.conv_incl(x))  # (4**K, F, 1), >= 0
            a_skip = F.softplus(model.conv_skip(x))  # (4**K, F, 1), >= 0
            # -> (4**K, 2F): k-mers are rows, incl then skip filters are columns.
            X = torch.cat([a_incl, a_skip], dim=1).squeeze(-1).cpu().numpy()
        X = X.astype(np.float64)

        if args.weighting == "uniform":
            weights = np.ones(len(kmers), dtype=np.float64)
        else:
            # Both remaining rungs need the alignment's observed composition.
            matrix = np.load(args.matrix)
            kmer_index = {k: i for i, k in enumerate(kmers)}
            counts = np.zeros(len(kmers), dtype=np.float64)
            base_counts = np.zeros(4, dtype=np.float64)
            n_windows = 0
            for j in range(matrix.shape[1]):
                row = matrix[:, j]
                seq = "".join(row[(row != "-") & (row != "N")]).upper()
                for b, base in enumerate("ACGT"):
                    base_counts[b] += seq.count(base)
                for p in range(len(seq) - kernel_size + 1):
                    h = kmer_index.get(seq[p : p + kernel_size])
                    if h is not None:
                        counts[h] += 1
                        n_windows += 1
            if args.weighting in ("malat1", "real"):
                weights = counts
                logger.info(
                    f"Weighting by {n_windows:,} observed windows; "
                    f"{int((counts > 0).sum())}/{len(kmers)} {kernel_size}-mers present."
                )
            else:  # composition: independent bases at alignment frequencies
                p_base = base_counts / base_counts.sum()
                weights = np.ones(len(kmers), dtype=np.float64)
                for pos in range(kernel_size):
                    weights *= np.array(
                        [p_base["ACGT".index(k[pos])] for k in kmers]
                    )
                logger.info(
                    "Weighting by mononucleotide composition A/C/G/T = "
                    + ", ".join(f"{p:.4f}" for p in p_base)
                )
    else:
        # ── Load alignment and pool per-window activations across all species ────
        # matrix: (n_aligned, n_species) of single characters; gaps are "-"/"N",
        # lowercase soft-masked a/c/g/t are nucleotides (upper-cased before one-hot).
        matrix = np.load(args.matrix)
        n_aligned, n_species = matrix.shape
        logger.info(
            f"Alignment {matrix.shape} (positions x species) on {device}; "
            f"pooling per-window activations from {num_filters}+{num_filters} filters"
            + (
                f"; Gaussian smoothing along position at sigma {smooth_sigma:g}."
                if smooth_sigma
                else "; no smoothing (raw per-window activations)."
            )
        )

        # Fixed Gaussian kernel over position, built once — the same operator the
        # training loop applies to the SR track: sum-normalized (a weighted
        # average, so activation scale is preserved) with a ~3-sigma half-width.
        # One copy per filter channel, applied with groups= so the 2F tracks are
        # smoothed independently.
        if smooth_sigma:
            smooth_radius = math.ceil(3 * smooth_sigma)
            offsets = torch.arange(
                -smooth_radius, smooth_radius + 1, device=device, dtype=torch.float32
            )
            gauss_kernel = torch.exp(-(offsets**2) / (2 * smooth_sigma**2))
            gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)
            gauss_kernel = gauss_kernel.repeat(2 * num_filters, 1, 1)

        # X accumulates one row per (species, window): 2F filter activations each.
        window_blocks = []
        with torch.no_grad():
            for j in range(n_species):
                row = matrix[:, j]
                is_nuc = (row != "-") & (row != "N")
                seq = "".join(row[is_nuc]).upper()
                x = torch.tensor(str_to_vector(seq), dtype=torch.float32, device=device)
                x = x.unsqueeze(0)  # (1, 4, Lj)
                a_incl = F.softplus(model.conv_incl(x))  # (1, F, Lj-K+1), >= 0
                a_skip = F.softplus(model.conv_skip(x))  # (1, F, Lj-K+1), >= 0
                act = torch.cat([a_incl, a_skip], dim=1)  # (1, 2F, Lj-K+1)

                # Smooth each of the 2F tracks along position before pooling, so
                # a row of X is a filter's local neighbourhood rather than one
                # window's jitter. Reflect padding (replicate for a track shorter
                # than the kernel half-width) keeps the ends from being damped by
                # zero edges, exactly as in training; the species' track is dense
                # and gap-free here, so no gap ever enters the convolution.
                if smooth_sigma:
                    pad_mode = (
                        "reflect" if act.shape[-1] > smooth_radius else "replicate"
                    )
                    act = F.conv1d(
                        F.pad(act, (smooth_radius, smooth_radius), mode=pad_mode),
                        gauss_kernel,
                        groups=2 * num_filters,
                    )

                # -> (Lj-K+1, 2F): windows are rows, incl then skip filters are columns.
                block = act.squeeze(0).T.cpu().numpy()
                window_blocks.append(block.astype(np.float64))

        X = np.concatenate(window_blocks, axis=0)  # (n_windows, 2F)
        logger.info(f"Pooled {X.shape[0]:,} windows across {n_species} species.")

    # ── Covariance and correlation over the 2F filter channels ───────────────
    if weights is None:
        cov = np.cov(X, rowvar=False)  # (2F, 2F)
        corr = np.corrcoef(X, rowvar=False)  # (2F, 2F), scale-free
        mean_act = X.mean(axis=0)
        std_act = X.std(axis=0)
    else:
        # Weighted population (ddof=0) covariance: the 4**K rows are an
        # exhaustive enumeration, not a sample, so no Bessel correction applies.
        w_norm = weights / weights.sum()
        mean_act = w_norm @ X
        X_centered = X - mean_act
        cov = (X_centered * w_norm[:, None]).T @ X_centered  # (2F, 2F)
        std_act = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_act, std_act)  # (2F, 2F), scale-free

    np.save(os.path.join(args.out_dir, "covariance.npy"), cov)
    np.save(os.path.join(args.out_dir, "correlation.npy"), corr)
    _save_labeled_csv(os.path.join(args.out_dir, "correlation.csv"), corr, labels)
    with open(os.path.join(args.out_dir, "mean_activation.csv"), "w") as fh:
        fh.write("filter,mean_activation,std_activation\n")
        for lab, mu, sg in zip(labels, mean_act, std_act):
            fh.write(f"{lab},{mu:.6f},{sg:.6f}\n")

    # ── Order rows/cols by hierarchical clustering so structure is visible ───
    # With --natural-order, keep the fixed incl_00..skip_NN order instead so the
    # heatmap axes are identical across checkpoints/seeds.
    order = list(range(len(labels))) if args.natural_order else _cluster_order(corr)
    with open(os.path.join(args.out_dir, "filter_order.txt"), "w") as fh:
        fh.write("\n".join(labels[i] for i in order) + "\n")

    if args.pooled_space:
        source = (
            f"pooled {args.pool_length}-mers, {args.weighting}, "
            f"n={X.shape[0]:,}"
        )
    elif kmers:
        source = f"{kernel_size}-mer space, {args.weighting}"
    else:
        source = "MALAT1 alignment" + (
            f", smoothed sigma={smooth_sigma:g}" if smooth_sigma else ", unsmoothed"
        )
    _plot_heatmap(
        corr[np.ix_(order, order)],
        [labels[i] for i in order],
        os.path.join(args.out_dir, "coactivation_correlation.png"),
        # The architecture is in the title because the checkpoint basename alone need
        # not identify a run — a pipeline names every run's checkpoint weights.pt and
        # tells them apart by directory, so two heatmaps would otherwise be captioned
        # identically.
        title=f"Filter co-activation correlation — "
        f"{os.path.basename(args.checkpoint)} "
        f"({num_filters}+{num_filters} filters, {kernel_size}-nt kernel; {source})",
    )

    # ── Block structure: mean correlation within and between the two banks ───
    # Asks whether the additively-combined banks divide sequence space (negative
    # incl-skip block) or scan the same space and partially cancel (positive).
    tri = np.triu_indices(num_filters, k=1)
    blocks = {
        "incl-incl": corr[:num_filters, :num_filters][tri],
        "skip-skip": corr[num_filters:, num_filters:][tri],
        "incl-skip": corr[:num_filters, num_filters:].ravel(),
    }
    logger.info("\nMean co-activation by block:")
    with open(os.path.join(args.out_dir, "block_structure.csv"), "w") as fh:
        fh.write("block,n_pairs,mean_r,mean_abs_r,sd_r,min_r,max_r,frac_positive\n")
        for name, vals in blocks.items():
            fh.write(
                f"{name},{vals.size},{vals.mean():+.6f},{np.abs(vals).mean():.6f},"
                f"{vals.std():.6f},{vals.min():+.6f},{vals.max():+.6f},"
                f"{np.mean(vals > 0):.6f}\n"
            )
            logger.info(
                f"  {name}: mean r {vals.mean():+.4f}   mean |r| {np.abs(vals).mean():.4f}"
                f"   sd {vals.std():.4f}   ({vals.size} pairs, "
                f"{100 * np.mean(vals > 0):.1f}% positive)"
            )

    # ── Report the strongest interactions ────────────────────────────────────
    iu = np.triu_indices(len(labels), k=1)
    pair_corr = corr[iu]
    strongest = np.argsort(-np.abs(pair_corr))[:15]
    logger.info("\nStrongest filter co-activation pairs (|correlation|):")
    for k in strongest:
        i, j = iu[0][k], iu[1][k]
        logger.info(f"  {labels[i]} <-> {labels[j]}   r = {corr[i, j]:+.3f}")

    # ── Per-6-mer attribution for the strongest pairs ────────────────────────
    # cov[i, j] is a sum of per-6-mer terms, so the enumeration says exactly
    # which sequences drive a pair — concentrated for genuine shared motifs,
    # diffuse for pairs that merely share broad base composition.
    if kmers:
        np.save(os.path.join(args.out_dir, "kmer_activations.npy"), X)
        np.save(os.path.join(args.out_dir, "kmer_weights.npy"), weights)
        with open(os.path.join(args.out_dir, "kmer_drivers.csv"), "w") as fh:
            fh.write("filter_i,filter_j,correlation,top20_share,top_kmers\n")
            for k in strongest:
                i, j = iu[0][k], iu[1][k]
                contrib = w_norm * X_centered[:, i] * X_centered[:, j]
                top = np.argsort(-np.abs(contrib))[:20]
                share = np.abs(contrib)[top].sum() / np.abs(contrib).sum()
                fh.write(
                    f"{labels[i]},{labels[j]},{corr[i, j]:+.6f},{share:.4f},"
                    + " ".join(kmers[t] for t in top)
                    + "\n"
                )
        logger.info(
            "\nWrote per-6-mer drivers for those pairs to kmer_drivers.csv "
            "(top20_share near 1 = motif-driven, near 0 = diffuse)."
        )

    logger.info(f"\nWrote covariance/correlation matrices + heatmap to {args.out_dir}")


def _save_labeled_csv(path, mat, labels):
    """Write a square matrix with a header row/column of filter labels."""
    with open(path, "w") as fh:
        fh.write("," + ",".join(labels) + "\n")
        for lab, rowvals in zip(labels, mat):
            fh.write(lab + "," + ",".join(f"{v:.6f}" for v in rowvals) + "\n")


def _cluster_order(corr):
    """Order filters so co-activating ones sit together in the heatmap.

    Prefers SciPy average-linkage hierarchical clustering on ``1 - corr``.
    Without SciPy, falls back to numpy-only spectral (Fiedler-vector) seriation,
    then to the identity order.
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # enforce symmetry for squareform
        link = linkage(squareform(dist, checks=False), method="average")
        return list(leaves_list(link))
    except Exception as exc:  # scipy missing or degenerate distances
        logger.warning(f"SciPy clustering unavailable ({exc}); using spectral order.")

    # Fiedler-vector seriation: order by the eigenvector of the graph Laplacian
    # with the 2nd-smallest eigenvalue, on the non-negative similarity (corr+1)/2.
    try:
        sim = (corr + 1.0) / 2.0
        np.fill_diagonal(sim, 0.0)
        laplacian = np.diag(sim.sum(axis=1)) - sim
        _, eigvecs = np.linalg.eigh(laplacian)
        fiedler = eigvecs[:, 1]  # second-smallest eigenvalue
        return list(np.argsort(fiedler))
    except Exception as exc:
        logger.warning(f"Spectral seriation failed ({exc}); using original order.")
        return list(range(corr.shape[0]))


def _plot_heatmap(mat, labels, path, title):
    """Save a diverging correlation heatmap; skip cleanly if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"matplotlib unavailable ({exc}); skipping heatmap PNG.")
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.34 * n + 2, 0.34 * n + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
