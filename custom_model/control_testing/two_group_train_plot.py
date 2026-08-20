"""Two-group (phylogenetically balanced) cross-species AAD training + combined filter readout.

A sibling of ``multi_seed_train_plot.py``. That script is a *seed*-robustness control
(same species pool, many random seeds). This one flips the varying axis: it holds the
seed fixed and instead splits the 62 alignment species into **two disjoint groups**,
training one fresh ``PNASModel`` per group and rendering **one combined figure per
readout** so the two groups can be compared side by side:

  * ``combined_filters_heatmap.png`` — every INCL/SKIP filter of every group as a
    4x6 weight heat map, groups stacked vertically (one labelled block per group,
    shared symmetric color scale so panels are comparable across groups).
  * ``combined_filter_logos.png``    — each group's *significant* filters as
    information-content (bits) sequence logos, one labelled row-block per group.

The comparison asks "do we recover the same filters from two disjoint halves of the
species?" — a data-subset robustness check rather than an init-noise check.

**Why the split is balanced, not random.** The loss is a cross-species deviation down
each alignment column, normalized by the within-species dynamic range, so a group's
*phylogenetic diversity* (branch-length spread) directly drives the training signal. A
random 31/31 draw risks handing one group the tightly-related primates+rodents (weak
signal) and the other the divergent outgroups — the two models would then train on
different statistical regimes, confounding "which species" with "how much diversity."
Instead we stratify: the species are stored in phylogenetic order (human -> primates ->
rodents -> carnivores -> laurasiatheres/bats -> afrotheres -> marsupials -> platypus),
so alternating along that order (even index -> group A, odd -> group B) gives each group
~31 species spanning every clade with matched diversity and matched coverage.

The training math (per-species SR scatter into alignment coordinates, NaN-aware
cross-species mean-absolute-deviation numerator normalized by the within-species dynamic
range, L1 activation penalty, Gaussian SR smoothing) is identical to
``custom_real_train_absolute_deviation.py`` / ``multi_seed_train_plot.py``; only the
sampling pool (restricted to the group) and the varied axis (group instead of seed)
change. Each group's weights are saved to a group-suffixed checkpoint next to the
figures, without touching the canonical ``weights/real_weights_aad.pt``.

Run (defaults: seed 0, 10000 epochs per group — this trains two models, so it is
~2x the single-run cost):

    python two_group_train_plot.py
    python two_group_train_plot.py --seed 0 --num-epochs 10000
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

# ── Training hyperparameters (identical to custom_real_train_absolute_deviation.py
# so each group's run is faithful) ─────────────────────────────────────────────
MATRIX_PATH = os.path.join(
    CUSTOM_DIR, "..", "data", "multiz100", "alignment_matrix.npy"
)
N_SAMPLE = 10  # number of species sampled per epoch
MIN_SPECIES = 5  # a column counts toward the loss only if >= this many species present
LR = 1e-2
L1_LAMBDA = 0.002  # strength of L1 penalty on the softplus activations
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)
LOG_EVERY = 100  # console log cadence (epochs); larger than the trainer's 10 to keep
# the combined two-group log readable

NUCLEOTIDES = ["A", "C", "G", "T"]

# Species names in alignment-column order (copied names-only from the SPECIES dict in
# dataset_preparations/maf_processing.py, which is the authority on column order). Used
# purely for figure labels + the metadata sidecar; NOT imported from maf_processing.py
# because that module imports Bio at import time. The order is phylogenetic, which is
# what makes the even/odd interleave a diversity-balanced split.
SPECIES_NAMES = [
    "human",
    "chimp",
    "mouse",
    "rat",
    "cat",
    "elephant",
    "gorilla",
    "orangutan",
    "gibbon",
    "rhesus",
    "macaque",
    "baboon",
    "green_monkey",
    "marmoset",
    "squirrel_monkey",
    "bushbaby",
    "chinese_tree_shrew",
    "squirrel",
    "lesser_egyptian_jerboa",
    "prairie_vole",
    "chinese_hamster",
    "golden_hamster",
    "naked_mole-rat",
    "guinea_pig",
    "chinchilla",
    "brush-tailed_rat",
    "rabbit",
    "pika",
    "pig",
    "alpaca",
    "bactrian_camel",
    "dolphin",
    "killer_whale",
    "tibetan_antelope",
    "cow",
    "sheep",
    "domestic_goat",
    "horse",
    "white_rhinoceros",
    "dog",
    "ferret",
    "panda",
    "pacific_walrus",
    "weddell_seal",
    "black_flying-fox",
    "megabat",
    "big_brown_bat",
    "david's_myotis_bat",
    "little_brown_bat",
    "hedgehog",
    "shrew",
    "star-nosed_mole",
    "manatee",
    "cape_golden_mole",
    "tenrec",
    "aardvark",
    "armadillo",
    "opossum",
    "tasmanian_devil",
    "wallaby",
    "Platypus",
    "cape_elephant_shrew",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single random seed used identically for BOTH groups, so the only "
        "difference between the two runs is which species each can sample (default 0).",
    )
    parser.add_argument(
        "--split-mode",
        default="phylo-interleave",
        choices=["phylo-interleave"],
        help="How to split species into two groups. 'phylo-interleave' (default) alternates "
        "species along the phylogenetic column order (even idx -> A, odd idx -> B), giving "
        "each group matched clade diversity and coverage.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10000,
        help="Training epochs per group (default 10000, matching the trainer).",
    )
    parser.add_argument(
        "--out-heatmaps",
        default=os.path.join(_HERE, "combined_species_filters_heatmap.png"),
        help="Output path for the combined (group-tiled) filter heat-map figure.",
    )
    parser.add_argument(
        "--out-logos",
        default=os.path.join(_HERE, "combined_species_filter_logos.png"),
        help="Output path for the combined (group-tiled) information-content logo figure.",
    )
    parser.add_argument(
        "--min-contribution-frac",
        type=float,
        default=0.2,
        help="Per group, only render logos for filters whose peak softplus activation is "
        "at least this fraction of that group's strongest filter's peak (default 0.2). "
        "Use 0 to show every filter.",
    )
    parser.add_argument(
        "--logo-activation-frac",
        type=float,
        default=0.75,
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

    # ── One-time, group-independent setup ────────────────────────────────────
    # The alignment and every species' gap-removed input encoding are the same
    # for both groups, so build them once and reuse across the group loop.
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

    # ── Phylogenetically balanced two-group split ────────────────────────────
    # Even column index -> group A, odd -> group B. Because the columns are in
    # phylogenetic order, alternating gives each group ~half of every clade, so
    # both groups have matched diversity + coverage and the split is deterministic.
    group_a = torch.arange(0, n_total, 2, device=device)  # even idx
    group_b = torch.arange(1, n_total, 2, device=device)  # odd idx
    groups = [("A", group_a), ("B", group_b)]

    # Species names cover the alignment columns 1:1; fall back to a generic label if the
    # matrix ever carries more columns than the documented name list.
    def _members(idx_tensor):
        return [
            SPECIES_NAMES[i] if i < len(SPECIES_NAMES) else f"species_{i}"
            for i in idx_tensor.tolist()
        ]

    logger.info(
        f"Real alignment matrix: {matrix.shape} (aligned positions x species) "
        f"from {os.path.normpath(MATRIX_PATH)} on {device} — sampling {N_SAMPLE} "
        f"species/epoch, column valid when >= {MIN_SPECIES} species present"
    )
    logger.info(
        f"Split mode '{args.split_mode}': seed {args.seed} shared across both groups"
    )
    for label, gidx in groups:
        logger.info(
            f"  group {label}: {gidx.numel()} species (cols {gidx.tolist()}) "
            f"-> {', '.join(_members(gidx))}"
        )

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once
    # (it carries no learnable parameters and is identical across groups). Sum-
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

    # ── Group loop: train one fresh model per group (training logic kept inline) ──
    trained = (
        []
    )  # list of {"group", "model", "final", ...} for the plotting passes below
    for label, group_idx in groups:
        # Same seed for both groups: identical init + identical RNG stream, so the only
        # difference is which species indices each group is allowed to sample.
        torch.manual_seed(args.seed)
        model = PNASModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Running sums of the per-epoch loss statistics over the first 10 epochs, so
        # the metadata can report the early-training average (a stable snapshot of the
        # from-scratch starting regime) alongside the final-epoch values.
        first10_sums = {
            "loss": 0.0,
            "sd_loss": 0.0,
            "l1": 0.0,
            "within_species_scale": 0.0,
        }
        n_first10 = 0

        for epoch in range(1, args.num_epochs + 1):
            optimizer.zero_grad()

            # Fresh random combination of N_SAMPLE species drawn from THIS group only.
            sel = group_idx[torch.randperm(group_idx.numel(), device=device)[:N_SAMPLE]]

            # Build the aligned (N_SAMPLE, n_aligned) matrix of SR values. We never
            # put literal NaN into the differentiable tensor; `mask` marks present
            # entries. Masked-out entries stay 0 and receive no gradient.
            values = torch.zeros(N_SAMPLE, n_aligned, device=device)
            mask = torch.zeros(N_SAMPLE, n_aligned, device=device)

            # Accumulate the summed softplus activations over all sampled species so
            # the L1 penalty is a mean over every (species, window) activation.
            act_sum = torch.zeros((), device=device)
            act_count = 0
            for r, j in enumerate(sel.tolist()):
                # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
                # both (1, Lj-5) and >= 0; used by the activation L1 penalty below.
                sr_j, a_incl, a_skip = model.compute_sr_profile(
                    species_oh[j], return_activations=True
                )
                sr_j = sr_j.squeeze(0)  # (Lj-5,)
                act_sum = act_sum + (a_incl + a_skip).sum()
                act_count += a_incl.numel()

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
                mask[r, cols] = 1.0

            # Numerator: NaN-aware cross-species mean absolute deviation per column
            # (about each column's cross-species mean), valid only where at least
            # MIN_SPECIES of the sampled species are present.
            count = mask.sum(0)  # (n_aligned,)
            col_mean = (values * mask).sum(0) / count.clamp(min=1)
            col_std = ((values - col_mean).abs() * mask).sum(0) / count.clamp(min=1)
            valid = count >= MIN_SPECIES

            # Denominator: within-species dynamic range = each row's mean absolute
            # deviation across positions (NaN-aware), averaged over rows. Dividing by
            # it makes the loss scale-free so the model can't cheat to a low abs-dev
            # by shrinking all activations toward a constant.
            rc = mask.sum(1)  # (N_SAMPLE,)
            row_mean = (values * mask).sum(1) / rc.clamp(min=1)
            within_species_scale = (
                ((values - row_mean.unsqueeze(1)).abs() * mask).sum(1) / rc.clamp(min=1)
            ).mean()  # scalar

            sd_loss = (
                (col_std[valid]) / within_species_scale
            ).mean()  # normalized average column abs-dev

            # L1 penalty on the summed softplus activations (activity sparsity ->
            # peaked, data-driven SR profiles). a_incl/a_skip are already >= 0.
            l1_penalty = L1_LAMBDA * (act_sum / act_count)
            loss = sd_loss + l1_penalty
            loss.backward()
            optimizer.step()

            # Accumulate the first-10-epoch statistics for the metadata average below.
            if epoch <= 10:
                first10_sums["loss"] += loss.item()
                first10_sums["sd_loss"] += sd_loss.item()
                first10_sums["l1"] += l1_penalty.item()
                first10_sums["within_species_scale"] += within_species_scale.item()
                n_first10 += 1

            if epoch == 1 or epoch % LOG_EVERY == 0:
                mean_abs_w = (
                    model.conv_incl.weight.abs().mean().item()
                    + model.conv_skip.weight.abs().mean().item()
                ) / 2
                logger.info(
                    f"[group {label}] epoch {epoch:5d} | loss = {loss.item():.6f} "
                    f"| sd loss = {sd_loss.item():.6f} "
                    f"| l1 = {l1_penalty.item():.6f} "
                    f"| within-species scale = {within_species_scale.item():.6f} "
                    f"| valid cols = {int(valid.sum().item())} "
                    f"| mean|conv_w| = {mean_abs_w:.6f}"
                )

        # Final-epoch loss values (loop variables still in scope after the loop).
        final = {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "l1": float(l1_penalty.item()),
            "within_species_scale": float(within_species_scale.item()),
        }
        # Average of each statistic over the first min(10, num_epochs) epochs.
        first_10_epochs_avg = {k: v / n_first10 for k, v in first10_sums.items()}
        model.eval()
        members = _members(group_idx)
        trained.append(
            {
                "group": label,
                "species_idx": group_idx.tolist(),
                "members": members,
                "model": model,
                "final": final,
                "first_10_epochs_avg": first_10_epochs_avg,
            }
        )

        # Save this group's weights to a group-suffixed checkpoint next to the figures
        # (non-destructive: does not touch the canonical weights/real_weights_aad.pt).
        # Format matches train.py so plot_filters.py / load_partial_state_dict can reload.
        metadata = {
            "dataset": "real",
            "script": os.path.basename(__file__),
            "motif": None,
            "matrix_path": MATRIX_PATH,
            "hparams": {
                "num_epochs": args.num_epochs,
                "lr": LR,
                "l1_lambda": L1_LAMBDA,
                "smooth_sigma": SMOOTH_SIGMA,
                "seed": args.seed,
                "split_mode": args.split_mode,
                "group": label,
                "species_idx": group_idx.tolist(),
                "members": members,
                "n_sample": N_SAMPLE,
                "min_species": MIN_SPECIES,
            },
            "final": final,
            "first_10_epochs_avg": first_10_epochs_avg,
        }
        weights_path = os.path.join(_HERE, f"real_weights_aad_group{label}.pt")
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
        logger.info(
            f"[group {label}] done — final loss {final['loss']:.6f} "
            f"(first-{n_first10}-epoch avg loss {first_10_epochs_avg['loss']:.6f}); "
            f"saved weights to {os.path.normpath(weights_path)}"
        )

    num_filters, channels, kernel_size = model.conv_incl.weight.shape
    n_cols = 5

    # Shared run-metadata footer used on both figures.
    caption = (
        f"real MALAT1 cross-species AAD loss | {args.split_mode} split | seed {args.seed} | "
        f"epochs {args.num_epochs} | lr {LR}, l1λ {L1_LAMBDA}, σ {SMOOTH_SIGMA} | "
        + "  ".join(
            f"grp {t['group']} (n={len(t['species_idx'])}): loss={t['final']['loss']:.3f}"
            for t in trained
        )
    )

    # ── Combined heat-map figure: groups stacked vertically, one block per group ──
    # Shared symmetric color scale (diverging around 0) over ALL groups' both banks
    # so every panel is directly comparable across groups.
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
    rows_per_group = 2 * rows_per_conv  # INCL bank on top, SKIP bank below
    n_rows = len(trained) * rows_per_group
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
        group_row0 = s * rows_per_group
        convs = [("INCL", t["model"].conv_incl), ("SKIP", t["model"].conv_skip)]
        for block, (name, conv) in enumerate(convs):
            weight = conv.weight.detach().cpu()
            bias = conv.bias.detach().cpu()
            for filter_idx in range(num_filters):
                ax = axes[group_row0 + block * rows_per_conv + filter_idx // n_cols][
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
    # Rotated group labels on the far left of each group's block (computed from the
    # block's top/bottom axis positions, so they never collide with panel titles).
    for s, t in enumerate(trained):
        group_row0 = s * rows_per_group
        y_top = axes[group_row0][0].get_position().y1
        y_bot = axes[group_row0 + rows_per_group - 1][0].get_position().y0
        fig.text(
            0.01,
            (y_top + y_bot) / 2,
            f"group {t['group']}",
            rotation=90,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
    # Horizontal divider lines separating consecutive group blocks, drawn in the gap
    # between the last row of one group and the first row of the next (figure coords,
    # like plot_filters.py's INCL/SKIP divider).
    for s in range(1, len(trained)):
        y_above = (
            axes[s * rows_per_group - 1][0].get_position().y0
        )  # bottom of prev group
        y_below = axes[s * rows_per_group][0].get_position().y1  # top of next group
        y_line = (y_above + y_below) / 2
        x_right = axes[s * rows_per_group][n_cols - 1].get_position().x1
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
        "Two-group conv_incl / conv_skip filters "
        "(rows A/C/G/T, cols = kernel position)"
    )
    fig.text(0.5, 0.004, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.savefig(args.out_heatmaps, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Wrote {len(trained) * 2 * num_filters} filter heat maps "
        f"({len(trained)} groups) to {os.path.normpath(args.out_heatmaps)}"
    )

    # ── Combined logos figure: one row-block per group ───────────────────────
    # Because the kernel (6) spans the whole input window, a filter's response over
    # every possible input is an exhaustive scan of the 4**6 = 4096 six-mers. The
    # all-6-mers one-hot encoding is identical for every filter/group, so build once.
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    # Per group: keep its significant filters (peak softplus >= min_contribution_frac
    # of that group's own strongest filter), INCL bank first then SKIP, each sorted
    # strongest-first — matching plot_filters.py's within-model semantics.
    group_blocks = []
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
        group_blocks.append(
            {
                "group": t["group"],
                "kept": kept,
                "rows": max(1, math.ceil(len(kept) / n_cols)),
            }
        )

    num_total = len(trained) * 2 * num_filters
    num_kept = sum(len(b["kept"]) for b in group_blocks)
    logger.info(
        f"Rendering {num_kept}/{num_total} filters across {len(trained)} groups with "
        f"peak softplus >= {args.min_contribution_frac:g}x each group's strongest"
    )

    # Grid sized to the kept filters: one row-block per group, each packed row-major
    # into n_cols columns (wrapping to extra rows when a group keeps > n_cols filters).
    logo_n_rows = sum(b["rows"] for b in group_blocks)
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
    for bi, b in enumerate(group_blocks):
        # Horizontal divider line above every group block after the first, drawn in the
        # gap between the previous block's last row and this block's first row.
        if bi > 0:
            y_above = axes[row_offset - 1][0].get_position().y0  # bottom of prev group
            y_below = axes[row_offset][0].get_position().y1  # top of this group
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
        # Rotated group label on the far left of this group's block.
        y_top = axes[row_offset][0].get_position().y1
        y_bot = axes[row_offset + b["rows"] - 1][0].get_position().y0
        fig.text(
            0.01,
            (y_top + y_bot) / 2,
            f"group {b['group']}",
            rotation=90,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
        row_offset += b["rows"]

    fig.suptitle(
        "Two-group conv_incl / conv_skip information-content (bits) logos\n"
        f"({num_kept}/{num_total} filters across {len(trained)} groups; peak softplus "
        f">= {args.min_contribution_frac:g}x each group's strongest; from 6-mers >= "
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

    # ── Sidecar JSON: shared hparams, per-group final losses + checkpoints, and the
    # figures this run produced, written next to the figures. ──
    meta_out = args.meta_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out_logos)), "combined_plot_metadata.json"
    )
    sidecar = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "matrix_path": os.path.abspath(MATRIX_PATH),
        "hparams": {
            "seed": args.seed,
            "split_mode": args.split_mode,
            "num_epochs": args.num_epochs,
            "lr": LR,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "n_sample": N_SAMPLE,
            "min_species": MIN_SPECIES,
            "min_contribution_frac": args.min_contribution_frac,
            "logo_activation_frac": args.logo_activation_frac,
        },
        "groups": [
            {
                "group": t["group"],
                "species_idx": t["species_idx"],
                "members": t["members"],
            }
            for t in trained
        ],
        "per_group": [
            {
                "group": t["group"],
                "final": t["final"],
                "first_10_epochs_avg": t["first_10_epochs_avg"],
                "weights": os.path.abspath(t["weights"]),
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
