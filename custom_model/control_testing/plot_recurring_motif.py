"""Recurring-motif logo grid: motif families (rows) x seeds (columns).

Companion readout to ``multi_seed_train_plot.py``. That script trains one fresh
``PNASModel`` per seed and dumps *every* significant filter of *every* seed as a
logo; the seeds are stacked but the figure says nothing about *which* motifs are
the same across seeds. This script answers that: it loads the already-trained
per-seed checkpoints (``real_weights_aad_seed{0..3}.pt`` — no retraining), finds
the filters that detect the *same* sequence across seeds, and lays them out as a
grid so a reader can see the shared motif reproduce (or not) at a glance.

Method (all seed-order / filter-index independent):

  * Fingerprint every filter by its softplus response over all 4**6 = 4096
    six-mers — an exhaustive scan, since the kernel (6) spans the whole input
    window. This vector *is* what the filter detects, independent of its index.
  * Keep only significant filters (peak softplus >= ``--sig-frac`` x that seed's
    strongest); dead filters have flat, noise-only fingerprints.
  * Compare filters across seeds by the Pearson correlation of their (mean-
    centered) fingerprints: scale/offset-invariant, so a weak and a strong copy
    of the same detector still match and a large bias can't inflate the score.
  * Discover families by greedy anchor-peeling (not single-linkage, which would
    chain broad G-rich filters together): repeatedly take the highest-"seed
    reach" filter not already used and not within the correlation threshold of an
    existing anchor, then attach, per seed, that seed's most-correlated filter
    above the threshold (blank if none). Every member is tied to its anchor by a
    *direct* correlation, so no transitive chaining.

Because members are chosen by *un-shifted* 6-mer-response correlation, a high
correlation already implies the motif sits at the same offset inside the 6-wide
window (a shifted motif fires on different 6-mers and would not correlate), so
the logos in a row are already in register — no alignment shift is applied.

Run (defaults: seeds 0-3, top 3 families):

    python plot_recurring_motif.py
    python plot_recurring_motif.py --n-motifs 6 --corr-threshold 0.8
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
import numpy as np
import pandas as pd
import torch

# Import custom_utils the same way the sibling scripts do: put custom_model/ first
# on sys.path so the module custom_model/custom_utils.py resolves.
CUSTOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CUSTOM_DIR)
from custom_utils import DNA_ALPHABET, one_hot_batch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)
NUCLEOTIDES = ["A", "C", "G", "T"]


def logo_from_resp(resp, all_seqs, kernel_size, frac):
    """Information-content logo for one filter, from its 6-mer response vector.

    Mirrors the logo pipeline in multi_seed_train_plot.py: build the logo from
    every 6-mer whose softplus activation reaches ``frac`` x this filter's own
    peak, take per-position base frequencies (no pseudocount), convert to Shannon
    information content (bits), and scale the frequencies by it.

    Returns (heights (4, kernel_size), n_sel, consensus_str, info). The consensus
    is information-aware: a position is upper-cased only when it carries >= 0.5 bit,
    otherwise lower-cased, so a label can't overstate a weak flanking position.
    """
    peak = float(resp.max())
    thresh = frac * peak
    sel_seqs = [all_seqs[i] for i in np.argsort(-resp) if resp[i] >= thresh]
    n_sel = len(sel_seqs)
    counts = one_hot_batch(sel_seqs).sum(axis=0)  # (4, kernel_size)
    freq = counts / n_sel
    info = 2.0 + np.sum(
        np.where(freq > 0, freq * np.log2(freq, where=freq > 0), 0.0), axis=0
    )
    heights = freq * info[None, :]  # (4, kernel_size), all >= 0
    consensus = "".join(
        NUCLEOTIDES[b] if info[p] >= 0.5 else NUCLEOTIDES[b].lower()
        for p, b in enumerate(freq.argmax(axis=0))
    )
    return heights, n_sel, consensus, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Seeds whose real_weights_aad_seed{S}.pt checkpoints to compare (default 0 1 2 3).",
    )
    parser.add_argument(
        "--n-motifs",
        type=int,
        default=3,
        help="Number of recurring-motif families (rows) to show, strongest first (default 3).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Two filters count as the same detector when their 6-mer-response Pearson "
        "correlation exceeds this (default 0.8; the cross-seed null is ~0).",
    )
    parser.add_argument(
        "--sig-frac",
        type=float,
        default=0.2,
        help="Per seed, a filter is significant when its peak softplus is at least this "
        "fraction of that seed's strongest filter's peak (default 0.2).",
    )
    parser.add_argument(
        "--logo-activation-frac",
        type=float,
        default=0.75,
        help="Per filter, build its logo from every 6-mer whose softplus activation is at "
        "least this fraction of that filter's own peak (default 0.75).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "recurring_motif_logos.png"),
        help="Output path for the family x seed logo grid.",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Output path for the run-metadata sidecar JSON. Defaults to "
        "recurring_motif_metadata.json next to --out.",
    )
    args = parser.parse_args()

    # ── Load each seed's checkpoint and collect every filter as a behavioral
    # fingerprint. Weights are read straight from the state dict (no PNASModel). ──
    ckpts = {}
    for seed in args.seeds:
        path = os.path.join(_HERE, f"real_weights_aad_seed{seed}.pt")
        ckpts[seed] = torch.load(path, map_location="cpu", weights_only=False)

    # Kernel size / filter count from the first checkpoint (same architecture for all).
    w0 = ckpts[args.seeds[0]]["model_state_dict"]["conv_incl.weight"]
    num_filters, _, kernel_size = w0.shape

    # All 4**kernel_size six-mers, one-hot (identical for every filter) built once.
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    filters = []  # one dict per filter across all seeds
    for seed in args.seeds:
        sd = ckpts[seed]["model_state_dict"]
        for bank in ["INCL", "SKIP"]:
            weight = sd[f"conv_{bank.lower()}.weight"].numpy()  # (num_filters, 4, K)
            bias = sd[f"conv_{bank.lower()}.bias"].numpy()
            for fi in range(num_filters):
                raw = np.einsum("ncp,cp->n", onehot, weight[fi]) + bias[fi]
                resp = np.logaddexp(0.0, raw)  # softplus, numerically stable
                filters.append(
                    dict(
                        seed=seed,
                        bank=bank,
                        idx=fi,
                        peak=float(resp.max()),
                        bias=float(bias[fi]),
                        resp=resp,
                    )
                )

    # Significance gate (per seed): drop dead filters whose flat fingerprints would
    # just add correlation noise.
    seed_maxpeak = {
        s: max(f["peak"] for f in filters if f["seed"] == s) for s in args.seeds
    }
    for f in filters:
        f["sig"] = f["peak"] >= args.sig_frac * seed_maxpeak[f["seed"]]
    sig_idx = [i for i, f in enumerate(filters) if f["sig"]]

    # Pairwise Pearson correlation of the (mean-centered) fingerprints = normalized
    # dot product of the response vectors, over ALL filters (indexed into below).
    R = np.stack([f["resp"] for f in filters])  # (N, 4096)
    Rc = R - R.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Rc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Rn = Rc / norm
    CORR = Rn @ Rn.T  # (N, N)

    thr = args.corr_threshold

    # "Seed reach" of a significant filter: how many *other* seeds contain some
    # significant filter correlated above threshold with it.
    def seed_reach(i):
        reached = set()
        for j in sig_idx:
            if filters[j]["seed"] != filters[i]["seed"] and CORR[i, j] > thr:
                reached.add(filters[j]["seed"])
        return len(reached)

    reach = {i: seed_reach(i) for i in sig_idx}

    # ── Greedy anchor-peeling into families ─────────────────────────────────────
    # Repeatedly pick the highest-reach filter that is neither already a member of a
    # family nor within `thr` of an existing anchor (so families stay distinct), then
    # attach its best above-threshold partner in each seed.
    member_used = set()
    anchors = []
    families = []
    while len(families) < args.n_motifs:
        candidates = [
            i
            for i in sig_idx
            if i not in member_used
            and all(CORR[i, a] <= thr for a in anchors)
            and reach[i] >= 1  # must reproduce in >= 1 other seed to be "recurring"
        ]
        if not candidates:
            break
        anchor = max(candidates, key=lambda i: (reach[i], filters[i]["peak"]))
        anchors.append(anchor)

        members = {}  # seed -> filter index
        for s in args.seeds:
            pool = [
                j
                for j in sig_idx
                if filters[j]["seed"] == s
                and j not in member_used
                and CORR[anchor, j] > thr
            ]
            if pool:
                best = max(pool, key=lambda j: (CORR[anchor, j], filters[j]["peak"]))
                members[s] = best
                member_used.add(best)
        # The anchor's own seed is guaranteed a member (self-corr = 1 > thr).
        span = len(members)
        _, _, anchor_consensus, _ = logo_from_resp(
            filters[anchor]["resp"], all_seqs, kernel_size, args.logo_activation_frac
        )
        families.append(
            dict(anchor=anchor, members=members, span=span, consensus=anchor_consensus)
        )

    # Order rows by span (seeds reproduced) then anchor strength — strongest first.
    families.sort(key=lambda fam: (-fam["span"], -filters[fam["anchor"]]["peak"]))

    logger.info(
        f"Found {len(families)} recurring-motif families across seeds {args.seeds} "
        f"(corr>{thr}, sig>={args.sig_frac}x seed max):"
    )
    for rank, fam in enumerate(families):
        parts = []
        for s in args.seeds:
            if s in fam["members"]:
                m = filters[fam["members"][s]]
                parts.append(
                    f"s{s}:{m['bank']}#{m['idx']}(r={CORR[fam['anchor'], fam['members'][s]]:.2f})"
                )
            else:
                parts.append(f"s{s}:-")
        logger.info(
            f"  [{fam['consensus']}] span {fam['span']}/{len(args.seeds)}  " + "  ".join(parts)
        )

    if not families:
        logger.warning("No recurring families found; nothing to plot.")
        return

    # ── Render the grid: families (rows) x seeds (columns) ──────────────────────
    n_rows, n_cols = len(families), len(args.seeds)
    row_h, header_in, footer_in = 1.9, 1.3, 0.7
    fig_h = n_rows * row_h + header_in + footer_in
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.3, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.7, "wspace": 0.3},
    )
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=footer_in / fig_h,
        left=0.075,
        right=0.98,
    )

    for r, fam in enumerate(families):
        for c, s in enumerate(args.seeds):
            ax = axes[r][c]
            if s not in fam["members"]:
                # Non-reproduction is informative: leave the cell empty but labelled.
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"absent\n(<{thr:g})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.6",
                    style="italic",
                    transform=ax.transAxes,
                )
                continue
            m = filters[fam["members"][s]]
            heights, n_sel, _, _ = logo_from_resp(
                m["resp"], all_seqs, kernel_size, args.logo_activation_frac
            )
            df = pd.DataFrame(heights.T, columns=NUCLEOTIDES)
            logomaker.Logo(df, ax=ax, color_scheme="classic")
            ax.set_ylim(0, 2)  # standard information-content scale (bits)
            if c == 0:
                ax.set_ylabel("bits", fontsize=6)
            ax.set_title(
                f"{m['bank']}#{m['idx']}  r={CORR[fam['anchor'], fam['members'][s]]:.2f}  "
                f"pk={m['peak']:.2f}",
                fontsize=8,
            )
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)

    # Column headers (seed k) above the top row, and rotated family labels at left.
    for c, s in enumerate(args.seeds):
        pos = axes[0][c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.45 * header_in / fig_h,
            f"seed {s}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
    for r, fam in enumerate(families):
        pos = axes[r][0].get_position()
        fig.text(
            0.015,
            (pos.y0 + pos.y1) / 2,
            f"{fam['consensus']}  ({fam['span']}/{n_cols})",
            rotation=90,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=11,
        )

    # Pull the shared training hparams from a checkpoint's metadata for the caption.
    hp = ckpts[args.seeds[0]].get("metadata", {}).get("hparams", {})
    fig.suptitle(
        "Recurring-motif families across seeds (information-content logos)\n"
        f"row = family (label = consensus, span), col = seed; each cell is that seed's "
        f"filter with 6-mer-response corr > {thr:g} to the family anchor; "
        f"blank = no such filter (motif did not reproduce in that seed)",
        y=1 - 0.14 * header_in / fig_h,
        va="top",
        fontsize=10,
    )
    caption = (
        f"real MALAT1 cross-species AAD loss | seeds {args.seeds} | "
        f"epochs {hp.get('num_epochs', '?')} | lr {hp.get('lr', '?')}, "
        f"l1λ {hp.get('l1_lambda', '?')}, σ {hp.get('smooth_sigma', '?')} | "
        f"corr>{thr:g}, sig>={args.sig_frac:g}x seed max, logos from 6-mers "
        f">={args.logo_activation_frac:g}x each filter's peak | note: the detecting "
        f"bank (INCL vs SKIP) can differ across a row — the sequence preference recurs, "
        f"its inclusion/skipping role need not."
    )
    fig.text(
        0.5, 0.4 * footer_in / fig_h, caption, ha="center", va="center", fontsize=7, wrap=True
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Wrote {len(families)}x{n_cols} recurring-motif grid to {os.path.normpath(args.out)}"
    )

    # ── Sidecar JSON: the families, their anchors and per-seed members. ─────────
    meta_out = args.meta_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "recurring_motif_metadata.json"
    )
    sidecar = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "hparams": {
            "seeds": args.seeds,
            "n_motifs": args.n_motifs,
            "corr_threshold": thr,
            "sig_frac": args.sig_frac,
            "logo_activation_frac": args.logo_activation_frac,
            "training": hp,
        },
        "families": [
            {
                "consensus": fam["consensus"],
                "span": fam["span"],
                "anchor": {
                    "seed": filters[fam["anchor"]]["seed"],
                    "bank": filters[fam["anchor"]]["bank"],
                    "idx": filters[fam["anchor"]]["idx"],
                    "peak": filters[fam["anchor"]]["peak"],
                },
                "members": {
                    str(s): (
                        {
                            "bank": filters[fam["members"][s]]["bank"],
                            "idx": filters[fam["members"][s]]["idx"],
                            "peak": filters[fam["members"][s]]["peak"],
                            "corr_to_anchor": float(CORR[fam["anchor"], fam["members"][s]]),
                        }
                        if s in fam["members"]
                        else None
                    )
                    for s in args.seeds
                },
            }
            for fam in families
        ],
        "provenance": {
            "logos_png": os.path.abspath(args.out),
            "checkpoints": {
                str(s): os.path.abspath(
                    os.path.join(_HERE, f"real_weights_aad_seed{s}.pt")
                )
                for s in args.seeds
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(meta_out, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    logger.info(f"Wrote run metadata to {os.path.normpath(meta_out)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
