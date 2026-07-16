"""Filter readout for a trained custom_model checkpoint.

Loads weights saved by custom_real_train.py (``weights/custom_model.pt`` by
default), prints each learned conv filter's consensus motif (the argmax
nucleotide at every kernel position), renders every INCL/SKIP filter as a
heat map, and renders each filter as an enrichment/depletion logo (EDLogo)
built from the 6-mers it maximally activates on (see experiment/filter_max_scan.py).
Real cross-species training has no planted motif, so the readout is
motif-agnostic: it reports what each filter converged to rather than scoring it
against a known target.

Run after training:

    python plot_filters.py
    python plot_filters.py --weights weights/custom_model.pt --out filters.png
"""

import argparse
import itertools
import json
import logging
import math
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from custom_model import PNASModel
from custom_utils import DNA_ALPHABET, one_hot_batch

NUCLEOTIDES = ["A", "C", "G", "T"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)


def filter_consensus(conv_weight) -> tuple[list[str], list[float]]:
    """Per-filter consensus motif and L1 strength.

    For each filter (4 x K) the consensus is the argmax nucleotide at every
    kernel position; strength is the filter's summed absolute weight. With no
    planted motif to score against, this reports what each filter converged to.
    """
    _, _, k = conv_weight.shape
    consensus, strength = [], []
    for f in range(conv_weight.shape[0]):
        w = conv_weight[f]
        consensus.append(
            "".join(NUCLEOTIDES[w[:, p].argmax().item()] for p in range(k))
        )
        strength.append(w.abs().sum().item())
    return consensus, strength


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=os.path.join(_HERE, "weights", "custom_model.pt"),
        help="Checkpoint saved by custom_train.py (weights under 'model_state_dict').",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "filters.png"),
        help="Output path for the filter heat-map figure.",
    )
    parser.add_argument(
        "--txt",
        default=os.path.join(_HERE, "real_filters.txt"),
        help="Output path for the raw filter-weight text dump.",
    )
    parser.add_argument(
        "--logos",
        default=os.path.join(_HERE, "filter_logos.png"),
        help="Output path for the enrichment/depletion (EDLogo) figure, built "
        "from each filter's top-4 max-activating 6-mers.",
    )
    parser.add_argument(
        "--min-contribution-frac",
        type=float,
        default=0.05,
        help="Only render EDLogos for filters whose peak softplus activation is at "
        "least this fraction of the strongest filter's peak (default: 0.05, i.e. "
        "5 percent of the strongest). Use 0 to show every filter.",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Output path for the run-metadata sidecar JSON. Defaults to "
        "plot_metadata.json next to --out.",
    )
    args = parser.parse_args()

    model = PNASModel()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded weights from {args.weights}")

    # ── Run metadata stamped into the checkpoint by the training scripts (dataset
    # identity, planted motif, hyperparameters, final-epoch loss). Build one caption
    # string, reused as the figure footer on both plots and in the sidecar JSON.
    # Checkpoints saved before metadata stamping have no "metadata" key, so fall back
    # to a clear "unavailable" note rather than failing. ──
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    if meta:
        hp = meta.get("hparams", {})
        final = meta.get("final", {})
        motif = meta.get("motifs", meta.get("motif"))
        if motif is None:
            motif_str = "no planted motif"
        else:
            motif_str = str(motif)
            num_blocks = meta.get("num_blocks")
            if num_blocks is not None:
                motif_str += f" ({num_blocks} blocks)"
        variant = meta.get("loss_variant")
        variant_str = f", loss={variant}" if variant else ""
        model_type = meta.get("model_type", "?")
        caption = (
            f"dataset: {meta.get('dataset', '?')} | model: {model_type} | "
            f"motif: {motif_str}{variant_str} | "
            f"epochs: {hp.get('num_epochs', '?')} | "
            f"final loss {final.get('loss', float('nan')):.4f} "
            f"(sd {final.get('sd_loss', float('nan')):.4f}, "
            f"l1 {final.get('l1', float('nan')):.4f}) | "
            f"lr {hp.get('lr', '?')}, l1λ {hp.get('l1_lambda', '?')}, "
            f"σ {hp.get('smooth_sigma', '?')}, seed {hp.get('seed', '?')}"
        )
    else:
        caption = (
            f"metadata unavailable (epoch {ckpt.get('epoch', '?')}; "
            "checkpoint predates metadata stamping)"
        )
    logger.info(f"Run metadata: {caption}")

    convs = [("INCL", model.conv_incl), ("SKIP", model.conv_skip)]
    num_filters, channels, kernel_size = model.conv_incl.weight.shape

    # ── Learned-filter readout (motif-agnostic, since real training has no planted
    # motif): each filter's consensus motif (argmax nucleotide per kernel position)
    # and its strength, listed strongest-first within each conv bank. Ranked by the
    # gauge-invariant strength |w_c|_1 (raw |w|_1 counts the arbitrary per-column
    # offset the bias absorbs, so it over/under-states real discriminative power). ──
    readout_lines = [
        "Learned filter consensus motifs (argmax nucleotide per kernel position), "
        "ranked by gauge-invariant strength |w_c|_1 (sum of |median-centered weights|):"
    ]
    for name, conv in convs:
        consensus, strength = filter_consensus(conv.weight.detach().cpu())
        bias = conv.bias.detach().cpu()
        # Median-center each filter column (removing the per-column offset the bias
        # absorbs), then sum |centered weights| for a gauge-invariant strength.
        w = conv.weight.detach().cpu().numpy()
        w_c = w - np.median(w, axis=1, keepdims=True)
        cstrength = np.abs(w_c).sum(axis=(1, 2))
        ranked = sorted(range(len(consensus)), key=lambda f: cstrength[f], reverse=True)
        readout_lines.append(f"  [{name}] (strongest first):")
        for f in ranked:
            readout_lines.append(
                f"    #{f:2d}  {consensus[f]}  "
                f"(|w_c|_1={cstrength[f]:.3f}, |w|_1={strength[f]:.3f}, bias={bias[f]:+.4f})"
            )
    for line in readout_lines:
        logger.info(line)

    # ── Write the raw filter weights to a text file: the readout header, a shape
    # line, then a pos1..posK grid per filter (rows A/C/G/T). Column widths match
    # the existing *_filters.txt dumps (8-wide label, 9-wide value cells). ──
    pos_header = "".ljust(8) + "".join(
        f"pos{p + 1}".ljust(9) for p in range(kernel_size)
    )
    with open(args.txt, "w") as fh:
        for line in readout_lines:
            fh.write(line + "\n")
        fh.write(
            f"conv_incl/conv_skip weight shape: ({num_filters}, {channels}, {kernel_size})"
            "  (num_filters, channels, kernel_size)\n"
        )
        for name, conv in convs:
            weight = conv.weight.detach().cpu()
            bias = conv.bias.detach().cpu()
            for filter_idx in range(num_filters):
                fh.write(
                    f"\n=== {name} filter #{filter_idx} "
                    f"(bias={bias[filter_idx]:+.4f}) ===\n"
                )
                fh.write(pos_header + "\n")
                for ch_idx, nt in enumerate(NUCLEOTIDES):
                    cells = "".join(
                        f"{weight[filter_idx, ch_idx, p]:+.4f}".ljust(9)
                        for p in range(kernel_size)
                    )
                    fh.write(f"  {nt}".ljust(8) + cells + "\n")
    logger.info(f"Wrote raw filter values to {args.txt}")

    # ── Plot all filters as heat maps (4 x 6 each: rows A/C/G/T, columns = 6 kernel positions) ──

    # Shared symmetric color scale (diverging around 0) so filters are comparable.
    all_w = torch.cat([conv.weight.detach().cpu().reshape(-1) for _, conv in convs])
    vmax = all_w.abs().max().item()

    n_cols = 5
    rows_per_conv = math.ceil(num_filters / n_cols)
    n_rows = len(convs) * rows_per_conv
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
    for block, (name, conv) in enumerate(convs):
        weight = conv.weight.detach().cpu()
        bias = conv.bias.detach().cpu()
        for filter_idx in range(num_filters):
            ax = axes[block * rows_per_conv + filter_idx // n_cols][filter_idx % n_cols]
            ax.axis("on")
            im = ax.imshow(
                weight[filter_idx], cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto"
            )
            ax.set_title(f"{name} #{filter_idx}  b={bias[filter_idx]:+.2f}", fontsize=8)
            ax.set_yticks(range(len(NUCLEOTIDES)))
            ax.set_yticklabels(NUCLEOTIDES, fontsize=6)
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)

    fig.colorbar(im, ax=axes, shrink=0.6, label="filter weight")
    fig.suptitle("conv_incl / conv_skip filters (rows A/C/G/T, cols = kernel position)")
    # Run-metadata footer (bbox_inches="tight" below keeps it in frame).
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {2 * num_filters} filter heat maps to {args.out}")

    # ── Plot each *significant* filter as an enrichment/depletion logo (EDLogo) built
    # from the 6-mers it maximally activates on, mirroring experiment/filter_max_scan.py.
    # Because the kernel (6) spans the whole input window, a filter's response over
    # every possible input is an exhaustive scan of the 4**6 = 4096 six-mers. For
    # each filter we take the top 4 highest-output 6-mers and turn their base
    # composition into letter heights: log2(freq / 0.25 uniform background) — bases
    # the top hits are enriched for sit above the axis, depleted below. A Jeffreys
    # pseudocount (0.5) avoids log2(0) and keeps the scale symmetric (a base in
    # exactly 1 of the 4 hits maps to 0 = uniform). Each logo autoscales to its own
    # y-range so weak filters stay readable, and the peak softplus output is printed
    # per panel. The all-6-mers one-hot encoding is identical for every filter, so
    # build it once.
    #
    # A filter's contribution to the model output is its softplus activation
    # (compute_sr_profile sums softplus over the 20 filters), so its peak softplus
    # over all 4096 6-mers is that filter's largest possible contribution to the SR
    # sum — a filter whose peak is ~0 is effectively dead. We render only filters
    # whose peak reaches --min-contribution-frac of the single strongest filter's
    # peak (softplus outputs are directly comparable across the INCL and SKIP banks),
    # strongest-first within each bank. ──
    all_seqs = [
        "".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)
    ]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    # First pass: per-filter top-4 6-mers, enrichment matrix, and peak softplus.
    filter_logos = {name: [] for name, _ in convs}
    for name, conv in convs:
        weight = conv.weight.detach().cpu().numpy()
        bias = conv.bias.detach().cpu().numpy()
        for filter_idx in range(num_filters):
            # Filter output per 6-mer: for one-hot inputs the conv is just the sum of
            # the weights under the active base at each position, plus bias; softplus
            # matches the model. softplus is monotonic so it does not change the
            # ranking, only the reported peak value.
            raw = np.einsum("ncp,cp->n", onehot, weight[filter_idx]) + bias[filter_idx]
            outputs = np.logaddexp(0.0, raw)  # softplus, numerically stable
            top4_idx = np.argsort(-outputs)[:4]
            top4_seqs = [all_seqs[i] for i in top4_idx]

            # Base composition of the top-4 hits → enrichment vs uniform background.
            counts = one_hot_batch(top4_seqs).sum(axis=0)  # (4, kernel_size)
            freq = (counts + 0.5) / (4 + 4 * 0.5)  # Jeffreys pseudocount 0.5
            enrichment = np.log2(freq / 0.25)  # up = enriched, down = depleted
            filter_logos[name].append(
                {
                    "filter_idx": filter_idx,
                    "bias": float(bias[filter_idx]),
                    "peak": float(outputs[top4_idx[0]]),
                    "enrichment": enrichment,
                }
            )

    # Keep only significant filters (peak >= frac of the strongest filter's peak),
    # strongest-first within each bank. frac=0 keeps every filter.
    max_peak = max(d["peak"] for name in filter_logos for d in filter_logos[name])
    threshold = args.min_contribution_frac * max_peak
    kept = {
        name: sorted(
            (d for d in filter_logos[name] if d["peak"] >= threshold),
            key=lambda d: d["peak"],
            reverse=True,
        )
        for name, _ in convs
    }
    num_total = len(convs) * num_filters
    num_kept = sum(len(v) for v in kept.values())
    logger.info(
        f"Rendering {num_kept}/{num_total} filters with peak softplus >= "
        f"{args.min_contribution_frac:g} x strongest ({max_peak:.3f}) = {threshold:.3f}"
    )

    # Grid sized to the kept filters: INCL block on top, SKIP block below, each packed
    # row-major into n_cols columns (a bank with no significant filter is omitted).
    bank_rows = {name: math.ceil(len(kept[name]) / n_cols) for name, _ in convs}
    logo_n_rows = sum(bank_rows.values()) or 1
    # Reserve fixed-height bands (in inches) for the suptitle and metadata footer, so
    # they never land on the panels when only a few rows are shown — a well-trained
    # model can leave just one significant filter, i.e. a single short row. The panel
    # area then gets logo_n_rows * row_h inches between those bands.
    row_h, header_in, footer_in = 1.9, 1.0, 0.55
    fig_h = logo_n_rows * row_h + header_in + footer_in
    fig, axes = plt.subplots(
        logo_n_rows,
        n_cols,
        figsize=(n_cols * 2.2, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=footer_in / fig_h,
        left=0.05,
        right=0.97,
    )
    for ax in axes.ravel():
        ax.axis("off")  # hide unused cells

    row_offset = 0
    for name, _ in convs:
        for i, d in enumerate(kept[name]):
            ax = axes[row_offset + i // n_cols][i % n_cols]
            ax.axis("on")
            df = pd.DataFrame(d["enrichment"].T, columns=NUCLEOTIDES)
            logomaker.Logo(
                df,
                ax=ax,
                flip_below=False,  # keep depleted letters upright, below the axis
                color_scheme="classic",
                shade_below=0.0,
                fade_below=0.0,
            )
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(
                f"{name} #{d['filter_idx']}  b={d['bias']:+.2f}  max={d['peak']:.2f}",
                fontsize=8,
            )
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
        row_offset += bank_rows[name]

    # Divider between the INCL (top) and SKIP (bottom) blocks, drawn in the gap
    # between the last INCL row and the first SKIP row. Only when both banks have
    # significant filters — otherwise there is a single block and nothing to split.
    if kept["INCL"] and kept["SKIP"]:
        boundary_row = bank_rows["INCL"]
        y_above = axes[boundary_row - 1][0].get_position().y0  # bottom of last INCL row
        y_below = axes[boundary_row][0].get_position().y1  # top of first SKIP row
        y_line = (y_above + y_below) / 2
        fig.add_artist(
            Line2D(
                [0.05, 0.95],
                [y_line, y_line],
                transform=fig.transFigure,
                color="0.4",
                lw=1.2,
            )
        )

    # Two-line suptitle inside the reserved top band (keeps it narrow so a few-panel
    # figure isn't forced ultra-wide, and clear of the top row's panel titles).
    fig.suptitle(
        "conv_incl / conv_skip EDLogos from top-4 max-activating 6-mers\n"
        f"({num_kept}/{num_total} filters, peak softplus >= "
        f"{args.min_contribution_frac:g}x strongest; up = enriched, down = depleted; "
        f"per-filter y-scale)",
        y=1 - 0.12 * header_in / fig_h,
        va="top",
        fontsize=10,
    )
    # Run-metadata footer, centered in the reserved bottom band so it clears the
    # bottom row's x-axis labels.
    fig.text(
        0.5,
        0.45 * footer_in / fig_h,
        caption,
        ha="center",
        va="center",
        fontsize=7,
        wrap=True,
    )
    fig.savefig(args.logos, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {num_kept} filter EDLogos to {args.logos}")

    # ── Sidecar JSON: the full stamped metadata plus provenance (source checkpoint,
    # the figures/text this run produced, and a timestamp), written next to the plots
    # so the run is machine-readable and travels with the images. ──
    meta_out = args.meta_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "plot_metadata.json"
    )
    sidecar = {
        "metadata": meta,
        "caption": caption,
        "provenance": {
            "weights": os.path.abspath(args.weights),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "heatmap_png": os.path.abspath(args.out),
            "logos_png": os.path.abspath(args.logos),
            "filter_txt": os.path.abspath(args.txt),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(meta_out, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    logger.info(f"Wrote run metadata to {meta_out}")


if __name__ == "__main__":
    main()
