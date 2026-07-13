"""Filter readout for a trained custom_model checkpoint.

Loads weights saved by custom_real_train.py (``weights/custom_model.pt`` by
default), prints each learned conv filter's consensus motif (the argmax
nucleotide at every kernel position), and renders every INCL/SKIP filter as a
heat map. Real cross-species training has no planted motif, so the readout is
motif-agnostic: it reports what each filter converged to rather than scoring it
against a known target.

Run after training:

    python plot_filters.py
    python plot_filters.py --weights weights/custom_model.pt --out filters.png
"""

import argparse
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from custom_model import PNASModel

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
        default=os.path.join(_HERE, "weights", "real_local_baseline_weights.pt"),
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
        help="Output path for the enrichment/depletion (EDLogo) figure.",
    )
    args = parser.parse_args()

    model = PNASModel()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded weights from {args.weights}")

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
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {2 * num_filters} filter heat maps to {args.out}")

    # ── Plot the same filters as enrichment/depletion logos (EDLogo). Median-centering
    # each filter column removes the arbitrary per-column offset the bias absorbs; the
    # remaining weights are the interpretable letter heights — positive above the axis
    # = bases the filter is enriched for, negative below = depleted. Same 5-wide grid
    # (INCL bank then SKIP bank). Each logo autoscales to its own y-range so weak
    # motifs stay readable (one poly-A filter is ~35x stronger than the rest and a
    # shared scale would flatten everything else); the gauge-invariant strength
    # |w_c|_1 is printed per panel to keep cross-filter magnitude visible. ──
    centered = {
        name: conv.weight.detach().cpu().numpy()
        - np.median(conv.weight.detach().cpu().numpy(), axis=1, keepdims=True)
        for name, conv in convs
    }

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, n_rows * 1.9),
        squeeze=False,
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )
    for ax in axes.ravel():
        ax.axis("off")  # hide unused cells

    for block, (name, conv) in enumerate(convs):
        bias = conv.bias.detach().cpu()
        w_c = centered[name]
        for filter_idx in range(num_filters):
            ax = axes[block * rows_per_conv + filter_idx // n_cols][filter_idx % n_cols]
            ax.axis("on")
            df = pd.DataFrame(w_c[filter_idx].T, columns=NUCLEOTIDES)
            logomaker.Logo(
                df,
                ax=ax,
                flip_below=False,  # keep depleted letters upright, below the axis
                color_scheme="classic",
                shade_below=0.0,
                fade_below=0.0,
            )
            ax.axhline(0, color="k", lw=0.8)
            cstrength = float(np.abs(w_c[filter_idx]).sum())
            ax.set_title(
                f"{name} #{filter_idx}  b={bias[filter_idx]:+.2f}  |w_c|={cstrength:.1f}",
                fontsize=8,
            )
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)

    fig.suptitle(
        "conv_incl / conv_skip EDLogos (median-centered weights; up = enriched, "
        "down = depleted; per-filter y-scale, strength |w_c|_1 in title)"
    )
    fig.savefig(args.logos, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {2 * num_filters} filter EDLogos to {args.logos}")


if __name__ == "__main__":
    main()
