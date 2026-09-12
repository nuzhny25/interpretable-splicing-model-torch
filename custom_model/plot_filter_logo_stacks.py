"""Multi-threshold sequence logos for a hand-picked set of filters.

plot_filters.py renders every significant filter as one information-content logo at a
single activation threshold (--logo-activation-frac, default 0.5 = half-max). Judging how
sharp a filter really is means re-running that script at another fraction and comparing two
PNGs. This script instead takes a handful of filters chosen by hand and, for each one, draws
the logo at several fractions of its own peak activation stacked vertically in one figure:
rows = activation fractions (0.75 / 0.50 / 0.25 by default, loosening downward), columns =
the selected filters. A motif that stays crisp as the threshold loosens is genuinely
conserved; one that blurs into a mush by the bottom row only looked tight at its very peak.

The logos come from the exhaustive 4096 six-mer scan (the same source as plot_filters.py's
--logos figure), not from the real MALAT1 alignment, so every 6-mer is weighted equally and
the panels say what a filter *could* detect.

    python plot_filter_logo_stacks.py --filters INCL:8,12 SKIP:11
    python plot_filter_logo_stacks.py --filters INCL:8 --fracs 0.9 0.6 0.3 0.1
"""

import argparse
import itertools
import logging
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from custom_model import PNASModel
from custom_utils import DNA_ALPHABET, one_hot_batch

NUCLEOTIDES = ["A", "C", "G", "T"]
BANKS = ("INCL", "SKIP")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=os.path.join(_HERE, "weights", "real_weights_ortho.pt"),
        help="Checkpoint saved by a training script (weights under 'model_state_dict').",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        required=True,
        metavar="BANK:IDX[,IDX...]",
        help="Filters to render, one column each, in the order given. Bank is INCL or "
        "SKIP (case-insensitive), indices are comma-separated: "
        "--filters INCL:8,12 SKIP:11",
    )
    parser.add_argument(
        "--fracs",
        nargs="+",
        type=float,
        default=[0.75, 0.5, 0.25],
        help="Activation fractions, one row each in the order given (default: 0.75 0.5 "
        "0.25). Each row's logo is built from every 6-mer whose softplus activation is "
        "at least that fraction of the filter's own peak, so higher = fewer, sharper "
        "6-mers. Any number of fractions works, not just three.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "filter_logo_stacks.png"),
        help="Output path for the stacked-logo figure.",
    )
    args = parser.parse_args()

    for frac in args.fracs:
        if not 0.0 < frac <= 1.0:
            parser.error(f"--fracs values must be in (0, 1]; got {frac:g}")

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Size the model to the checkpoint before loading: strict=False forgives missing and
    # unexpected keys but not shape mismatches, so a bank trained at a different filter
    # count or kernel width has to be matched here or the load aborts.
    n_filt, _, ksize = state_dict["conv_incl.weight"].shape
    model = PNASModel(num_filters=n_filt, kernel_size=ksize)
    # strict=False so checkpoints from architecture variants that carry extra parameters
    # (e.g. custom_model_capped.py's per-filter gain logits) still load their conv banks
    # here; the extras are pulled out by hand below rather than crashing the load.
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(
            f"Checkpoint is missing {len(load_result.missing_keys)} model "
            f"parameter(s), left at random init: {sorted(load_result.missing_keys)}"
        )
    model.eval()
    logger.info(f"Loaded weights from {args.weights}")

    convs = {"INCL": model.conv_incl, "SKIP": model.conv_skip}
    num_filters, _, kernel_size = model.conv_incl.weight.shape

    # ── Parse --filters into an ordered list of (bank, filter_idx) columns ──
    selected = []
    for token in args.filters:
        bank, sep, idx_str = token.partition(":")
        bank = bank.strip().upper()
        if not sep or bank not in BANKS:
            parser.error(
                f"--filters entry {token!r} is not BANK:IDX; bank must be one of "
                f"{', '.join(BANKS)} (e.g. INCL:8,12)"
            )
        for piece in idx_str.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                idx = int(piece)
            except ValueError:
                parser.error(
                    f"--filters entry {token!r} has non-integer index {piece!r}"
                )
            if not 0 <= idx < num_filters:
                parser.error(
                    f"--filters entry {token!r} index {idx} is outside 0..{num_filters - 1}"
                )
            if (bank, idx) in selected:
                logger.warning(
                    f"{bank} #{idx} listed more than once; keeping one column"
                )
                continue
            selected.append((bank, idx))
    if not selected:
        parser.error("--filters selected no filters; give at least one BANK:IDX")

    # ── Per-filter gains (capped-model checkpoints only) ──
    # custom_model_capped.py scales each filter's activation by a learned gain on a fixed
    # budget. Recover the gains from the checkpoint's logits with the same formula as
    # PNASModel.filter_gains so the column headers report what the model learned.
    # Checkpoints without the logits (every uncapped run) get no g= at all.
    gain_temp = ckpt.get("metadata", {}).get("hparams", {}).get("gain_temp", 1.0)
    gain_floor = ckpt.get("metadata", {}).get("hparams", {}).get("gain_floor", 0.0)
    gains = {}
    for bank, key in (("INCL", "gain_logit_incl"), ("SKIP", "gain_logit_skip")):
        if key in state_dict:
            k = state_dict[key].numel()
            gains[bank] = (
                k
                * (
                    (1.0 - gain_floor)
                    * torch.softmax(state_dict[key].float() / gain_temp, dim=0)
                    + gain_floor / k
                )
            ).numpy()

    # ── Run metadata stamped into the checkpoint by the training scripts, rendered as the
    # figure footer. Checkpoints saved before metadata stamping have no "metadata" key, so
    # fall back to a clear "unavailable" note rather than failing. ──
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

    # ── Exhaustive 6-mer scan. Because the kernel spans the whole input window, a filter's
    # response over every possible input is a scan of the 4**6 = 4096 six-mers. The
    # encoding is identical for every filter, so build it once. ──
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    # Per selected filter: the softplus response over all 6-mers and its peak (which is
    # threshold-independent, so it is computed once and reused across every fraction),
    # then one information-content logo per requested fraction.
    columns = []
    for bank, filter_idx in selected:
        conv = convs[bank]
        weight = conv.weight.detach().cpu().numpy()[filter_idx]
        bias = float(conv.bias.detach().cpu().numpy()[filter_idx])
        # For one-hot inputs the conv is just the sum of the weights under the active base
        # at each position, plus bias; softplus matches the model. softplus is monotonic,
        # so it does not change the ranking, only the reported peak value.
        raw = np.einsum("ncp,cp->n", onehot, weight) + bias
        outputs = np.logaddexp(0.0, raw)  # softplus, numerically stable
        peak = float(outputs.max())

        rows = []
        for frac in args.fracs:
            thresh = frac * peak
            # Every 6-mer reaching this fraction of the filter's own peak (>=1: the peak
            # itself always passes). Looser fractions give supersets of tighter ones.
            sel_seqs = [all_seqs[i] for i in np.flatnonzero(outputs >= thresh)]
            n_sel = len(sel_seqs)
            # Per-position base frequencies of the above-threshold 6-mers. No pseudocount,
            # so nucleotides absent from every selected 6-mer stay exactly 0 (no letter).
            counts = one_hot_batch(sel_seqs).sum(axis=0)  # (4, kernel_size)
            freq = counts / n_sel  # each column sums to 1
            # Shannon information content per position (bits): 2 + Σ f·log2 f, max 2 for a
            # fully conserved column, ~0 for a uniform one. Letter height = freq * info.
            info = 2.0 + np.sum(
                np.where(freq > 0, freq * np.log2(freq, where=freq > 0), 0.0), axis=0
            )
            rows.append(
                {
                    "frac": frac,
                    "n_sel": n_sel,
                    "heights": freq * info[None, :],  # (4, kernel_size), all >= 0
                    "consensus": "".join(NUCLEOTIDES[c] for c in freq.argmax(axis=0)),
                }
            )

        gain = float(gains[bank][filter_idx]) if bank in gains else 1.0
        columns.append(
            {
                "bank": bank,
                "filter_idx": filter_idx,
                "bias": bias,
                "peak": peak,
                "gain": gain,
                "rows": rows,
            }
        )

    # Numbers behind each panel, so the selection sizes are readable without squinting at
    # the PNG (and so a filter whose set explodes at the loosest fraction is obvious).
    logger.info(
        "Selected 6-mers per (filter, fraction), out of "
        f"{len(all_seqs)} total (consensus = argmax base per position):"
    )
    for col in columns:
        logger.info(
            f"  [{col['bank']} #{col['filter_idx']}]  bias={col['bias']:+.4f}  "
            f"peak softplus={col['peak']:.3f}"
        )
        for row in col["rows"]:
            logger.info(
                f"    frac={row['frac']:<5g}  n={row['n_sel']:5d}  "
                f"consensus={row['consensus']}"
            )

    # ── Figure: rows = fractions (in the order given, so the default loosens downward),
    # columns = selected filters. Fixed-height bands (in inches) are reserved for the
    # suptitle and metadata footer so they never land on the panels when only a couple of
    # rows are shown; the header band is sized to clear the two-line column headers. ──
    n_rows, n_cols = len(args.fracs), len(columns)

    frac_str = ", ".join(f"{f:g}" for f in args.fracs)
    # Wrap the subtitle to the figure's own width (~30 chars per 2.2in column at
    # fontsize 10). Without this a one- or two-column figure gets stretched to the
    # width of the text by bbox_inches="tight", leaving the panels marooned in
    # whitespace; a wide selection still fits on one line.
    subtitle = textwrap.fill(
        f"(rows = 6-mers reaching {frac_str} x each filter's own peak softplus; "
        f"columns = selected filters; {os.path.basename(args.weights)}; "
        "present nucleotides only)",
        width=max(40, 30 * n_cols),
    )

    row_h, footer_in = 1.9, 0.55
    # The header band holds the suptitle plus the top row's two-line column headers.
    # 1.2in covers the one-line-subtitle case; each extra wrapped line needs its own
    # room or the suptitle rides down onto those headers.
    header_in = 1.2 + 0.18 * subtitle.count("\n")
    fig_h = n_rows * row_h + header_in + footer_in
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, fig_h),
        squeeze=False,
        # Tighter hspace than the plot_filters.py grids: only the top row carries a
        # title here, so the rows can sit close enough to read as one stack.
        gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )
    fig.subplots_adjust(
        top=1 - header_in / fig_h,
        bottom=footer_in / fig_h,
        left=0.08,
        right=0.97,
    )

    for c, col in enumerate(columns):
        for r, row in enumerate(col["rows"]):
            ax = axes[r][c]
            df = pd.DataFrame(row["heights"].T, columns=NUCLEOTIDES)
            logomaker.Logo(
                df,
                ax=ax,
                color_scheme="classic",  # heights are all >= 0: nothing below the axis
            )
            ax.set_ylim(
                0, 2
            )  # standard information-content scale (bits), shared by all
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            if c == 0:
                # Row identity lives on the left edge rather than in a per-panel title,
                # which would collide with the column headers on the top row.
                ax.set_ylabel(f"{row['frac']:g}× peak\n(bits)", fontsize=7)
            # Sample size in-panel, right-aligned in the top corner for the same reason.
            ax.text(
                0.98,
                0.94,
                f"n={row['n_sel']}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                color="0.35",
            )
            if r == 0:
                # Learned gain, on capped-model checkpoints only (1.00 is the model's
                # neutral allocation: every filter's equal share of the fixed budget).
                # Two-line header, because the single-line form overruns the ~2.2in panel
                # width and neighboring titles collide into an unreadable smear.
                gain_str = f"  g={col['gain']:.2f}" if gains else ""
                ax.set_title(
                    f"{col['bank']} #{col['filter_idx']}  b={col['bias']:+.2f}{gain_str}\n"
                    f"max={col['peak']:.2f}",
                    fontsize=7,
                )

    fig.suptitle(
        f"Information-content (bits) logos per activation threshold\n{subtitle}",
        y=1 - 0.12 * header_in / fig_h,
        va="top",
        fontsize=10,
    )
    # Run-metadata footer, centered in the reserved bottom band so it clears the bottom
    # row's x-axis labels.
    fig.text(
        0.5,
        0.45 * footer_in / fig_h,
        caption,
        ha="center",
        va="center",
        fontsize=7,
        wrap=True,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Wrote {n_cols} filter(s) x {n_rows} fraction(s) of stacked logos to {args.out}"
    )


if __name__ == "__main__":
    main()
