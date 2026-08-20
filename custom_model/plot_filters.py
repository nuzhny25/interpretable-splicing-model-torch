"""Filter readout for a trained custom_model checkpoint.

Loads weights saved by custom_real_train.py (``weights/custom_model.pt`` by
default), prints each learned conv filter's consensus motif (the argmax
nucleotide at every kernel position), renders every INCL/SKIP filter as a
heat map, and renders each filter as an information-content (bits) sequence logo
built from the 6-mers whose softplus activation reaches at least a fraction (default
half, see --logo-activation-frac) of that filter's peak (see experiment/filter_max_scan.py).
The same logos are also rendered a second time (``--real-logos``) from the real MALAT1
alignment instead of the exhaustive 6-mer scan: there each above-threshold *window* of
the aligned species counts once, so the logo is weighted by how often the filter's
targets actually occur in MALAT1 rather than treating all 4096 6-mers as equally likely.
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
import torch.nn.functional as F

from custom_model import PNASModel
from custom_utils import DNA_ALPHABET, one_hot_batch, str_to_vector

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
        help="Output path for the information-content (bits) logo figure, built from "
        "each filter's above-threshold max-activating 6-mers (see --logo-activation-frac).",
    )
    parser.add_argument(
        "--real-logos",
        default=os.path.join(_HERE, "filter_logos_real.png"),
        help="Output path for the real-data information-content (bits) logo figure: same "
        "filters and layout as --logos, but each logo is built from the real MALAT1 "
        "windows that fire the filter (see --matrix) instead of the exhaustive 4096 "
        "6-mer scan. Written only when the alignment matrix exists.",
    )
    parser.add_argument(
        "--min-contribution-frac",
        type=float,
        default=0.2,
        help="Only render logos for filters whose peak softplus activation is at "
        "least this fraction of the strongest filter's peak (default: 0.05, i.e. "
        "5 percent of the strongest). Use 0 to show every filter.",
    )
    parser.add_argument(
        "--logo-activation-frac",
        type=float,
        default=0.5,
        help="Per filter, build its logo from every 6-mer whose softplus activation is "
        "at least this fraction of that filter's own peak activation (default: 0.5 = "
        "half-max, matching experiment/filter_max_scan.py). Higher = fewer, sharper "
        "6-mers.",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Output path for the run-metadata sidecar JSON. Defaults to "
        "plot_metadata.json next to --out.",
    )
    parser.add_argument(
        "--matrix",
        default=os.path.join(_HERE, "..", "data", "multiz100", "alignment_matrix.npy"),
        help="Real alignment matrix .npy (aligned positions x species) used to count how "
        "many times each filter fires on the original data (softplus >= "
        "--logo-activation-frac x that filter's theoretical peak). A missing file just "
        "skips the counts and panels show the theoretical max only.",
    )
    args = parser.parse_args()

    model = PNASModel()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # strict=False so checkpoints from architecture variants that carry extra
    # parameters still load their conv banks here. custom_model_capped.py adds
    # per-filter gain logits; this plotter's PNASModel has no such parameter, so
    # they arrive as unexpected keys and are pulled out by hand below rather than
    # crashing the load.
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(
            f"Checkpoint is missing {len(load_result.missing_keys)} model "
            f"parameter(s), left at random init: {sorted(load_result.missing_keys)}"
        )
    model.eval()
    logger.info(f"Loaded weights from {args.weights}")

    # ── Per-filter gains (capped-model checkpoints only) ──
    # custom_model_capped.py scales each filter's activation by a learned gain on a
    # fixed budget, so a filter's actual contribution to the SR sum is gain x
    # activation, not activation alone. Recover the gains from the checkpoint's logits
    # with the same formula as PNASModel.filter_gains so the cross-filter ranking below
    # reflects what the model learned. Checkpoints without the logits (every uncapped
    # run) fall back to a gain of 1.0 per filter, i.e. the previous behaviour exactly.
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
    if gains:
        logger.info(
            "Per-filter gains found in checkpoint (temp="
            f"{gain_temp}, floor={gain_floor}); filter contribution is gain x peak "
            "activation. Effective filter count per bank: "
            + ", ".join(
                f"{bank} {np.exp(-(p * np.log(p)).sum()):.2f}"
                for bank, g in gains.items()
                for p in [g / g.sum()]
            )
        )

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
        gridspec_kw={"hspace": 0.85, "wspace": 0.3},
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
            # Learned gain, on capped-model checkpoints only (1.00 is the model's
            # neutral allocation: every filter's equal share of the fixed budget).
            # Two-line title, because appending g= overruns the ~2.2in panel width
            # and neighboring titles collide into an unreadable smear — the same
            # reason the logo panels below use a two-line title.
            if gains:
                ax.set_title(
                    f"{name} #{filter_idx}\n"
                    f"b={bias[filter_idx]:+.2f}  g={gains[name][filter_idx]:.2f}",
                    fontsize=8,
                )
            else:
                ax.set_title(
                    f"{name} #{filter_idx}  b={bias[filter_idx]:+.2f}", fontsize=8
                )
            ax.set_yticks(range(len(NUCLEOTIDES)))
            ax.set_yticklabels(NUCLEOTIDES, fontsize=6)
            ax.set_xticks(range(kernel_size))
            ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)

    fig.colorbar(im, ax=axes, shrink=0.6, label="filter weight")
    # Second suptitle line summarizing the gain allocation, so the figure states how
    # concentrated the model is without the reader having to compare 40 g= values.
    if gains:
        alloc = "; ".join(
            f"{bank} {np.exp(-(p * np.log(p)).sum()):.1f} effective of {num_filters} "
            f"(max g={g.max():.1f})"
            for bank, g in gains.items()
            for p in [g / g.sum()]
        )
        fig.suptitle(
            "conv_incl / conv_skip filters (rows A/C/G/T, cols = kernel position)\n"
            f"gain allocation — {alloc}",
            fontsize=10,
        )
    else:
        fig.suptitle(
            "conv_incl / conv_skip filters (rows A/C/G/T, cols = kernel position)"
        )
    # Run-metadata footer (bbox_inches="tight" below keeps it in frame).
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7, wrap=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {2 * num_filters} filter heat maps to {args.out}")

    # ── Plot each *significant* filter as an information-content (bits) sequence logo
    # built from the 6-mers it maximally activates on, mirroring experiment/filter_max_scan.py.
    # Because the kernel (6) spans the whole input window, a filter's response over
    # every possible input is an exhaustive scan of the 4**6 = 4096 six-mers. For each
    # filter we keep every 6-mer whose softplus activation reaches --logo-activation-frac
    # of that filter's own peak (default 0.5 = the half-max set), then turn the base
    # composition of that above-threshold set into a standard Shannon information logo:
    # per-position frequencies with no pseudocount (so a base absent from every selected
    # 6-mer stays exactly 0 and draws no letter — present nucleotides only), scaled by the
    # column's information content 2 + Σ f·log2 f (max 2 bits for a fully conserved
    # column, ~0 for a uniform one). Letter heights are all >= 0, so nothing sits below
    # the axis. The number of selected 6-mers and the peak softplus are printed per panel.
    # The all-6-mers one-hot encoding is identical for every filter, so build it once.
    #
    # A filter's contribution to the model output is its softplus activation
    # (compute_sr_profile sums softplus over the 20 filters), so its peak softplus
    # over all 4096 6-mers is that filter's largest possible contribution to the SR
    # sum — a filter whose peak is ~0 is effectively dead. We render only filters
    # whose peak reaches --min-contribution-frac of the single strongest filter's
    # peak (softplus outputs are directly comparable across the INCL and SKIP banks),
    # strongest-first within each bank. ──
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, kernel_size)

    # First pass: per-filter above-threshold 6-mers, information-content matrix, and peak.
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
            peak = float(outputs.max())
            thresh = args.logo_activation_frac * peak
            # Every 6-mer reaching the fraction of this filter's peak (>=1: the peak
            # itself always passes), strongest-first — the half-max set at frac=0.5.
            sel_idx = [i for i in np.argsort(-outputs) if outputs[i] >= thresh]
            sel_seqs = [all_seqs[i] for i in sel_idx]
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
            heights = freq * info[None, :]  # (4, kernel_size), all >= 0
            # "contrib" is the filter's largest possible contribution to the SR sum:
            # its peak activation scaled by its learned gain (1.0 when the checkpoint
            # has none). Only cross-filter comparisons need it — the
            # --logo-activation-frac threshold above is a fraction of this filter's
            # own peak, where a constant gain cancels.
            gain = float(gains[name][filter_idx]) if name in gains else 1.0
            filter_logos[name].append(
                {
                    "filter_idx": filter_idx,
                    "bias": float(bias[filter_idx]),
                    "peak": peak,
                    "gain": gain,
                    "contrib": gain * peak,
                    "n_sel": n_sel,
                    "heights": heights,
                }
            )

    # ── Real-data activation counts ("fires"): how many sliding-window positions across
    # the real alignment actually activate each filter. The 4096-mer scan above gives
    # each filter's theoretical peak softplus; here we run the real MALAT1 species
    # sequences through the same conv banks and count windows whose softplus reaches
    # --logo-activation-frac of that filter's peak — the very threshold that selects the
    # filter's logo 6-mers, so the count reads as "how many real windows come within that
    # fraction of this filter's best-possible activation." Every real 6-nt window is one
    # of the 4096 scanned 6-mers, so a real activation is always <= the theoretical peak.
    # Gap-strip -> one-hot -> conv -> softplus mirrors filter_coactivation_covariance.py.
    # If the alignment file is absent (e.g. a synthetic checkpoint), the counts are
    # skipped and panels show the theoretical max only.
    real_matrix_path = os.path.abspath(args.matrix)
    have_real = os.path.exists(real_matrix_path)
    n_windows_total = 0
    if have_real:
        # matrix: (n_aligned, n_species) of single chars; gaps are "-"/"N", lowercase
        # a/c/g/t are nucleotides (upper-cased before one-hot).
        matrix = np.load(real_matrix_path)
        n_species = matrix.shape[1]
        with torch.no_grad():
            for name, conv in convs:
                peaks = np.array(
                    [d["peak"] for d in filter_logos[name]]
                )  # (num_filters,)
                thresh = (
                    args.logo_activation_frac * peaks
                )  # per-filter, index == filter_idx
                counts = np.zeros(num_filters, dtype=np.int64)
                windows = 0
                # Per-filter base composition of the firing windows, flattened to
                # (num_filters, 4 * kernel_size) so it can be accumulated with one
                # matmul per species (reshaped to (4, kernel_size) after the loop).
                # This is what turns the "fires" set into a real-data sequence logo:
                # unlike the 4096-mer scan above, a 6-mer that occurs 500 times in the
                # alignment contributes 500 counts, so the logo is weighted by what the
                # filter actually sees in MALAT1.
                base_counts = np.zeros((num_filters, 4 * kernel_size), dtype=np.float64)
                for j in range(n_species):
                    col = matrix[:, j]
                    is_nuc = (col != "-") & (col != "N")
                    seq = "".join(col[is_nuc]).upper()
                    if len(seq) < kernel_size:
                        continue  # too short to hold a single 6-nt window
                    oh = str_to_vector(seq)  # (4, Lj), reused for the base counts below
                    x = torch.tensor(oh, dtype=torch.float32).unsqueeze(0)
                    act = (
                        F.softplus(conv(x)).squeeze(0).numpy()
                    )  # (num_filters, Lj-5), >= 0
                    fired = act >= thresh[:, None]  # (num_filters, Lj-5)
                    counts += fired.sum(axis=1)
                    windows += act.shape[1]
                    # Every 6-nt window as a flat one-hot row, then one matmul sums the
                    # one-hots of exactly the windows each filter fired on. Equivalent to
                    # looping over fired windows and adding str_to_vector(window), but
                    # BLAS-fast; float32 is exact here (counts stay far below 2**24).
                    sw = np.lib.stride_tricks.sliding_window_view(
                        oh, kernel_size, axis=1
                    )  # (4, Lj-5, kernel_size)
                    win = np.ascontiguousarray(sw.transpose(1, 0, 2)).reshape(
                        act.shape[1], -1
                    )  # (Lj-5, 4 * kernel_size)
                    base_counts += fired.astype(np.float32) @ win
                for d in filter_logos[name]:
                    f = d["filter_idx"]
                    d["n_real"] = int(counts[f])
                    # Same information-content math as the 4096-mer logos above, applied
                    # to the real firing windows: per-position frequencies with no
                    # pseudocount (absent bases stay 0 and draw no letter), scaled by the
                    # column's Shannon information 2 + Σ f·log2 f (uniform background, so
                    # heights stay on the standard 0-2 bit scale and are comparable to
                    # the theoretical panels).
                    if d["n_real"] == 0:
                        d["real_heights"] = None
                        d["real_consensus"] = None
                        continue
                    bc = base_counts[f].reshape(4, kernel_size)
                    freq = bc / d["n_real"]  # each column sums to 1
                    info = 2.0 + np.sum(
                        np.where(freq > 0, freq * np.log2(freq, where=freq > 0), 0.0),
                        axis=0,
                    )
                    d["real_heights"] = freq * info[None, :]  # (4, kernel_size), >= 0
                    d["real_consensus"] = "".join(
                        NUCLEOTIDES[c] for c in freq.argmax(axis=0)
                    )
                n_windows_total = windows  # same window set for both banks
        logger.info(
            f"Real-data activation counts from {real_matrix_path}: {n_windows_total:,} "
            f"windows/bank across {n_species} species; each filter's 'fires' = windows with "
            f"softplus >= {args.logo_activation_frac:g} x that filter's peak."
        )
        # Append a ranked real-data section to the filter-weight dump (additive; the
        # existing readout was written above in "w" mode, so open in "a" here).
        with open(args.txt, "a") as fh:
            fh.write(
                f"\n\nReal-data activation counts on {real_matrix_path}\n"
                f"({n_windows_total:,} windows/bank across {n_species} species; a filter "
                f"'fires' at a window when its softplus >= {args.logo_activation_frac:g} x "
                "its theoretical peak; consensus = argmax base per position over those "
                "firing windows, i.e. the real-data logo's consensus), strongest-first:\n"
            )
            for name, _ in convs:
                fh.write(f"  [{name}]:\n")
                for d in sorted(
                    filter_logos[name], key=lambda d: d["n_real"], reverse=True
                ):
                    pct = (
                        100.0 * d["n_real"] / n_windows_total
                        if n_windows_total
                        else 0.0
                    )
                    fh.write(
                        f"    #{d['filter_idx']:2d}  fires={d['n_real']:6d}  "
                        f"({pct:5.2f}% of windows)  "
                        f"consensus={d['real_consensus'] or '-' * kernel_size}  "
                        f"peak={d['peak']:.3f}\n"
                    )
    else:
        logger.warning(
            f"Alignment matrix not found at {real_matrix_path}; skipping real-data "
            "activation counts (panels show theoretical max only). Pass --matrix to set it."
        )

    # Keep only significant filters (contribution >= frac of the strongest filter's
    # contribution), strongest-first within each bank. frac=0 keeps every filter.
    # Ranking on gain-scaled contribution is what makes this selection agree with the
    # model on which filters matter; without gains it is identical to ranking on peak.
    max_contrib = max(d["contrib"] for name in filter_logos for d in filter_logos[name])
    threshold = args.min_contribution_frac * max_contrib
    kept = {
        name: sorted(
            (d for d in filter_logos[name] if d["contrib"] >= threshold),
            key=lambda d: d["contrib"],
            reverse=True,
        )
        for name, _ in convs
    }
    num_total = len(convs) * num_filters
    num_kept = sum(len(v) for v in kept.values())
    logger.info(
        f"Rendering {num_kept}/{num_total} filters with "
        f"{'gain x peak' if gains else 'peak'} softplus >= "
        f"{args.min_contribution_frac:g} x strongest ({max_contrib:.3f}) = {threshold:.3f}"
    )

    # Grid sized to the kept filters: INCL block on top, SKIP block below, each packed
    # row-major into n_cols columns (a bank with no significant filter is omitted).
    bank_rows = {name: math.ceil(len(kept[name]) / n_cols) for name, _ in convs}
    logo_n_rows = sum(bank_rows.values()) or 1
    # Reserve fixed-height bands (in inches) for the suptitle and metadata footer, so
    # they never land on the panels when only a few rows are shown — a well-trained
    # model can leave just one significant filter, i.e. a single short row. The panel
    # area then gets logo_n_rows * row_h inches between those bands. The header band is
    # sized to clear the two-line panel titles of the top row below the suptitle.
    row_h, header_in, footer_in = 1.9, 1.2, 0.55
    fig_h = logo_n_rows * row_h + header_in + footer_in
    fig, axes = plt.subplots(
        logo_n_rows,
        n_cols,
        figsize=(n_cols * 2.2, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.85, "wspace": 0.3},
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
            df = pd.DataFrame(d["heights"].T, columns=NUCLEOTIDES)
            logomaker.Logo(
                df,
                ax=ax,
                color_scheme="classic",  # heights are all >= 0: nothing below the axis
            )
            ax.set_ylim(0, 2)  # standard information-content scale (bits), comparable
            if i % n_cols == 0:
                ax.set_ylabel("bits", fontsize=6)
            fires = f"  fires={d['n_real']}" if "n_real" in d else ""
            # Learned gain, shown only for checkpoints that have one (1.00 = the model's
            # neutral allocation, so values above/below it read as the model promoting or
            # demoting that filter relative to its 19 bank-mates).
            gain_str = f"  g={d['gain']:.2f}" if gains else ""
            # Two-line title: the single-line form overruns the ~2.2in panel width so
            # neighboring titles overlap into an unreadable smear (worse once fires= is
            # appended). Identity/bias on the first line, activation stats on the second.
            ax.set_title(
                f"{name} #{d['filter_idx']}  b={d['bias']:+.2f}{gain_str}\n"
                f"max={d['peak']:.2f}  n={d['n_sel']}{fires}",
                fontsize=7,
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
        "conv_incl / conv_skip information-content (bits) logos\n"
        f"({num_kept}/{num_total} filters, peak softplus >= "
        f"{args.min_contribution_frac:g}x strongest; from 6-mers >= "
        f"{args.logo_activation_frac:g}x each filter's peak activation; "
        f"present nucleotides only)",
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
    logger.info(f"Wrote {num_kept} filter logos to {args.logos}")

    # ── Same logos, built from the real MALAT1 alignment instead of the 4096-mer scan ──
    # The figure above gives every 6-mer equal weight, so it says what a filter *could*
    # detect: 4091 of the 4096 6-mers occur somewhere in this alignment, so the set of
    # selected 6-mers is nearly the same either way. What differs is the weighting —
    # here each *window* of the real alignment counts once, so a 6-mer occurring 1776
    # times (TTTTTT) contributes 1776 counts and one occurring twice contributes two.
    # The result is what the filter actually sees in MALAT1, shaped by the locus's
    # conservation across the 62 species and its skewed base composition.
    #
    # The windows are exactly the "fires" set counted above (softplus >=
    # --logo-activation-frac x that filter's theoretical peak), so each panel's fires=
    # is both the activation count and the logo's sample size. Same kept filters, same
    # order, same 0-2 bit scale as the figure above, so the two line up panel-for-panel.
    if have_real:
        r_bank_rows = {name: math.ceil(len(kept[name]) / n_cols) for name, _ in convs}
        r_n_rows = sum(r_bank_rows.values()) or 1
        r_fig_h = r_n_rows * row_h + header_in + footer_in
        r_fig, r_axes = plt.subplots(
            r_n_rows,
            n_cols,
            figsize=(n_cols * 2.2, r_fig_h),
            squeeze=False,
            gridspec_kw={"hspace": 0.85, "wspace": 0.3},
        )
        r_fig.subplots_adjust(
            top=1 - header_in / r_fig_h,
            bottom=footer_in / r_fig_h,
            left=0.05,
            right=0.97,
        )
        for ax in r_axes.ravel():
            ax.axis("off")  # hide unused cells

        row_offset = 0
        for name, _ in convs:
            for i, d in enumerate(kept[name]):
                ax = r_axes[row_offset + i // n_cols][i % n_cols]
                ax.axis("on")
                gain_str = f"  g={d['gain']:.2f}" if gains else ""
                if d["real_heights"] is None:
                    # No real window reaches this filter's threshold, so there is no
                    # composition to draw — an empty panel labelled as such, rather than
                    # a misleading flat logo.
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(
                        f"{name} #{d['filter_idx']}  b={d['bias']:+.2f}{gain_str}\n"
                        "no real windows",
                        fontsize=7,
                    )
                    continue
                df = pd.DataFrame(d["real_heights"].T, columns=NUCLEOTIDES)
                logomaker.Logo(df, ax=ax, color_scheme="classic")
                ax.set_ylim(0, 2)  # same information-content scale as the figure above
                if i % n_cols == 0:
                    ax.set_ylabel("bits", fontsize=6)
                pct = 100.0 * d["n_real"] / n_windows_total if n_windows_total else 0.0
                # Two-line title, same convention as the theoretical logos: identity and
                # bias on the first line, the real-data sample on the second.
                ax.set_title(
                    f"{name} #{d['filter_idx']}  b={d['bias']:+.2f}{gain_str}\n"
                    f"max={d['peak']:.2f}  fires={d['n_real']} ({pct:.2f}%)",
                    fontsize=7,
                )
                ax.set_xticks(range(kernel_size))
                ax.set_xticklabels(range(1, kernel_size + 1), fontsize=6)
                ax.tick_params(axis="y", labelsize=6)
            row_offset += r_bank_rows[name]

        if kept["INCL"] and kept["SKIP"]:
            boundary_row = r_bank_rows["INCL"]
            y_above = r_axes[boundary_row - 1][0].get_position().y0
            y_below = r_axes[boundary_row][0].get_position().y1
            y_line = (y_above + y_below) / 2
            r_fig.add_artist(
                Line2D(
                    [0.05, 0.95],
                    [y_line, y_line],
                    transform=r_fig.transFigure,
                    color="0.4",
                    lw=1.2,
                )
            )

        r_fig.suptitle(
            "conv_incl / conv_skip information-content (bits) logos — real MALAT1 windows\n"
            f"({num_kept}/{num_total} filters; from the {n_windows_total:,} windows across "
            f"{n_species} species reaching {args.logo_activation_frac:g}x each filter's "
            "peak; occurrence-weighted, uniform background, present nucleotides only)",
            y=1 - 0.12 * header_in / r_fig_h,
            va="top",
            fontsize=10,
        )
        r_fig.text(
            0.5,
            0.45 * footer_in / r_fig_h,
            caption,
            ha="center",
            va="center",
            fontsize=7,
            wrap=True,
        )
        r_fig.savefig(args.real_logos, dpi=150, bbox_inches="tight")
        plt.close(r_fig)
        logger.info(f"Wrote {num_kept} real-data filter logos to {args.real_logos}")
    else:
        logger.warning(
            "No alignment matrix, so the real-data logo figure is skipped; "
            f"{args.real_logos} was not written."
        )

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
            "real_logos_png": os.path.abspath(args.real_logos) if have_real else None,
            "filter_txt": os.path.abspath(args.txt),
            "logo_activation_frac": args.logo_activation_frac,
            "min_contribution_frac": args.min_contribution_frac,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        # Learned per-filter gains, keyed by bank -> filter_idx, or None for
        # checkpoints without them. Recorded so filter importance can be compared
        # across runs/seeds without re-deriving it from the logits.
        "gains": (
            {
                name: {i: float(g) for i, g in enumerate(gains[name])}
                for name, _ in convs
                if name in gains
            }
            if gains
            else None
        ),
        # Per-filter real-data activation counts (windows firing >= activation_frac x peak),
        # or None when the alignment matrix was absent. Keyed by bank -> filter_idx -> count.
        "real_activation": (
            {
                "matrix": real_matrix_path,
                "windows_per_bank": n_windows_total,
                "activation_frac": args.logo_activation_frac,
                "fires": {
                    name: {d["filter_idx"]: d["n_real"] for d in filter_logos[name]}
                    for name, _ in convs
                },
                # Consensus of each filter's real-data logo: argmax base per position
                # over the firing windows (None for a filter no real window fires).
                "consensus": {
                    name: {
                        d["filter_idx"]: d["real_consensus"] for d in filter_logos[name]
                    }
                    for name, _ in convs
                },
            }
            if have_real
            else None
        ),
    }
    with open(meta_out, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)
    logger.info(f"Wrote run metadata to {meta_out}")


if __name__ == "__main__":
    main()
