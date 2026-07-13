"""Scan every 6-mer through one hand-set conv filter and report its maximum.

A single conv filter is a (4, 6) weight grid (rows A, C, G, T = DNA_ALPHABET) plus
one scalar bias. For a one-hot 6-nt input the filter's output is just

    output(seq) = sum_p  weight[base_at_pos_p, p]  +  bias

so there are exactly 4**6 = 4096 possible inputs. This script sets the filter's
weights/bias by hand (the EDIT HERE block below), evaluates all 4096 6-mers, and
reports:

    * the highest output and the 6-mer that produces it,
    * 50% of that highest output (the half-max threshold), and
    * every 6-mer whose output reaches that half-max threshold.

The weight format matches set_weights.py, so a filter you planted there can be
pasted straight in here.

Run:
    python filter_max_scan.py                 # scan the EDIT HERE filter
    python filter_max_scan.py --out scan.txt   # also write the full report to a file
"""

import argparse
import itertools
import logging
import os
import sys

import numpy as np

# Make the parent custom_model package importable when run from experiment/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from custom_utils import DNA_ALPHABET, one_hot_batch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────── EDIT HERE ───────────────────────────────
# The filter to scan. Keys:
#   "bias"          -> scalar bias for the filter (optional, default 0)
#   "A"/"C"/"G"/"T" -> list of 6 weights along the kernel for that row (optional)
# Any row you omit stays all-zero. Row/channel order is A, C, G, T.
FILTER = {
    # Example: a poly-A detector — weight 1 on the A-row at every position.
    "bias": -5.5881,
    "A": [-12.6254, -9.8335, 3.0086, -14.4291, -3.7976, 2.9816],
    "C": [-3.5763, -10.7536, -9.7905, -8.8350, -15.3888, -5.7046],
    "G": [-8.0035, -12.6329, 1.3165, -13.1783, -14.7402, -11.1311],
    "T": [2.5082, 2.4828, -7.6769, 2.9084, 1.3513, -11.4041],
}
# ──────────────────────────── END EDIT HERE ──────────────────────────────

KERNEL_SIZE = 6


def build_filter(spec):
    """Turn the EDIT HERE dict into a (4, 6) weight grid and a scalar bias."""
    weight = np.zeros((len(DNA_ALPHABET), KERNEL_SIZE), dtype=np.float32)
    for nt in DNA_ALPHABET:
        if nt not in spec:
            continue
        values = spec[nt]
        if len(values) != KERNEL_SIZE:
            raise ValueError(
                f"Row {nt}: expected {KERNEL_SIZE} weights, got {len(values)}"
            )
        weight[DNA_ALPHABET.index(nt), :] = values
    bias = float(spec.get("bias", 0.0))
    return weight, bias


def print_filter(weight, bias):
    """Print the filter's 4x6 weight grid and bias (plot_filters.py layout)."""
    pos_header = "".ljust(8) + "".join(
        f"pos{p + 1}".ljust(9) for p in range(KERNEL_SIZE)
    )
    logger.info(f"Filter (bias={bias:+.4f}):")
    logger.info("  " + pos_header)
    for ch_idx, nt in enumerate(DNA_ALPHABET):
        cells = "".join(
            f"{weight[ch_idx, p]:+.4f}".ljust(9) for p in range(KERNEL_SIZE)
        )
        logger.info(f"    {nt}".ljust(10) + cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional text file to also write the full report to.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top-scoring 6-mers to list (default: 10).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG file to save a sorted-output plot to.",
    )
    args = parser.parse_args()

    weight, bias = build_filter(FILTER)

    # ── Enumerate all 4**6 = 4096 six-mers and one-hot encode them. ──
    all_seqs = ["".join(p) for p in itertools.product(DNA_ALPHABET, repeat=KERNEL_SIZE)]
    onehot = one_hot_batch(all_seqs)  # (4096, 4, 6)

    # ── Filter output per 6-mer. For one-hot inputs a conv is just the sum of the
    # weights sitting under the active base at each position, plus the bias. The
    # model then passes this through softplus (see compute_sr_profile), so we apply
    # it here too. softplus is monotonic, so the ranking/argmax are unchanged, but
    # the values — and the 50%-of-max threshold — reflect the actual activation. ──
    raw = np.einsum("ncp,cp->n", onehot, weight) + bias  # (4096,), pre-activation
    outputs = np.logaddexp(0.0, raw)  # softplus(raw) = log(1 + exp(raw)), stable

    best_idx = int(np.argmax(outputs))
    best_seq = all_seqs[best_idx]
    best_out = float(outputs[best_idx])
    best_raw = float(raw[best_idx])
    half_max = 0.5 * best_out

    # 6-mers reaching the half-max threshold, ranked highest-first.
    reach_idx = np.argsort(-outputs)
    reach_idx = [i for i in reach_idx if outputs[i] >= half_max]

    # The 6-mer whose output sits closest to the half-max value — the "half-maximal"
    # input, i.e. the 50% counterpart to the best 6-mer above. For a sharp filter
    # nothing lands near half-max, so we also report the gap to be honest about it.
    half_idx = int(np.argmin(np.abs(outputs - half_max)))
    half_seq = all_seqs[half_idx]
    half_out = float(outputs[half_idx])
    half_gap = half_out - half_max

    # ── Report. ──
    lines = []
    lines.append("=" * 60)
    print_filter(weight, bias)  # prints via logger; grid also captured below
    lines.append(f"Scanned all {len(all_seqs)} six-mers (4**6).")
    lines.append("Output = softplus(sum of filter weights under each base + bias).")
    lines.append("")
    lines.append(
        f"HIGHEST output : {best_out:+.4f}  (softplus; pre-activation {best_raw:+.4f})"
    )
    lines.append(f"  input 6-mer  : {best_seq}")
    lines.append(
        f"50% of highest : {half_max:+.4f}  (half-max threshold, on softplus output)"
    )
    lines.append(
        f"  closest input: {half_seq}  (output {half_out:+.4f}, "
        f"{half_gap:+.4f} vs half-max)"
    )
    lines.append(f"  {len(reach_idx)} six-mer(s) reach >= 50% of the highest output.")
    lines.append("")
    lines.append(
        f"Top {min(args.top, len(all_seqs))} six-mers by output (softplus, raw):"
    )
    for rank, i in enumerate(np.argsort(-outputs)[: args.top], start=1):
        flag = "  <-- >=50% max" if outputs[i] >= half_max else ""
        lines.append(
            f"  {rank:2d}. {all_seqs[i]}  {outputs[i]:+.4f}  (raw {raw[i]:+.4f}){flag}"
        )

    for line in lines:
        logger.info(line)

    if args.out:
        with open(args.out, "w") as fh:
            pos_header = "".ljust(8) + "".join(
                f"pos{p + 1}".ljust(9) for p in range(KERNEL_SIZE)
            )
            fh.write(f"Filter (bias={bias:+.4f}):\n  {pos_header}\n")
            for ch_idx, nt in enumerate(DNA_ALPHABET):
                cells = "".join(
                    f"{weight[ch_idx, p]:+.4f}".ljust(9) for p in range(KERNEL_SIZE)
                )
                fh.write(f"    {nt}".ljust(10) + cells + "\n")
            for line in lines:
                fh.write(line + "\n")
            # Full ranked list of every 6-mer at/above the half-max threshold.
            fh.write(
                "\nAll six-mers reaching >= 50% of the highest output (softplus, raw):\n"
            )
            for i in reach_idx:
                fh.write(f"  {all_seqs[i]}  {outputs[i]:+.4f}  (raw {raw[i]:+.4f})\n")
        logger.info(f"\nWrote full report to {args.out}")

    # ── Sorted-output plot: every 6-mer's softplus output, ranked highest-first,
    # with the peak, the half-max threshold, and how many 6-mers clear it. ──
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")  # save to file, no display needed
        import matplotlib.pyplot as plt

        sorted_out = np.sort(outputs)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sorted_out, lw=1, color="tab:blue")
        ax.axhline(
            half_max,
            color="tab:red",
            ls="--",
            label=f"half-max = {half_max:+.4f}",
        )
        ax.axvline(
            len(reach_idx),
            color="tab:green",
            ls=":",
            label=f"{len(reach_idx)} six-mer(s) >= half-max",
        )
        ax.set_xlabel("6-mer rank (highest output first)")
        ax.set_ylabel("softplus output")
        ax.set_title(f"Sorted filter outputs over all {len(all_seqs)} six-mers")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        logger.info(f"\nWrote sorted-output plot to {args.plot}")


if __name__ == "__main__":
    main()
