"""Run PNASModel with hand-set weights on a 4xN aligned matrix and report everything.

A diagnostic companion to set_weights.py. Instead of saving a checkpoint, this
plants hand-picked conv filter weights/biases (no training), runs a small matrix
of equal-length "aligned" sequences through compute_sr_profile, and writes — in a
single run, to a text file (--out) — the filters and biases, the
(n_species, n_windows) SR-balance matrix, every intermediate quantity that feeds
the cross-species SD loss, and the final loss. The loss math mirrors the no-gap trainer
(custom_model/training/custom_synthetic_train.py); it is the right variant for a
gapless, equal-length matrix (the real-alignment loss needs gaps and >= 5 species).

Everything you tweak lives in the EDIT HERE block below: the sequences, which
filters to plant, and the loss knobs. Channel/row order is A, C, G, T
(custom_utils.DNA_ALPHABET). Each conv bank is (20, 4, 6) = (num_filters, channels,
kernel_size); the SR matrix has L - 6 + 1 = L - 5 windows per sequence.

Run:
    python run_manual_matrix.py                      # writes run_manual_matrix_output.txt
    python run_manual_matrix.py --out report.txt
    python run_manual_matrix.py --from-checkpoint manual_model.pt
"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F

# Make the parent custom_model package importable when run from experiment/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from custom_model import PNASModel  # noqa: E402
from custom_utils import DNA_ALPHABET, one_hot_batch  # noqa: E402

# Reuse the weight-planting helpers from set_weights.py (importing it does not run
# its main()). zero_conv wipes a bank to 0; apply_filter_specs writes the dicts below.
from set_weights import apply_filter_specs, zero_conv  # noqa: E402

# Silence the incidental INFO logs from the imported modules ("total parameters",
# "set INCL filter #...") so the report below is the only output, in stream order.
# The planted filters are still confirmed by the FILTERS section's [edited] tags.
logging.getLogger().setLevel(logging.WARNING)


# ─────────────────────────────── EDIT HERE ───────────────────────────────
# The aligned matrix: one sequence per "species", all the same length (gapless).
SEQUENCES = [
    "ACGTACAAAAAAACGTACCCCCCCACGTAC",
    "CGTACGAAAAAACGTACGCCCCCCCGTACG",
    "GTACGTAAAAAAGTACGTCCCCCCGTACGT",
    "TACGTAAAAAAATACGTACCCCCCTACGTA",
]

# Start from all-zero weights/biases ("zeros") and plant the filters below.
# --from-checkpoint overrides this entirely (loads a set_weights.py output as-is).
START_FROM = "zeros"

# One dict per filter to set. Keys:
#   "bias"          -> scalar bias for that filter (optional)
#   "A"/"C"/"G"/"T" -> list of 6 weights along the kernel for that row (optional)
# Anything omitted stays 0 (the zeros base).
INCL_FILTERS = {
    # Filter #1: detects a run of A's. Weight 1 on the A-row at every position.
    0: {
        "bias": 0.0,
        "A": [1, 1, 1, 1, 1, 1],
    },
}

SKIP_FILTERS = {
    # Example: uncomment to plant a poly-G skip filter with a negative bias.
    # 7: {"bias": -3.0, "G": [1, 1, 1, 1, 1, 1]},
}

# Loss knobs — same names/values as the trainer.
L1_LAMBDA = 1e-2  # strength of L1 penalty on the conv filter weights
SMOOTH_SIGMA = (
    0  # Gaussian smoothing of each SR track (nt); 0 = raw, 1.5 = training parity
)

# Print only filters that were hand-set / have any non-zero weight or bias.
# Set True to dump all 20 INCL + 20 SKIP filters.
PRINT_ALL_FILTERS = True
# ──────────────────────────── END EDIT HERE ──────────────────────────────


def print_filters(conv, specs, bank_name):
    """Print each filter's 4x6 weight grid and bias for one conv bank.

    Mirrors the text layout in plot_filters.py (8-wide nt label, 9-wide value
    cells, pos1..posK header). By default only filters that were hand-set (present
    in ``specs``) or have any non-zero weight/bias are shown; the rest are counted.
    """
    weight = conv.weight.detach().cpu()
    bias = conv.bias.detach().cpu()
    num_filters, _, kernel_size = weight.shape
    edited = set(specs.keys())

    print(
        f"── {bank_name} filters (weight shape {tuple(weight.shape)} = "
        f"num_filters, channels, kernel_size) ──"
    )
    pos_header = "".ljust(8) + "".join(
        f"pos{p + 1}".ljust(9) for p in range(kernel_size)
    )
    shown, omitted = 0, 0
    for f in range(num_filters):
        nonzero = weight[f].abs().sum().item() > 0 or bias[f].item() != 0
        if not (PRINT_ALL_FILTERS or f in edited or nonzero):
            omitted += 1
            continue
        shown += 1
        tag = "  [edited]" if f in edited else ""
        print(f"\n  {bank_name} filter #{f} (bias={bias[f]:+.4f}){tag}")
        print("  " + pos_header)
        for ch_idx, nt in enumerate(DNA_ALPHABET):
            cells = "".join(
                f"{weight[f, ch_idx, p]:+.4f}".ljust(9) for p in range(kernel_size)
            )
            print(f"    {nt}".ljust(10) + cells)
    if omitted:
        print(
            f"\n  ({omitted} all-zero {bank_name} filters omitted; "
            f"set PRINT_ALL_FILTERS=True to show them)"
        )
    print()


def print_sr_matrix(title, sr, seqs):
    """Print a (n_species, n_windows) SR matrix with a per-row min/max/mean summary."""
    n_rows, n_win = sr.shape
    print(title)
    header = " " * 6 + "".join(f"{'w' + str(c):>8}" for c in range(n_win))
    print(header)
    for r in range(n_rows):
        cells = "".join(f"{sr[r, c].item():+8.3f}" for c in range(n_win))
        print(f"  s{r}: {cells}")
    print()
    for r in range(n_rows):
        row = sr[r]
        print(
            f"  s{r}  min={row.min().item():+.4f}  max={row.max().item():+.4f}  "
            f"mean={row.mean().item():+.4f}  seq={seqs[r]}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        help="Load weights from this checkpoint (a set_weights.py output) instead of "
        "the zeros base + EDIT HERE dicts.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "run_manual_matrix_output.txt"),
        help="Text file to write the full report to (default: run_manual_matrix_output.txt).",
    )
    args = parser.parse_args()

    # Send the whole report to the output text file instead of the terminal.
    # Tracebacks still surface on the terminal (they go to stderr). Restored at the
    # end of main, where we print the output path to the real terminal.
    report_file = open(args.out, "w")
    sys.stdout = report_file

    model = PNASModel()

    # ── Weights: checkpoint as-is, or zeros base + planted EDIT HERE filters. ──
    if args.from_checkpoint:
        ckpt = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.from_checkpoint}\n")
    elif START_FROM == "zeros":
        zero_conv(model.conv_incl)
        zero_conv(model.conv_skip)
        print("Started from all-zero weights and biases\n")
        print("Applying manual filter edits:")
        apply_filter_specs(model.conv_incl, INCL_FILTERS, "INCL")
        apply_filter_specs(model.conv_skip, SKIP_FILTERS, "SKIP")
        print()
    else:
        raise ValueError(f"Unknown START_FROM={START_FROM!r} (use 'zeros')")
    model.eval()

    # ── 1) Filters and biases. ──
    incl_specs = {} if args.from_checkpoint else INCL_FILTERS
    skip_specs = {} if args.from_checkpoint else SKIP_FILTERS
    print("=" * 78)
    print("FILTERS AND BIASES")
    print("=" * 78)
    print_filters(model.conv_incl, incl_specs, "INCL")
    print_filters(model.conv_skip, skip_specs, "SKIP")

    # ── 2) SR-balance matrix from compute_sr_profile. ──
    x = torch.from_numpy(one_hot_batch(SEQUENCES))  # (n_species, 4, L)
    with torch.no_grad():
        sr_raw, a_incl, a_skip = model.compute_sr_profile(
            x, return_activations=True
        )  # (n_species, L-5)
    n_win = sr_raw.shape[1]
    print("=" * 78)
    print(
        f"SR-BALANCE MATRIX  (compute_sr_profile, shape {tuple(sr_raw.shape)} = "
        f"n_species x n_windows)"
    )
    print("=" * 78)
    print_sr_matrix(
        "Per-window SR-balance (a_incl - a_skip), row = species, col = window:",
        sr_raw,
        SEQUENCES,
    )

    # ── 3) Loss components (mirrors custom_synthetic_train.py). ──
    print("=" * 78)
    print("LOSS COMPONENTS")
    print("=" * 78)

    # Optional Gaussian smoothing of each SR track along position (0 = skip).
    sr = sr_raw
    if SMOOTH_SIGMA:
        smooth_radius = math.ceil(3 * SMOOTH_SIGMA)  # kernel half-width (~3 sigma)
        offsets = torch.arange(-smooth_radius, smooth_radius + 1, dtype=torch.float32)
        gauss_kernel = torch.exp(-(offsets**2) / (2 * SMOOTH_SIGMA**2))
        gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)  # (1, 1, K)
        with torch.no_grad():
            sr = F.conv1d(
                F.pad(
                    sr_raw.unsqueeze(1), (smooth_radius, smooth_radius), mode="reflect"
                ),
                gauss_kernel,
            ).squeeze(1)
        print(
            f"Gaussian smoothing: sigma={SMOOTH_SIGMA} nt, kernel size={gauss_kernel.shape[-1]}\n"
        )
        print_sr_matrix(
            "Smoothed SR-balance (this is what the loss sees):", sr, SEQUENCES
        )
    else:
        print("Smoothing: OFF (SMOOTH_SIGMA=0) — loss uses the raw SR matrix above.\n")

    with torch.no_grad():
        # Numerator: cross-species SD per window (torch.std uses ddof=1, so with
        # n_species rows the column SD divides by n_species - 1).
        col_std = sr.std(dim=0)  # (n_windows,)
        # Denominator: within-species dynamic range = each row's SD across windows,
        # averaged over rows. Dividing by it makes the loss scale-free.
        row_std = sr.std(dim=1)  # (n_species,)
        within_species_scale = row_std.mean()  # scalar
        sd_loss = col_std.mean() / within_species_scale

        l1_penalty = L1_LAMBDA * (a_incl + a_skip).mean()
        loss = sd_loss + l1_penalty

    print(
        f"[numerator] col_std — cross-species SD per window (ddof=1 over {sr.shape[0]} species):"
    )
    print("  " + "".join(f"{col_std[c].item():8.3f}" for c in range(n_win)))
    print(f"  col_std.mean() = {col_std.mean().item():.6f}\n")

    print(f"[denominator] within-species SD per species (ddof=1 over {n_win} windows):")
    print("  " + "  ".join(f"s{r}={row_std[r].item():.4f}" for r in range(sr.shape[0])))
    print(
        f"  within_species_scale = mean(row SDs) = {within_species_scale.item():.6f}\n"
    )

    print(f"sd_loss    = col_std.mean() / within_species_scale = {sd_loss.item():.6f}")
    print(
        f"mean a_incl (inclusion activation, summed softplus) = {a_incl.mean().item():.6f}"
    )
    print(
        f"mean a_skip (skipping activation, summed softplus)  = {a_skip.mean().item():.6f}"
    )
    print(
        f"l1_penalty = L1_LAMBDA({L1_LAMBDA}) * mean(a_incl + a_skip) = {l1_penalty.item():.6f}\n"
    )

    print("=" * 78)
    print(f"LOSS = sd_loss + l1_penalty = {loss.item():.6f}")
    print("=" * 78)

    # Restore the terminal and report where the output was written.
    sys.stdout = sys.__stdout__
    report_file.close()
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
