"""Manually set the conv weights and biases of a PNASModel, then save it.

Build a checkpoint with hand-picked filter weights/biases instead of training.
Useful for probing compute_sr_profile: e.g. plant an all-ones "A" filter and
watch how a poly-A window scores.

Layout of each conv bank (conv_incl / conv_skip):
    weight  shape (20, 4, 6)  = (num_filters, channels, kernel_size)
    bias    shape (20,)       = one scalar per filter
Channel/row order is A, C, G, T (custom_utils.DNA_ALPHABET).

Everything you set lives in the EDIT HERE block below. Unspecified rows default
to 0; unspecified filters keep whatever the starting point gave them (all zeros
by default, or the values from --from-checkpoint).

Run:
    python set_weights.py                       # zeros base, apply edits, save
    python set_weights.py --from-checkpoint ../weights/custom_model.pt
    python set_weights.py --out my_model.pt --check-seq AAAAAACGTACGT
"""

import argparse
import logging
import os
import sys

import torch

# Make the parent custom_model package importable when run from experiment/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from custom_model import PNASModel  # noqa: E402
from custom_utils import DNA_ALPHABET, one_hot_batch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────── EDIT HERE ───────────────────────────────
# Start from all-zero weights/biases ("zeros") or warm-start from a checkpoint
# passed via --from-checkpoint. --from-checkpoint overrides this.
START_FROM = "zeros"

# One dict per filter you want to set. Keys:
#   "bias"          -> scalar bias for that filter (optional)
#   "A"/"C"/"G"/"T" -> list of 6 weights along the kernel for that row (optional)
# Anything omitted stays at its starting value (0 for the zeros base).
INCL_FILTERS = {
    # Filter #1: detects a run of A's. Weight 1 on the A-row at every position.
    1: {
        "bias": 1.0,
        "A": [1, 1, 1, 1, 1, 1],
    },
}

SKIP_FILTERS = {
    # Example: uncomment to plant a poly-G skip filter with a negative bias.
    # 7: {"bias": -3.0, "G": [1, 1, 1, 1, 1, 1]},
}
# ──────────────────────────── END EDIT HERE ──────────────────────────────


def zero_conv(conv):
    """Set every weight and bias in a conv bank to 0."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()


def apply_filter_specs(conv, specs, bank_name):
    """Write hand-set weights/biases into one conv bank.

    Args:
        conv: The nn.Conv1d bank (conv_incl or conv_skip).
        specs: Mapping {filter_idx: {"bias": float, "A": [...], ...}}.
        bank_name: "INCL"/"SKIP", for log messages and error context.
    """
    num_filters, _, kernel_size = conv.weight.shape
    with torch.no_grad():
        for idx, spec in specs.items():
            if not 0 <= idx < num_filters:
                raise ValueError(
                    f"{bank_name} filter #{idx} out of range 0..{num_filters - 1}"
                )
            if "bias" in spec:
                conv.bias[idx] = float(spec["bias"])
            for nt in DNA_ALPHABET:
                if nt not in spec:
                    continue
                values = spec[nt]
                if len(values) != kernel_size:
                    raise ValueError(
                        f"{bank_name} filter #{idx} row {nt}: expected "
                        f"{kernel_size} weights, got {len(values)}"
                    )
                channel = DNA_ALPHABET.index(nt)
                conv.weight[idx, channel, :] = torch.as_tensor(
                    values, dtype=conv.weight.dtype
                )
            logger.info(f"  set {bank_name} filter #{idx}: {spec}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        help="Warm-start from this checkpoint before applying edits "
        "(otherwise START_FROM decides).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "manual_model.pt"),
        help="Where to save the resulting checkpoint.",
    )
    parser.add_argument(
        "--check-seq",
        default=None,
        help="Optional sequence to run through compute_sr_profile as a sanity check.",
    )
    args = parser.parse_args()

    model = PNASModel()

    # ── Starting point: checkpoint warm-start, or an all-zero base. ──
    if args.from_checkpoint:
        ckpt = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Warm-started from {args.from_checkpoint}")
    elif START_FROM == "zeros":
        zero_conv(model.conv_incl)
        zero_conv(model.conv_skip)
        logger.info("Started from all-zero weights and biases")
    else:
        raise ValueError(f"Unknown START_FROM={START_FROM!r} (use 'zeros')")

    # ── Apply the hand-set filters from the EDIT HERE block. ──
    logger.info("Applying manual filter edits:")
    apply_filter_specs(model.conv_incl, INCL_FILTERS, "INCL")
    apply_filter_specs(model.conv_skip, SKIP_FILTERS, "SKIP")
    model.eval()

    # ── Optional sanity check: score a sequence through compute_sr_profile. ──
    if args.check_seq:
        x = torch.from_numpy(one_hot_batch([args.check_seq]))
        with torch.no_grad():
            profile = model.compute_sr_profile(x)[0]  # (L-5,)
        logger.info(f"\ncompute_sr_profile('{args.check_seq}'):")
        logger.info(f"  per-window SR-balance: {profile.tolist()}")
        logger.info(
            f"  sum={profile.sum().item():+.4f}  "
            f"mean={profile.mean().item():+.4f}  "
            f"max={profile.max().item():+.4f}"
        )

    # ── Save in the {"model_state_dict": ...} format the rest of the repo loads. ──
    torch.save({"model_state_dict": model.state_dict()}, args.out)
    logger.info(f"\nWrote checkpoint to {args.out}")


if __name__ == "__main__":
    main()
