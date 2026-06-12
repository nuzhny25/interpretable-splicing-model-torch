"""Visually test the `permute_filters` function on the trained model.

Each sequence filter is a 4 x 6 matrix: rows are the nucleotide channels
(A, C, G, T) and columns are the 6 kernel positions. This script runs the real
`permute_filters` exactly as the analysis does (pristine weight copies + a
seeded RNG) and prints each filter before and after, plus the inferred column
order, so you can confirm:

  * columns move as intact units (nucleotides stay locked within a position),
  * conv_skip and conv_incl receive the SAME permutation,
  * scheme="independent" differs per filter; scheme="shared" does not.

Run from the repo root or from this directory.
"""

import os
import sys

import numpy as np
import torch

# Make the repo root importable and locate the checkpoint regardless of cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model import PNASModel
from filter_permutations import permute_filters

NUCLEOTIDES = ["A", "C", "G", "T"]

# --- what to display ---
SCHEME = "independent"  # "independent" or "shared"
SEED = 0
FILTER_IDS = [0, 1]  # print these filters so per-filter independence is visible


def format_filter(weight_4x6):
    """Return a labeled string for one 4 x 6 filter tensor."""
    kernel_size = weight_4x6.shape[1]
    header = "        " + "".join(f"pos{p + 1:<6}" for p in range(kernel_size))
    lines = [header]
    for nt, row in zip(NUCLEOTIDES, weight_4x6):
        cells = "".join(f"{v:+9.4f}" for v in row.tolist())
        lines.append(f"  {nt}   {cells}")
    return "\n".join(lines)


def infer_column_order(orig_4x6, after_4x6):
    """Recover which original column each post-permutation column came from.

    Returns a list where entry ``c`` is the original (0-based) position now
    sitting at column ``c``. Columns are matched whole, so this only succeeds if
    each column moved as an intact unit.
    """
    kernel_size = orig_4x6.shape[1]
    order = []
    for c in range(kernel_size):
        col = after_4x6[:, c]
        matches = [j for j in range(kernel_size) if torch.allclose(col, orig_4x6[:, j])]
        order.append(matches[0] if matches else None)
    return order


def main():
    state_dict = torch.load(
        os.path.join(REPO_ROOT, "model_weights.pt"), map_location="cpu"
    )

    model = PNASModel()
    model.load_state_dict(state_dict)
    model.eval()

    # Pristine copies and seeded RNG — exactly how the analysis drives it.
    orig_skip = model.conv_skip.weight.detach().clone()
    orig_incl = model.conv_incl.weight.detach().clone()
    rng = np.random.default_rng(SEED)

    permute_filters(model, orig_skip, orig_incl, SCHEME, rng)

    print(
        f"conv_skip.weight shape: {tuple(model.conv_skip.weight.shape)}  "
        f"(num_filters, channels, kernel_size)"
    )
    print(f"scheme={SCHEME!r}, seed={SEED}\n")

    for filter_idx in FILTER_IDS:
        for name, orig_w, conv in [
            ("SKIP", orig_skip, model.conv_skip),
            ("INCL", orig_incl, model.conv_incl),
        ]:
            before = orig_w[filter_idx]            # (4, 6)
            after = conv.weight[filter_idx]        # (4, 6), now permuted
            order = infer_column_order(before, after)
            order_1based = [None if o is None else o + 1 for o in order]

            print(f"=== {name} filter #{filter_idx} — BEFORE ===")
            print(format_filter(before))
            print(f"--- {name} filter #{filter_idx} — AFTER (columns now {order_1based}) ---")
            print(format_filter(after))
            print()


if __name__ == "__main__":
    main()
