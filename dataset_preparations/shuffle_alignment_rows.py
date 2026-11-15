"""Write row-shuffled copies of an alignment matrix for the negative-control runs.

    python dataset_preparations/shuffle_alignment_rows.py --gene-dir multiz100 --seeds 0 1 2 3 4 5

Reads ``data/<gene-dir>/alignment_matrix.npy`` (``(n_aligned, n_species)``, one row per
aligned column) and, for each shuffle seed, permutes its rows once, writing the result to
``data/<gene-dir>_shuffled/shuf<seed>/alignment_matrix.npy``. The whole row moves, so
every aligned column keeps its cross-species nucleotide/gap composition and the species
stay aligned to one another — only the *order* of the columns is destroyed, so no
conserved motif survives as a contiguous block. Shuffle seed 0 is the same permutation as
control_testing/shuffle_train.py
(``matrix[np.random.default_rng(seed).permutation(matrix.shape[0])]``, SHUFFLE_SEED 0).

One permutation per training seed, so the control's seed-to-seed spread covers
permutation noise as well as initialization noise. The pipeline trains each one via
``--matrix`` while ``--gene malat1_shuffled`` keeps a single run name and label, so every
seed groups as the same gene downstream:

    cd custom_model
    python pipeline/run_pipeline.py --batch pipeline/shuffled_control.txt \\
        --runs-dir pipeline/runs_shuffled

Each permutation is saved next to its matrix (``row_permutation.npy``: shuffled row i is
original row perm[i]) along with a ``shuffle.json`` provenance record, so a shuffled run
can always be traced back to the real alignment.
"""

import argparse
import json
import os

import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gene-dir",
        default="multiz100",
        help="data/ subdirectory holding the source alignment_matrix.npy. "
        "Default: multiz100 (MALAT1).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Shuffle seeds, one permutation (and one shuf<seed>/ directory) each. "
        "Default: 0-5, one per training seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output data/ subdirectory name. Default: <gene-dir>_shuffled.",
    )
    args = parser.parse_args()

    src = os.path.join(DATA_DIR, args.gene_dir, "alignment_matrix.npy")
    out_root = os.path.join(DATA_DIR, args.out_dir or f"{args.gene_dir}_shuffled")
    matrix = np.load(src)
    # A species.json next to the source describes its columns; the shuffle leaves the
    # columns untouched, so it carries over verbatim for interactive_sr_heatmap.py.
    src_species = os.path.join(DATA_DIR, args.gene_dir, "species.json")

    for seed in args.seeds:
        out_dir = os.path.join(out_root, f"shuf{seed}")
        perm = np.random.default_rng(seed).permutation(matrix.shape[0])
        shuffled = matrix[perm]

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "alignment_matrix.npy"), shuffled)
        np.save(os.path.join(out_dir, "row_permutation.npy"), perm)
        if os.path.exists(src_species):
            with open(src_species) as fh, open(
                os.path.join(out_dir, "species.json"), "w"
            ) as out:
                out.write(fh.read())
        with open(os.path.join(out_dir, "shuffle.json"), "w") as fh:
            json.dump(
                {
                    "source_matrix": src,
                    "shuffle_seed": seed,
                    "axis": "rows (aligned positions); species columns untouched",
                    "shape": list(shuffled.shape),
                },
                fh,
                indent=2,
            )
        print(f"Row-shuffled {src} {matrix.shape} with seed {seed} -> {out_dir}")


if __name__ == "__main__":
    main()
