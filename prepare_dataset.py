"""CLI for preparing model-ready datasets from sequence CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import DEFAULT_SEQUENCE_COLUMN, dataframe_to_dataset, save_dataset_npz


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        Configured argument parser for dataset preparation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV of unflanked exon sequences, compute structure and "
            "wobble features, and save a compressed NPZ dataset."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV file containing an unflanked sequence column.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output .npz path for the prepared dataset.",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help=(
            "Column name for the unflanked input sequence. Defaults to "
            f"{DEFAULT_SEQUENCE_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--no-flanks",
        action="store_true",
        help="Skip adding the fixed model flanks before computing features.",
    )
    parser.add_argument(
        "--rnafold-bin",
        default="RNAfold",
        help="Executable name or path for ViennaRNA RNAfold.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=37.0,
        help="RNAfold temperature in Celsius.",
    )
    parser.add_argument(
        "--max-bp-span",
        type=int,
        default=0,
        help="Optional RNAfold maximum base-pair span. Zero disables the flag.",
    )
    parser.add_argument(
        "--commands-file",
        default="",
        help="Optional ViennaRNA commands file passed through to RNAfold.",
    )
    parser.add_argument(
        "--num-threads",
        default=8,
        type=int,
        help="Number of threads to use for ViennaRNA.",
    )
    return parser


def main() -> None:
    """Run the dataset preparation CLI."""
    args = build_parser().parse_args()

    df = pd.read_csv(args.input_csv)
    dataset = dataframe_to_dataset(
        df,
        sequence_column=args.sequence_column,
        add_flanks=not args.no_flanks,
        rnafold_bin=args.rnafold_bin,
        temperature=args.temperature,
        maxBPspan=args.max_bp_span,
        commands_file=args.commands_file,
        num_threads=args.num_threads,
    )
    output_path = save_dataset_npz(dataset, args.output_path)

    print(f"Saved dataset to {Path(output_path).resolve()}")
    print(f"Examples: {dataset['seq_oh'].shape[0]}")
    print(f"Sequence tensor shape (N, 4, L): {dataset['seq_oh'].shape}")
    print(f"Structure tensor shape (N, 3, L): {dataset['struct_oh'].shape}")
    print(f"Wobble tensor shape (N, 1, L): {dataset['wobbles'].shape}")
    print("Feature arrays are NumPy arrays; convert them to torch tensors for model inference.")


if __name__ == "__main__":
    main()
