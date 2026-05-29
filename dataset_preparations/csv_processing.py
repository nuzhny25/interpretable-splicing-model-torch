import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from utils import DEFAULT_SEQUENCE_COLUMN, dataframe_to_dataset, save_dataset_npz
from maf_processing import SPECIES

DATA_DIR = "../data/multiz100"


def prepare_data(
    input_path,
    output_path,
    sequence_column=DEFAULT_SEQUENCE_COLUMN,
    flanks=True,
    rnafold="RNAfold",
    temperature=37.0,
    max_bp_span=0,
    commands_file="",
    num_threads=8,
) -> None:
    df = pd.read_csv(input_path)
    dataset = dataframe_to_dataset(
        df,
        sequence_column=sequence_column,
        add_flanks=flanks,
        rnafold_bin=rnafold,
        temperature=temperature,
        maxBPspan=max_bp_span,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    output_path = save_dataset_npz(dataset, output_path)

    print(f"Saved dataset to {Path(output_path).resolve()}")
    print(f"Examples: {dataset['seq_oh'].shape[0]}")
    print(f"Sequence tensor shape (N, 4, L): {dataset['seq_oh'].shape}")
    print(f"Structure tensor shape (N, 3, L): {dataset['struct_oh'].shape}")
    print(f"Wobble tensor shape (N, 1, L): {dataset['wobbles'].shape}")
    print(
        "Feature arrays are NumPy arrays; convert them to torch tensors for model inference."
    )


def main():
    for name in SPECIES:
        input_path = f"{DATA_DIR}/{name}_malat1_complement_chunks.csv"
        output_path = f"{DATA_DIR}/{name}_malat1_complement_chunks.npz"

        try:
            with open(input_path, "r"):
                prepare_data(input_path, output_path)
            os.remove(input_path)
        except FileNotFoundError:
            print(f"Skipping {name}: {input_path} not found")
            continue


if __name__ == "__main__":
    main()
