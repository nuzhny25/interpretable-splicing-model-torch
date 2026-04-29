import pandas as pd

from maf_processing import SPECIES
from chunking import chunk_sequence

DATA_DIR = "../data/multiz100"
WINDOW_SIZE = 70
STEP_SIZE = 10


def main():

    for name in SPECIES:
        input_path = f"{DATA_DIR}/{name}_malat1.txt"
        output_path = f"{DATA_DIR}/{name}_malat1_reversed_chunks.csv"

        try:
            with open(input_path, "r") as file:
                seq = file.read().replace("\n", "")
        except FileNotFoundError:
            print(f"Skipping {name}: {input_path} not found")
            continue

        seq = seq[::-1]

        chunks, positions = chunk_sequence(seq, WINDOW_SIZE, STEP_SIZE)
        pd.DataFrame({"exon": chunks}).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
