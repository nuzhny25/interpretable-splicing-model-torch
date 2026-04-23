import pandas as pd

from maf_processing import SPECIES

DATA_DIR = "../data/multiz100"
WINDOW_SIZE = 70
STEP_SIZE = 10


def chunk_sequence(seq, window_size, step_size):
    chunks, positions = [], []
    for i in range(0, len(seq) - window_size + 1, step_size):
        chunks.append(seq[i : i + window_size])
        positions.append(i)
    return chunks, positions


for name in SPECIES:
    input_path = f"{DATA_DIR}/{name}_malat1.txt"
    output_path = f"{DATA_DIR}/{name}_malat1_chunks.csv"

    try:
        with open(input_path, "r") as file:
            seq = file.read().replace("\n", "")
    except FileNotFoundError:
        print(f"Skipping {name}: {input_path} not found")
        continue

    chunks, positions = chunk_sequence(seq, WINDOW_SIZE, STEP_SIZE)
    pd.DataFrame({"exon": chunks}).to_csv(output_path, index=False)
