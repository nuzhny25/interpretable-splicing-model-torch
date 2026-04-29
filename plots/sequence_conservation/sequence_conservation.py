import sys, os
import numpy as np
import json

ROOT = os.path.join(os.path.dirname(__file__), "../..")

MATRIX_PATH = os.path.join(ROOT, "data/multiz100/alignment_matrix.npy")
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "../creating_alignment/alignment_mapping.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "sequence_conservation.json")


def calculate_overlap_row(row):
    base_nucleotide = row[0]
    if base_nucleotide == "-":
        return None
    count = 0
    for nucleotide in row[1:]:
        if nucleotide != "-" and nucleotide == base_nucleotide:
            count += 1
    return count


def calculate_overlap_window(matrix):
    num_other_species = matrix.shape[1] - 1
    total = 0
    valid_rows = 0
    for row in matrix:
        result = calculate_overlap_row(row)
        if result is not None:
            total += result
            valid_rows += 1
    if valid_rows == 0:
        return 0.0
    return total / (valid_rows * num_other_species) * 100


def calculate_overlap_matrix(matrix, human_mapping):
    window_size = 70
    step_size = 10
    positions = []
    conservation = []

    for i in range(0, len(human_mapping) - window_size + 1, step_size):
        start_col = human_mapping[i]
        end_col = human_mapping[i + window_size - 1] + 1
        center_col = human_mapping[i + window_size // 2]
        window = matrix[start_col:end_col]
        positions.append(center_col)
        conservation.append(calculate_overlap_window(window))

    return positions, conservation


def main():
    matrix = np.load(MATRIX_PATH)
    with open(MAPPING_PATH) as f:
        human_mapping = json.load(f)[0]

    positions, conservation = calculate_overlap_matrix(matrix, human_mapping)

    with open(OUT_PATH, "w") as f:
        json.dump({"positions": positions, "conservation": conservation}, f)


if __name__ == "__main__":
    main()
