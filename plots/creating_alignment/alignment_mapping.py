from Bio import AlignIO
import sys, os
import numpy as np
import json


ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
from dataset_preparations.maf_processing import SPECIES

MAF_PATH = os.path.join(ROOT, "data/multiz100/MALAT1_orthologues_multiz100.maf")


def main():
    # Parsing the maf file to extract the aligned sequences for selected species
    alignments = list(AlignIO.parse(MAF_PATH, "maf"))

    sequences = {name: "" for name in SPECIES}

    for alignment in alignments:
        block_len = alignment.get_alignment_length()
        present = {line.id: str(line.seq) for line in alignment}
        for name, maf_id in SPECIES.items():
            if maf_id in present:
                sequences[name] += present[maf_id]
            else:
                sequences[name] += "-" * block_len

    matrix_np = np.array([list(s) for s in sequences.values()])

    print(matrix_np.shape)

    # transposing the array so I can iterate trhough the rows and delete those ones that are all == to "-"
    matrix_np = matrix_np.T

    print(matrix_np.shape)

    mask = ~np.all(matrix_np == "-", axis=1)
    matrix_np = matrix_np[mask]

    np.save(os.path.join(ROOT, "data/multiz100/alignment_matrix.npy"), matrix_np)

    print(matrix_np.shape)

    # Iterate through every aligned sequence and find the location for every nucleotide
    mapping = [[] for _ in range(matrix_np.shape[1])]
    for aligned_idx, row in enumerate(matrix_np):
        for species_idx, char in enumerate(row):
            if char != "-":
                mapping[species_idx].append(aligned_idx)

    print(mapping[0][:100])

    with open("alignment_mapping.json", "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    main()
