"""NEAT1 counterpart of alignment_mapping.py — builds data/neat1/alignment_matrix.npy.

Same output contract as the MALAT1 matrix that custom_real_train_absolute_deviation.py
already consumes: a ``(n_aligned, n_species)`` array of single characters whose columns
are in ``neat1_maf_processing.SPECIES`` insertion order, with ``-`` for gaps, ``N`` for
ambiguous bases, and lowercase ``a/c/g/t`` preserved as soft-masked nucleotides.

One difference from the MALAT1 build: NEAT1.maf contains lowercase soft-masked ``n`` as
well as uppercase ``N`` (10 vs 218 occurrences). The training script classifies a
character as a gap by testing ``char != "-" and char != "N"`` on the raw matrix entry and
only then uppercases it, so a lowercase ``n`` would survive that test and reach
``str_to_vector``, which rejects anything outside ACGT. Uppercasing ``n`` to ``N`` here
keeps that gap test correct and means the training script needs no change.
"""

from Bio import AlignIO
import sys, os
import numpy as np
import json

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
from dataset_preparations.neat1_maf_processing import SPECIES

MAF_PATH = os.path.join(ROOT, "data/neat1/NEAT1.maf")


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

    # Fold lowercase soft-masked n into the uppercase N the training script treats as a
    # gap; every other lowercase letter stays as-is, exactly like the MALAT1 matrix.
    matrix_np[matrix_np == "n"] = "N"

    # transposing the array so I can iterate trhough the rows and delete those ones that are all == to "-"
    matrix_np = matrix_np.T

    print(matrix_np.shape)

    mask = ~np.all(matrix_np == "-", axis=1)
    matrix_np = matrix_np[mask]

    np.save(os.path.join(ROOT, "data/neat1/alignment_matrix.npy"), matrix_np)

    print(matrix_np.shape)

    # Iterate through every aligned sequence and find the location for every nucleotide
    mapping = [[] for _ in range(matrix_np.shape[1])]
    for aligned_idx, row in enumerate(matrix_np):
        for species_idx, char in enumerate(row):
            if char != "-" and char != "N":
                mapping[species_idx].append(aligned_idx)

    print(mapping[0][:100])

    with open(os.path.join(ROOT, "data/neat1/alignment_mapping.json"), "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    main()
