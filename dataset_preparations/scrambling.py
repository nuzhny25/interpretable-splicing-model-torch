import pandas as pd

from maf_processing import SPECIES
from chunking import chunk_sequence
import numpy as np
import random
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import PNASModel

from chunking import chunk_sequence
from csv_processing import prepare_data

OUTPUT_DATA_DIR = "../data/human_shuffling/"
N_OF_SHUFFLINGS = 1000
INPUT_FILE = "../data/multiz100/human_malat1.txt"
WINDOW_SIZE = 70
STEP_SIZE = 10


def dinuc_shuffle(seq, rng):
    from collections import defaultdict

    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])
    for k in edges:
        rng.shuffle(edges[k])
    stack = [seq[0]]
    path = []
    while stack:
        v = stack[-1]
        if edges.get(v):
            stack.append(edges[v].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    assert len(path) == len(seq), f"Hierholzer truncated: {len(path)} vs {len(seq)}"
    assert (
        path[0] == seq[0] and path[-1] == seq[-1]
    ), "Terminal nucleotides not preserved"
    return "".join(path)


def main():
    with open(INPUT_FILE, "r") as file:
        seq = file.read().replace("\n", "")

    state_dict = torch.load("../model_weights.pt", map_location="cpu")
    results = {}

    for i in range(N_OF_SHUFFLINGS):
        shuffle_path = f"{OUTPUT_DATA_DIR}shuffle{i}"  # creating a sequential path

        shuffling = dinuc_shuffle(
            seq, np.random.default_rng()
        )  # creating a random dinucleotide shuffling from the human malat1 txt

        chunks, pos = chunk_sequence(
            shuffling, WINDOW_SIZE, STEP_SIZE
        )  # chunking the random shuffled sequence

        pd.DataFrame({"exon": chunks}).to_csv(
            f"{shuffle_path}.csv", index=False
        )  # saving the chunks as a csv file

        prepare_data(
            f"{shuffle_path}.csv", f"{shuffle_path}.npz"
        )  # preprocessing the csv file to be used for the model

        os.remove(f"{shuffle_path}.csv")  # deleting the cvs file

        dataset = np.load(f"{shuffle_path}.npz")  # loading the preprocessed file
        os.remove(f"{shuffle_path}.npz")  # deleting the preprocessed file

        # pytorch initialization

        x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)

        model = PNASModel(input_length=x_seq.shape[-1])
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            sr_balance = model.compute_sr_balance(x_seq, agg="mean")

        results[f"{i}_sr"] = sr_balance.numpy()

    np.savez(f"{OUTPUT_DATA_DIR}/shuffle_embeddings.npz", **results)


if __name__ == "__main__":
    main()
