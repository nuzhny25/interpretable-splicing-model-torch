import numpy as np
import torch
from model import PNASModel

from dataset_preparations.maf_processing import SPECIES

DATA_DIR = "data/multiz100"

state_dict = torch.load("model_weights.pt", map_location="cpu")

results = {}

for name in SPECIES:
    path = f"{DATA_DIR}/{name}_malat1_chunks.npz"

    try:
        dataset = np.load(path)
    except FileNotFoundError:
        print(f"Skipping {name}: {path} not found")
        continue

    x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)

    model = PNASModel(input_length=x_seq.shape[-1])
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")
        sr_balance = model.compute_sr_balance(x_seq, agg="mean")

    results[f"{name}_sr"] = sr_balance.numpy()
    results[f"{name}_incl_mean"] = a_incl.numpy().mean(axis=1)
    results[f"{name}_skip_mean"] = a_skip.numpy().mean(axis=1)

np.savez(f"{DATA_DIR}/embeddings.npz", **results)
