import numpy as np
import torch
from model import PNASModel

datasets = ["data/human_malat1_chunks.npz", "data/mouse_malat1_chunks.npz"]
a_incl_list = []
a_skip_list = []
sr_balance_list = []

state_dict = torch.load("model_weights.pt", map_location="cpu")

for path in datasets:
    dataset = np.load(path)

    x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)

    model = PNASModel(input_length=x_seq.shape[-1])
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # Compute inclusion, skipping sequence activations
        a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")
        # Compute SR balance
        sr_balance = model.compute_sr_balance(x_seq, agg="mean")

    a_incl_list.append(a_incl.numpy())
    a_skip_list.append(a_skip.numpy())
    sr_balance_list.append(sr_balance.numpy())

human_sr = sr_balance_list[0]
mouse_sr = sr_balance_list[1]

human_incl_mean = a_incl_list[0].mean(axis=1)
mouse_incl_mean = a_incl_list[1].mean(axis=1)

np.savez(
    "data/embeddings.npz",
    human_sr=human_sr,
    mouse_sr=mouse_sr,
    human_incl_mean=human_incl_mean,
    mouse_incl_mean=mouse_incl_mean,
)
