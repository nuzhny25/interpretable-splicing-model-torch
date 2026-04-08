import numpy as np
import torch
import matplotlib.pyplot as plt
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

step_size = 10 

# Create x-axis positions for the plots (Position in the transcript)
# Note: Human and Mouse MALAT1 might be slightly different lengths!
x_human = np.arange(len(human_sr)) * step_size
x_mouse = np.arange(len(mouse_sr)) * step_size

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- PLOT 1: SR Balance (Enhancers vs Silencers) ---
ax1.plot(x_human, human_sr, label="Human MALAT1", color="blue", alpha=0.7)
ax1.plot(x_mouse, mouse_sr, label="Mouse MALAT1", color="orange", alpha=0.7)
ax1.set_title("SR Balance Across MALAT1 Transcript")
ax1.set_ylabel("SR Balance Score")
ax1.axhline(0, color="black", linestyle="--", alpha=0.5) # Baseline
ax1.legend()

# --- PLOT 2: Mean Inclusion Activation ---
ax2.plot(x_human, human_incl_mean, label="Human MALAT1", color="blue", alpha=0.7)
ax2.plot(x_mouse, mouse_incl_mean, label="Mouse MALAT1", color="orange", alpha=0.7)
ax2.set_title("Mean Inclusion Activation Across MALAT1 Transcript")
ax2.set_ylabel("Activation Intensity")
ax2.set_xlabel("Approximate Nucleotide Position")
ax2.legend()

plt.tight_layout()
plt.show()
