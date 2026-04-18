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


x_human = np.arange(len(human_sr)) * step_size
x_mouse = np.arange(len(mouse_sr)) * step_size

# BLAST anchors
# Format: [0, Region1_Start, Region1_End, Region2_Start, Region2_End, ..., Max_Length]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

anchors_human = [0, 1146, 1958, 2186, 2975, 3141, 7141, max(x_human)]
anchors_mouse = [0, 2022, 2818, 2951, 3687, 3780, 7661, max(x_mouse)]

# 2. Warp the mouse x-axis coordinates to match the human coordinate space
x_mouse_warped = np.interp(x_mouse, anchors_mouse, anchors_human)

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- PLOT 1: SR Balance (Enhancers vs Silencers) ---
ax1.plot(x_human, human_sr, label="Human MALAT1 (Reference)", color="blue", alpha=0.7)
# Use x_mouse_warped instead of x_mouse!
ax1.plot(
    x_mouse_warped, mouse_sr, label="Mouse MALAT1 (Aligned)", color="orange", alpha=0.7
)
ax1.set_title("Aligned SR Balance Across MALAT1 Transcript")
ax1.set_ylabel("SR Balance Score")
ax1.axhline(0, color="black", linestyle="--", alpha=0.5)

# --- PLOT 2: Mean Inclusion Activation ---
ax2.plot(x_human, human_incl_mean, label="Human MALAT1", color="blue", alpha=0.7)
# Use x_mouse_warped here too!
ax2.plot(
    x_mouse_warped,
    mouse_incl_mean,
    label="Mouse MALAT1 (Aligned)",
    color="orange",
    alpha=0.7,
)
ax2.set_title("Aligned Mean Inclusion Activation Across MALAT1 Transcript")
ax2.set_ylabel("Activation Intensity")
ax2.set_xlabel("Human Transcript Nucleotide Position")

# 4. Draw shaded background regions to highlight the aligned BLAST sections
for i in range(1, len(anchors_human) - 1, 2):
    start = anchors_human[i]
    end = anchors_human[i + 1]
    ax1.axvspan(
        start, end, color="green", alpha=0.1, label="BLAST Alignment" if i == 1 else ""
    )
    ax2.axvspan(start, end, color="green", alpha=0.1)

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()
