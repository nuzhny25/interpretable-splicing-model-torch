import matplotlib.pyplot as plt
import numpy as np


data = np.load("data/embeddings.npz")

human_sr = data["human_sr"]
mouse_sr = data["mouse_sr"]
human_incl_mean = data["human_incl_mean"]
mouse_incl_mean = data["mouse_incl_mean"]

step_size = 10


# Create x-axis positions for the plots (Position in the transcript)
# Note: Human and Mouse MALAT1 might be slightly different lengths!
x_human = np.arange(len(human_sr)) * step_size
x_mouse = np.arange(len(mouse_sr)) * step_size

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- PLOT 1: SR Balance (Enhancers vs Silencers) ---
ax1.plot(x_human, human_sr, label="Human MALAT1", color="blue", alpha=0.7)
ax1.plot(x_mouse, mouse_sr, label="Mouse MALAT1", color="orange", alpha=0.7)
ax1.set_title("Unaligned SR Balance Across MALAT1 Transcript")
ax1.set_ylabel("SR Balance Score")
ax1.axhline(0, color="black", linestyle="--", alpha=0.5)  # Baseline
ax1.legend()

# --- PLOT 2: Mean Inclusion Activation ---
ax2.plot(x_human, human_incl_mean, label="Human MALAT1", color="blue", alpha=0.7)
ax2.plot(x_mouse, mouse_incl_mean, label="Mouse MALAT1", color="orange", alpha=0.7)
ax2.set_title("Unaligned Mean Inclusion Activation Across MALAT1 Transcript")
ax2.set_ylabel("Activation Intensity")
ax2.set_xlabel("Approximate Nucleotide Position")
ax2.legend()

plt.tight_layout()
plt.show()
