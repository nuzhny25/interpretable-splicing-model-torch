import json
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
from dataset_preparations.maf_processing import SPECIES

DATA_DIR = os.path.join(ROOT, "data/multiz100")
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "../creating_alignment/alignment_mapping.json"
)
SEQ_CONS_PATH = os.path.join(
    os.path.dirname(__file__), "../sequence_conservation/sequence_conservation.json"
)
WINDOW_SIZE = 70
STEP_SIZE = 10


def load_aligned_tracks(data, mapping):
    sr, incl, excl = {}, {}, {}
    for idx, name in enumerate(SPECIES):
        if f"{name}_sr" not in data:
            continue
        vals_sr = data[f"{name}_sr"]
        vals_incl = data[f"{name}_incl_mean"]
        vals_excl = data[f"{name}_skip_mean"]
        nuc_map = mapping[idx]

        aligned_positions, sr_vals, incl_vals, excl_vals = [], [], [], []
        for i in range(len(vals_sr)):
            center_nuc = len(nuc_map) - 1 - (i * STEP_SIZE + WINDOW_SIZE // 2)
            if center_nuc >= 0:
                aligned_positions.append(nuc_map[center_nuc])
                sr_vals.append(vals_sr[i])
                incl_vals.append(vals_incl[i])
                excl_vals.append(vals_excl[i])

        if aligned_positions:
            # reverse so positions run low→high (5'→3' on the x-axis)
            pos = np.array(aligned_positions[::-1])
            sr[name] = (pos, np.array(sr_vals[::-1]))
            incl[name] = (pos, np.array(incl_vals[::-1]))
            excl[name] = (pos, np.array(excl_vals[::-1]))

    return sr, incl, excl


def break_at_gaps(pos, vals, gap_factor=5):
    diffs = np.diff(pos)
    threshold = gap_factor * np.median(diffs)
    new_pos, new_vals = [pos[0]], [vals[0]]
    for i in range(1, len(pos)):
        if diffs[i - 1] > threshold:
            new_pos.append(np.nan)
            new_vals.append(np.nan)
        new_pos.append(pos[i])
        new_vals.append(vals[i])
    return np.array(new_pos, dtype=float), np.array(new_vals, dtype=float)


# --- Load data ---
with open(MAPPING_PATH) as f:
    mapping = json.load(f)

with open(SEQ_CONS_PATH) as f:
    seq_cons_data = json.load(f)
seq_cons_pos = np.array(seq_cons_data["positions"])
seq_cons_vals = np.array(seq_cons_data["conservation"])

data = np.load(os.path.join(DATA_DIR, "reversed_embeddings.npz"))
sr_tracks, incl_tracks, excl_tracks = load_aligned_tracks(data, mapping)

colors = cm.tab10(np.linspace(0, 0.9, len(sr_tracks)))
species_colors = dict(zip(sr_tracks.keys(), colors))

# --- Figure layout ---
fig, (ax_sr, ax_incl, ax_excl, ax_seq_cons) = plt.subplots(
    4,
    1,
    figsize=(16, 18),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 3, 3, 1]},
)

ax_sr.margins(x=0)

# --- SR Balance ---
for name, (pos, vals) in sr_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_sr.plot(
        p, v, label=name.capitalize(), color=species_colors[name], lw=1, alpha=0.7
    )
ax_sr.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr.set_title("Aligned SR Balance Across MALAT1 Transcript (Reversed)")
ax_sr.set_ylabel("SR Balance Score")
ax_sr.legend(loc="upper right", fontsize=8)

# --- Inclusion Activation ---
for name, (pos, vals) in incl_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_incl.plot(
        p, v, label=name.capitalize(), color=species_colors[name], lw=1, alpha=0.7
    )
ax_incl.set_title(
    "Aligned Mean Inclusion Activation Across MALAT1 Transcript (Reversed)"
)
ax_incl.set_ylabel("Activation Intensity")
ax_incl.legend(loc="upper right", fontsize=8)

# --- Exclusion Activation ---
for name, (pos, vals) in excl_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_excl.plot(
        p, v, label=name.capitalize(), color=species_colors[name], lw=1, alpha=0.7
    )
ax_excl.set_title(
    "Aligned Mean Exclusion Activation Across MALAT1 Transcript (Reversed)"
)
ax_excl.set_ylabel("Activation Intensity")
ax_excl.legend(loc="upper right", fontsize=8)

# --- Sequence Conservation ---
seq_cons_p, seq_cons_v = break_at_gaps(seq_cons_pos, seq_cons_vals)
ax_seq_cons.fill_between(seq_cons_p, seq_cons_v, alpha=0.75, color="seagreen")
ax_seq_cons.set_ylim(0, 100)
ax_seq_cons.set_ylabel("Sequence\nConservation (%)", fontsize=8)
ax_seq_cons.set_yticks([0, 50, 100])
ax_seq_cons.set_xlabel("Aligned Nucleotide Position")
ax_seq_cons.xaxis.set_major_locator(ticker.MultipleLocator(1000))
ax_seq_cons.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax_seq_cons.tick_params(axis="x", which="minor", length=3)
ax_seq_cons.tick_params(axis="x", which="major", length=6)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "malat1_aligned_multispecies_reversed.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
