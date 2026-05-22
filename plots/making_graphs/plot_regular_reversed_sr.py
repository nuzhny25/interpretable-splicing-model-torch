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


def load_aligned_tracks(data, mapping, reversed_orient=False):
    """Map per-window predictions onto MSA coordinates.

    For reversed-orientation embeddings, prediction i corresponds to a window
    over the reverse-complement of the original sequence, so its center in
    forward (original) ungapped coordinates is mirrored.
    """
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
            if reversed_orient:
                center_nuc = len(nuc_map) - 1 - (i * STEP_SIZE + WINDOW_SIZE // 2)
            else:
                center_nuc = i * STEP_SIZE + WINDOW_SIZE // 2
            if 0 <= center_nuc < len(nuc_map):
                aligned_positions.append(nuc_map[center_nuc])
                sr_vals.append(vals_sr[i])
                incl_vals.append(vals_incl[i])
                excl_vals.append(vals_excl[i])

        if aligned_positions:
            # sort by position so reversed orientation still runs 5'→3' on x
            order = np.argsort(aligned_positions)
            pos = np.array(aligned_positions)[order]
            sr[name] = (pos, np.array(sr_vals)[order])
            incl[name] = (pos, np.array(incl_vals)[order])
            excl[name] = (pos, np.array(excl_vals)[order])

    return sr, incl, excl


def cross_species_sd(species_dict, step=STEP_SIZE):
    """Per-position SD of SR balance across species on a common aligned grid.

    Nearest-neighbor lookup with a step/2 tolerance: a species contributes to
    grid column X only if it has a window center within step/2 MSA columns of
    X. Avoids fabricating values across mapping jumps caused by gaps.
    """
    all_pos = np.concatenate([pos for pos, _ in species_dict.values()])
    grid = np.arange(all_pos.min(), all_pos.max() + 1, step)
    tol = step / 2

    stacked = []
    for pos, vals in species_dict.values():
        order = np.argsort(pos)
        pos_s = pos[order]
        vals_s = vals[order].astype(float)
        ins = np.searchsorted(pos_s, grid)
        left = np.clip(ins - 1, 0, len(pos_s) - 1)
        right = np.clip(ins, 0, len(pos_s) - 1)
        dist_left = np.abs(grid - pos_s[left])
        dist_right = np.abs(grid - pos_s[right])
        use_right = dist_right < dist_left
        nearest_idx = np.where(use_right, right, left)
        nearest_dist = np.where(use_right, dist_right, dist_left)
        col = np.where(nearest_dist <= tol, vals_s[nearest_idx], np.nan)
        stacked.append(col)
    stacked = np.array(stacked)
    sd = np.nanstd(stacked, axis=0)
    valid = np.sum(~np.isnan(stacked), axis=0) >= 2
    sd[~valid] = np.nan
    return grid, sd


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

data = np.load(os.path.join(DATA_DIR, "embeddings.npz"))
sr_tracks, _, _ = load_aligned_tracks(data, mapping)
sd_pos, sd_vals = cross_species_sd(sr_tracks)

colors = cm.tab10(np.linspace(0, 0.9, len(sr_tracks)))
species_colors = dict(zip(sr_tracks.keys(), colors))

# --- Load data (reversed) ---
reversed_data = np.load(os.path.join(DATA_DIR, "reversed_embeddings.npz"))
reversed_sr_tracks, _, _ = load_aligned_tracks(
    reversed_data, mapping, reversed_orient=True
)
reversed_sd_pos, reversed_sd_vals = cross_species_sd(reversed_sr_tracks)


# --- Figure layout ---
fig, (ax_sr_fwd, ax_sr_rev, ax_sr_sd, ax_seq_cons) = plt.subplots(
    4,
    1,
    figsize=(16, 16),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 3, 1, 1]},
)

ax_sr_fwd.margins(x=0)
ax_sr_rev.margins(x=0)

# --- SR Balance (forward) ---
for name, (pos, vals) in sr_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_sr_fwd.plot(
        p,
        v,
        label=name.capitalize(),
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_sr_fwd.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr_fwd.set_title("Aligned SR Balance Across MALAT1 Transcript (Forward)")
ax_sr_fwd.set_ylabel("SR Balance Score")
ax_sr_fwd.legend(loc="upper right", fontsize=8)

# --- SR Balance (reversed) ---
for name, (pos, vals) in reversed_sr_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_sr_rev.plot(
        p,
        v,
        label=name.capitalize(),
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_sr_rev.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr_rev.set_title("Aligned SR Balance Across MALAT1 Transcript (Reversed)")
ax_sr_rev.set_ylabel("SR Balance Score")
ax_sr_rev.legend(loc="upper right", fontsize=8)

# Align y-axes so forward/reversed are visually comparable.
sr_ymin = min(ax_sr_fwd.get_ylim()[0], ax_sr_rev.get_ylim()[0])
sr_ymax = max(ax_sr_fwd.get_ylim()[1], ax_sr_rev.get_ylim()[1])
ax_sr_fwd.set_ylim(sr_ymin, sr_ymax)
ax_sr_rev.set_ylim(sr_ymin, sr_ymax)

# --- SR Balance cross-species SD (forward vs reversed) ---
ax_sr_sd.fill_between(sd_pos, sd_vals, alpha=0.5, color="steelblue", label="Forward")
ax_sr_sd.fill_between(
    reversed_sd_pos, reversed_sd_vals, alpha=0.5, color="darkorange", label="Reversed"
)
ax_sr_sd.set_title("Cross-Species SD of SR Balance")
ax_sr_sd.set_ylabel("SR SD\n(across species)", fontsize=8)
ax_sr_sd.legend(loc="upper right", fontsize=8)
ax_sr_sd.margins(x=0)

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
    os.path.join(os.path.dirname(__file__), "malat1_aligned_regular_reversed_sr.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
