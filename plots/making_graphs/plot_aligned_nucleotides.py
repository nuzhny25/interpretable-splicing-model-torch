import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import sys, os

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
from dataset_preparations.maf_processing import SPECIES

DATA_DIR = os.path.join(ROOT, "data/multiz100")
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "../creating_alignment/alignment_mapping.json"
)
WINDOW_SIZE = 70
STEP_SIZE = 10
CONSERVATION_GRID = 1000


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
            center_nuc = i * STEP_SIZE + WINDOW_SIZE // 2
            if center_nuc < len(nuc_map):
                aligned_positions.append(nuc_map[center_nuc])
                sr_vals.append(vals_sr[i])
                incl_vals.append(vals_incl[i])
                excl_vals.append(vals_excl[i])

        if aligned_positions:
            pos = np.array(aligned_positions)
            sr[name] = (pos, np.array(sr_vals))
            incl[name] = (pos, np.array(incl_vals))
            excl[name] = (pos, np.array(excl_vals))

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


def conservation_score(species_dict, grid):
    """1 - normalised std across species; high = conserved."""
    interp_vals = [np.interp(grid, pos, vals) for pos, vals in species_dict.values()]
    std = np.nanstd(np.array(interp_vals), axis=0)
    std_norm = (std - std.min()) / (std.max() - std.min() + 1e-8)
    return 1.0 - std_norm


with open(MAPPING_PATH) as f:
    mapping = json.load(f)

data = np.load(os.path.join(DATA_DIR, "embeddings.npz"))
sr_tracks, incl_tracks, excl_tracks = load_aligned_tracks(data, mapping)

ref_pos = sr_tracks.get("human", next(iter(sr_tracks.values())))[0]
grid = np.linspace(ref_pos.min(), ref_pos.max(), CONSERVATION_GRID)
sr_cons = conservation_score(sr_tracks, grid)
incl_cons = conservation_score(incl_tracks, grid)
excl_cons = conservation_score(excl_tracks, grid)

colors = cm.tab10(np.linspace(0, 0.9, len(sr_tracks)))
species_colors = dict(zip(sr_tracks.keys(), colors))

fig, axes = plt.subplots(
    6,
    1,
    figsize=(16, 20),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 3, 1, 3, 1]},
)
ax_sr, ax_sr_cons, ax_incl, ax_incl_cons, ax_excl, ax_excl_cons = axes

# --- SR Balance ---
for name, (pos, vals) in sr_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_sr.plot(
        p,
        v,
        label=name.capitalize(),
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_sr.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr.set_title("Aligned SR Balance Across MALAT1 Transcript")
ax_sr.set_ylabel("SR Balance Score")
ax_sr.legend(loc="upper right", fontsize=8)

# --- SR Conservation ---
ax_sr_cons.fill_between(grid, sr_cons, alpha=0.75, color="steelblue")
ax_sr_cons.set_ylim(0, 1)
ax_sr_cons.set_ylabel("Conservation\n(SR)", fontsize=8)
ax_sr_cons.set_yticks([0, 0.5, 1])

# --- Inclusion Activation ---
for name, (pos, vals) in incl_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_incl.plot(
        p,
        v,
        label=name.capitalize(),
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_incl.set_title("Aligned Mean Inclusion Activation Across MALAT1 Transcript")
ax_incl.set_ylabel("Activation Intensity")
ax_incl.legend(loc="upper right", fontsize=8)

# --- Inclusion Conservation ---
ax_incl_cons.fill_between(grid, incl_cons, alpha=0.75, color="mediumpurple")
ax_incl_cons.set_ylim(0, 1)
ax_incl_cons.set_ylabel("Conservation\n(Inclusion)", fontsize=8)
ax_incl_cons.set_yticks([0, 0.5, 1])

# --- Exclusion Activation ---
for name, (pos, vals) in excl_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_excl.plot(
        p,
        v,
        label=name.capitalize(),
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_excl.set_title("Aligned Mean Exclusion Activation Across MALAT1 Transcript")
ax_excl.set_ylabel("Activation Intensity")
ax_excl.legend(loc="upper right", fontsize=8)

# --- Exclusion Conservation ---
ax_excl_cons.fill_between(grid, excl_cons, alpha=0.75, color="darkorange")
ax_excl_cons.set_ylim(0, 1)
ax_excl_cons.set_ylabel("Conservation\n(Exclusion)", fontsize=8)
ax_excl_cons.set_yticks([0, 0.5, 1])
ax_excl_cons.set_xlabel("Aligned Nucleotide Position")

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "malat1_aligned_multispecies.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
