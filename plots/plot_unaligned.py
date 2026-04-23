import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset_preparations.maf_processing import SPECIES

DATA_DIR = "../data/multiz100"
WINDOW_SIZE = 70
STEP_SIZE = 10
CONSERVATION_GRID = 1000


def load_species_tracks(data):
    sr, incl = {}, {}
    for name in SPECIES:
        if f"{name}_sr" not in data:
            continue
        vals_sr = data[f"{name}_sr"]
        vals_incl = data[f"{name}_incl_mean"]
        n = len(vals_sr)
        # Center of each window: more precise position than window start
        centers = np.arange(n) * STEP_SIZE + WINDOW_SIZE // 2
        frac = centers / (centers[-1] + WINDOW_SIZE // 2)
        sr[name] = (frac, vals_sr)
        incl[name] = (frac, vals_incl)
    return sr, incl


def conservation_score(species_dict, grid):
    """1 - normalised std across species; high = conserved."""
    interp_vals = []
    for frac, vals in species_dict.values():
        interp_vals.append(np.interp(grid, frac, vals))
    stack = np.array(interp_vals)
    std = np.nanstd(stack, axis=0)
    std_norm = (std - np.nanmin(std)) / (np.nanmax(std) - np.nanmin(std) + 1e-8)
    return 1.0 - std_norm


data = np.load(f"{DATA_DIR}/embeddings.npz")
sr_tracks, incl_tracks = load_species_tracks(data)

grid = np.linspace(0, 1, CONSERVATION_GRID)
sr_cons = conservation_score(sr_tracks, grid)
incl_cons = conservation_score(incl_tracks, grid)

colors = cm.tab10(np.linspace(0, 0.9, len(sr_tracks)))
species_colors = dict(zip(sr_tracks.keys(), colors))

fig, axes = plt.subplots(
    4,
    1,
    figsize=(16, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 3, 1]},
)
ax_sr, ax_sr_cons, ax_incl, ax_incl_cons = axes

# --- SR Balance (scatter) ---
for name, (frac, vals) in sr_tracks.items():
    ax_sr.scatter(
        frac,
        vals,
        label=name.capitalize(),
        color=species_colors[name],
        s=4,
        alpha=0.65,
        linewidths=0,
    )
ax_sr.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr.set_title("Unaligned SR Balance Across MALAT1 Transcript")
ax_sr.set_ylabel("SR Balance Score")
ax_sr.legend(markerscale=4, loc="upper right", fontsize=8)

# --- SR Conservation ---
ax_sr_cons.fill_between(grid, sr_cons, alpha=0.75, color="steelblue")
ax_sr_cons.set_ylim(0, 1)
ax_sr_cons.set_ylabel("Conservation\n(SR)", fontsize=8)
ax_sr_cons.set_yticks([0, 0.5, 1])

# --- Mean Inclusion Activation (scatter) ---
for name, (frac, vals) in incl_tracks.items():
    ax_incl.scatter(
        frac,
        vals,
        label=name.capitalize(),
        color=species_colors[name],
        s=4,
        alpha=0.65,
        linewidths=0,
    )
ax_incl.set_title("Unaligned Mean Inclusion Activation Across MALAT1 Transcript")
ax_incl.set_ylabel("Activation Intensity")
ax_incl.legend(markerscale=4, loc="upper right", fontsize=8)

# --- Inclusion Conservation ---
ax_incl_cons.fill_between(grid, incl_cons, alpha=0.75, color="mediumpurple")
ax_incl_cons.set_ylim(0, 1)
ax_incl_cons.set_ylabel("Conservation\n(Inclusion)", fontsize=8)
ax_incl_cons.set_yticks([0, 0.5, 1])
ax_incl_cons.set_xlabel("Fractional Position in Transcript  (5′ → 3′)")

# Tick labels as percentage of transcript length for readability
ax_incl_cons.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
)

plt.tight_layout()
plt.savefig("malat1_unaligned_multispecies.png", dpi=150, bbox_inches="tight")
plt.show()
