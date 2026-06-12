"""SR balance across all species: unpermuted original vs N filter permutations.

Builds a single page of stacked panels sharing one x-axis: the top panel is the
unpermuted original, and each panel below is one random ``independent`` filter
permutation (produced by ``permute_filters``, which shuffles the 6 kernel columns
of the sequence convs -- a fresh column order per filter). Every panel overlays
all species' per-window SR balance (inclusion - skipping), aligned to MSA
coordinates, plus a mean line; the unpermuted reference is read straight from
``data/multiz100/embeddings.npz`` (no recompute needed) and repeated as a dashed
line in each permutation panel for comparison.

Aligned positions depend only on each species' sequence (window centers mapped
through the MSA), not on the filter weights, so the x-coordinates are identical
across permutations -- only the SR values move.
"""

import json
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "filter_permutations"))

from model import PNASModel
from filter_permutations import permute_filters
from dataset_preparations.maf_processing import SPECIES

DATA_DIR = os.path.join(ROOT, "data/multiz100")
WEIGHTS_PATH = os.path.join(ROOT, "model_weights.pt")
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "../creating_alignment/alignment_mapping.json"
)
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npz")
OUT_DIR = os.path.dirname(__file__)

SCHEME = "independent"  # "shared" or "independent" (see permute_filters)
N_PERMS = 10
WINDOW_SIZE = 70
STEP_SIZE = 10
SEED = 0
MIN_SPECIES = 10  # min species covering a grid column for a valid mean


def align_sr(vals_sr, species_idx):
    """Map a species' per-window SR balance onto MSA coordinates.

    Window ``i`` is centered at ``i*STEP_SIZE + WINDOW_SIZE//2`` in the species'
    ungapped sequence; ``mapping[species_idx]`` translates that to an aligned
    column. Returns ``(positions, values)`` sorted by position.
    """
    nuc_map = mapping[species_idx]
    aligned, sr_vals = [], []
    for i in range(len(vals_sr)):
        center_nuc = i * STEP_SIZE + WINDOW_SIZE // 2
        if 0 <= center_nuc < len(nuc_map):
            aligned.append(nuc_map[center_nuc])
            sr_vals.append(vals_sr[i])
    aligned = np.array(aligned)
    order = np.argsort(aligned)
    return aligned[order], np.array(sr_vals)[order]


def mean_on_grid(species_dict, grid, step=STEP_SIZE):
    """Cross-species mean SR at each grid column (nearest-neighbor, step/2 tol)."""
    tol = step / 2
    stacked = []
    for pos, vals in species_dict.values():
        vals_s = vals.astype(float)
        ins = np.searchsorted(pos, grid)
        left = np.clip(ins - 1, 0, len(pos) - 1)
        right = np.clip(ins, 0, len(pos) - 1)
        use_right = np.abs(grid - pos[right]) < np.abs(grid - pos[left])
        nearest = np.where(use_right, right, left)
        dist = np.where(use_right, np.abs(grid - pos[right]), np.abs(grid - pos[left]))
        stacked.append(np.where(dist <= tol, vals_s[nearest], np.nan))
    stacked = np.array(stacked)
    avg = np.nanmean(stacked, axis=0)
    avg[np.sum(~np.isnan(stacked), axis=0) < MIN_SPECIES] = np.nan
    return avg


def break_at_gaps(pos, vals, gap_factor=5):
    """Insert NaNs across large coordinate jumps so lines don't bridge gaps."""
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


# --- load mapping, originals, model ---
with open(MAPPING_PATH) as f:
    mapping = json.load(f)

original_data = np.load(EMBEDDINGS_PATH)
species_idx = {name: i for i, name in enumerate(SPECIES)}
names = [n for n in SPECIES if f"{n}_sr" in original_data.files]

state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
# Cache each species' one-hot windows; all share input_length=90.
seq_by_species = {
    n: torch.tensor(
        np.load(os.path.join(DATA_DIR, f"{n}_malat1_chunks.npz"))["seq_oh"],
        dtype=torch.float32,
    )
    for n in names
}
input_length = seq_by_species[names[0]].shape[-1]
model = PNASModel(input_length=input_length)
model.load_state_dict(state_dict)
model.eval()

# Pristine sequence-conv weights to permute from each iteration.
orig_skip = model.conv_skip.weight.detach().clone()
orig_incl = model.conv_incl.weight.detach().clone()
rng = np.random.default_rng(SEED)

# A consistent color per species across all windows.
colors = cm.turbo(np.linspace(0.05, 0.95, len(names)))
species_colors = dict(zip(names, colors))

# --- original (unpermuted) aligned tracks + shared grid + mean reference ---
original_aligned = {n: align_sr(original_data[f"{n}_sr"], species_idx[n]) for n in names}
all_pos = np.concatenate([pos for pos, _ in original_aligned.values()])
grid = np.arange(all_pos.min(), all_pos.max() + 1, STEP_SIZE)
original_mean = mean_on_grid(original_aligned, grid)

# --- single page: original panel on top, one panel per permutation below ---
fig, axes = plt.subplots(
    N_PERMS + 1, 1, figsize=(18, 2.4 * (N_PERMS + 1)), sharex=True
)

# top panel: unpermuted original
ax = axes[0]
for n in names:
    p, v = break_at_gaps(*original_aligned[n])
    ax.plot(p, v, color=species_colors[n], lw=0.4, alpha=0.35)
ax.plot(grid, original_mean, color="black", lw=1.4, label="original mean")
ax.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.6)
ax.set_ylabel("original\nSR balance", fontsize=8)
ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
ax.margins(x=0)

# one panel per permutation
for i in range(N_PERMS):
    permute_filters(model, orig_skip, orig_incl, rng, SCHEME)

    aligned_perm = {}
    for n in names:
        with torch.no_grad():
            sr = model.compute_sr_balance(seq_by_species[n], agg="mean").numpy()
        aligned_perm[n] = align_sr(sr, species_idx[n])

    ax = axes[i + 1]
    for n in names:
        p, v = break_at_gaps(*aligned_perm[n])
        ax.plot(p, v, color=species_colors[n], lw=0.4, alpha=0.35)
    ax.plot(grid, mean_on_grid(aligned_perm, grid), color="black", lw=1.4,
            label="permuted mean")
    ax.plot(grid, original_mean, color="red", lw=1.2, ls="--",
            label="original mean")
    ax.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.6)
    ax.set_ylabel(f"perm {i + 1}\nSR balance", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.margins(x=0)

# share y-limits across panels so amplitudes are comparable
ymin = min(ax.get_ylim()[0] for ax in axes)
ymax = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(ymin, ymax)

axes[-1].set_xlabel("aligned nucleotide position")
fig.suptitle(
    f"MALAT1 SR balance across {len(names)} species: "
    f"original vs {N_PERMS} {SCHEME} filter permutations"
)
fig.tight_layout()

# Restore pristine weights so the model isn't left in a permuted state.
with torch.no_grad():
    model.conv_skip.weight.copy_(orig_skip)
    model.conv_incl.weight.copy_(orig_incl)

out_png = os.path.join(OUT_DIR, "permuted_sr_balance_all.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"saved plot -> {out_png}")
plt.show()
