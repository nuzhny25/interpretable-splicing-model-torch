"""Compare cross-species SD of forward vs reversed SR balance.

Prints the percentage of aligned positions where forward SD is lower than
reversed SD, considering only positions where both have a valid SD.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, ROOT)
from dataset_preparations.maf_processing import SPECIES

DATA_DIR = os.path.join(ROOT, "data/multiz100")
MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "../creating_alignment/alignment_mapping.json"
)
WINDOW_SIZE = 70
STEP_SIZE = 10


def load_aligned_sr(data, mapping, reversed_orient=False):
    sr = {}
    for idx, name in enumerate(SPECIES):
        if f"{name}_sr" not in data:
            continue
        vals_sr = data[f"{name}_sr"]
        nuc_map = mapping[idx]
        aligned, sr_vals = [], []
        for i in range(len(vals_sr)):
            if reversed_orient:
                center_nuc = len(nuc_map) - 1 - (i * STEP_SIZE + WINDOW_SIZE // 2)
            else:
                center_nuc = i * STEP_SIZE + WINDOW_SIZE // 2
            if 0 <= center_nuc < len(nuc_map):
                aligned.append(nuc_map[center_nuc])
                sr_vals.append(vals_sr[i])
        if aligned:
            order = np.argsort(aligned)
            sr[name] = (np.array(aligned)[order], np.array(sr_vals)[order])
    return sr


def sd_on_grid(species_dict, grid, step=STEP_SIZE):
    """Cross-species SD at each grid column, nearest-neighbor within step/2."""
    tol = step / 2
    stacked = []
    for pos, vals in species_dict.values():
        vals_s = vals.astype(float)
        ins = np.searchsorted(pos, grid)
        left = np.clip(ins - 1, 0, len(pos) - 1)
        right = np.clip(ins, 0, len(pos) - 1)
        dl = np.abs(grid - pos[left])
        dr = np.abs(grid - pos[right])
        use_right = dr < dl
        nearest = np.where(use_right, right, left)
        dist = np.where(use_right, dr, dl)
        col = np.where(dist <= tol, vals_s[nearest], np.nan)
        stacked.append(col)
    stacked = np.array(stacked)
    sd = np.nanstd(stacked, axis=0)
    valid = np.sum(~np.isnan(stacked), axis=0) >= 2
    sd[~valid] = np.nan
    return sd


with open(MAPPING_PATH) as f:
    mapping = json.load(f)

fwd_data = np.load(os.path.join(DATA_DIR, "embeddings.npz"))
rev_data = np.load(os.path.join(DATA_DIR, "reversed_embeddings.npz"))

fwd_tracks = load_aligned_sr(fwd_data, mapping, reversed_orient=False)
rev_tracks = load_aligned_sr(rev_data, mapping, reversed_orient=True)

all_pos = np.concatenate(
    [pos for pos, _ in fwd_tracks.values()] + [pos for pos, _ in rev_tracks.values()]
)
grid = np.arange(all_pos.min(), all_pos.max() + 1, STEP_SIZE)

fwd_sd = sd_on_grid(fwd_tracks, grid)
rev_sd = sd_on_grid(rev_tracks, grid)

both_valid = ~np.isnan(fwd_sd) & ~np.isnan(rev_sd)
n_both = int(both_valid.sum())
n_fwd_lower = int(((fwd_sd < rev_sd) & both_valid).sum())
n_rev_lower = int(((rev_sd < fwd_sd) & both_valid).sum())
n_tie = n_both - n_fwd_lower - n_rev_lower

pct_fwd = 100.0 * n_fwd_lower / n_both if n_both else float("nan")
pct_rev = 100.0 * n_rev_lower / n_both if n_both else float("nan")

print(f"Grid columns total:        {len(grid)}")
print(f"Columns valid in both:     {n_both}")
print(f"Forward SD < Reversed SD:  {n_fwd_lower} ({pct_fwd:.2f}%)")
print(f"Reversed SD < Forward SD:  {n_rev_lower} ({pct_rev:.2f}%)")
print(f"Ties:                      {n_tie}")
print(
    f"Median SD — forward: {np.nanmedian(fwd_sd[both_valid]):.4f}, "
    f"reversed: {np.nanmedian(rev_sd[both_valid]):.4f}"
)
