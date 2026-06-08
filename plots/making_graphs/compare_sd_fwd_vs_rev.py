"""Compare cross-species SD of SR balance across orientations.

Prints, for each orientation pair, the percentage of aligned positions
where one series has lower SD than the other, considering only positions
where both have a valid SD. Pairs covered:
  - forward vs reversed
  - complement vs reverse complement
Also prints overall median SD per series for cross-series comparison.
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
    sd = np.nanstd(stacked, axis=0, ddof=1)
    valid = np.sum(~np.isnan(stacked), axis=0) >= 10
    sd[~valid] = np.nan
    return sd


with open(MAPPING_PATH) as f:
    mapping = json.load(f)

fwd_data = np.load(os.path.join(DATA_DIR, "embeddings.npz"))
rev_data = np.load(os.path.join(DATA_DIR, "reversed_embeddings.npz"))
comp_data = np.load(os.path.join(DATA_DIR, "complement_embeddings.npz"))
revcomp_data = np.load(os.path.join(DATA_DIR, "reversed_complement_embeddings.npz"))

fwd_tracks = load_aligned_sr(fwd_data, mapping, reversed_orient=False)
rev_tracks = load_aligned_sr(rev_data, mapping, reversed_orient=True)
comp_tracks = load_aligned_sr(comp_data, mapping, reversed_orient=False)
revcomp_tracks = load_aligned_sr(revcomp_data, mapping, reversed_orient=True)

all_tracks = [fwd_tracks, rev_tracks, comp_tracks, revcomp_tracks]
all_pos = np.concatenate([pos for tracks in all_tracks for pos, _ in tracks.values()])
grid = np.arange(all_pos.min(), all_pos.max() + 1, STEP_SIZE)

fwd_sd = sd_on_grid(fwd_tracks, grid)
rev_sd = sd_on_grid(rev_tracks, grid)
comp_sd = sd_on_grid(comp_tracks, grid)
revcomp_sd = sd_on_grid(revcomp_tracks, grid)


def compare(name_a, sd_a, name_b, sd_b):
    both_valid = ~np.isnan(sd_a) & ~np.isnan(sd_b)
    n_both = int(both_valid.sum())
    n_a_lower = int(((sd_a < sd_b) & both_valid).sum())
    n_b_lower = int(((sd_b < sd_a) & both_valid).sum())
    n_tie = n_both - n_a_lower - n_b_lower
    pct_a = 100.0 * n_a_lower / n_both if n_both else float("nan")
    pct_b = 100.0 * n_b_lower / n_both if n_both else float("nan")

    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"Columns valid in both:        {n_both}")
    print(f"{name_a} SD < {name_b} SD:  {n_a_lower} ({pct_a:.2f}%)")
    print(f"{name_b} SD < {name_a} SD:  {n_b_lower} ({pct_b:.2f}%)")
    print(f"Ties:                         {n_tie}")
    print(
        f"Median SD — {name_a}: {np.nanmedian(sd_a[both_valid]):.4f}, "
        f"{name_b}: {np.nanmedian(sd_b[both_valid]):.4f}"
    )


print(f"Grid columns total:        {len(grid)}")

compare("Forward         ", fwd_sd, "Reversed       ", rev_sd)
compare("Complement      ", comp_sd, "RevComplement  ", revcomp_sd)

print("\n--- Overall median SD per series (valid positions only) ---")
for name, sd in [
    ("Forward         ", fwd_sd),
    ("Reversed        ", rev_sd),
    ("Complement      ", comp_sd),
    ("RevComplement   ", revcomp_sd),
]:
    valid = ~np.isnan(sd)
    print(
        f"{name}  median: {np.nanmedian(sd[valid]):.4f}  "
        f"mean: {np.nanmean(sd[valid]):.4f}  "
        f"n_valid: {int(valid.sum())}"
    )
