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
MAF_PATH = os.path.join(DATA_DIR, "MALAT1_orthologues_multiz100.maf")
HUMAN_MAF_ID = SPECIES["human"]
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


def cross_species_stats(species_dict, step=STEP_SIZE):
    """Per-position median, IQR, and SD of SR balance across species.

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
    valid = np.sum(~np.isnan(stacked), axis=0) >= 10

    med = np.nanmedian(stacked, axis=0)
    q25 = np.nanpercentile(stacked, 25, axis=0)
    q75 = np.nanpercentile(stacked, 75, axis=0)
    sd = np.nanstd(stacked, axis=0, ddof=1)
    for arr in (med, q25, q75, sd):
        arr[~valid] = np.nan
    return grid, med, q25, q75, sd


def plot_sr_panel(ax, species_dict, species_colors, title):
    """Draw per-species SR traces with a median + IQR consensus overlay.

    Individual traces are thin and translucent so the band of variation reads
    as texture rather than a tangle. The median line and IQR ribbon carry the
    consensus signal.
    """
    for name, (pos, vals) in species_dict.items():
        p, v = break_at_gaps(pos, vals)
        ax.plot(p, v, color=species_colors[name], lw=0.4, alpha=0.35)

    grid, med, _, _, _ = cross_species_stats(species_dict)
    ax.plot(grid, med, color="black", lw=1.4, label="Median")

    ax.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_title(f"{title} (n={len(species_dict)} species)")
    ax.set_ylabel("SR Balance Score")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


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


def parse_maf_block_scores(maf_path, human_id=HUMAN_MAF_ID):
    """Read raw block 'a score' values and the human-base length of each block.

    Streams the MAF line by line. Each block starts with `a score=<float>`; the
    human reference line `s <human_id> ...` carries the aligned sequence whose
    non-gap, non-N characters count the human bases in that block (matching how
    alignment_mapping was built). Returns parallel arrays of scores and human
    base lengths, in file order.
    """
    scores, h_lens = [], []
    pending_score = None
    with open(maf_path) as f:
        for line in f:
            if line.startswith("a score="):
                pending_score = float(line.strip().split("score=")[1])
            elif line.startswith("s ") and pending_score is not None:
                fields = line.split()
                if fields[1] == human_id:
                    seq = fields[6]
                    h_lens.append(sum(c not in "-N" for c in seq))
                    scores.append(pending_score)
                    pending_score = None
    return np.array(scores), np.array(h_lens)


# --- Load data ---
with open(MAPPING_PATH) as f:
    mapping = json.load(f)
human_map = mapping[0]

data = np.load(os.path.join(DATA_DIR, "embeddings.npz"))
sr_tracks, _, _ = load_aligned_tracks(data, mapping)
sd_pos, _, _, _, sd_vals = cross_species_stats(sr_tracks)

# Sequential colormap scales to arbitrary species counts; tab10 wraps at 10.
# Use SPECIES order (typically phylogenetic) so neighboring species get
# neighboring colors, making the band of traces read coherently.
ordered_names = [n for n in SPECIES if n in sr_tracks]
colors = cm.turbo(np.linspace(0.05, 0.95, len(ordered_names)))
species_colors = dict(zip(ordered_names, colors))

# --- MAF block scores mapped to aligned columns ---
maf_scores, maf_h_lens = parse_maf_block_scores(MAF_PATH)
h_starts = np.concatenate([[0], np.cumsum(maf_h_lens)[:-1]])
maf_bars = []  # (x_left, x_right, score) per block
for score, h_start, h_len in zip(maf_scores, h_starts, maf_h_lens):
    if h_len == 0:
        continue
    x_left = human_map[h_start]
    x_right = human_map[h_start + h_len - 1]
    maf_bars.append((x_left, x_right, score))

# --- Figure layout ---
fig, (ax_sr, ax_sd, ax_maf) = plt.subplots(
    3,
    1,
    figsize=(18, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1]},
)

ax_sr.margins(x=0)
plot_sr_panel(
    ax_sr,
    sr_tracks,
    species_colors,
    "Aligned SR Balance Across MALAT1 Transcript (Forward)",
)

# --- SR Balance cross-species SD ---
ax_sd.fill_between(sd_pos, sd_vals, alpha=0.5, color="steelblue")
ax_sd.set_title("Cross-Species SD of SR Balance")
ax_sd.set_ylabel("SR SD\n(across species)", fontsize=8)
ax_sd.margins(x=0)

# --- MAF block alignment score (raw) ---
for x_left, x_right, score in maf_bars:
    ax_maf.fill_between([x_left, x_right], [score, score], alpha=0.7, color="mediumpurple")
ax_maf.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_maf.set_title("MALAT1 Conservation — multiz100 Alignment Score (raw)")
ax_maf.set_ylabel("multiz\nAlign Score", fontsize=8)
ax_maf.set_xlabel("Aligned Nucleotide Position")
ax_maf.margins(x=0)
ax_maf.xaxis.set_major_locator(ticker.MultipleLocator(1000))
ax_maf.xaxis.set_minor_locator(ticker.MultipleLocator(100))
ax_maf.tick_params(axis="x", which="minor", length=3)
ax_maf.tick_params(axis="x", which="major", length=6)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "malat1_sr_sd_maf_conservation.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
