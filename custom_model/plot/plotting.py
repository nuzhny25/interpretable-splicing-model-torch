import json
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "../..")
CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
# Ahead of ROOT: "custom_model" names both ROOT/custom_model/ (a namespace
# portion) and custom_model/custom_model.py (the module we want). The regular
# module only wins if its directory comes first on sys.path.
sys.path.insert(0, CUSTOM_DIR)
from custom_model import PNASModel
from custom_utils import str_to_vector
from dataset_preparations.maf_processing import SPECIES

MATRIX_PATH = os.path.join(ROOT, "data/multiz100/alignment_matrix.npy")
WEIGHTS_PATH = os.path.join(CUSTOM_DIR, "weights/real_weights_aad.pt")
SEQ_CONS_PATH = os.path.join(
    ROOT, "plots/sequence_conservation/sequence_conservation.json"
)
# Mirrors custom_real_train.py.
MIN_SPECIES = 5


def compute_sr_tracks(matrix, model):
    """Run the custom model on each species and map onto MSA coordinates.

    Each species' gap-removed sequence is scored by compute_sr_profile, giving
    one SR value per 6-nt sliding window. Window k is placed at the aligned
    column of its *first* nucleotide, matching how custom_real_train.py
    scatters the profile for the SD loss.
    """
    raw = {}

    for idx, name in enumerate(SPECIES):
        row = matrix[:, idx]
        # Gaps are "-" and "N"; lowercase a/c/g/t are soft-masked nucleotides.
        is_nuc = (row != "-") & (row != "N")
        cols = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()

        x_seq = torch.tensor(str_to_vector(seq), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            sr = model.compute_sr_profile(x_seq).squeeze(0)

        # Trailing nucleotides that never start a full window have no value.
        window_cols = cols[: sr.shape[0]]
        raw[name] = (window_cols, sr.numpy())

    return raw


def cross_species_sd(species_dict, n_aligned):
    """Per-column SD of SR balance across species, over MSA coordinates.

    Mirrors the training loss: species are scattered into a (n_species,
    n_aligned) matrix, and the SD (ddof=1) is taken down each column over the
    species actually present there. Columns with fewer than MIN_SPECIES
    present are NaN so they render as gaps rather than fabricated values.
    """
    values = np.zeros((len(species_dict), n_aligned))
    mask = np.zeros((len(species_dict), n_aligned), dtype=bool)
    for i, (cols, vals) in enumerate(species_dict.values()):
        values[i, cols] = vals
        mask[i, cols] = True

    count = mask.sum(axis=0)
    col_mean = (values * mask).sum(axis=0) / np.clip(count, 1, None)
    col_var = (((values - col_mean) ** 2) * mask).sum(axis=0) / np.clip(
        count - 1, 1, None
    )
    sd = np.sqrt(col_var)
    sd[count < MIN_SPECIES] = np.nan
    return np.arange(n_aligned), sd


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
with open(SEQ_CONS_PATH) as f:
    seq_cons_data = json.load(f)
seq_cons_pos = np.array(seq_cons_data["positions"])
seq_cons_vals = np.array(seq_cons_data["conservation"])

matrix = np.load(MATRIX_PATH)
n_aligned = matrix.shape[0]

model = PNASModel()
ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
# Checkpoints saved before metadata stamping are a bare state_dict.
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

sr_tracks = compute_sr_tracks(matrix, model)
sd_pos, sd_vals = cross_species_sd(sr_tracks, n_aligned)

colors = cm.tab10(np.linspace(0, 0.9, len(sr_tracks)))
species_colors = dict(zip(sr_tracks.keys(), colors))


# --- Figure layout ---
# Stretch the figure horizontally so the ~15k aligned positions aren't crushed
# together: aim for roughly 300 nucleotides per inch (never narrower than 16").
fig_width = max(16, n_aligned / 300)
fig, (ax_sr_raw, ax_sr_sd, ax_seq_cons) = plt.subplots(
    3,
    1,
    figsize=(fig_width, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1]},
)

ax_sr_raw.margins(x=0)

# --- SR Balance ---
for name, (pos, vals) in sr_tracks.items():
    p, v = break_at_gaps(pos, vals)
    ax_sr_raw.plot(
        p,
        v,
        color=species_colors[name],
        lw=1,
        alpha=0.7,
    )
ax_sr_raw.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
ax_sr_raw.set_title("Custom Model (Real Data) SR Balance Across MALAT1")
ax_sr_raw.set_ylabel("SR Balance Score")

# --- SR Balance cross-species SD ---
ax_sr_sd.fill_between(sd_pos, sd_vals, alpha=0.5, color="steelblue")
ax_sr_sd.set_title("Cross-Species SD of SR Balance")
ax_sr_sd.set_ylabel("SR SD\n(across species)", fontsize=8)
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
    os.path.join(os.path.dirname(__file__), "malat1_custom_real_sr.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
