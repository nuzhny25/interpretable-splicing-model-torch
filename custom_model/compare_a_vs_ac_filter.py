"""Compare the trained "A-only" model against an "A+C" variant on the SD loss.

Motivation: the synthetic training plants a balanced mix of conserved poly-A and
poly-C blocks, yet the trained checkpoint (``weights/custom_model.pt``) appears to
capture only the all-A motif. This script tests whether the A-only solution
genuinely achieves a lower loss than one that also carries a dedicated C filter.

Two models are built from the same trained weights:
  * Model A ("A-only")  — loaded as-is from weights/custom_model.pt.
  * Model AC ("A + C")  — a copy of Model A where inclusion filter #11 is
    overwritten with a copy of the all-A inclusion filter #10, with its A and C
    rows swapped so it becomes an all-C detector (bias copied too).

Both models are run on the same synthetic matrix. We report the full training
loss (Gaussian-smoothed, normalized cross-species SD + activation L1) for each,
plot per-position cross-species SD, print the average SD, and plot the affected
filters as heat maps so the row-swap can be visually verified.

Run:
    python compare_a_vs_ac_filter.py                    # blocks conserved in all species
    python compare_a_vs_ac_filter.py --matrix imperfect # blocks conserved in 9/10 species
"""

import argparse
import copy
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from custom_model import PNASModel

# Loss/config constants (identical in custom_synthetic_train and imperfect_train). The
# matrix builder itself is imported per-run in main() based on the --matrix flag.
from custom_synthetic_train import (
    L1_LAMBDA,
    MOTIF,
    N_SPECIES,
    NUCLEOTIDES,
    NUM_BLOCKS,
    SEED,
    SEQ_LEN,
    SMOOTH_SIGMA,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)

# Nucleotide channel indices (ACGT order); A/C are the rows we swap.
A_ROW = NUCLEOTIDES.index("A")  # 0
C_ROW = NUCLEOTIDES.index("C")  # 1


def full_loss(model, x, device):
    """Reproduce the exact custom_synthetic_train.py loss for a model on ``x``.

    Returns a dict with the total loss and its components, plus the per-position
    smoothed cross-species SD (``col_std``) as a numpy array for plotting.
    """
    with torch.no_grad():
        # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
        # both (n_species, L-5) and >= 0; used by the activation L1 penalty.
        sr, a_incl, a_skip = model.compute_sr_profile(x, return_activations=True)
        raw_col_std = sr.std(dim=0)  # unsmoothed cross-species SD per column

        # Gaussian-smooth each row's SR track along position (reflect-padded so the
        # conserved block at position 0 isn't damped by zero edges), matching the
        # training pipeline.
        if SMOOTH_SIGMA:
            smooth_radius = math.ceil(3 * SMOOTH_SIGMA)
            offsets = torch.arange(
                -smooth_radius, smooth_radius + 1, device=device, dtype=torch.float32
            )
            gauss_kernel = torch.exp(-(offsets**2) / (2 * SMOOTH_SIGMA**2))
            gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)
            sr = F.conv1d(
                F.pad(sr.unsqueeze(1), (smooth_radius, smooth_radius), mode="reflect"),
                gauss_kernel,
            ).squeeze(1)

        # Numerator: cross-species SD per column (across the n_species rows).
        col_std = sr.std(dim=0)  # (num_windows,)
        # Denominator: within-species dynamic range, averaged over rows (scale-free).
        within_species_scale = sr.std(dim=1).mean()

        sd_loss = col_std.mean() / within_species_scale
        l1_penalty = L1_LAMBDA * (a_incl + a_skip).mean()
        loss = sd_loss + l1_penalty

    return {
        "loss": loss.item(),
        "sd_loss": sd_loss.item(),
        "l1_penalty": l1_penalty.item(),
        "raw_mean_col_sd": raw_col_std.mean().item(),
        "smoothed_mean_col_sd": col_std.mean().item(),
        "within_species_scale": within_species_scale.item(),
        "col_std": col_std.detach().cpu().numpy(),
    }


def log_breakdown(name, res):
    logger.info(
        f"{name:>18s} | loss = {res['loss']:.6f} "
        f"| sd loss = {res['sd_loss']:.6f} "
        f"| l1 = {res['l1_penalty']:.6f} "
        f"| raw mean col SD = {res['raw_mean_col_sd']:.6f} "
        f"| smoothed mean col SD = {res['smoothed_mean_col_sd']:.6f} "
        f"| within-species scale = {res['within_species_scale']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=["synthetic", "imperfect"],
        default="synthetic",
        help="Matrix construction: 'synthetic' = each conserved block written to all "
        "species (custom_synthetic_train); 'imperfect' = each block dropped in one random "
        "species (9/10 conserved, imperfect_train). Default: synthetic.",
    )
    args = parser.parse_args()

    # Select the matrix builder. The loss/config constants are identical across both
    # modules, so only the construction of x changes.
    if args.matrix == "imperfect":
        from imperfect_train import make_synthetic_matrix
    else:
        from custom_synthetic_train import make_synthetic_matrix

    torch.manual_seed(SEED)
    device = torch.device("cpu")  # matches the trained checkpoint; keeps matrix reproducible

    # ── Synthetic matrix (same construction / seed as the chosen training script) ──
    x = make_synthetic_matrix(N_SPECIES, SEQ_LEN, MOTIF, NUM_BLOCKS, device)
    logger.info(
        f"Synthetic matrix [{args.matrix}]: {tuple(x.shape)} on {device} — "
        f"{NUM_BLOCKS} conserved blocks of length {len(MOTIF)}, balanced poly-A / poly-C split"
    )

    # ── Model A: the trained "A-only" model, loaded as-is ──
    weights_path = os.path.join(_HERE, "weights", "custom_model.pt")
    model_a = PNASModel().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_a.load_state_dict(state_dict)
    model_a.eval()
    logger.info(f"Loaded Model A (A-only) from {weights_path}")

    # ── Model AC: copy of Model A with incl filter #11 = all-C copy of incl #10 ──
    model_ac = copy.deepcopy(model_a)
    with torch.no_grad():
        w10 = model_ac.conv_incl.weight[10].clone()  # (4, 6): rows A/C/G/T
        # Swap the A and C rows so the all-A detector becomes an all-C detector.
        row_perm = list(range(w10.shape[0]))
        row_perm[A_ROW], row_perm[C_ROW] = row_perm[C_ROW], row_perm[A_ROW]
        w10_swapped = w10[row_perm, :]
        model_ac.conv_incl.weight[11] = w10_swapped
        model_ac.conv_incl.bias[11] = model_ac.conv_incl.bias[10]
    model_ac.eval()

    # Programmatic swap check (in addition to the filter plot below).
    wa = model_a.conv_incl.weight.detach()
    wac = model_ac.conv_incl.weight.detach()
    swap_ok = (
        torch.equal(wac[10], wa[10])  # #10 unchanged
        and torch.equal(wac[11, C_ROW], wa[10, A_ROW])  # AC #11 C-row == A #10 A-row
        and torch.equal(wac[11, A_ROW], wa[10, C_ROW])  # AC #11 A-row == A #10 C-row
        and torch.equal(wac[11, 2:], wa[10, 2:])  # G/T rows unchanged
        and torch.equal(
            model_ac.conv_incl.bias.detach()[11], model_a.conv_incl.bias.detach()[10]
        )
    )
    logger.info(f"Built Model AC (A+C): incl #11 <- A/C-swapped copy of incl #10 (swap_ok={swap_ok})")

    # ── Loss comparison ──
    res_a = full_loss(model_a, x, device)
    res_ac = full_loss(model_ac, x, device)

    logger.info("── Loss breakdown ──")
    log_breakdown("Model A (A-only)", res_a)
    log_breakdown("Model AC (A+C)", res_ac)

    d_loss = res_ac["loss"] - res_a["loss"]
    lower = "Model A (A-only)" if res_a["loss"] < res_ac["loss"] else "Model AC (A+C)"
    logger.info(f"── Verdict [{args.matrix} matrix] ──")
    logger.info(
        f"Lower FULL loss: {lower}  "
        f"(A-only={res_a['loss']:.6f}, A+C={res_ac['loss']:.6f}, "
        f"A+C - A-only = {d_loss:+.6f})"
    )
    logger.info(
        f"  sd_loss:  A-only={res_a['sd_loss']:.6f}, A+C={res_ac['sd_loss']:.6f}, "
        f"diff={res_ac['sd_loss'] - res_a['sd_loss']:+.6f}"
    )
    logger.info(
        f"  avg cross-species SD (smoothed): A-only={res_a['smoothed_mean_col_sd']:.6f}, "
        f"A+C={res_ac['smoothed_mean_col_sd']:.6f}"
    )

    # ── Plot 1: per-position cross-species SD, both models overlaid ──
    sd_path = os.path.join(_HERE, f"a_vs_ac_position_sd_{args.matrix}.png")
    fig, ax = plt.subplots(figsize=(12, 4))
    positions = range(len(res_a["col_std"]))
    ax.plot(
        positions,
        res_a["col_std"],
        lw=0.8,
        color="tab:blue",
        label=f"Model A (A-only)  avg SD={res_a['smoothed_mean_col_sd']:.4f}",
    )
    ax.plot(
        positions,
        res_ac["col_std"],
        lw=0.8,
        color="tab:red",
        alpha=0.8,
        label=f"Model AC (A+C)  avg SD={res_ac['smoothed_mean_col_sd']:.4f}",
    )
    ax.set_xlabel("window position")
    ax.set_ylabel("cross-species SD (smoothed)")
    ax.set_title(f"Per-position cross-species SD: A-only vs A+C  [{args.matrix} matrix]")
    ax.legend()
    fig.savefig(sd_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote per-position SD plot to {sd_path}")

    # ── Plot 2: filter verification (conv_incl #10/#11 for both models) ──
    filt_path = os.path.join(_HERE, f"a_vs_ac_filters_{args.matrix}.png")
    panels = [
        (model_a, 10, "Model A  INCL #10  (all-A)"),
        (model_a, 11, "Model A  INCL #11  (original, trained)"),
        (model_ac, 10, "Model AC  INCL #10  (unchanged)"),
        (model_ac, 11, "Model AC  INCL #11  (A/C-swap of #10 -> all-C)"),
    ]
    shown = torch.stack(
        [m.conv_incl.weight.detach().cpu()[f] for m, f, _ in panels]
    )
    vmax = shown.abs().max().item()

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), gridspec_kw={"hspace": 0.6, "wspace": 0.3})
    im = None
    for ax, (model, f, title) in zip(axes.ravel(), panels):
        w = model.conv_incl.weight.detach().cpu()[f]
        bias = model.conv_incl.bias.detach().cpu()[f].item()
        im = ax.imshow(w, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}\nb={bias:+.2f}", fontsize=9)
        ax.set_yticks(range(len(NUCLEOTIDES)))
        ax.set_yticklabels(NUCLEOTIDES, fontsize=8)
        ax.set_xticks(range(w.shape[1]))
        ax.set_xticklabels(range(1, w.shape[1] + 1), fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.7, label="filter weight")
    fig.suptitle(
        "conv_incl filters #10/#11 (rows A/C/G/T, cols = kernel position)\n"
        "AC #11 should be A #10 with the A and C rows swapped",
        fontsize=11,
        y=1.02,
    )
    fig.savefig(filt_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote filter verification plot to {filt_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
