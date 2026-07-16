"""Real cross-species training for the normalized SD-minimization loss.

Loads the real MALAT1 multiz alignment matrix (``data/multiz100/alignment_matrix.npy``,
shape ``(n_aligned, n_species)`` of single characters with ``A/C/G/T`` plus
lowercase soft-masked ``a/c/g/t`` and the gap symbols ``-`` / ``N``). Each epoch
trains on a fresh random combination of 10 species drawn from the alignment.

For every sampled species the SR-balance profile is computed by the simplified
20+20-filter model on that species' gap-removed sequence (gaps are ``-`` and
``N``; lowercase letters are nucleotides). Each 6-nt sliding-window value is then
scattered back into alignment coordinates at the aligned column of the window's
*first* nucleotide. Aligned columns that are gaps in a species — and the trailing
nucleotides that never start a full window — are left absent (conceptually NaN).

This produces a ``(10, n_aligned)`` matrix with NaN at gaps; the loss is the
NaN-aware cross-species standard deviation taken down each column, restricted to
columns where at least ``MIN_SPECIES`` of the 10 sampled species are present.

The per-column cross-species SD is divided by the within-species dynamic range
(mean over rows of each row's SD across positions), mirroring the normalization
in filter_permutations/filter_permutations.py. This makes the loss scale-free:
shrinking all activations toward zero shrinks the numerator and denominator
equally, so the trivial "constant SR" collapse no longer lowers the loss.

The NaN matrix is implemented as a ``values`` + ``mask`` pair rather than literal
NaN, so gradients stay finite: masked entries contribute 0 and receive no
gradient, exactly matching ``np.nanstd(..., ddof=1)`` over the non-NaN entries.
"""

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from custom_model import PNASModel
from custom_utils import str_to_vector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Real-alignment setup ─────────────────────────────────────────────────────
MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "multiz100", "alignment_matrix.npy"
)
N_SAMPLE = 10  # number of species sampled per epoch
MIN_SPECIES = 5  # a column counts toward the loss only if >= this many species present
NUM_EPOCHS = 10000
LR = 1e-2
L1_LAMBDA = (
    1e-3  # strength of L1 penalty on the softplus activations (retune by watching logs)
)
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)
SEED = 0

NUCLEOTIDES = ["A", "C", "G", "T"]


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load the real alignment matrix and precompute per-species inputs ──
    # matrix: (n_aligned, n_species) of single characters. Gaps are "-" and "N";
    # everything else (including lowercase soft-masked a/c/g/t) is a nucleotide.
    matrix = np.load(MATRIX_PATH)
    n_aligned, n_total = matrix.shape

    # For each species: the gap-removed one-hot sequence and, for every nucleotide,
    # the aligned column it occupies. species_cols[j][k] is the aligned column of
    # the k-th nucleotide = the first nucleotide of sliding window k.
    species_oh, species_cols = [], []
    for j in range(n_total):
        row = matrix[:, j]
        is_nuc = (row != "-") & (row != "N")
        nuc_aligned = np.nonzero(is_nuc)[0]
        seq = "".join(row[is_nuc]).upper()
        oh = torch.tensor(
            str_to_vector(seq), dtype=torch.float32, device=device
        )  # (4, Lj)
        species_oh.append(oh.unsqueeze(0))  # (1, 4, Lj)
        species_cols.append(torch.tensor(nuc_aligned, dtype=torch.long, device=device))

    logger.info(
        f"Real alignment matrix: {matrix.shape} (aligned positions x species) "
        f"from {MATRIX_PATH} on {device} — sampling {N_SAMPLE} species/epoch, "
        f"column valid when >= {MIN_SPECIES} species present"
    )

    model = PNASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once.
    # Sum-normalized so it's a weighted average (preserves overall scale). It carries
    # no learnable parameters and is applied with reflect padding at the use site so
    # the conserved block at position 0 isn't damped by zero edges. Smoothing denoises
    # single-position jitter so the conserved-motif signal dominates the column SD.
    # It is applied to each species' dense (gap-free) SR track before scattering, so
    # no NaNs ever enter the convolution.
    if SMOOTH_SIGMA:
        smooth_radius = math.ceil(3 * SMOOTH_SIGMA)  # kernel half-width (~3 sigma)
        offsets = torch.arange(
            -smooth_radius, smooth_radius + 1, device=device, dtype=torch.float32
        )
        gauss_kernel = torch.exp(-(offsets**2) / (2 * SMOOTH_SIGMA**2))
        gauss_kernel = (gauss_kernel / gauss_kernel.sum()).view(1, 1, -1)  # (1, 1, K)
        logger.info(
            f"SR-profile smoothing: Gaussian sigma={SMOOTH_SIGMA} nt, "
            f"kernel size={gauss_kernel.shape[-1]}"
        )

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()

        # Fresh random combination of N_SAMPLE species for this epoch.
        sel = torch.randperm(n_total)[:N_SAMPLE]

        # Build the aligned (N_SAMPLE, n_aligned) matrix of SR values. We never put
        # literal NaN into the differentiable tensor; `mask` marks present entries
        # (== non-NaN). Masked-out entries stay 0 and receive no gradient.
        values = torch.zeros(N_SAMPLE, n_aligned, device=device)
        mask = torch.zeros(N_SAMPLE, n_aligned, device=device)

        # Accumulate the summed softplus activations over all sampled species so the
        # L1 penalty below matches custom_synthetic_train.py: a mean over every
        # (species, window) activation. Sequences differ in length, so we track the
        # running activation sum and element count and divide once at the end.
        act_sum = torch.zeros((), device=device)
        act_count = 0
        for r, j in enumerate(sel.tolist()):
            # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
            # both (1, Lj-5) and >= 0; used by the activation L1 penalty below.
            sr_j, a_incl, a_skip = model.compute_sr_profile(
                species_oh[j], return_activations=True
            )
            sr_j = sr_j.squeeze(0)  # (Lj-5,)
            act_sum = act_sum + (a_incl + a_skip).sum()
            act_count += a_incl.numel()

            # Gaussian-smooth the dense SR track along position (gap-free, no NaN).
            # Reflect-pad by the kernel half-width, then conv1d with the single-channel
            # kernel. Differentiable, so gradients flow back to the conv filters.
            if SMOOTH_SIGMA:
                sr_j = F.conv1d(
                    F.pad(
                        sr_j.view(1, 1, -1),
                        (smooth_radius, smooth_radius),
                        mode="reflect",
                    ),
                    gauss_kernel,
                ).view(-1)

            # Place each window's value at the aligned column of its first nucleotide.
            cols = species_cols[j][: sr_j.shape[0]]
            values[r, cols] = sr_j
            mask[r, cols] = 1.0

        # Numerator: NaN-aware cross-species SD per column (ddof=1), valid only where
        # at least MIN_SPECIES of the sampled species are present.
        count = mask.sum(0)  # (n_aligned,)
        col_mean = (values * mask).sum(0) / count.clamp(min=1)
        col_var = (((values - col_mean) ** 2) * mask).sum(0) / (count - 1).clamp(min=1)
        col_std = torch.sqrt(col_var + 1e-8)
        valid = count >= MIN_SPECIES

        # Denominator: within-species dynamic range = each row's SD across positions
        # (NaN-aware, ddof=1), averaged over rows. Dividing by it makes the loss
        # scale-free (matches filter_permutations/filter_permutations.py), so the
        # model can't cheat to a low SD by shrinking all activations toward a constant.
        rc = mask.sum(1)  # (N_SAMPLE,)
        row_mean = (values * mask).sum(1) / rc.clamp(min=1)
        row_var = (((values - row_mean.unsqueeze(1)) ** 2) * mask).sum(1) / (
            rc - 1
        ).clamp(min=1)
        within_species_scale = torch.sqrt(row_var + 1e-8).mean()  # scalar

        sd_loss = (
            col_std[valid] / within_species_scale
        ).mean()  # normalized average column SD

        # L1 penalty on the summed softplus activations (activity sparsity -> peaked,
        # data-driven SR profiles). a_incl/a_skip are already >= 0, so no abs() is
        # needed; dividing by the total element count keeps the penalty independent of
        # seq length / number of species (matches custom_synthetic_train.py).
        l1_penalty = L1_LAMBDA * (act_sum / act_count)
        loss = sd_loss + l1_penalty
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            mean_abs_w = (
                model.conv_incl.weight.abs().mean().item()
                + model.conv_skip.weight.abs().mean().item()
            ) / 2
            logger.info(
                f"epoch {epoch:4d} | loss = {loss.item():.6f} "
                f"| sd loss = {sd_loss.item():.6f} "
                f"| l1 = {l1_penalty.item():.6f} "
                f"| raw mean col SD = {col_std[valid].mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| valid cols = {int(valid.sum().item())} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

    # ── Run metadata for the plotter/sidecar: dataset identity (real alignment, so
    # no planted motif), the source matrix, the hyperparameters, and the final-epoch
    # loss values. The "final" numbers are read from the loop variables still in scope
    # after the loop (their last-iteration values), so nothing in the training loop
    # above changes; raw_mean_col_sd is over valid columns to match the training log. ──
    metadata = {
        "dataset": "real",
        "script": os.path.basename(__file__),
        "motif": None,
        "matrix_path": MATRIX_PATH,
        "hparams": {
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "seed": SEED,
            "n_sample": N_SAMPLE,
            "min_species": MIN_SPECIES,
        },
        "final": {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "l1": float(l1_penalty.item()),
            "raw_mean_col_sd": float(col_std[valid].mean().item()),
            "within_species_scale": float(within_species_scale.item()),
        },
    }

    # ── Save the trained weights to the weights/ directory ──
    # Checkpoint format matches train.py (weights nested under "model_state_dict")
    # so it can be reloaded by plot_filters.py and PNASModel.load_partial_state_dict.
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "real_weights.pt")
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        weights_path,
    )
    logger.info(f"Saved model weights to {weights_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
