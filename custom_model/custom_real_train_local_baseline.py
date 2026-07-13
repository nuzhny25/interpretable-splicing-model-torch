"""Real cross-species training with a LOCAL-baseline (excess-conservation) loss.

This is ``custom_real_train.py`` with exactly one change — the normalization of
the per-column cross-species SD — so the two can be compared head to head.

Motivation
----------
``custom_real_train.py`` divides every column's cross-species SD by a *single
global* scalar (the within-species dynamic range). On the real MALAT1 alignment
the background is ~86% conserved across species (phylogeny + overall selection),
so almost every column already has a low cross-species SD regardless of the
filters. Dividing by one global number cannot tell a functionally conserved
column apart from a column that is merely sitting in a slow-evolving, closely
related neighborhood. The loss is dominated by that shared phylogenetic
conservation and collapses onto the single strongest motif.

Fix (a): local baseline instead of a global scalar
--------------------------------------------------
Here each column's cross-species SD is divided by a *per-column local baseline*
— the masked rolling mean of ``col_std`` over a window of
``+/- LOCAL_BASELINE_WINDOW`` aligned columns, counting only present columns.
The loss then measures **excess** conservation: a column only lowers the loss
when it is more cross-species-consistent than its own local background, rather
than being rewarded for the phylogenetic conservation the whole region shares.
This is the training analogue of what phyloP does when it scores a column
against its neutral/local expectation instead of in absolute terms.

Both numerator (``col_std``) and denominator (``local_baseline``) are derived
from the same SR track and scale together, so the loss stays scale-free: uniform
shrinkage of all activations toward a constant does not lower it.

Caveat
------
This addresses the "background correlation swamps the contrast" failure. If
motifs still collapse to one, that points to the deeper issue that uniformly
conserved motifs give no *per-motif* gradient under an SD-minimization loss — the
next step there is to score the SR track directly against a precomputed
phyloP-style per-column conservation target (fix (b)).

Everything else — species sampling, NaN-aware masking, Gaussian SR smoothing, and
the L1 activation penalty — is identical to ``custom_real_train.py``.
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
# Half-width (in aligned columns) of the local background window used to normalize
# each column's cross-species SD. Wide enough to give a stable neighborhood
# estimate, narrow enough to still be "local" relative to the ~15k-column gene.
LOCAL_BASELINE_WINDOW = 45
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
    logger.info(
        f"Local-baseline normalization: masked rolling mean of col_std over "
        f"+/-{LOCAL_BASELINE_WINDOW} aligned columns (excess-conservation loss)"
    )

    model = PNASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Fixed box kernel for the local-baseline rolling mean, built once. Ones of
    # width 2*W+1; applied with zero padding to both col_std*valid and the valid
    # mask so edge columns average over the (fewer) neighbors that actually exist.
    box_kernel = torch.ones(1, 1, 2 * LOCAL_BASELINE_WINDOW + 1, device=device)

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

        # Denominator: per-column LOCAL baseline = masked rolling mean of col_std over
        # a +/-LOCAL_BASELINE_WINDOW window of *valid* columns. Convolving col_std*valid
        # with a box kernel gives the neighborhood sum; convolving the valid mask with
        # the same kernel gives how many valid columns fell in each window, so dividing
        # yields the mean over present neighbors only (edges self-correct). Dividing
        # each column by its own baseline makes the loss an excess-conservation measure
        # (see module docstring); it stays differentiable and scale-free because the
        # baseline is built from the same col_std as the numerator.
        valid_f = valid.float()
        local_sum = F.conv1d(
            (col_std * valid_f).view(1, 1, -1),
            box_kernel,
            padding=LOCAL_BASELINE_WINDOW,
        ).view(-1)
        local_cnt = F.conv1d(
            valid_f.view(1, 1, -1), box_kernel, padding=LOCAL_BASELINE_WINDOW
        ).view(-1)
        local_baseline = local_sum / local_cnt.clamp(min=1)

        sd_loss = (
            col_std[valid] / (local_baseline[valid] + 1e-8)
        ).mean()  # mean excess-conservation ratio over valid columns

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
                f"| mean local baseline = {local_baseline[valid].mean().item():.6f} "
                f"| valid cols = {int(valid.sum().item())} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

    # ── Save the trained weights to the weights/ directory ──
    # Checkpoint format matches train.py (weights nested under "model_state_dict")
    # so it can be reloaded by plot_filters.py and PNASModel.load_partial_state_dict.
    # Distinct filename so it does not clobber custom_real_train.py's real_weights.pt.
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "real_local_baseline_weights.pt")
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        weights_path,
    )
    logger.info(f"Saved model weights to {weights_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
