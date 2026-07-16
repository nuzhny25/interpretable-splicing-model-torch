"""Synthetic training on per-location A/C-split conserved blocks.

Builds a ``10 x seq_len`` matrix of aligned sequences (10 "species" rows, no
gaps) with ``NUM_BLOCKS`` evenly spaced conserved locations, then runs the same
normalized cross-species SD-minimization training loop as
``custom_synthetic_train.py`` on it.

Unlike ``custom_synthetic_train.make_synthetic_matrix`` — which writes the *same*
homopolymer into every row at a location, so all species agree there — here each
conserved location is *split across species*: every species independently flips a
fair coin and gets either an ``AAAAAA`` block (probability 0.5) or a ``CCCCCC``
block. The coins are redrawn independently per species and per location, so the
A/C partition differs from location to location and the poly-A / poly-C counts are
not forced balanced (any ratio can occur). Everywhere else each row has an
independent random nucleotide background (channel order ACGT).

The model still processes the 10 rows as an independent batch and never sees the
matrix structure; the alignment is used only in the loss, where the 10 SR tracks
are stacked into a ``(10, num_windows)`` matrix and the per-column cross-species
SD (normalized by the within-species dynamic range) is minimized — identical to
``custom_synthetic_train.py``. Note the signal here is bimodal across species at
each conserved column (a random mix of poly-A and poly-C rows), so driving the
column SD down pushes the filters toward giving poly-A and poly-C blocks a matched
SR-balance readout.
"""

import logging
import math
import os

import torch
import torch.nn.functional as F

from custom_model import PNASModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Hardcoded synthetic setup ───────────────────────────────────────────────
N_SPECIES = 10
SEQ_LEN = 5000
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
BLOCK_LEN = 6  # length of each conserved block (AAAAAA / CCCCCC)
NUM_BLOCKS = 250  # number of evenly spaced conserved locations


def make_split_synthetic_matrix(
    n_species: int, seq_len: int, block_len: int, num_blocks: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot ``(n_species, 4, seq_len)`` with per-location A/C-split blocks.

    Each row gets an independent random nucleotide background (channel order
    ACGT). Then ``num_blocks`` conserved blocks are written at evenly spaced,
    non-overlapping columns. At each location every species independently flips a
    fair coin: with probability 0.5 it gets an all-A block, otherwise an all-C
    block. The coin is redrawn per species and per location, so the poly-A /
    poly-C counts at a location are no longer forced balanced — any ratio can
    occur, including (rarely) an all-poly-A or all-poly-C location.

    Returns the one-hot matrix and a boolean ``(num_blocks, n_species)`` mask
    that is ``True`` where a species carries the poly-A block at that location
    (its complement marks the poly-C species).
    """
    idx = torch.randint(0, 4, (n_species, seq_len), device=device)

    a_idx = NUCLEOTIDES.index("A")
    c_idx = NUCLEOTIDES.index("C")

    starts = torch.linspace(0, seq_len - block_len, steps=num_blocks).long()
    # At each location every species independently flips a fair coin: poly-A block
    # (True) with probability 0.5, otherwise poly-C. Counts are no longer forced
    # balanced, so a location may land on any poly-A / poly-C ratio (rarely even
    # all-poly-A or all-poly-C).
    is_poly_a = torch.rand(num_blocks, n_species, device=device) < 0.5
    for b, s in enumerate(starts.tolist()):
        a_species = is_poly_a[b]
        idx[a_species, s : s + block_len] = a_idx
        idx[~a_species, s : s + block_len] = c_idx

    x = F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()
    return x, is_poly_a


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, is_poly_a = make_split_synthetic_matrix(
        N_SPECIES, SEQ_LEN, BLOCK_LEN, NUM_BLOCKS, device
    )  # (10, 4, 5000)
    starts = torch.linspace(0, SEQ_LEN - BLOCK_LEN, steps=NUM_BLOCKS).long().tolist()

    # ── Verify the matrix has the intended per-location A/C split ──
    # Decode the one-hot back to nucleotide indices and check, at every conserved
    # location, that (a) it's a valid one-hot column and (b) the poly-A species are
    # all-A across the whole block while the poly-C species are all-C. Counts are no
    # longer balanced (each species is an independent coin flip), so we don't check
    # the split size.
    assert (x.sum(dim=1) == 1).all(), "matrix is not valid one-hot"
    idx_decoded = x.argmax(dim=1)  # (n_species, seq_len)
    a_idx = NUCLEOTIDES.index("A")
    c_idx = NUCLEOTIDES.index("C")
    for b, s in enumerate(starts):
        block = idx_decoded[:, s : s + BLOCK_LEN]  # (n_species, block_len)
        a_mask = is_poly_a[b]  # (n_species,) True = poly-A species here
        assert (block[a_mask] == a_idx).all(), f"location {b}: poly-A block not all-A"
        assert (block[~a_mask] == c_idx).all(), f"location {b}: poly-C block not all-C"
    logger.info(
        f"Verification passed: {NUM_BLOCKS} conserved locations, each an independent "
        f"per-species 50/50 poly-A / poly-C split with clean length-{BLOCK_LEN} "
        f"homopolymer blocks."
    )

    logger.info(
        f"Synthetic split matrix: {tuple(x.shape)} on {device} — "
        f"{NUM_BLOCKS} conserved locations of length {BLOCK_LEN}, each species "
        f"independently poly-A with prob 0.5 (overall "
        f"{is_poly_a.float().mean().item():.2f} poly-A fraction)"
    )

    # Per-location A/C assignment: one row per conserved location, one column per
    # species. 'A' = poly-A block at that location, 'C' = poly-C block.
    header = "location   start  | " + " ".join(f"s{j}" for j in range(N_SPECIES))
    logger.info(header)
    logger.info("-" * len(header))
    for b, s in enumerate(starts):
        letters = " ".join(" A" if is_poly_a[b, j] else " C" for j in range(N_SPECIES))
        logger.info(f"loc {b:<2d}    {s:6d}  | {letters}")

    # ── Training: normalized cross-species SD minimization (see module docstring) ──
    model = PNASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Fixed Gaussian kernel for smoothing the SR profile along position, built once.
    # Sum-normalized so it's a weighted average (preserves overall scale). It carries
    # no learnable parameters and is applied with reflect padding at the use site so
    # the conserved block at position 0 isn't damped by zero edges. Smoothing denoises
    # single-position jitter so the conserved-motif signal dominates the column SD.
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
        # a_incl / a_skip are the raw (unsmoothed) summed softplus activations,
        # both (10, num_windows) and >= 0; used by the activation L1 penalty below.
        sr, a_incl, a_skip = model.compute_sr_profile(x, return_activations=True)

        # Gaussian-smooth each row's SR track along position (independent per row).
        # Reflect-pad by the kernel half-width, then depthwise-style conv1d with the
        # single-channel kernel, keeping the (10, num_windows) shape. Differentiable,
        # so gradients flow back to the conv filters.
        if SMOOTH_SIGMA:
            sr = F.conv1d(
                F.pad(sr.unsqueeze(1), (smooth_radius, smooth_radius), mode="reflect"),
                gauss_kernel,
            ).squeeze(1)

        # Numerator: cross-species SD per column (across the 10 rows).
        col_std = sr.std(dim=0)  # (num_windows,)

        # Denominator: within-species dynamic range = each row's SD across
        # positions, averaged over rows. Dividing by it makes the loss scale-free
        # (matches filter_permutations/filter_permutations.py), so the model can't
        # cheat to a low SD by shrinking all activations toward a constant.
        within_species_scale = sr.std(dim=1).mean()  # scalar

        sd_loss = (
            col_std
        ).mean() / within_species_scale  # normalized average column SD

        # L1 penalty on the summed softplus activations (activity sparsity -> peaked,
        # data-driven SR profiles). a_incl/a_skip are already >= 0, so no abs() is
        # needed; .mean() keeps the penalty independent of seq length / batch size.

        l1_penalty = L1_LAMBDA * (a_incl + a_skip).mean()

        loss = sd_loss + l1_penalty
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            mean_abs_w = (
                model.conv_incl.weight.abs().mean().item()
                + model.conv_skip.weight.abs().mean().item()
            ) / 2
            mean_act = (a_incl + a_skip).mean().item()  # the now-regularized quantity
            logger.info(
                f"epoch {epoch:4d} | loss = {loss.item():.6f} "
                f"| sd loss = {sd_loss.item():.6f} "
                f"| l1 = {l1_penalty.item():.6f} "
                f"| raw mean col SD = {col_std.mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| mean act = {mean_act:.6f} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

    # ── Run metadata for the plotter/sidecar: dataset identity, the planted motif
    # (a per-species poly-A / poly-C coin flip per location, so there is no single
    # MOTIF string), the hyperparameters, and the final-epoch loss values. The
    # "final" numbers are read from the loop variables still in scope after the loop
    # (their last-iteration values), so nothing in the training loop above changes. ──
    metadata = {
        "dataset": "synthetic",
        "script": os.path.basename(__file__),
        "motif": "poly-A/poly-C 50/50 per-species split",
        "block_len": BLOCK_LEN,
        "num_blocks": NUM_BLOCKS,
        "hparams": {
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "seed": SEED,
            "n_species": N_SPECIES,
            "seq_len": SEQ_LEN,
        },
        "final": {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "l1": float(l1_penalty.item()),
            "raw_mean_col_sd": float(col_std.mean().item()),
            "within_species_scale": float(within_species_scale.item()),
        },
    }

    # ── Save the trained weights to the weights/ directory ──
    # Checkpoint format matches train.py (weights nested under "model_state_dict")
    # so it can be reloaded by plot_filters.py and PNASModel.load_partial_state_dict.
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "custom_split_model.pt")
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
