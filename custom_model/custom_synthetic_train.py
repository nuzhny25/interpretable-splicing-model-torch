"""Synthetic training for the normalized cross-species SD-minimization loss.

Builds a hardcoded ``10 x 5000`` matrix of aligned sequences (10 "species" rows,
5000 nucleotides each, no gaps), runs the simplified 20+20-filter model to get a
per-position SR-balance track for each row, and minimizes a normalized
cross-species standard deviation.

The model processes the 10 rows as an independent batch — it never sees that
they form a matrix. The matrix/alignment structure is used only in the loss,
where the 10 SR tracks are stacked into a ``(10, num_windows)`` matrix and the
SD is taken down each column. The rows are therefore coupled only through the
loss gradient.

The per-column cross-species SD is divided by the within-species dynamic range
(mean over rows of each row's SD across positions), mirroring the normalization
in filter_permutations/filter_permutations.py. This makes the loss scale-free:
shrinking all activations toward zero shrinks the numerator and denominator
equally, so the trivial "constant SR" collapse no longer lowers the loss.
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
    1e-2  # strength of L1 penalty on the softplus activations (retune by watching logs)
)
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)
SEED = 0

NUCLEOTIDES = ["A", "C", "G", "T"]
MOTIF = (
    "AAAA"  # sets the conserved block length; blocks are an even poly-A / poly-C split
)
NUM_BLOCKS = 250  # number of evenly spaced conserved blocks


def motif_to_indices(motif: str) -> list[int]:
    """Map a nucleotide string to channel indices in the ACGT order."""
    return [NUCLEOTIDES.index(nt) for nt in motif.upper()]


def make_synthetic_matrix(
    n_species: int, seq_len: int, motif: str, num_blocks: int, device
) -> torch.Tensor:
    """One-hot ``(n_species, 4, seq_len)``: random rows sharing conserved motifs.

    Each row gets an independent random nucleotide background (channel order
    ACGT). Then conserved blocks are written into every row at ``num_blocks``
    evenly spaced, non-overlapping columns — the conserved regions. The blocks
    are split evenly between all-A and all-C (length ``len(motif)``; the extra
    block goes to poly-A when ``num_blocks`` is odd) and then shuffled across the
    positions, but the same choice is written into every row, so all rows are
    identical at those columns and differ everywhere else. Cross-species
    variation thus lives only in the random background, while the
    position-varying, species-invariant signal is a balanced mix of poly-A and
    poly-C conserved blocks.
    """
    idx = torch.randint(0, 4, (n_species, seq_len), device=device)

    motif_len = len(motif)
    a_idx = NUCLEOTIDES.index("A")
    c_idx = NUCLEOTIDES.index("C")

    # Balanced split: half the blocks poly-C, the rest poly-A, then shuffled so
    # the two homopolymers are interleaved at random positions (not clustered).
    n_c = num_blocks // 2
    block_choices = torch.tensor([c_idx] * n_c + [a_idx] * (num_blocks - n_c))
    block_choices = block_choices[torch.randperm(num_blocks)]

    starts = torch.linspace(0, seq_len - motif_len, steps=num_blocks).long()
    for s, block_idx in zip(starts.tolist(), block_choices.tolist()):
        idx[:, s : s + motif_len] = block_idx  # same homopolymer in every row

    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = make_synthetic_matrix(
        N_SPECIES, SEQ_LEN, MOTIF, NUM_BLOCKS, device
    )  # (10, 4, 5000)
    logger.info(
        f"Synthetic matrix: {tuple(x.shape)} on {device} — "
        f"{NUM_BLOCKS} conserved blocks of length {len(MOTIF)}, "
        f"balanced poly-A / poly-C split"
    )

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

    # ── Run metadata for the plotter/sidecar: dataset identity, planted motif, the
    # hyperparameters, and the final-epoch loss values. The "final" numbers are read
    # from the loop variables still in scope after the loop (their last-iteration
    # values), so nothing in the training loop above changes. ──
    metadata = {
        "dataset": "synthetic",
        "script": os.path.basename(__file__),
        "motif": MOTIF,
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
    weights_path = os.path.join(weights_dir, "custom_model.pt")
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
