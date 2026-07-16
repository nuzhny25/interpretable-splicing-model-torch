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
# Motifs to plant as the conserved blocks. Type in whatever sequences you want
# conserved (ACGT, any length, may be mixed lengths). Blocks are distributed as
# evenly as possible across this list and shuffled across positions; the same
# motif is written into every row, so all rows match at the conserved columns.
MOTIFS = ["AAAAAA", "CCCCCC"]
NUM_BLOCKS = 300  # number of evenly spaced conserved blocks

# Background: instead of an independent random background per species, all species start
# from one shared ancestral sequence tiled into blocks of BG_BLOCK_LEN. Within each block
# each species independently marks BG_MUTABLE_PER_BLOCK positions mutable; each mutable
# position is resampled from all 4 bases with prob BG_MUT_PROB (so it may stay the same).
# This yields a partially conserved, MALAT1-like background instead of pure noise.
BG_BLOCK_LEN = 6
BG_MUTABLE_PER_BLOCK = 3
BG_MUT_PROB = 0.25


def motif_to_indices(motif: str) -> list[int]:
    """Map a nucleotide string to channel indices in the ACGT order."""
    return [NUCLEOTIDES.index(nt) for nt in motif.upper()]


def make_synthetic_matrix(
    n_species: int, seq_len: int, motifs: list[str], num_blocks: int, device
) -> torch.Tensor:
    """One-hot ``(n_species, 4, seq_len)``: mutated rows sharing conserved motifs.

    The background is a single shared *ancestral* nucleotide sequence (channel
    order ACGT) copied into every row, then lightly mutated per species: the
    sequence is tiled into blocks of ``BG_BLOCK_LEN``, and within each block each
    species independently marks ``BG_MUTABLE_PER_BLOCK`` of the positions mutable
    and resamples each mutable position from all 4 bases with probability
    ``BG_MUT_PROB`` (so it may land on the same base). Then conserved blocks are
    written into every row at ``num_blocks`` evenly spaced, non-overlapping
    columns — the conserved regions. Each block is assigned one motif from
    ``motifs`` (distributed as evenly as possible across the list, then shuffled
    across positions), and the same motif is written into every row, so all rows
    are identical at those columns. Cross-species variation thus lives in the
    lightly mutated background (partial conservation, MALAT1-like) rather than in
    pure per-species noise, while the position-varying, species-invariant signal
    is the planted motifs.
    """
    # Shared ancestral background: one random sequence copied to every species, so the
    # unconserved regions start perfectly conserved and only drift via the mutation below.
    ancestral = torch.randint(0, 4, (seq_len,), device=device)
    idx = ancestral.unsqueeze(0).repeat(n_species, 1).clone()  # (n_species, seq_len)

    # Per-species, per-block mutation. Tile into blocks of BG_BLOCK_LEN (any trailing
    # < BG_BLOCK_LEN positions stay unmutated). For each (species, block), mark
    # BG_MUTABLE_PER_BLOCK positions mutable via top-k over random scores — an independent
    # random choice per species and per block — then resample each mutable position from
    # all 4 bases with probability BG_MUT_PROB.
    n_bg_blocks = seq_len // BG_BLOCK_LEN
    if n_bg_blocks:
        tiled = n_bg_blocks * BG_BLOCK_LEN
        scores = torch.rand(n_species, n_bg_blocks, BG_BLOCK_LEN, device=device)
        mutable = torch.zeros_like(scores, dtype=torch.bool)
        mutable.scatter_(-1, scores.topk(BG_MUTABLE_PER_BLOCK, dim=-1).indices, True)
        mutable = mutable.reshape(n_species, tiled)

        mut_event = torch.zeros(n_species, seq_len, dtype=torch.bool, device=device)
        mut_event[:, :tiled] = mutable & (
            torch.rand(n_species, tiled, device=device) < BG_MUT_PROB
        )
        resampled = torch.randint(0, 4, (n_species, seq_len), device=device)
        idx[mut_event] = resampled[mut_event]

    # Pre-map each motif to its ACGT channel indices; space blocks by the longest
    # motif so no two conserved regions overlap regardless of mixed lengths.
    motif_indices = [motif_to_indices(m) for m in motifs]
    max_len = max(len(m) for m in motifs)

    # Balanced assignment: cycle through the motif list so each appears about
    # num_blocks / len(motifs) times, then shuffle so the motifs are interleaved
    # at random positions (not clustered).
    block_choices = torch.tensor([i % len(motifs) for i in range(num_blocks)])
    block_choices = block_choices[torch.randperm(num_blocks)]

    starts = torch.linspace(0, seq_len - max_len, steps=num_blocks).long()
    for s, choice in zip(starts.tolist(), block_choices.tolist()):
        block = motif_indices[choice]
        for j, ch in enumerate(block):
            idx[:, s + j] = ch  # same motif in every row

    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = make_synthetic_matrix(
        N_SPECIES, SEQ_LEN, MOTIFS, NUM_BLOCKS, device
    )  # (10, 4, 5000)
    logger.info(
        f"Synthetic matrix: {tuple(x.shape)} on {device} — "
        f"{NUM_BLOCKS} conserved blocks drawn evenly from motifs {MOTIFS}; "
        f"shared ancestral background, blocks of {BG_BLOCK_LEN}, "
        f"{BG_MUTABLE_PER_BLOCK}/{BG_BLOCK_LEN} mutable @ p={BG_MUT_PROB}"
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
        # On the first iteration, dump the (constant) synthetic matrix as one nucleotide
        # string per species so the planted motifs and the mutated background can be
        # inspected by eye. Rows are un-prefixed so column N lines up across every row.
        if epoch == 1:
            seq_idx = x.argmax(dim=1)  # (n_species, seq_len) base indices
            matrix_path = os.path.join(
                os.path.dirname(__file__), "synthetic_matrix_realistic.txt"
            )
            with open(matrix_path, "w") as f:
                f.write(
                    f"# Synthetic realistic matrix ({N_SPECIES} species x {SEQ_LEN} nt)\n"
                )
                for row in seq_idx.tolist():
                    f.write("".join(NUCLEOTIDES[i] for i in row) + "\n")
            logger.info(f"Wrote synthetic matrix to {matrix_path}")

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
        "motifs": MOTIFS,
        "num_blocks": NUM_BLOCKS,
        "hparams": {
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "l1_lambda": L1_LAMBDA,
            "smooth_sigma": SMOOTH_SIGMA,
            "seed": SEED,
            "n_species": N_SPECIES,
            "seq_len": SEQ_LEN,
            "bg_block_len": BG_BLOCK_LEN,
            "bg_mutable_per_block": BG_MUTABLE_PER_BLOCK,
            "bg_mut_prob": BG_MUT_PROB,
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
    weights_path = os.path.join(weights_dir, "custom_model_realistic.pt")
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
