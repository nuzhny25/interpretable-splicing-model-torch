"""Synthetic training for the normalized cross-species abs-dev-minimization loss.

Builds a hardcoded ``10 x 5000`` matrix of aligned sequences (10 "species" rows,
5000 nucleotides each, no gaps), runs the simplified 20+20-filter model — whose
per-filter activations are tanh-capped to ``[0, 1)`` (``tanh(softplus(conv))``) —
to get a per-position SR-balance track for each row, and minimizes a normalized
cross-species mean absolute deviation.

The model processes the 10 rows as an independent batch — it never sees that
they form a matrix. The matrix/alignment structure is used only in the loss,
where the 10 SR tracks are stacked into a ``(10, num_windows)`` matrix and the
mean absolute deviation is taken down each column. The rows are therefore coupled
only through the loss gradient.

The per-column cross-species abs-dev is divided by the within-species dynamic range
(mean over rows of each row's abs-dev across positions), mirroring the normalization
in filter_permutations/filter_permutations.py. This makes the loss scale-free:
shrinking all activations toward zero shrinks the numerator and denominator
equally, so the trivial "constant SR" collapse no longer lowers the loss.
"""

import logging
import math
import os

import torch
import torch.nn.functional as F

from custom_model_capped import PNASModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Hardcoded synthetic setup ───────────────────────────────────────────────
N_SPECIES = 10
SEQ_LEN = 5000
NUM_EPOCHS = 10000
LR = 1e-2
L1_LAMBDA = 1e-2  # strength of L1 penalty on the tanh-capped activations (retune by watching logs)
SMOOTH_SIGMA = (
    1.5  # Gaussian smoothing of the SR profile along position (nt); 0 disables
)
SEED = 0

# ── Per-filter gain settings (see PNASModel.filter_gains) ────────────────────
GAIN_TEMP = 1.0  # softmax temperature; larger = flatter, slower reallocation
GAIN_FLOOR = 0.02  # minimum gain per filter, so starved filters keep a gradient
# Epochs to hold the gain logits frozen at zero (uniform gains) before letting the
# filters compete. Without this the allocation starts moving from epoch 1, when
# marginal utility is dominated by random init, and a couple of filters can starve
# the rest before any of them has found a motif. Training during the freeze is
# identical to the ungained model. 0 disables.
GAIN_FREEZE_EPOCHS = 500
# Strength of the entropy penalty on the gain allocation. The gains form a
# probability vector (g / num_filters), so its entropy H measures how spread out
# the budget is; adding +lambda*H to the loss rewards *low* entropy, i.e. a peaked
# allocation on few filters. Entropy on the simplex is inherently scale-free, so it
# fits the scale-free abs-dev loss without reintroducing a shrink-to-zero direction.
# 0 disables — the softmax competition already produces some sparsity on its own,
# so turn this on only if the logged effective-filter count stays too high.
GAIN_ENTROPY_LAMBDA = 0.0

# Descriptor of the model's filter-activation / output scheme, recorded in the
# checkpoint metadata so the plotter can label which variant produced the
# filters. Set it to match compute_sr_profile's activation: "softplus" for the
# default (uncapped) model, or e.g. "capped-tanh" / "capped-sigmoid" when the
# per-filter activations are capped to [0, 1].
MODEL_TYPE = "capped-tanh"

NUCLEOTIDES = ["A", "C", "G", "T"]
# Motifs to plant as the conserved blocks. Type in whatever sequences you want
# conserved (ACGT, any length, may be mixed lengths). Blocks are distributed as
# evenly as possible across this list and shuffled across positions; the same
# motif is written into every row, so all rows match at the conserved columns.
MOTIFS = ["AAAATA", "CCGGCC"]
NUM_BLOCKS = 30  # number of evenly spaced conserved blocks

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

    model = PNASModel(gain_temp=GAIN_TEMP, gain_floor=GAIN_FLOOR).to(device)
    # Adam's default weight_decay=0 matters here: decay on the gain logits would pull
    # them back toward zero, which is exactly the uniform allocation — an anti-sparsity
    # prior working against the competition the gains are meant to create.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Hold the gain logits at their zero init (uniform gains of 1.0) for the first
    # GAIN_FREEZE_EPOCHS epochs. Params with requires_grad=False get no .grad, and Adam
    # skips them, so the frozen phase trains exactly like the ungained model.
    if GAIN_FREEZE_EPOCHS:
        model.gain_logit_incl.requires_grad_(False)
        model.gain_logit_skip.requires_grad_(False)
        logger.info(
            f"Per-filter gains: temp={GAIN_TEMP}, floor={GAIN_FLOOR}, "
            f"entropy lambda={GAIN_ENTROPY_LAMBDA} — logits frozen for the first "
            f"{GAIN_FREEZE_EPOCHS} epochs"
        )
    else:
        logger.info(
            f"Per-filter gains: temp={GAIN_TEMP}, floor={GAIN_FLOOR}, "
            f"entropy lambda={GAIN_ENTROPY_LAMBDA} — logits trainable from epoch 1"
        )

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

        # End of the frozen phase: let the filters start competing for gain budget.
        if GAIN_FREEZE_EPOCHS and epoch == GAIN_FREEZE_EPOCHS + 1:
            model.gain_logit_incl.requires_grad_(True)
            model.gain_logit_skip.requires_grad_(True)
            logger.info(f"epoch {epoch:4d} | gain logits unfrozen")

        # a_incl / a_skip are the raw (unsmoothed), *ungained* summed activations,
        # both (10, num_windows) and >= 0; used by the activation L1 penalty below.
        # Ungained on purpose: a penalty on the gain-weighted sum would be minimized
        # by parking the gain budget on the least active filter, so the activity tax
        # would end up steering the allocation onto dead filters.
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

        # Numerator: cross-species mean absolute deviation per column (across the 10
        # rows), measured about each column's cross-species mean.
        col_std = (
            (sr - sr.mean(dim=0, keepdim=True)).abs().mean(dim=0)
        )  # (num_windows,)

        # Denominator: within-species scale = each row's mean absolute deviation across
        # positions (about that row's own mean), averaged over rows. Dividing by it keeps
        # the loss scale-free (the constant-SR collapse no longer lowers the loss),
        # matching the intent of the SD normalization in
        # filter_permutations/filter_permutations.py.
        within_species_scale = (
            (sr - sr.mean(dim=1, keepdim=True)).abs().mean(dim=1).mean()
        )  # scalar

        sd_loss = (
            (col_std).mean()
        ) / within_species_scale  # normalized average column abs-dev

        # L1 penalty on the summed tanh-capped activations (activity sparsity -> peaked,
        # data-driven SR profiles). a_incl/a_skip are already >= 0, so no abs() is
        # needed; .mean() keeps the penalty independent of seq length / batch size.

        l1_penalty = L1_LAMBDA * (a_incl + a_skip).mean()

        # Gain allocation per bank as a probability vector (each sums to 1 by
        # construction). Its Shannon entropy measures how spread the budget is:
        # log(20) when uniform, 0 when one filter takes everything. exp(H) is the
        # effective number of filters actually carrying the track — the single
        # number to watch for whether the competition is working (falling) or
        # locking in too early (crashing toward 1 right after the unfreeze).
        g_incl, g_skip = model.filter_gains()
        p_incl = g_incl / model.num_seq_filters
        p_skip = g_skip / model.num_seq_filters
        gain_entropy_incl = -(p_incl * p_incl.log()).sum()
        gain_entropy_skip = -(p_skip * p_skip.log()).sum()

        loss = sd_loss + l1_penalty
        if GAIN_ENTROPY_LAMBDA:
            # Positive lambda penalizes high entropy, i.e. rewards concentrating the
            # budget on few filters.
            loss = loss + GAIN_ENTROPY_LAMBDA * (gain_entropy_incl + gain_entropy_skip)
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
                f"| raw mean col abs-dev = {col_std.mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| mean act = {mean_act:.6f} "
                f"| mean|conv_w| = {mean_abs_w:.6f} "
                f"| eff filters incl/skip = {gain_entropy_incl.exp().item():.2f}"
                f"/{gain_entropy_skip.exp().item():.2f} "
                f"| max gain incl/skip = {g_incl.max().item():.2f}"
                f"/{g_skip.max().item():.2f}"
            )

    # ── Run metadata for the plotter/sidecar: dataset identity, planted motif, the
    # hyperparameters, and the final-epoch loss values. The "final" numbers are read
    # from the loop variables still in scope after the loop (their last-iteration
    # values), so nothing in the training loop above changes. ──
    metadata = {
        "dataset": "synthetic",
        "script": os.path.basename(__file__),
        "model_type": MODEL_TYPE,
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
            "gain_temp": GAIN_TEMP,
            "gain_floor": GAIN_FLOOR,
            "gain_freeze_epochs": GAIN_FREEZE_EPOCHS,
            "gain_entropy_lambda": GAIN_ENTROPY_LAMBDA,
        },
        "final": {
            "loss": float(loss.item()),
            "sd_loss": float(sd_loss.item()),
            "l1": float(l1_penalty.item()),
            "raw_mean_col_sd": float(col_std.mean().item()),
            "within_species_scale": float(within_species_scale.item()),
            # Final gain allocation, so the plotter can rank filters by learned
            # importance and runs can be compared across seeds by effective count.
            "gain_incl": [float(v) for v in g_incl.detach().cpu()],
            "gain_skip": [float(v) for v in g_skip.detach().cpu()],
            "eff_filters_incl": float(gain_entropy_incl.exp().item()),
            "eff_filters_skip": float(gain_entropy_skip.exp().item()),
        },
    }

    # ── Save the trained weights to the weights/ directory ──
    # Checkpoint format matches train.py (weights nested under "model_state_dict")
    # so it can be reloaded by plot_filters.py and PNASModel.load_partial_state_dict.
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "custom_model_capped.pt")
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
