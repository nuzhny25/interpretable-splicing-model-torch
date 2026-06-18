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

import torch
import torch.nn.functional as F

from custom_model import PNASModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Hardcoded synthetic setup ───────────────────────────────────────────────
N_SPECIES = 10
SEQ_LEN = 5000
NUM_EPOCHS = 20000
LR = 1e-2
SEED = 0

NUCLEOTIDES = ["A", "C", "G", "T"]
MOTIF = "AAAAAA"  # conserved block, planted identically into every row
NUM_BLOCKS = 10  # number of evenly spaced conserved blocks


def motif_to_indices(motif: str) -> list[int]:
    """Map a nucleotide string to channel indices in the ACGT order."""
    return [NUCLEOTIDES.index(nt) for nt in motif.upper()]


def motif_match_scores(conv_weight, motif_idx) -> list[float]:
    """Cosine similarity of each filter (4 x K) with the motif's one-hot (4 x K).

    A high positive score means the filter's weights resemble the planted motif,
    i.e. the filter has converged toward detecting it.
    """
    _, channels, k = conv_weight.shape
    motif_oh = torch.zeros(channels, k)
    motif_oh[motif_idx, torch.arange(k)] = 1.0
    motif_flat = motif_oh.reshape(-1)
    scores = []
    for f in range(conv_weight.shape[0]):
        w = conv_weight[f].reshape(-1)
        denom = w.norm() * motif_flat.norm() + 1e-8
        scores.append((w @ motif_flat / denom).item())
    return scores


def format_filter(weight_4x6) -> str:
    """Return a labeled string for one 4 x 6 filter tensor.

    Rows are the nucleotide channels (A, C, G, T) and columns are the 6 kernel
    positions, matching filter_permutations/print_filters.py.
    """
    kernel_size = weight_4x6.shape[1]
    header = "        " + "".join(f"pos{p + 1:<6}" for p in range(kernel_size))
    lines = [header]
    for nt, row in zip(NUCLEOTIDES, weight_4x6):
        cells = "".join(f"{v:+9.4f}" for v in row.tolist())
        lines.append(f"  {nt}   {cells}")
    return "\n".join(lines)


def make_synthetic_matrix(
    n_species: int, seq_len: int, motif: str, num_blocks: int, device
) -> torch.Tensor:
    """One-hot ``(n_species, 4, seq_len)``: random rows sharing a conserved motif.

    Each row gets an independent random nucleotide background (channel order
    ACGT). Then ``motif`` is written into every row at ``num_blocks`` evenly
    spaced, non-overlapping columns — the conserved regions. All rows are
    identical at those columns and differ everywhere else, so cross-species
    variation lives only in the random background and the only position-varying,
    species-invariant signal is the motif.
    """
    idx = torch.randint(0, 4, (n_species, seq_len), device=device)

    motif_idx = torch.tensor(motif_to_indices(motif), device=device)
    motif_len = motif_idx.numel()

    starts = torch.linspace(0, seq_len - motif_len, steps=num_blocks).long()
    for s in starts.tolist():
        idx[:, s : s + motif_len] = motif_idx  # same motif in every row

    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = make_synthetic_matrix(
        N_SPECIES, SEQ_LEN, MOTIF, NUM_BLOCKS, device
    )  # (10, 4, 5000)
    logger.info(
        f"Synthetic matrix: {tuple(x.shape)} on {device} — "
        f"motif '{MOTIF}' planted in {NUM_BLOCKS} conserved blocks"
    )

    model = PNASModel(input_length=SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        sr = model.compute_sr_profile(x)  # (10, num_windows): per-row SR track

        # Numerator: cross-species SD per column (across the 10 rows).
        col_std = sr.std(dim=0)  # (num_windows,)

        # Denominator: within-species dynamic range = each row's SD across
        # positions, averaged over rows. Dividing by it makes the loss scale-free
        # (matches filter_permutations/filter_permutations.py), so the model can't
        # cheat to a low SD by shrinking all activations toward a constant.
        within_species_scale = sr.std(dim=1).mean()  # scalar

        loss = (col_std / within_species_scale).mean()  # normalized average column SD
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            mean_abs_w = (
                model.conv_incl.weight.abs().mean().item()
                + model.conv_skip.weight.abs().mean().item()
            ) / 2
            logger.info(
                f"epoch {epoch:4d} | norm loss = {loss.item():.6f} "
                f"| raw mean col SD = {col_std.mean().item():.6f} "
                f"| within-species scale = {within_species_scale.item():.6f} "
                f"| mean|conv_w| = {mean_abs_w:.6f}"
            )

    logger.info("Done.")

    # ── Filter -> motif convergence readout (cosine similarity to the planted motif) ──
    motif_idx = torch.tensor(motif_to_indices(MOTIF))
    logger.info(f"Filter convergence toward motif '{MOTIF}' (cosine similarity):")
    for name, conv in [("INCL", model.conv_incl), ("SKIP", model.conv_skip)]:
        scores = motif_match_scores(conv.weight.detach().cpu(), motif_idx)
        ranked = sorted(range(len(scores)), key=lambda f: scores[f], reverse=True)
        top = ", ".join(f"#{f}={scores[f]:+.3f}" for f in ranked[:5])
        logger.info(f"  [{name}] top matches: {top}")

    # ── Write all filters to a text file (4 x 6 each: rows A/C/G/T, columns = 6 kernel positions) ──
    out_path = "filters.txt"
    num_filters = model.conv_incl.weight.shape[0]
    with open(out_path, "w") as fh:
        fh.write(
            f"conv_incl/conv_skip weight shape: {tuple(model.conv_incl.weight.shape)}  "
            f"(num_filters, channels, kernel_size)\n\n"
        )
        for name, conv in [("INCL", model.conv_incl), ("SKIP", model.conv_skip)]:
            for filter_idx in range(num_filters):
                fh.write(f"=== {name} filter #{filter_idx} ===\n")
                fh.write(format_filter(conv.weight[filter_idx].detach().cpu()) + "\n\n")
    logger.info(f"Wrote {2 * num_filters} filters to {out_path}")


if __name__ == "__main__":
    main()
