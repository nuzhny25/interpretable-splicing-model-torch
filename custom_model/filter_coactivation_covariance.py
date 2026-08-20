"""Per-window co-activation covariance between the 20+20 sequence filters.

Because the model combines filters purely additively (the SR profile is
``sum_i softplus(incl_i) - sum_i softplus(skip_i)``), there is no pairwise
filter-interaction term inside the model itself. "How filter i interacts with
filter j" is therefore defined empirically here as how their per-window
activations co-vary across the real MALAT1 alignment.

For every species in ``data/multiz100/alignment_matrix.npy`` the gap-removed
sequence is run through ``conv_incl`` and ``conv_skip``; each 6-nt sliding
window produces one non-negative ``softplus`` activation per filter. Stacking
the two banks gives 40 filter channels (``incl_00..incl_19``,
``skip_00..skip_19``). Every window across every species is one sample (row) of
the data matrix ``X`` of shape ``(n_windows, 40)``. The deliverable is the
40x40 covariance matrix ``cov(X)`` and its normalized correlation ``corr(X)``:
entry (i, j) says whether filters i and j fire together at the same window.

Outputs (written to ``--out-dir``):
  * ``covariance.npy`` / ``correlation.npy`` — 40x40 matrices (filter order).
  * ``correlation.csv`` — labeled correlation matrix for quick inspection.
  * ``filter_order.txt`` — row/column order of the heatmap (clustering leaf
    order, or the fixed filter order under ``--natural-order``).
  * ``coactivation_correlation.png`` — correlation heatmap (clustered by
    default; ``--natural-order`` keeps the fixed incl_00..skip_19 axes).
  * ``mean_activation.csv`` — per-filter mean/std activation (scale reference).
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from custom_model import PNASModel
from custom_utils import str_to_vector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "multiz100", "alignment_matrix.npy"
)
NUM_FILTERS = 20  # per bank; incl + skip -> 40 channels total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            os.path.dirname(__file__), "weights", "real_weights_aad.pt"
        ),
        help="Model checkpoint (.pt). Default: the SD baseline custom_model.pt.",
    )
    parser.add_argument("--matrix", default=MATRIX_PATH, help="Alignment matrix .npy.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "filter_covariance"),
        help="Directory for the covariance/correlation matrices and heatmap.",
    )
    parser.add_argument(
        "--natural-order",
        action="store_true",
        help="Plot the heatmap in the fixed incl_00..skip_19 order instead of "
        "the clustered order (fixed axes are comparable across checkpoints).",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────────────
    sd = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model = PNASModel().to(device)
    model.load_partial_state_dict(sd)
    model.eval()

    # ── Load alignment and pool per-window activations across all species ────
    # matrix: (n_aligned, n_species) of single characters; gaps are "-"/"N",
    # lowercase soft-masked a/c/g/t are nucleotides (upper-cased before one-hot).
    matrix = np.load(args.matrix)
    n_aligned, n_species = matrix.shape
    logger.info(
        f"Alignment {matrix.shape} (positions x species) on {device}; "
        f"pooling per-window activations from {NUM_FILTERS}+{NUM_FILTERS} filters."
    )

    # X accumulates one row per (species, window): 40 filter activations each.
    window_blocks = []
    with torch.no_grad():
        for j in range(n_species):
            row = matrix[:, j]
            is_nuc = (row != "-") & (row != "N")
            seq = "".join(row[is_nuc]).upper()
            x = torch.tensor(str_to_vector(seq), dtype=torch.float32, device=device)
            x = x.unsqueeze(0)  # (1, 4, Lj)
            a_incl = F.softplus(model.conv_incl(x))  # (1, 20, Lj-5), >= 0
            a_skip = F.softplus(model.conv_skip(x))  # (1, 20, Lj-5), >= 0
            # -> (Lj-5, 40): windows are rows, incl then skip filters are columns.
            block = torch.cat([a_incl, a_skip], dim=1).squeeze(0).T.cpu().numpy()
            window_blocks.append(block.astype(np.float64))

    X = np.concatenate(window_blocks, axis=0)  # (n_windows, 40)
    labels = [f"incl_{i:02d}" for i in range(NUM_FILTERS)] + [
        f"skip_{i:02d}" for i in range(NUM_FILTERS)
    ]
    logger.info(f"Pooled {X.shape[0]:,} windows across {n_species} species.")

    # ── Covariance and correlation over the 40 filter channels ───────────────
    cov = np.cov(X, rowvar=False)  # (40, 40)
    corr = np.corrcoef(X, rowvar=False)  # (40, 40), scale-free
    mean_act = X.mean(axis=0)
    std_act = X.std(axis=0)

    np.save(os.path.join(args.out_dir, "covariance.npy"), cov)
    np.save(os.path.join(args.out_dir, "correlation.npy"), corr)
    _save_labeled_csv(os.path.join(args.out_dir, "correlation.csv"), corr, labels)
    with open(os.path.join(args.out_dir, "mean_activation.csv"), "w") as fh:
        fh.write("filter,mean_activation,std_activation\n")
        for lab, mu, sg in zip(labels, mean_act, std_act):
            fh.write(f"{lab},{mu:.6f},{sg:.6f}\n")

    # ── Order rows/cols by hierarchical clustering so structure is visible ───
    # With --natural-order, keep the fixed incl_00..skip_19 order instead so the
    # heatmap axes are identical across checkpoints/seeds.
    order = list(range(len(labels))) if args.natural_order else _cluster_order(corr)
    with open(os.path.join(args.out_dir, "filter_order.txt"), "w") as fh:
        fh.write("\n".join(labels[i] for i in order) + "\n")

    _plot_heatmap(
        corr[np.ix_(order, order)],
        [labels[i] for i in order],
        os.path.join(args.out_dir, "coactivation_correlation.png"),
        title=f"Filter co-activation correlation — {os.path.basename(args.checkpoint)}",
    )

    # ── Report the strongest interactions ────────────────────────────────────
    iu = np.triu_indices(len(labels), k=1)
    pair_corr = corr[iu]
    strongest = np.argsort(-np.abs(pair_corr))[:15]
    logger.info("\nStrongest filter co-activation pairs (|correlation|):")
    for k in strongest:
        i, j = iu[0][k], iu[1][k]
        logger.info(f"  {labels[i]} <-> {labels[j]}   r = {corr[i, j]:+.3f}")
    logger.info(f"\nWrote covariance/correlation matrices + heatmap to {args.out_dir}")


def _save_labeled_csv(path, mat, labels):
    """Write a square matrix with a header row/column of filter labels."""
    with open(path, "w") as fh:
        fh.write("," + ",".join(labels) + "\n")
        for lab, rowvals in zip(labels, mat):
            fh.write(lab + "," + ",".join(f"{v:.6f}" for v in rowvals) + "\n")


def _cluster_order(corr):
    """Order filters so co-activating ones sit together in the heatmap.

    Prefers SciPy average-linkage hierarchical clustering on ``1 - corr``.
    Without SciPy, falls back to numpy-only spectral (Fiedler-vector) seriation,
    then to the identity order.
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # enforce symmetry for squareform
        link = linkage(squareform(dist, checks=False), method="average")
        return list(leaves_list(link))
    except Exception as exc:  # scipy missing or degenerate distances
        logger.warning(f"SciPy clustering unavailable ({exc}); using spectral order.")

    # Fiedler-vector seriation: order by the eigenvector of the graph Laplacian
    # with the 2nd-smallest eigenvalue, on the non-negative similarity (corr+1)/2.
    try:
        sim = (corr + 1.0) / 2.0
        np.fill_diagonal(sim, 0.0)
        laplacian = np.diag(sim.sum(axis=1)) - sim
        _, eigvecs = np.linalg.eigh(laplacian)
        fiedler = eigvecs[:, 1]  # second-smallest eigenvalue
        return list(np.argsort(fiedler))
    except Exception as exc:
        logger.warning(f"Spectral seriation failed ({exc}); using original order.")
        return list(range(corr.shape[0]))


def _plot_heatmap(mat, labels, path, title):
    """Save a diverging correlation heatmap; skip cleanly if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"matplotlib unavailable ({exc}); skipping heatmap PNG.")
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.34 * n + 2, 0.34 * n + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
