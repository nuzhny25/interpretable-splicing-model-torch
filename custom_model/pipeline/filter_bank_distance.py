"""Pairwise filter-bank distances across the pipeline runs: seed effect vs gene effect.

Every ``pipeline/runs/<gene>_f<F>_k<K>_s<S>/`` directory holds one trained filter bank.
Asking whether the *gene* or the *seed* drives what a run learns means comparing banks
to each other, and the obstacle is that a bank is an unordered *set*: ``incl_03`` in one
run has no relationship to ``incl_03`` in another. The model has two exact symmetries,
and a distance that ignores either one mostly measures the symmetry rather than the
biology:

  * **Filter permutation.** Relabelling the filters within a bank leaves the model
    unchanged, so the distance matches filters optimally (Hungarian assignment) instead
    of comparing them index by index.
  * **Bank swap.** ``custom_model.py`` builds ``sr = a_incl - a_skip`` and the loss in
    ``run_pipeline.py`` is a mean *absolute* deviation about each alignment column's
    cross-species mean, so negating the whole SR profile leaves it exactly invariant --
    as does the orthogonality penalty, which sums ``|r|`` over unordered channel pairs.
    Swapping the two banks wholesale is therefore a symmetry of the objective, and the
    distance takes the better of (incl->incl, skip->skip) and (incl->skip, skip->incl).

Filters are compared in the same gauge the readouts use: ``plot_filters.py`` centers each
kernel column on its median over the four nucleotides, because a constant added down a
column is absorbed by the bias and carries no discriminative power. Centered filters are
L2-normalized and compared by cosine, so the per-filter distance ``1 - cos`` lives in
[0, 2] and ignores overall filter magnitude. ``--max-shift`` additionally allows a motif
to be matched at an offset inside the kernel (the comparison is renormalized over the
overlap), which matters when a short motif sits somewhere inside a wide kernel.

Runs are only comparable at equal kernel size, and mixing filter counts makes the mean
matched distance mean different things on either side (a 5-filter bank matched into a
20-filter bank ignores 15 of them), so the unit of analysis is one **architecture group**:
the 6 genes x 6 seeds = 36 runs sharing an ``(F, K)``. Each group yields a full 36x36
distance matrix and its 630 pairs split into exactly three labelled groups:

  * ``same gene, diff seed``   (90 pairs)  -- how much the seed alone moves the bank
  * ``diff gene, same seed``   (90 pairs)  -- how much the gene alone moves it
  * ``diff gene, diff seed``   (450 pairs) -- both at once

Reading the result: if the first two distributions overlap, seed and gene matter about
equally. The third group is the reference for "everything differs". The second vs third
contrast is a free bonus: ``run_pipeline.py`` calls ``torch.manual_seed`` before the model
is constructed and the only other RNG consumer is the per-epoch species ``randperm``, so
at a fixed ``(F, K)`` a seed gives *identical initial weights across all six genes* --
if "diff gene, same seed" sits below "diff gene, diff seed", the initialization basin is
still visible in the trained filters.

    python pipeline/filter_bank_distance.py
    python pipeline/filter_bank_distance.py --max-shift 2 --out-dir /tmp/shifted

Outputs land in ``pipeline/analysis/filter_bank_distance/``:

    distances.npz                 per-arch distance matrices + run labels
    pairs.csv                     one row per pair (arch, genes, seeds, distance, swapped)
    summary.txt                   per-arch group means and the two contrasts
    pair_distance_histograms.png  the three distributions, one panel per architecture

Note that this measures *learned filters*, not held-out predictive behaviour, and that a
"gene" effect here is inseparable from that gene's alignment composition -- the species
sets and depths differ per gene -- so a large gene effect is a dataset effect until a
species-matched rerun says otherwise. It also has no null floor built in: to know what
"no signal" looks like, point ``--runs-dir`` at a directory of shuffled-alignment runs
and compare the same-gene/diff-seed distribution against it.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
ROOT = os.path.abspath(os.path.join(CUSTOM_DIR, ".."))
# Same ordering rationale as run_pipeline.py: "custom_model" names both the package
# directory and the module inside it, and only this order resolves to the module.
sys.path.insert(0, ROOT)
sys.path.insert(0, CUSTOM_DIR)

from custom_model import PNASModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HERE = _HERE
DEFAULT_RUNS_DIR = os.path.join(HERE, "runs")
DEFAULT_OUT_DIR = os.path.join(HERE, "analysis", "filter_bank_distance")

RUN_NAME_RE = re.compile(r"^(?P<gene>.+?)_f(?P<f>\d+)_k(?P<k>\d+)_s(?P<s>\d+)$")

# Categorical slots 1-3 of the validated default palette. The first three slots are the
# ones that clear the all-pairs colorblind floors, and three is all this figure needs.
GROUP_STYLE = {
    "same gene, diff seed": "#2a78d6",  # blue
    "diff gene, same seed": "#eb6834",  # orange
    "diff gene, diff seed": "#1baf7a",  # aqua
}
GRID_COLOR = "#d8d7d2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


# ── Optimal assignment ───────────────────────────────────────────────────────────────
def lap_min(cost):
    """Minimum-cost assignment via the Jonker-Volgenant shortest-augmenting-path method.

    scipy is not in this project's environment and the matrices here are tiny (at most
    20x20), so the O(n^3) algorithm is implemented directly rather than pulled in.

    Args:
        cost: ``(n, m)`` cost array. The smaller side is matched in full; when
            ``n > m`` the matrix is transposed internally and the result mapped back.

    Returns:
        ``(rows, cols)`` integer arrays of length ``min(n, m)``, the matched pairs.
    """
    cost = np.asarray(cost, dtype=float)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    # Column 0 is a sentinel: potentials u/v are 1-indexed over rows/columns, p[j] is the
    # 1-based row matched to column j (0 = free), and way[j] records the augmenting path.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            # Reduced cost of reaching each still-free column from the current row.
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0

            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]

            # Shift the potentials so the chosen column becomes tight. Rows reachable
            # through used columns are distinct (p[0] = i is unmatched), so this is a
            # safe vectorized update rather than an accumulate.
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Walk the recorded path back to the sentinel, flipping matched edges.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    matched = np.nonzero(p[1:])[0]  # 0-based columns that ended up matched
    rows = p[1:][matched] - 1
    cols = matched
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


# ── Loading and gauge ────────────────────────────────────────────────────────────────
def gauge_normalize(w, eps=1e-9):
    """Median-center each kernel column, then L2-normalize each filter.

    ``w`` is ``(F, 4, K)``. Centering matches ``plot_filters.py``: a constant added to all
    four nucleotides at one position is absorbed by the bias, so it is gauge and must not
    contribute to a distance. Normalizing afterwards makes the comparison about motif
    shape rather than filter magnitude.

    Returns:
        ``(normalized, degenerate)`` -- the ``(F, 4, K)`` unit filters, and a boolean mask
        marking filters whose centered norm was ~0 (no motif preference at all; their
        cosine against anything is defined as 0 below).
    """
    w = np.asarray(w, dtype=float)
    w_c = w - np.median(w, axis=1, keepdims=True)
    norms = np.sqrt((w_c**2).sum(axis=(1, 2)))
    degenerate = norms < eps
    w_n = w_c / np.where(degenerate, 1.0, norms)[:, None, None]
    w_n[degenerate] = 0.0
    return w_n, degenerate


def load_run(run_dir):
    """Read one run's two filter banks plus its gene / architecture / seed labels.

    Labels come from ``run_config.json`` where it exists (the directory prefix is the
    data directory, not the gene -- MALAT1 lives in ``multiz100/``), falling back to the
    run-name prefix otherwise.
    """
    name = os.path.basename(run_dir.rstrip(os.sep))
    m = RUN_NAME_RE.match(name)
    if m is None:
        return None
    ckpt = torch.load(
        os.path.join(run_dir, "weights.pt"), map_location="cpu", weights_only=False
    )
    sd = ckpt["model_state_dict"]

    label = m.group("gene")
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            label = json.load(fh)["gene"]["label"]

    incl, incl_deg = gauge_normalize(sd["conv_incl.weight"].numpy())
    skip, skip_deg = gauge_normalize(sd["conv_skip.weight"].numpy())
    return {
        "name": name,
        "gene": label,
        "num_filters": int(m.group("f")),
        "kernel_size": int(m.group("k")),
        "seed": int(m.group("s")),
        "incl": incl,
        "skip": skip,
        "n_degenerate": int(incl_deg.sum() + skip_deg.sum()),
    }


# ── Distances ────────────────────────────────────────────────────────────────────────
def cost_matrix(a, b, max_shift):
    """Per-filter distance ``1 - cos`` between two banks, minimized over kernel offsets.

    Args:
        a: ``(F_a, 4, K)`` gauge-normalized filters.
        b: ``(F_b, 4, K)`` gauge-normalized filters.
        max_shift: largest offset, in nucleotides, at which ``b`` may be slid against
            ``a``. 0 compares kernels position-for-position. At a nonzero shift only the
            overlapping columns are compared, each side renormalized over that overlap,
            so shifted matches are not rewarded for the truncation.

    Returns:
        ``(F_a, F_b)`` distances in [0, 2]; 0 means identical motif shape.
    """
    k = a.shape[2]
    max_shift = min(max_shift, k - 1)
    best = np.full((a.shape[0], b.shape[0]), -1.0)
    for shift in range(-max_shift, max_shift + 1):
        a_lo, b_lo = max(0, shift), max(0, -shift)
        overlap = k - abs(shift)
        a_sub = a[:, :, a_lo : a_lo + overlap].reshape(a.shape[0], -1)
        b_sub = b[:, :, b_lo : b_lo + overlap].reshape(b.shape[0], -1)
        # Renormalize over the overlap; a filter that is all-zero there stays zero, which
        # yields cosine 0 (treated as unrelated) rather than a divide-by-zero.
        a_sub = a_sub / np.maximum(np.linalg.norm(a_sub, axis=1, keepdims=True), 1e-9)
        b_sub = b_sub / np.maximum(np.linalg.norm(b_sub, axis=1, keepdims=True), 1e-9)
        np.maximum(best, a_sub @ b_sub.T, out=best)
    return 1.0 - best


def bank_distance(run_a, run_b, max_shift):
    """Symmetry-aware distance between two runs' filter banks.

    Matches filters optimally within each bank (permutation symmetry) and takes the
    better of the identity and swapped bank pairings (the incl<->skip symmetry of the
    objective). The reported distance is the mean matched per-filter distance, averaged
    over the two banks.

    Returns:
        ``(distance, swapped)`` -- the distance, and whether the swapped pairing won.
    """

    def matched_mean(x, y):
        c = cost_matrix(x, y, max_shift)
        rows, cols = lap_min(c)
        return float(c[rows, cols].mean())

    d_identity = 0.5 * (
        matched_mean(run_a["incl"], run_b["incl"])
        + matched_mean(run_a["skip"], run_b["skip"])
    )
    d_swapped = 0.5 * (
        matched_mean(run_a["incl"], run_b["skip"])
        + matched_mean(run_a["skip"], run_b["incl"])
    )
    if d_swapped < d_identity:
        return d_swapped, True
    return d_identity, False


def untrained_floor(num_filters, kernel_size, n_banks, max_shift):
    """Distance between pairs of *untrained* banks at this architecture: the null floor.

    Without this the distances have no scale. Two freshly initialized banks are as
    unrelated as two banks can be, so their mean matched distance is what "these runs
    learned nothing in common" looks like -- and because the Hungarian matching picks the
    best of F! pairings, that floor sits well below the 1.0 a naive random-vector cosine
    would suggest, and it moves with F and K. A trained group near this line shares no
    more structure than chance; the gap below it is the only real signal.

    Seeds start at 1000 to stay clear of the 0-5 the runs themselves used.

    Returns:
        ``(mean, sd)`` over the ``C(n_banks, 2)`` untrained pairs.
    """
    model_logger = logging.getLogger("custom_model")
    prior = model_logger.level
    model_logger.setLevel(logging.WARNING)  # mute the per-construction parameter count
    banks = []
    for seed in range(1000, 1000 + n_banks):
        torch.manual_seed(seed)
        sd = PNASModel(num_filters=num_filters, kernel_size=kernel_size).state_dict()
        banks.append(
            {
                "incl": gauge_normalize(sd["conv_incl.weight"].numpy())[0],
                "skip": gauge_normalize(sd["conv_skip.weight"].numpy())[0],
            }
        )
    model_logger.setLevel(prior)
    d = np.array([bank_distance(a, b, max_shift)[0] for a, b in combinations(banks, 2)])
    return float(d.mean()), float(d.std(ddof=1))


def pair_group(run_a, run_b):
    """Label a pair by what differs between the two runs, or None for the diagonal."""
    same_gene = run_a["gene"] == run_b["gene"]
    same_seed = run_a["seed"] == run_b["seed"]
    if same_gene and same_seed:
        return None
    if same_gene:
        return "same gene, diff seed"
    return "diff gene, same seed" if same_seed else "diff gene, diff seed"


# ── Figure ───────────────────────────────────────────────────────────────────────────
def plot_histograms(per_arch, path, max_shift):
    """Small multiples: the three pair distributions, one panel per architecture.

    Direct mean labels ride on every panel because the aqua slot sits under 3:1 against
    a white surface -- identity is never carried by color alone here.
    """
    archs = sorted(per_arch, key=lambda a: (a[1], a[0]))  # by kernel, then filters
    ncols = min(4, len(archs))
    nrows = int(np.ceil(len(archs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.1 * ncols, 3.4 * nrows), squeeze=False
    )

    all_d = np.concatenate(
        [g for arch in archs for g in per_arch[arch]["groups"].values() if len(g)]
    )
    bins = np.linspace(all_d.min(), all_d.max(), 34)

    for idx, arch in enumerate(archs):
        ax = axes[idx // ncols][idx % ncols]
        groups = per_arch[arch]["groups"]
        for gname, color in GROUP_STYLE.items():
            d = groups.get(gname)
            if d is None or not len(d):
                continue
            ax.hist(
                d,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                label=gname,
            )
            # ymax stops the rules short of the stats block in the top-right corner.
            ax.axvline(
                d.mean(), color=color, linewidth=1.0, linestyle=":", alpha=0.9, ymax=0.68
            )

        # The null floor: where two untrained banks of this size land. Anything sitting
        # on this line shares no more structure than chance. Its value goes in the stats
        # block rather than beside the line, which collided with the curves.
        floor = per_arch[arch].get("floor")
        if floor is not None:
            ax.axvline(
                floor[0],
                color=TEXT_PRIMARY,
                linewidth=1.4,
                linestyle="--",
                ymax=0.68,
            )

        # The relief the contrast check obligates: every group's mean, spelled out, so
        # identity never rests on the line color alone.
        lines = [
            f"{gname:<21}{groups[gname].mean():.3f}"
            for gname in GROUP_STYLE
            if len(groups.get(gname, ()))
        ]
        if floor is not None:
            lines.append(f"{'chance floor':<21}{floor[0]:.3f}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)  # headroom so the block clears the curves
        ax.text(
            0.97,
            0.96,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            family="monospace",
            color=TEXT_SECONDARY,
        )

        ax.set_title(
            f"{arch[0]}+{arch[0]} filters, kernel {arch[1]} nt",
            fontsize=10,
            color=TEXT_PRIMARY,
        )
        ax.set_xlabel("matched filter-bank distance (1 - cos)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)

    for idx in range(len(archs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles = [
        plt.Line2D([], [], color=c, linewidth=2.0, label=g)
        for g, c in GROUP_STYLE.items()
    ] + [
        plt.Line2D(
            [],
            [],
            color=TEXT_PRIMARY,
            linewidth=1.4,
            linestyle="--",
            label="untrained-bank floor (chance)",
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )
    shift_note = (
        "kernels compared position-for-position"
        if max_shift == 0
        else f"best match over offsets up to +/-{max_shift} nt"
    )
    fig.suptitle(
        "Does the seed or the gene move the learned filters more?\n"
        f"pairwise filter-bank distance within each architecture ({shift_note})",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="Directory of <gene>_f<F>_k<K>_s<S> run directories.",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help="Where to write the four outputs."
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=0,
        help="Allow a motif match at an offset of up to this many nucleotides inside "
        "the kernel (default 0: compare kernels position-for-position).",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=4,
        help="Skip architecture groups with fewer runs than this.",
    )
    parser.add_argument(
        "--null-banks",
        type=int,
        default=16,
        help="Untrained banks per architecture used for the chance-level distance floor "
        "(0 disables the floor).",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load every run, grouped by architecture ──────────────────────────────────────
    run_dirs = sorted(
        os.path.join(args.runs_dir, n)
        for n in os.listdir(args.runs_dir)
        if RUN_NAME_RE.match(n)
        and os.path.exists(os.path.join(args.runs_dir, n, "weights.pt"))
    )
    logger.info(f"Loading {len(run_dirs)} runs from {args.runs_dir}")
    by_arch = {}
    degenerate_total = 0
    for run_dir in run_dirs:
        run = load_run(run_dir)
        if run is None:
            continue
        degenerate_total += run["n_degenerate"]
        by_arch.setdefault((run["num_filters"], run["kernel_size"]), []).append(run)
    if degenerate_total:
        logger.warning(
            f"{degenerate_total} filters had ~zero centered weight (no motif preference); "
            "their cosine against anything is treated as 0."
        )

    # ── Pairwise distances within each architecture ──────────────────────────────────
    per_arch, pair_rows = {}, []
    for arch in sorted(by_arch, key=lambda a: (a[1], a[0])):
        runs = sorted(by_arch[arch], key=lambda r: (r["gene"], r["seed"]))
        if len(runs) < args.min_runs:
            logger.info(f"  f{arch[0]}_k{arch[1]}: only {len(runs)} runs, skipped")
            continue
        n = len(runs)
        dist = np.zeros((n, n))
        groups = {g: [] for g in GROUP_STYLE}
        for i, j in combinations(range(n), 2):
            d, swapped = bank_distance(runs[i], runs[j], args.max_shift)
            dist[i, j] = dist[j, i] = d
            gname = pair_group(runs[i], runs[j])
            if gname is not None:
                groups[gname].append(d)
            pair_rows.append(
                {
                    "num_filters": arch[0],
                    "kernel_size": arch[1],
                    "run_a": runs[i]["name"],
                    "run_b": runs[j]["name"],
                    "gene_a": runs[i]["gene"],
                    "gene_b": runs[j]["gene"],
                    "seed_a": runs[i]["seed"],
                    "seed_b": runs[j]["seed"],
                    "group": gname or "same gene, same seed",
                    "distance": f"{d:.6f}",
                    "swapped": int(swapped),
                }
            )
        per_arch[arch] = {
            "runs": runs,
            "dist": dist,
            "groups": {g: np.asarray(v) for g, v in groups.items()},
            "floor": (
                untrained_floor(arch[0], arch[1], args.null_banks, args.max_shift)
                if args.null_banks >= 2
                else None
            ),
        }
        means = {g: v.mean() for g, v in per_arch[arch]["groups"].items() if len(v)}
        floor = per_arch[arch]["floor"]
        logger.info(
            f"  f{arch[0]}_k{arch[1]}: {n} runs, {n * (n - 1) // 2} pairs | "
            + " | ".join(f"{g}: {m:.3f}" for g, m in means.items())
            + (f" | chance floor: {floor[0]:.3f}" if floor else "")
        )

    if not per_arch:
        logger.error("No architecture group had enough runs; nothing to compare.")
        return

    # ── Write the outputs ────────────────────────────────────────────────────────────
    npz_payload = {}
    for arch, block in per_arch.items():
        tag = f"f{arch[0]}_k{arch[1]}"
        npz_payload[f"{tag}__dist"] = block["dist"]
        npz_payload[f"{tag}__runs"] = np.array([r["name"] for r in block["runs"]])
        npz_payload[f"{tag}__gene"] = np.array([r["gene"] for r in block["runs"]])
        npz_payload[f"{tag}__seed"] = np.array([r["seed"] for r in block["runs"]])
    npz_path = os.path.join(args.out_dir, "distances.npz")
    np.savez_compressed(npz_path, **npz_payload)

    csv_path = os.path.join(args.out_dir, "pairs.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    # The summary table doubles as the table view the contrast check asks for.
    summary = [
        "Pairwise filter-bank distance (1 - cos on gauge-centered filters, optimally "
        "matched,",
        "minimized over the incl<->skip bank swap). 0 = identical banks.",
        f"max-shift: {args.max_shift} nt | runs dir: {args.runs_dir}",
        "",
        f"{'arch':>10} {'group':<22} {'n':>5} {'mean':>8} {'median':>8} {'sd':>8}",
    ]
    for arch in sorted(per_arch, key=lambda a: (a[1], a[0])):
        for gname in GROUP_STYLE:
            d = per_arch[arch]["groups"][gname]
            if not len(d):
                continue
            summary.append(
                f"{f'f{arch[0]}_k{arch[1]}':>10} {gname:<22} {len(d):>5} "
                f"{d.mean():>8.4f} {np.median(d):>8.4f} {d.std(ddof=1):>8.4f}"
            )
        floor = per_arch[arch]["floor"]
        if floor is not None:
            summary.append(
                f"{f'f{arch[0]}_k{arch[1]}':>10} {'[untrained floor]':<22} "
                f"{args.null_banks:>5} {floor[0]:>8.4f} {'':>8} {floor[1]:>8.4f}"
            )
        summary.append("")

    summary += [
        "",
        "Contrasts (per architecture):",
        "  gene-vs-seed  = mean(diff gene, same seed) - mean(same gene, diff seed)",
        "                  > 0 means changing the gene moves the bank more than the seed",
        "  init-residue  = mean(diff gene, diff seed) - mean(diff gene, same seed)",
        "                  > 0 means a shared initialization is still visible after "
        "training",
        "  vs-chance     = mean(same gene, diff seed) - untrained floor",
        "                  how far two runs of the SAME gene beat chance; ~0 means the "
        "seed",
        "                  decides essentially everything the filters end up being",
        "",
        f"{'arch':>10} {'gene-vs-seed':>14} {'init-residue':>14} {'vs-chance':>14}",
    ]
    for arch in sorted(per_arch, key=lambda a: (a[1], a[0])):
        g = per_arch[arch]["groups"]
        floor = per_arch[arch]["floor"]
        gene_vs_seed = g["diff gene, same seed"].mean() - g["same gene, diff seed"].mean()
        init_residue = g["diff gene, diff seed"].mean() - g["diff gene, same seed"].mean()
        vs_chance = (
            f"{g['same gene, diff seed'].mean() - floor[0]:>+14.4f}" if floor else f"{'-':>14}"
        )
        summary.append(
            f"{f'f{arch[0]}_k{arch[1]}':>10} {gene_vs_seed:>+14.4f} "
            f"{init_residue:>+14.4f} {vs_chance}"
        )

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary) + "\n")
    logger.info("\n" + "\n".join(summary))

    png_path = os.path.join(args.out_dir, "pair_distance_histograms.png")
    plot_histograms(per_arch, png_path, args.max_shift)

    logger.info(
        f"\nWrote:\n  {npz_path}\n  {csv_path}\n  {summary_path}\n  {png_path}"
    )


if __name__ == "__main__":
    main()
