"""K-mer response correlation: do trained models respond to sequence alike?

``filter_bank_distance.py`` compares runs by their filter *weights*. This script compares
them by what they *compute*: every one of the 4**K k-mers is fed through each trained
model, and the responses are correlated across models. The convolution spans the whole
kernel on one-hot input, so a model's response to any window is a pure function of that
window's k-mer -- the 4**K rows are the model's entire input space, weighted uniformly.

Two levels, each compared two ways, give four bars per architecture:

  1. Filters, MALAT1 across seeds.   Table: 4**K rows x (2F filters x 6 seeds) columns,
     each cell ``softplus(w . x + b)``. |Pearson r| between every pair of columns that
     belong to *different* seeds (pairs inside one model are excluded), averaged.
  2. Filters, seed 0 across genes.   The same, with the six genes' seed-0 runs as the
     models.
  3. Whole model, MALAT1 across seeds.   Table: 4**K rows x 6 columns, each cell the
     model's net SR on that k-mer (``PNASModel.compute_sr_profile``: sum of softplus over
     the incl filters minus the skip filters). Mean |r| over the 15 column pairs.
  4. Whole model, seed 0 across genes.

All four bars average |r|. For the model bars this is required: the training loss is
invariant to SR -> -SR (the incl and skip banks can swap roles), so two runs can learn
the same function with opposite sign. The filter bars follow suit, which is the same
reduction the orthogonality penalty applies to its filter pairs (mean |r|): an
anti-correlated filter pair counts as related rather than cancelling a correlated one.

A dashed tick on every bar marks the same quantity computed on untrained models (seeds
1000+, clear of the 0-5 the runs used): what "no learned agreement" looks like at this
architecture.

Read the across-genes bars with one confound in mind: ``run_pipeline.py`` seeds before
building the model, so all six genes at one seed start from *identical* weights, while
the six MALAT1 seeds start from different ones. Some of any across-genes similarity is
that shared starting point rather than biology.

    python pipeline/kmer_response_correlation.py
    python pipeline/kmer_response_correlation.py --configs f10_k6 --seed 1

Outputs land in ``pipeline/analysis/kmer_response_correlation/``:

    kmer_response_correlation_<config>.png   one figure per config: a filter panel and a
                                             whole-model panel, two bars each
    summary.csv                              one row per (config, bar): mean, sd, chance
    pairs.csv                                one row per model pair per bar
"""

import argparse
import csv
import itertools
import json
import logging
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(_HERE, ".."))
ROOT = os.path.abspath(os.path.join(CUSTOM_DIR, ".."))
# Same ordering rationale as run_pipeline.py: "custom_model" names both the package
# directory and the module inside it, and only this order resolves to the module.
sys.path.insert(0, ROOT)
sys.path.insert(0, CUSTOM_DIR)

from custom_model import PNASModel  # noqa: E402
from custom_utils import DNA_ALPHABET, str_to_vector  # noqa: E402
from run_pipeline import GENES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# Every PNASModel construction logs its parameter count; 22 of them per config is noise.
logging.getLogger("custom_model").setLevel(logging.WARNING)

DEFAULT_RUNS_DIR = os.path.join(_HERE, "runs")
DEFAULT_OUT_DIR = os.path.join(_HERE, "analysis", "kmer_response_correlation")

CONFIG_RE = re.compile(r"^f(?P<f>\d+)_k(?P<k>\d+)$")
REFERENCE_GENE = "malat1"
NULL_SEED_START = 1000

# Categorical slots 1-2 of the validated default palette, the same two that
# filter_bank_distance.py uses. Color carries the comparison; the level (filters vs whole
# model) is carried by position and the group labels under the axis.
SEEDS_COLOR = "#2a78d6"  # blue
GENES_COLOR = "#eb6834"  # orange
GRID_COLOR = "#d8d7d2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


# ── Loading ──────────────────────────────────────────────────────────────────────────
def load_model(runs_dir, gene, num_filters, kernel_size, seed):
    """Build ``PNASModel`` at this size and load one run's checkpoint into it.

    Returns:
        ``(model, label)`` -- the model in eval mode, and the gene's display label
        (read from ``run_config.json``, falling back to the ``GENES`` table).
    """
    gene_dir, label = GENES[gene]
    name = f"{gene_dir}_f{num_filters}_k{kernel_size}_s{seed}"
    run_dir = os.path.join(runs_dir, name)
    weights_path = os.path.join(run_dir, "weights.pt")
    if not os.path.exists(weights_path):
        raise SystemExit(f"Missing run: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = PNASModel(num_filters=num_filters, kernel_size=kernel_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            label = json.load(fh)["gene"]["label"]
    return model, label


# ── Response tables ──────────────────────────────────────────────────────────────────
def response_tables(models, x):
    """Every model's per-filter and net-SR response to every k-mer.

    Args:
        models: list of ``PNASModel``, all at one architecture.
        x: ``(4**K, 4, K)`` one-hot k-mers.

    Returns:
        ``(filters, net)`` -- ``(4**K, n_models * 2F)`` with each model's incl filters
        then its skip filters as adjacent columns, and ``(4**K, n_models)`` net SR.
        Both float64 for the correlation.
    """
    filter_cols, net_cols = [], []
    with torch.no_grad():
        for model in models:
            a_incl = F.softplus(model.conv_incl(x)).squeeze(-1)  # (4**K, F), >= 0
            a_skip = F.softplus(model.conv_skip(x)).squeeze(-1)  # (4**K, F), >= 0
            net = model.compute_sr_profile(x).squeeze(-1)  # (4**K,)
            # The whole-model column must be exactly the filter columns' incl-minus-skip
            # sum; if it is not, the two levels are not measuring the same model.
            if not torch.allclose(
                net, a_incl.sum(1) - a_skip.sum(1), rtol=1e-5, atol=1e-4
            ):
                raise RuntimeError("compute_sr_profile disagrees with the filter sum.")
            filter_cols.append(torch.cat([a_incl, a_skip], dim=1))
            net_cols.append(net)
    filters = torch.cat(filter_cols, dim=1).numpy().astype(np.float64)
    net = torch.stack(net_cols, dim=1).numpy().astype(np.float64)
    return filters, net


def corr(table, what):
    """Pearson correlation between the columns of ``table``, refusing non-finite output."""
    r = np.corrcoef(table, rowvar=False)
    if not np.all(np.isfinite(r)):
        # A column with zero variance over all k-mers (a filter that responds to nothing)
        # has no defined correlation; say so instead of letting NaN reach the mean.
        raise RuntimeError(f"{what}: a column has zero variance across k-mers.")
    return r


def filter_pair_values(filters, n_models, n_channels):
    """Mean filter-filter |r| for each pair of *different* models.

    Every block is ``n_channels x n_channels``, so the mean over the per-pair block
    means equals the mean over every individual cross-model filter pair.
    """
    r = np.abs(corr(filters, "filter table"))
    out = {}
    for a, b in itertools.combinations(range(n_models), 2):
        block = r[
            a * n_channels : (a + 1) * n_channels, b * n_channels : (b + 1) * n_channels
        ]
        out[(a, b)] = float(block.mean())
    return out


def net_pair_values(net):
    """|r| of net SR for each pair of models."""
    r = np.abs(corr(net, "net SR table"))
    return {
        (a, b): float(r[a, b])
        for a, b in itertools.combinations(range(net.shape[1]), 2)
    }


# ── Figure ───────────────────────────────────────────────────────────────────────────
def plot(res, path, seed):
    """One config's figure: a filter panel and a whole-model panel, each on its own axis.

    The filter means are small and tightly bunched next to the whole-model ones -- every
    filter is averaged against the unrelated filters of the other model as well as its
    counterpart -- so the filter panel is scaled to its own data rather than sharing the
    whole-model axis.
    """
    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(9.6, 5.0))
    width = 0.7
    rng = np.random.default_rng(0)  # jitter only; fixed so reruns draw identically
    panels = (
        (
            ax_f,
            res["bars"][:2],
            "{:.3f}",
            f"Filters: {res['n_cols']} columns "
            f"({2 * res['num_filters']} per model x 6)",
            "mean |Pearson r| over every\ncross-model filter pair",
        ),
        (
            ax_m,
            res["bars"][2:],
            "{:.2f}",
            "Whole model: net SR, 6 columns",
            "mean |Pearson r| of net SR\nacross model pairs",
        ),
    )

    for ax, bars, fmt, title, ylabel in panels:
        for x, bar in enumerate(bars):
            color = SEEDS_COLOR if bar["axis"] == "seeds" else GENES_COLOR
            vals = np.array(list(bar["pairs"].values()))
            mean = vals.mean()
            ax.bar(x, mean, width=width, color=color, zorder=2)
            ax.scatter(
                x + rng.uniform(-0.18, 0.18, len(vals)),
                vals,
                s=14,
                color=TEXT_PRIMARY,
                alpha=0.75,
                linewidths=0,
                zorder=3,
            )
            ax.hlines(
                bar["chance"],
                x - width / 2,
                x + width / 2,
                colors=TEXT_PRIMARY,
                linestyles="--",
                linewidth=1.4,
                zorder=4,
            )
            # The mean rides above whichever is higher, the bar or its highest dot, so
            # the label never sits on top of a point. Offset in points, not data units,
            # because the two panels have very different y-ranges.
            ax.annotate(
                fmt.format(mean),
                (x, max(mean, vals.max())),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT_PRIMARY,
            )

        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(
            ["MALAT1\n6 seeds", f"seed {seed}\n6 genes"], fontsize=9, color=TEXT_PRIMARY
        )
        ax.set_xlim(-0.7, len(bars) - 0.3)
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8, colors=TEXT_SECONDARY)
        ax.tick_params(axis="x", length=0)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)

    # Both panels start at 0 (every value is an |r|), so the two baselines sit on one
    # line. The filter panel's top fits its highest dot with headroom for the labels;
    # the whole-model panel keeps the full |r| scale, so 1.0 always means identical.
    top_f = max(
        max(b["pairs"].values()) for b in res["bars"][:2]
    )
    ax_f.set_ylim(0, top_f * 1.2)
    ax_m.set_ylim(0, 1.08)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SEEDS_COLOR, label="across seeds (MALAT1)"),
        plt.Rectangle(
            (0, 0), 1, 1, color=GENES_COLOR, label=f"across genes (seed {seed})"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=4,
            color=TEXT_PRIMARY,
            label="one model pair",
        ),
        plt.Line2D(
            [],
            [],
            color=TEXT_PRIMARY,
            linestyle="--",
            linewidth=1.4,
            label="untrained models (chance)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.005,
        f"All six genes' seed-{seed} runs start from identical weights (the MALAT1 seeds "
        "do not), so part of any across-genes similarity is the shared starting point.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.suptitle(
        f"{res['num_filters']}+{res['num_filters']} filters, kernel "
        f"{res['kernel_size']} nt: do trained models respond to k-mers alike?\n"
        f"Pearson r over all {res['n_kmers']:,} k-mers",
        fontsize=12,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["f20_k6", "f20_k8"],
        help="Architectures as f<filters>_k<kernel>. Default: f20_k6 f20_k8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed whose runs are compared across genes. Default: 0.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="MALAT1 seeds compared against each other. Default: 0-5.",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=6,
        help="Untrained models per config for the chance tick. Default: 6.",
    )
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    genes = list(GENES)  # malat1, neat1, hotair, il6, vegfa, xist
    results, pair_rows, summary_rows = {}, [], []

    for config in args.configs:
        m = CONFIG_RE.match(config)
        if m is None:
            raise SystemExit(f"--configs entry {config!r} is not f<filters>_k<kernel>.")
        num_filters, kernel_size = int(m.group("f")), int(m.group("k"))
        n_channels = 2 * num_filters

        # Every k-mer, one-hot, built once per kernel size (4096 at k6, 65536 at k8).
        kmers = [
            "".join(p) for p in itertools.product(DNA_ALPHABET, repeat=kernel_size)
        ]
        x = torch.tensor(
            np.stack([str_to_vector(k) for k in kmers]), dtype=torch.float32
        )  # (4**K, 4, K)

        seed_models = [
            load_model(args.runs_dir, REFERENCE_GENE, num_filters, kernel_size, s)[0]
            for s in args.seeds
        ]
        gene_loaded = [
            load_model(args.runs_dir, g, num_filters, kernel_size, args.seed)
            for g in genes
        ]
        gene_models = [mdl for mdl, _ in gene_loaded]
        gene_labels = [label for _, label in gene_loaded]
        null_models = []
        for s in range(NULL_SEED_START, NULL_SEED_START + args.n_null):
            torch.manual_seed(s)
            null_models.append(
                PNASModel(num_filters=num_filters, kernel_size=kernel_size).eval()
            )

        # One chance value per level: for untrained models "across seeds" and "across
        # genes" are the same comparison.
        null_f, null_net = response_tables(null_models, x)
        chance_filter = float(
            np.mean(list(filter_pair_values(null_f, args.n_null, n_channels).values()))
        )
        chance_net = float(np.mean(list(net_pair_values(null_net).values())))

        seed_f, seed_net = response_tables(seed_models, x)
        gene_f, gene_net = response_tables(gene_models, x)
        seed_names = [f"s{s}" for s in args.seeds]

        bars = [
            {
                "level": "filters",
                "axis": "seeds",
                "names": seed_names,
                "pairs": filter_pair_values(seed_f, len(seed_models), n_channels),
                "chance": chance_filter,
            },
            {
                "level": "filters",
                "axis": "genes",
                "names": gene_labels,
                "pairs": filter_pair_values(gene_f, len(gene_models), n_channels),
                "chance": chance_filter,
            },
            {
                "level": "whole model",
                "axis": "seeds",
                "names": seed_names,
                "pairs": net_pair_values(seed_net),
                "chance": chance_net,
            },
            {
                "level": "whole model",
                "axis": "genes",
                "names": gene_labels,
                "pairs": net_pair_values(gene_net),
                "chance": chance_net,
            },
        ]
        results[config] = {
            "bars": bars,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "n_kmers": len(kmers),
            "n_cols": n_channels * len(seed_models),
        }

        logger.info(f"{config}: {len(kmers):,} k-mers")
        for bar in bars:
            vals = np.array(list(bar["pairs"].values()))
            logger.info(
                f"  {bar['level']:<12} across {bar['axis']:<6} "
                f"mean {vals.mean():+.4f}  sd {vals.std(ddof=1):.4f}  "
                f"chance {bar['chance']:+.4f}"
            )
            summary_rows.append(
                {
                    "config": config,
                    "level": bar["level"],
                    "across": bar["axis"],
                    "metric": "|r|",
                    "n_model_pairs": len(vals),
                    "mean": f"{vals.mean():.6f}",
                    "sd": f"{vals.std(ddof=1):.6f}",
                    "chance": f"{bar['chance']:.6f}",
                }
            )
            for (a, b), v in bar["pairs"].items():
                pair_rows.append(
                    {
                        "config": config,
                        "level": bar["level"],
                        "across": bar["axis"],
                        "model_a": bar["names"][a],
                        "model_b": bar["names"][b],
                        "value": f"{v:.6f}",
                    }
                )

    for fname, rows in (("summary.csv", summary_rows), ("pairs.csv", pair_rows)):
        with open(os.path.join(args.out_dir, fname), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for config, res in results.items():
        plot(
            res,
            os.path.join(args.out_dir, f"kmer_response_correlation_{config}.png"),
            args.seed,
        )
    logger.info(
        f"\nWrote to {args.out_dir}: summary.csv, pairs.csv, "
        + ", ".join(f"kmer_response_correlation_{c}.png" for c in results)
    )


if __name__ == "__main__":
    main()
