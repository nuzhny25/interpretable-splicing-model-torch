"""Cross-run comparison for a pipeline sweep: which filter-bank config actually wins?

run_pipeline.py trains one model per (gene, num_filters, kernel_size, seed) and every
existing plotter reads exactly one run. That leaves the question the sweep was run to
answer -- which architecture is best, and is the difference real or seed noise -- with no
readout at all. This script walks pipeline/runs/, averages each configuration's final loss
over its seeds, and renders the comparison.

Three things drive the design, all of them measured from the runs rather than assumed:

  * The ranking flips depending on which loss you rank by. ``aad_loss`` is the fit term
    alone, so it falls monotonically with filter count -- more capacity, lower deviation.
    The total loss adds the orthogonality penalty, which *grows* with filter count because
    it averages |Pearson r| over the C(2F, 2) filter pairs. On the current sweep the best
    config by aad_loss (f20_k6) is only 4th by total loss, and the best by total loss
    (f5_k6) is 5th by aad_loss. Selection here is by --metric (aad_loss by default), but
    every bar shows the full aad_loss + ortho decomposition so the trade is never hidden.

  * The gaps at the top are smaller than the seed-to-seed spread. The global 1st-vs-2nd
    gap is ~0.0004 against a seed sd of ~0.004-0.010, and the same holds inside every
    gene. A bar chart that just orders the configs would imply a winner that six seeds
    cannot actually resolve, so each top-n panel carries its explicit delta against the
    two configs' seed sds, and the pooled panel carries a +-1 sd tie band.

  * Genes differ from each other far more than configs differ within a gene -- the gene
    means span ~0.10 while the configs inside one gene span ~0.02. Per-gene panels
    therefore keep their own y-limits (a shared axis would compress every comparison the
    figure exists to make) and state their level in the panel title instead.

The loss identity ``loss = aad_loss + ortho + l1`` holds exactly -- ``ortho`` and ``l1`` are
stored already multiplied by their lambdas -- so a stacked bar sums to the total with no
residual. Following plot/plot_training_curves.py, a term that is identically zero across
the whole sweep is dropped from the figures and the CSV rather than drawn flat against the
axis; with --l1-lambda 0 (the pipeline default) that silently removes the l1 band.

``aad_loss`` is the display name for the quantity training records under the key
``sd_loss``; the rename happens at the load boundary so the internal name cannot reach a
column header, an axis label or a legend.

Outputs (all into --out-dir):

    compare_overview.png        all configs, seed-averaged: pooled panel + one per gene
    compare_top2_global.png     the globally best --top-n configs, broken down per gene
    compare_top2_per_gene.png   each gene's own best --top-n configs, broken down
    compare_runs_summary.csv    per gene x config: mean/sd over seeds, ranks, tie flags
    compare_runs_pooled.csv     the pooled-across-genes numbers the figures are drawn from

Run:

    python pipeline/compare_runs.py
    python pipeline/compare_runs.py --top-n 3
    python pipeline/compare_runs.py --metric loss
    python pipeline/compare_runs.py --genes MALAT1 NEAT1 --out-dir /tmp/cmp
    python pipeline/compare_runs.py --configs f20_k8 f20_k6 f10_k8 f10_k6 f10_k3 f5_k6 f5_k3
"""

import argparse
import json
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Copied verbatim from plot/plot_training_curves.py so these figures read against the
# per-run training curves: the same blue is always aad_loss, the same purple always ortho.
COLOR_SD = "steelblue"
COLOR_L1 = "darkorange"
COLOR_LOSS = "black"
COLOR_SCALE = "seagreen"
COLOR_ORTHO = "rebeccapurple"

# Declaration order of GENES in run_pipeline.py, which is also the section order of
# runs.txt. Labels found on disk but missing from this list are appended in sorted order,
# so a newly swept gene shows up rather than silently vanishing from the figures.
GENE_ORDER = ["MALAT1", "NEAT1", "HOTAIR", "IL6", "VEGFA", "XIST"]

# Below this, a seed sd is not drawable: the two error-bar caps land on top of each other
# and read as "the error bar is missing". f5_k1 converges to the same solution from every
# seed (sd ~1e-5), which is a real finding about kernel size 1, not a plotting bug -- so
# those bars get a printed annotation instead of an invisible whisker.
SD_FLOOR = 1e-4

# A stacked segment thinner than this fraction of the axis cannot hold its own value
# label; the label goes outside the bar instead.
THIN_BAND_FRAC = 0.06


def load_run(run_dir):
    """Return one run's final loss components + hyperparameters, or None if unreadable.

    plots/plot_metadata.json is the cheapest source -- plain JSON holding a verbatim copy
    of the checkpoint metadata, so 288 runs load in well under a second with no torch and
    no numpy unpickling. history.npz is the fallback for a run whose plotting stage never
    ran; weights.pt is deliberately not consulted, since importing torch to read seven
    floats would dominate the runtime of the whole script.
    """
    name = os.path.basename(run_dir)
    hparams, final = None, None

    meta_path = os.path.join(run_dir, "plots", "plot_metadata.json")
    if os.path.exists(meta_path):
        try:
            md = json.load(open(meta_path)).get("metadata", {})
            hparams, final = md.get("hparams", {}), md.get("final", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"  {name}: unreadable plot_metadata.json ({exc}); trying npz")

    if final is None:
        # Fallback: the last recorded epoch of the history arrays. A mid-training flush
        # writes the arrays without hparams_json, so every key is guarded individually
        # rather than assuming the npz is complete.
        hist_path = os.path.join(run_dir, "history.npz")
        if not os.path.exists(hist_path):
            logger.warning(f"  {name}: no plot_metadata.json and no history.npz; skipped")
            return None
        try:
            d = np.load(hist_path, allow_pickle=True)
            final = {
                k: float(d[k][-1])
                for k in ("loss", "sd_loss", "ortho", "l1")
                if k in d.files and len(d[k])
            }
            hparams = json.loads(str(d["hparams_json"])) if "hparams_json" in d.files else {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"  {name}: unreadable history.npz ({exc}); skipped")
            return None

    # Identity comes only from the recorded hyperparameters. The directory prefix is NOT
    # the label: malat1 runs live in multiz100_* directories but must plot as "MALAT1".
    gene_dir = hparams.get("gene")
    gene_label = hparams.get("gene_label") or (gene_dir.upper() if gene_dir else None)
    num_filters, kernel_size = hparams.get("num_filters"), hparams.get("kernel_size")
    seed = hparams.get("seed")
    if None in (gene_label, num_filters, kernel_size, seed):
        logger.warning(f"  {name}: incomplete hparams (gene/filters/kernel/seed); skipped")
        return None

    row = {
        "run_name": name,
        "gene_label": gene_label,
        "gene_dir": gene_dir or gene_label,
        "num_filters": int(num_filters),
        "kernel_size": int(kernel_size),
        "seed": int(seed),
        "loss": float(final.get("loss", np.nan)),
        # The rename lives here, and only here: downstream code never sees "sd_loss", so
        # the internal key cannot leak into a CSV column, an axis label or a legend entry.
        "aad_loss": float(final.get("sd_loss", np.nan)),
        "ortho": float(final.get("ortho", 0.0)),
        "l1": float(final.get("l1", 0.0)),
        "num_epochs": hparams.get("num_epochs"),
        "ortho_lambda": hparams.get("ortho_lambda"),
    }
    if not np.isfinite(row["loss"]) or not np.isfinite(row["aad_loss"]):
        logger.warning(f"  {name}: non-finite loss values; skipped")
        return None

    # The stack is only honest if the parts sum to the whole. Warn rather than raise: one
    # odd run should not cost the other 287 their figure.
    residual = abs(row["loss"] - (row["aad_loss"] + row["ortho"] + row["l1"]))
    if residual > 1e-6:
        logger.warning(f"  {name}: loss != aad + ortho + l1 (residual {residual:.2e})")
    return row


def draw_stacked_bars(ax, cells, x, width, draw_l1, label_values=False, emphasise=()):
    """Draw the aad_loss/ortho (/l1) stack for one panel, with seed error bars.

    `cells` is a list of dicts carrying the *_mean and *_sd aggregates in config order.
    Shared by all three figures so the bar anatomy -- segment colours, the two error bars,
    the thin-band label fallback -- is identical everywhere and the eye can carry meaning
    from one figure to the next. Returns the highest drawn y, for setting the y-limit.
    """
    aad = np.array([c["aad_loss_mean"] for c in cells])
    ortho = np.array([c["ortho_mean"] for c in cells])
    total = np.array([c["loss_mean"] for c in cells])
    # NaN sds (a cell with a single seed) draw nothing sane, so they become zero-length
    # whiskers here while the CSV keeps the honest NaN.
    aad_sd = np.nan_to_num(np.array([c["aad_loss_sd"] for c in cells]))
    loss_sd = np.nan_to_num(np.array([c["loss_sd"] for c in cells]))

    # Outlined bars for configs the caller wants traceable across figures (the global
    # top-n), so figure 2's selection can be found again in the overview.
    edges = [COLOR_LOSS if c["config"] in emphasise else "white" for c in cells]
    widths = [1.6 if c["config"] in emphasise else 0.6 for c in cells]

    ax.bar(x, aad, width, color=COLOR_SD, edgecolor=edges, linewidth=widths, alpha=0.9,
           zorder=2)
    ax.bar(x, ortho, width, bottom=aad, color=COLOR_ORTHO, edgecolor=edges,
           linewidth=widths, alpha=0.9, zorder=2)
    if draw_l1:
        l1 = np.array([c["l1_mean"] for c in cells])
        ax.bar(x, l1, width, bottom=aad + ortho, color=COLOR_L1, edgecolor=edges,
               linewidth=widths, alpha=0.9, zorder=2)

    # Two whiskers per bar, both black and unambiguous by position: the lower one sits at
    # the aad_loss top (the comparison that decides the ranking), the upper at the total.
    for xs, ys, sds in ((x, aad, aad_sd), (x, total, loss_sd)):
        drawable = sds >= SD_FLOOR
        if drawable.any():
            ax.errorbar(xs[drawable], ys[drawable], yerr=sds[drawable], fmt="none",
                        ecolor=COLOR_LOSS, elinewidth=1.2, capsize=3, capthick=1.2,
                        zorder=3)

    ymax = float(np.max(total + loss_sd))
    for i, cell in enumerate(cells):
        # A sd below the floor would render as a bar with no whisker at all, which reads
        # as missing data; say so in text instead.
        if aad_sd[i] < SD_FLOOR:
            # Sits well above the #n rank badge (drawn at +6pt by the overview) so the two
            # annotations stack rather than overprint.
            ax.annotate(f"sd < {SD_FLOOR:g}", xy=(x[i], total[i]), xytext=(0, 18),
                        textcoords="offset points", ha="center", fontsize=6,
                        color="dimgray", zorder=4)
        if not label_values:
            continue
        # Zero-based stacking costs resolution when bars differ by 0.005 on a 0.45 axis,
        # so every quantity the eye cannot measure is printed.
        ax.text(x[i], aad[i] / 2, f"{aad[i]:.4f}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)
        if ortho[i] >= THIN_BAND_FRAC * ymax:
            ax.text(x[i], aad[i] + ortho[i] / 2, f"{ortho[i]:.4f}", ha="center",
                    va="center", color="white", fontsize=8, zorder=4)
        else:
            ax.text(x[i] + width / 2 + 0.04, aad[i] + ortho[i] / 2, f"{ortho[i]:.4f}",
                    ha="left", va="center", color=COLOR_ORTHO, fontsize=8, zorder=4)
        ax.annotate(f"loss = {total[i]:.4f}", xy=(x[i], total[i] + loss_sd[i]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                    color=COLOR_LOSS, zorder=4)
    return ymax


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default=os.path.join(_HERE, "runs"),
                        help="root holding <gene_dir>_f<F>_k<K>_s<S>/ run directories")
    parser.add_argument("--out-dir", default=os.path.join(_HERE, "comparisons"),
                        help="where the figures and CSVs are written")
    parser.add_argument("--top-n", type=int, default=2,
                        help="how many configs to break down per gene panel")
    parser.add_argument("--metric", choices=("aad_loss", "loss"), default="aad_loss",
                        help="ranking criterion; the bars always show the full breakdown")
    parser.add_argument("--genes", nargs="+", default=None,
                        help="restrict to these gene labels (case-insensitive)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="restrict to these configs, e.g. f20_k6 f20_k8")
    parser.add_argument("--sort-configs", choices=("capacity", "metric"), default="capacity",
                        help="x-axis order: descending capacity, or ascending --metric")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metric_mean = f"{args.metric}_mean"
    metric_sd = f"{args.metric}_sd"

    # ── Load every run ──────────────────────────────────────────────────────────────
    rows, skipped = [], 0
    for name in sorted(os.listdir(args.runs_dir)):
        path = os.path.join(args.runs_dir, name)
        # Leading underscore skips the _batch_*.json provenance files for free.
        if not os.path.isdir(path) or name.startswith("_"):
            continue
        row = load_run(path)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    if not rows:
        parser.error(f"no readable runs under {args.runs_dir}")
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} runs ({skipped} skipped) from {args.runs_dir}")

    df["config"] = "f" + df.num_filters.astype(str) + "_k" + df.kernel_size.astype(str)
    if args.genes:
        wanted = {g.upper() for g in args.genes}
        df = df[df.gene_label.str.upper().isin(wanted)]
    if args.configs:
        df = df[df.config.isin(set(args.configs))]
    if df.empty:
        parser.error("no runs left after --genes/--configs filtering")

    # Mirror plot_training_curves.py: a term that is identically zero over the whole sweep
    # is dropped rather than drawn as an invisible band and a dead legend entry. Costs six
    # lines and keeps the script honest if someone later sweeps --l1-lambda.
    draw_l1 = not np.allclose(df["l1"].fillna(0.0), 0.0)
    if not draw_l1:
        logger.info(f"l1 is identically 0 across all {len(df)} runs; dropping the term "
                    "from every figure and from the CSV")

    # ── Aggregate over seeds ────────────────────────────────────────────────────────
    aggs = {
        "n_seeds": ("seed", "nunique"),
        "seeds": ("seed", lambda s: " ".join(map(str, sorted(s)))),
        "loss_mean": ("loss", "mean"), "loss_sd": ("loss", "std"),
        "aad_loss_mean": ("aad_loss", "mean"), "aad_loss_sd": ("aad_loss", "std"),
        "ortho_mean": ("ortho", "mean"), "ortho_sd": ("ortho", "std"),
    }
    if draw_l1:
        aggs.update({"l1_mean": ("l1", "mean"), "l1_sd": ("l1", "std")})
    agg = df.groupby(
        ["gene_label", "gene_dir", "num_filters", "kernel_size", "config"], as_index=False
    ).agg(**aggs)
    agg["aad_loss_sem"] = agg.aad_loss_sd / np.sqrt(agg.n_seeds)

    genes = [g for g in GENE_ORDER if g in set(agg.gene_label)]
    genes += sorted(set(agg.gene_label) - set(genes))
    if args.sort_configs == "capacity":
        # Structural order, deliberately independent of the data: the same x slot means
        # the same config in all seven panels, which is what makes them comparable. The
        # ranking is carried by the #n badges instead of by position.
        configs = [c for _, _, c in sorted(
            {(r.num_filters, r.kernel_size, r.config) for r in agg.itertuples()},
            key=lambda t: (-t[0], -t[1]))]
    else:
        configs = list(agg.groupby("config")[metric_mean].mean().sort_values().index)

    # Fast lookup: cell[(gene, config)] -> the aggregate row as a plain dict.
    cell = {(r["gene_label"], r["config"]): r for r in agg.to_dict("records")}

    # ── Pool across genes ───────────────────────────────────────────────────────────
    # The design is balanced (every gene x config cell has the same seeds), so the pooled
    # mean is just the mean of the per-gene means. The spread is the trap: np.std over the
    # raw values for a config is ~0.035 and is dominated by WHICH GENE each run came from
    # (gene levels span ~0.10, configs within a gene span ~0.02), so it would say nothing
    # about whether two configs are separable and would make them all look identical.
    # Remove the gene offsets first by pooling the within-gene seed variances.
    pooled = []
    for c in configs:
        cells = [cell[(g, c)] for g in genes if (g, c) in cell]
        if not cells:
            continue
        rec = {"config": c,
               "num_filters": cells[0]["num_filters"],
               "kernel_size": cells[0]["kernel_size"],
               "n_genes": len(cells),
               "n_runs": int(sum(x["n_seeds"] for x in cells))}
        for term in ("loss", "aad_loss", "ortho"):
            means = np.array([x[f"{term}_mean"] for x in cells])
            sds = np.nan_to_num(np.array([x[f"{term}_sd"] for x in cells]))
            rec[f"{term}_mean"] = float(means.mean())
            rec[f"{term}_sd"] = float(np.sqrt((sds ** 2).mean()))  # pooled seed sd
            rec[f"{term}_between_gene_sd"] = float(means.std(ddof=1)) if len(means) > 1 else np.nan
        pooled.append(rec)
    pooled_df = pd.DataFrame(pooled)
    pooled_rank = {r["config"]: i + 1 for i, r in enumerate(
        pooled_df.sort_values(metric_mean).to_dict("records"))}
    pooled_by_metric = pooled_df.sort_values(metric_mean).to_dict("records")
    global_top = [r["config"] for r in pooled_by_metric[:args.top_n]]
    pooled_cell = {r["config"]: r for r in pooled}

    # ── Per-gene ranks and tie flags ────────────────────────────────────────────────
    for g in genes:
        cells = [cell[(g, c)] for c in configs if (g, c) in cell]
        best = min(cells, key=lambda r: r[metric_mean])
        for rank, r in enumerate(sorted(cells, key=lambda r: r[metric_mean]), start=1):
            r[f"{args.metric}_rank_in_gene"] = rank
        for rank, r in enumerate(sorted(cells, key=lambda r: r["loss_mean"]), start=1):
            r["loss_rank_in_gene"] = rank
        for r in cells:
            delta = r[metric_mean] - best[metric_mean]
            r[f"delta_{args.metric}_vs_best_in_gene"] = delta
            # "Separable" means the gap clears the larger of the two configs' seed sds.
            # With six seeds this is a weak bar, and on the current sweep even that is not
            # cleared anywhere near the top -- which is the headline the figures carry.
            noise = max(np.nan_to_num(r[metric_sd]), np.nan_to_num(best[metric_sd]))
            r["within_seed_noise_of_best"] = bool(abs(delta) < noise)

    # ── CSVs ────────────────────────────────────────────────────────────────────────
    cols = ["gene_label", "gene_dir", "config", "num_filters", "kernel_size", "n_seeds",
            "seeds", "loss_mean", "loss_sd", "aad_loss_mean", "aad_loss_sd",
            "aad_loss_sem", "ortho_mean", "ortho_sd"]
    if draw_l1:
        cols += ["l1_mean", "l1_sd"]
    cols += [f"{args.metric}_rank_in_gene", "loss_rank_in_gene",
             f"delta_{args.metric}_vs_best_in_gene", "within_seed_noise_of_best"]
    summary = pd.DataFrame([cell[(g, c)] for g in genes for c in configs if (g, c) in cell])
    summary_path = os.path.join(args.out_dir, "compare_runs_summary.csv")
    summary[cols].to_csv(summary_path, index=False, float_format="%.6f")
    logger.info(f"Wrote {len(summary)} gene x config rows to {summary_path}")

    pooled_out = pooled_df.copy()
    pooled_out[f"{args.metric}_rank_global"] = pooled_out.config.map(pooled_rank)
    pooled_out["loss_rank_global"] = pooled_out.loss_mean.rank(method="min").astype(int)
    pooled_out[f"delta_{args.metric}_vs_best"] = (
        pooled_out[metric_mean] - pooled_by_metric[0][metric_mean])
    # Both spreads are carried side by side so a future reader cannot conflate them: the
    # pooled seed sd (~0.004) is the one that says whether configs are separable, the
    # between-gene sd (~0.035) only says that genes sit at different levels.
    pooled_path = os.path.join(args.out_dir, "compare_runs_pooled.csv")
    pooled_out.to_csv(pooled_path, index=False, float_format="%.6f")
    logger.info(f"Wrote {len(pooled_out)} pooled config rows to {pooled_path}")

    # Bar anatomy is shared by all three figures, so these handles are too. The datum
    # lines are drawn only by the overview, so they live in a separate list rather than
    # being advertised by a legend on a figure that does not draw them.
    bar_handles = [Patch(facecolor=COLOR_SD, alpha=0.9, label="aad_loss (fit term)"),
                   Patch(facecolor=COLOR_ORTHO, alpha=0.9, label="ortho (penalty term)")]
    if draw_l1:
        bar_handles.append(Patch(facecolor=COLOR_L1, alpha=0.9, label="l1"))
    bar_handles.append(Line2D([], [], color=COLOR_LOSS, lw=1.2,
                              label="±1 sd over seeds (at the aad_loss top and the total)"))
    datum_handles = [
        Line2D([], [], color=COLOR_SD, ls=":", lw=1.2, label="lowest aad_loss in panel"),
        Line2D([], [], color=COLOR_LOSS, ls=":", lw=1.2, label="lowest total loss in panel")]

    # Stamp facts only when the sweep agrees on them, rather than hardcoding.
    def _unique(col):
        vals = df[col].dropna().unique()
        return vals[0] if len(vals) == 1 else None

    stamp = ", ".join(
        s for s in (f"{len(df)} runs",
                    f"{_unique('num_epochs'):.0f} epochs" if _unique("num_epochs") else None,
                    f"ortho_lambda {_unique('ortho_lambda')}" if _unique("ortho_lambda") else None)
        if s)

    # ── Figure 1: overview of every config ──────────────────────────────────────────
    # The (F, K) grid is ragged -- only 8 of the 12 combinations were ever run -- so the
    # configs are one ordered categorical axis rather than an F x K matrix. A matrix would
    # spend a third of its area on empty cells and invite the eye to interpolate across
    # holes that are missing by design, not by accident.
    # Grid sized from the gene count: a full-width pooled row on top, then as many gene
    # rows as needed. Slicing to a fixed six would silently drop a seventh gene.
    gcol = min(3, len(genes))
    grow = math.ceil(len(genes) / gcol)
    fig = plt.figure(figsize=(5.4 * gcol, 5.2 + 4.0 * grow))
    gs = fig.add_gridspec(1 + grow, gcol, height_ratios=[1.3] + [1] * grow,
                          hspace=0.5, wspace=0.22)

    panels = [("__pooled__", fig.add_subplot(gs[0, :]))]
    panels += [(g, fig.add_subplot(gs[1 + i // gcol, i % gcol]))
               for i, g in enumerate(genes)]

    for gene, ax in panels:
        is_pooled = gene == "__pooled__"
        if is_pooled:
            cells = [pooled_cell[c] for c in configs if c in pooled_cell]
        else:
            cells = [cell[(gene, c)] for c in configs if (gene, c) in cell]
        xs = np.arange(len(cells))
        ymax = draw_stacked_bars(ax, cells, xs, 0.62, draw_l1, emphasise=set(global_top))

        # Rank badge: recovers the ranking on an axis ordered by capacity, and makes the
        # aad-vs-total disagreement visible (the #1 badge rarely sits on the shortest bar).
        order = {c["config"]: i + 1 for i, c in
                 enumerate(sorted(cells, key=lambda r: r[metric_mean]))}
        for i, c in enumerate(cells):
            ax.annotate(f"#{order[c['config']]}", xy=(xs[i], c["loss_mean"]),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=8, color=COLOR_LOSS, zorder=4)

        # Datum lines: bar tops differing by 0.01 on a 0.45 axis are indistinguishable,
        # but each bar's offset from a shared reference is readable.
        ax.axhline(min(c["loss_mean"] for c in cells), color=COLOR_LOSS, ls=":", lw=0.9,
                   alpha=0.5, zorder=1)
        ax.axhline(min(c[metric_mean] for c in cells), color=COLOR_SD, ls=":", lw=0.9,
                   alpha=0.6, zorder=1)

        ax.set_xticks(xs)
        ax.set_xticklabels([c["config"] for c in cells],
                           rotation=0 if is_pooled else 45,
                           ha="center" if is_pooled else "right",
                           fontsize=9 if is_pooled else 8)
        ax.set_ylim(0, ymax * 1.22)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        if is_pooled:
            # One band states the headline caveat without prose: everything it covers is
            # inside one seed sd of the best config, i.e. not separable by this sweep. It
            # is named in the legend rather than annotated in the axes, because the bars
            # fill the panel and any in-axes label would land on top of one.
            best = pooled_by_metric[0]
            band = best[metric_mean] + np.nan_to_num(best[metric_sd])
            ax.axhspan(best[metric_mean], band, color=COLOR_SD, alpha=0.12, zorder=0)
            tied = [r["config"] for r in pooled_by_metric if r[metric_mean] <= band]
            band_handle = Patch(facecolor=COLOR_SD, alpha=0.12,
                                label=f"±1 seed sd of the best config — covers "
                                      f"{', '.join(tied)}")
            ax.set_title(f"pooled across {len(genes)} genes "
                         f"(mean of {cells[0]['n_runs']} runs per config)",
                         fontsize=11, fontweight="bold")
            ax.set_ylabel("final loss")
        else:
            level = np.mean([c[metric_mean] for c in cells])
            ax.set_title(f"{gene}   mean {args.metric} {level:.3f}   "
                         f"(n = {cells[0]['n_seeds']} seeds per config)", fontsize=10)
            if genes.index(gene) % gcol == 0:  # left column only
                ax.set_ylabel("final loss")

    # Figure-level legend under the panels: the pooled panel's bars reach the top of its
    # axes, so an in-axes legend would cover f5_k1 and the tie band it is meant to explain.
    fig.legend(handles=bar_handles + datum_handles + [band_handle], loc="lower center",
               ncol=3, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"Final loss across {len(configs)} configs × {len(genes)} genes, "
        f"averaged over seeds ({stamp})\n"
        "bars stack aad_loss + ortho, which sum to the total loss exactly"
        + ("" if draw_l1 else " (l1 is 0 in every run, so the l1 term is dropped)") + "\n"
        "each panel has its own y-limits — gene levels span ~"
        f"{max(np.mean([cell[(g, c)][metric_mean] for c in configs if (g, c) in cell]) for g in genes) - min(np.mean([cell[(g, c)][metric_mean] for c in configs if (g, c) in cell]) for g in genes):.2f}"
        " while configs within a gene span far less; #n badges rank by " + args.metric,
        fontsize=11)
    # subplots_adjust rather than tight_layout: the full-width pooled panel spans all three
    # gridspec columns, which tight_layout cannot lay out (it warns and may misplace the
    # panels). The gridspec already carries its own hspace/wspace, so only the top margin
    # needs reserving for the three-line suptitle.
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.98)
    overview_path = os.path.join(args.out_dir, "compare_overview.png")
    fig.savefig(overview_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote config overview to {overview_path}")

    # ── Figures 2 and 3: the top-n breakdown, globally and per gene ─────────────────
    def breakdown_figure(per_gene, path, suptitle):
        # Grid follows the gene count rather than being fixed at 2x3, so --genes with a
        # short list gives a compact figure instead of one with rows of blank space.
        ncol = min(3, len(genes))
        nrow = math.ceil(len(genes) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.25 * nrow),
                                 squeeze=False)
        flat = axes.ravel()
        n_noise = 0

        for i, ax in enumerate(flat):
            if i >= len(genes):
                # Fewer than six genes: hide the leftovers rather than leaving empty
                # frames that read as "this gene had no runs".
                ax.set_visible(False)
                continue
            gene = genes[i]
            avail = [cell[(gene, c)] for c in configs if (gene, c) in cell]
            if per_gene:
                chosen = sorted(avail, key=lambda r: r[metric_mean])[:args.top_n]
            else:
                chosen = [cell[(gene, c)] for c in global_top if (gene, c) in cell]
            if not chosen:
                ax.set_visible(False)
                continue

            xs = np.arange(len(chosen))
            ymax = draw_stacked_bars(ax, chosen, xs, 0.5, draw_l1, label_values=True)
            # Enough headroom that the "loss = ..." labels sitting just above the bars
            # clear the tie box pinned to the top of the panel.
            ax.set_ylim(0, ymax * 1.6)
            ax.set_xlim(-0.7, len(chosen) - 0.3)
            ax.grid(True, axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.set_xticks(xs)

            # The tick label is where each figure earns its keep: in figure 2 it exposes a
            # globally-chosen config that ranks poorly in this particular gene, and in
            # figure 3 it shows where the gene's own pick sits globally.
            # Stacked vertically and without the gene name (which the panel title already
            # carries): laid out side by side these labels grow wide enough to collide
            # with their neighbours as soon as --top-n goes past 2.
            labels = []
            for c in chosen:
                rank = c[f"{args.metric}_rank_in_gene"]
                if per_gene:
                    labels.append(f"{c['config']}\ngene {rank}/{len(avail)}\n"
                                  f"global {pooled_rank[c['config']]}/{len(configs)}")
                else:
                    labels.append(f"{c['config']}\nrank {rank}/{len(avail)}")
            ax.set_xticklabels(labels, fontsize=9)

            title = gene
            if per_gene:
                same = {c["config"] for c in chosen} == set(global_top)
                # "same as global"/"gene-specific" rather than "...pair", which would be
                # wrong wording for any --top-n other than 2.
                title += "  — same as global" if same else "  — gene-specific"
            ax.set_title(title, fontsize=11, fontweight="bold")
            if i % ncol == 0:  # left column only
                ax.set_ylabel("final loss")

            # Delta against the seed noise, for the two configs actually drawn. This is the
            # number that decides whether the panel shows a result or a coin flip.
            if len(chosen) >= 2:
                a, b = chosen[0], chosen[1]
                delta = b[metric_mean] - a[metric_mean]
                sd_a, sd_b = np.nan_to_num(a[metric_sd]), np.nan_to_num(b[metric_sd])
                noisy = abs(delta) < max(sd_a, sd_b)
                n_noise += noisy
                text = (f"Δ{args.metric} ({b['config']} − {a['config']}) = {delta:+.4f}\n"
                        f"seed sd: {sd_a:.4f} / {sd_b:.4f}"
                        + ("  → within seed noise" if noisy else "  → separable"))
                # Black-bold for the noise flag rather than a warning colour: darkorange
                # already means "the l1 term" in this codebase and must not be borrowed.
                ax.text(0.5, 0.98, text, transform=ax.transAxes, va="top", ha="center",
                        fontsize=8.5, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec=COLOR_LOSS if noisy else "dimgray",
                                  lw=1.4 if noisy else 0.8, alpha=0.92))

        # bar_handles only: these panels draw no datum lines, so advertising them here
        # would name something the reader cannot find.
        fig.legend(handles=bar_handles, loc="lower center", ncol=3, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(suptitle(n_noise), fontsize=11)
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote top-{args.top_n} breakdown to {path}")

    best, second = pooled_by_metric[0], pooled_by_metric[1] if len(pooled_by_metric) > 1 else None
    gap = (second[metric_mean] - best[metric_mean]) if second else float("nan")
    ratio = (np.nan_to_num(best[metric_sd]) / gap) if gap else float("inf")

    def global_suptitle(n_noise):
        head = (f"Top {args.top_n} configs chosen globally by pooled mean {args.metric}: "
                + ", ".join(f"{c} ({pooled_cell[c][metric_mean]:.4f})" for c in global_top))
        if second is None:
            return head
        return (head + f"\nthe {gap:.4f} pooled gap is ~{ratio:.0f}× smaller than the "
                f"seed-to-seed sd ({np.nan_to_num(best[metric_sd]):.4f} / "
                f"{np.nan_to_num(second[metric_sd]):.4f}) — this pair is a tie, not a ranking\n"
                f"the same pair is shown for every gene; bars break the final loss into "
                f"aad_loss + ortho")

    def per_gene_suptitle(n_noise):
        differing = sum(
            1 for g in genes
            if {c["config"] for c in
                sorted([cell[(g, c)] for c in configs if (g, c) in cell],
                       key=lambda r: r[metric_mean])[:args.top_n]} != set(global_top))
        return (f"Each gene's own top {args.top_n} configs, by that gene's mean {args.metric}\n"
                f"{differing} of {len(genes)} genes prefer a set different from the global "
                f"{' / '.join(global_top)} — but in {n_noise} of {len(genes)} the 1st-vs-2nd "
                f"gap is smaller than the seed sd of either config,\nso none of those "
                f"preferences is resolved by {int(df.seed.nunique())} seeds")

    breakdown_figure(False, os.path.join(args.out_dir, "compare_top2_global.png"),
                     global_suptitle)
    breakdown_figure(True, os.path.join(args.out_dir, "compare_top2_per_gene.png"),
                     per_gene_suptitle)

    # ── Ranking table on stdout ─────────────────────────────────────────────────────
    logger.info(f"\nPooled config ranking by {args.metric} "
                f"(mean over {len(genes)} genes × {int(df.seed.nunique())} seeds):")
    logger.info(f"  {'config':<10} {args.metric:>10} {'seed sd':>9} {'total loss':>11} "
                f"{'ortho':>8}  {'loss rank':>9}")
    loss_rank = {r["config"]: i + 1 for i, r in
                 enumerate(sorted(pooled, key=lambda r: r["loss_mean"]))}
    for r in pooled_by_metric:
        logger.info(f"  {r['config']:<10} {r[metric_mean]:>10.4f} "
                    f"{np.nan_to_num(r[metric_sd]):>9.4f} {r['loss_mean']:>11.4f} "
                    f"{r['ortho_mean']:>8.4f}  {loss_rank[r['config']]:>9}")


if __name__ == "__main__":
    main()
