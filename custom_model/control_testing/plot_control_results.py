"""Figure readout for a run of control_sr_deviation.py.

Reads the control results npz that the scan writes next to itself
(``control_results.npz``) and renders the two comparisons that used to be printed
as console tables, as one figure of two stacked panels:

  * Top — the randomization controls (``random``, ``shuffled``), each the average
    over its ``N_RANDOM`` / ``N_SHUFFLE`` matrices, with +/-1 run-to-run standard
    deviation as a whisker and its distance from the original in standard
    deviations annotated beside it.

  * Bottom — the three transformed orientations of the real matrix (``reversed``,
    ``complement``, ``reverse_complement``). Each is a single deterministic matrix,
    so these carry no whisker.

Both panels plot **value / original**, not the raw value. The metrics span orders
of magnitude (``mean_col_abs_dev`` sits near 0.002 while ``norm_loss`` sits near
0.27), so a shared raw axis would flatten the small ones into the floor and a
second y-axis is never the answer; indexing every metric to the original puts them
all on one scale whose reference line at 1.0 *is* the original. The axis is
logarithmic by default so that 2x and 0.5x are the same distance from it
(``--linear-x`` opts out). Each y tick label carries the original's absolute value,
which is the one thing the ratio discards.

``ortho`` and ``l1`` are deliberately absent as rows. They are ``ortho_lambda *
mean_abs_r`` and ``l1_lambda * mean_activation``, so on a ratio axis the lambda
cancels and they would be pixel-identical duplicates of the two rows already
plotted; those rows are labelled to say so.

Metrics missing from the npz are skipped rather than raising, so a results file
from before the penalty terms were added still plots (as does one from a
checkpoint whose metrics differ).

Run after the scan, or any time to re-render without re-running it:

    python plot_control_results.py
    python plot_control_results.py --npz control_results.npz --out control_summary.png
    python plot_control_results.py --linear-x
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Deliberately imports nothing from control_sr_deviation: this module needs only the
# saved numbers, and reaching into the scan would drag torch and the model in just to
# draw a figure. PLOT_METRICS below is its own curated list, not that module's
# SCALAR_METRICS — it carries display labels and omits the two scaled twins.
_HERE = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Palette ──────────────────────────────────────────────────────────────────
# One fixed hue per condition, assigned by identity and never cycled. Checked with
# the dataviz skill's validate_palette.js against a light surface: all five pass the
# lightness band, chroma floor, colorblind separation, normal-vision floor and
# contrast checks. Re-run that validator before substituting any of them by eye —
# the obvious matplotlib names (steelblue, firebrick, chocolate) do not pass.
COLOR_RANDOM = "#B2182B"
COLOR_SHUFFLED = "#C2700F"
COLOR_REVERSED = "#1F78B4"
COLOR_COMPLEMENT = "#159E77"
COLOR_REVCOMP = "#7B3FA0"
# The original is the reference the whole figure is drawn against, so it is the
# axis line rather than a series. Black matches COLOR_LOSS in
# plot/plot_training_curves.py, where it likewise marks the quantity being explained.
COLOR_ORIGINAL = "black"
# Annotations wear text ink, never the series colour — the marker beside them
# already carries the identity.
COLOR_ANNOTATION = "#444444"
COLOR_GRID = "#BBBBBB"

# The randomization controls (top panel) and the orientations (bottom panel), each
# as (results-key prefix, legend label, colour). Order is the plotting order within
# a metric row, top to bottom.
CONTROL_CONDITIONS = (
    ("random", "random (gap-preserving)", COLOR_RANDOM),
    ("shuffled", "shuffled (aligned-position permutation)", COLOR_SHUFFLED),
)
ORIENTATION_CONDITIONS = (
    ("reversed", "reversed", COLOR_REVERSED),
    ("complement", "complement", COLOR_COMPLEMENT),
    ("reverse_complement", "reverse complement", COLOR_REVCOMP),
)

# Rows of the figure, top to bottom: the two loss-space quantities first, then the
# four raw quantities behind them. The scaled twins ortho and l1 are omitted — see
# the module docstring — and the rows that stand in for them say so in their label.
PLOT_METRICS = (
    ("total_loss", "total_loss"),
    ("norm_loss", "norm_loss"),
    ("mean_col_abs_dev", "mean_col_abs_dev"),
    ("within_species_scale", "within_species_scale"),
    ("mean_abs_r", "mean_abs_r\n(= ortho / λ)"),
    ("mean_activation", "mean_activation\n(= l1 / λ)"),
)


def _get(results, key):
    """Value for ``key`` as a plain float, or None when it is absent.

    Accepts both shapes the results take: the live dict the scan builds (Python
    floats and ``np.float64``) and the same dict round-tripped through ``np.savez``
    (0-d arrays). Missing keys return None so callers can skip a metric that this
    results file does not carry.
    """
    if key not in results:
        return None
    value = results[key]
    try:
        return float(np.asarray(value).item())
    except (TypeError, ValueError):
        return None


def _z_label(n_std):
    """Compact rendering of a distance-from-original in standard deviations.

    These range over several orders of magnitude — a control can sit a handful of
    standard deviations from the original or several thousand — so past 1000 the
    exact figure carries no information the reader can use and is abbreviated.
    """
    if n_std is None or not np.isfinite(n_std):
        return None
    if abs(n_std) >= 1000:
        return f"{n_std / 1000:+.0f}k σ"
    if abs(n_std) >= 10:
        return f"{n_std:+.0f} σ"
    return f"{n_std:+.1f} σ"


def _plot_panel(ax, results, metrics, conditions, annotate_z):
    """Draw one panel: every condition's ratio-to-original, one row per metric.

    Args:
        metrics: The ``(key, label)`` pairs to use as rows, already filtered to
            those this results file carries.
        conditions: ``(prefix, label, colour)`` triples to draw within each row.
        annotate_z: Whether to label each marker with its distance from the
            original in run-to-run standard deviations. Only the randomization
            controls have replicates, so only they can carry this.

    Returns:
        ``(min_ratio, max_ratio)`` over everything drawn, so the caller can size the
        shared axis to the data and leave room for the annotations.
    """
    n_cond = len(conditions)
    # Spread the conditions across the row without letting them touch the
    # neighbouring rows: the outermost sit at +/-0.26 of the 1.0 row spacing.
    offsets = (
        np.linspace(0.26, -0.26, n_cond) if n_cond > 1 else np.zeros(1)
    )
    min_ratio = max_ratio = 1.0

    for row, (metric, _) in enumerate(metrics):
        # Rows run top to bottom, so the first metric gets the highest y.
        y_row = len(metrics) - 1 - row
        original = _get(results, f"original_{metric}")
        for (prefix, _, color), dy in zip(conditions, offsets):
            value = _get(results, f"{prefix}_{metric}")
            if value is None or original is None or not original:
                continue
            ratio = value / original
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            max_ratio, min_ratio = max(max_ratio, ratio), min(min_ratio, ratio)

            # The run-to-run spread, carried into ratio units by the same divisor.
            std = _get(results, f"{prefix}_{metric}_std")
            xerr = (
                [[abs(std / original)], [abs(std / original)]]
                if std is not None and np.isfinite(std)
                else None
            )
            ax.errorbar(
                [ratio],
                [y_row + dy],
                xerr=xerr,
                fmt="o",
                markersize=6,
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=3,
                capthick=1.4,
                # A surface ring keeps a marker legible where it overlaps a whisker
                # or another condition, without drawing a border around it.
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=3,
            )

            if annotate_z:
                label = _z_label(_get(results, f"{prefix}_{metric}_n_std_from_original"))
                if label:
                    # Anchored past the whisker's end, on the side facing *away* from
                    # the reference line: a control below 1.0 would otherwise have its
                    # label written straight across that line.
                    half = abs(std / original) if xerr else 0.0
                    outward = 1 if ratio >= 1.0 else -1
                    ax.annotate(
                        label,
                        xy=(ratio + outward * half, y_row + dy),
                        xytext=(7 * outward, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left" if outward > 0 else "right",
                        fontsize=7.5,
                        color=COLOR_ANNOTATION,
                        zorder=4,
                    )

    # The original is the reference every marker is measured against.
    ax.axvline(1.0, color=COLOR_ORIGINAL, lw=1.2, zorder=2)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(
        [
            # The absolute value the ratio axis discards, kept on the label so the
            # rows stay interpretable without cross-referencing the npz.
            f"{label}\n({_get(results, f'original_{metric}'):.4g})"
            if _get(results, f"original_{metric}") is not None
            else label
            for metric, label in reversed(metrics)
        ],
        fontsize=8.5,
    )
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    # Hairline, solid, one shade off the surface — a grid should recede.
    ax.grid(True, axis="x", color=COLOR_GRID, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # The row label is the tick; a dash beside it is just chrome.
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return min_ratio, max_ratio


def render(results, out_path, log_x=True):
    """Render the control-summary figure from a results mapping.

    Args:
        results: The scan's results — either the live dict ``main`` builds or an
            ``np.load`` of the npz it saves. Metrics absent from it are skipped.
        out_path: Destination PNG.
        log_x: Log-scale the ratio axis, so a halving and a doubling sit the same
            distance from the reference line.

    Returns:
        ``out_path``.
    """
    # Only the metrics this results file actually carries, in PLOT_METRICS order.
    metrics = [
        (metric, label)
        for metric, label in PLOT_METRICS
        if _get(results, f"original_{metric}") is not None
    ]
    if not metrics:
        raise ValueError(
            "No plottable metrics: results carries no 'original_<metric>' key for "
            f"any of {[m for m, _ in PLOT_METRICS]}."
        )

    fig, (ax_controls, ax_orient) = plt.subplots(
        2,
        1,
        # ~0.5" per metric row per panel: enough to separate the conditions within a
        # row without stranding each row in whitespace.
        figsize=(11, 1.9 + 0.5 * len(metrics) * 2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    lo_control, hi_control = _plot_panel(
        ax_controls, results, metrics, CONTROL_CONDITIONS, annotate_z=True
    )
    lo_orient, hi_orient = _plot_panel(
        ax_orient, results, metrics, ORIENTATION_CONDITIONS, annotate_z=False
    )

    ax_controls.set_title(
        "Randomization controls — destroy the sequence, keep the gaps or the composition",
        fontsize=10,
        loc="left",
        pad=8,
    )
    ax_orient.set_title(
        "Orientations — the same sequence read backwards, complemented, or both",
        fontsize=10,
        loc="left",
        pad=8,
    )

    # Headroom for the sigma annotations, which hang off the outermost markers.
    lo = min(lo_control, lo_orient)
    hi = max(hi_control, hi_orient)
    if log_x:
        ax_orient.set_xscale("log")
        ax_orient.set_xlim(lo / 1.35, hi * 1.7)
        # The whole ratio range usually sits inside a single decade, where the log
        # locator puts one tick at 1.0 and nothing else — an axis nothing can be read
        # off. Tick the round multiples that fall in range instead, labelled as the
        # multiples they are rather than as the 10^n a log axis would print.
        candidates = np.array(
            [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7, 10]
        )
        left, right = ax_orient.get_xlim()
        ax_orient.set_xticks(candidates[(candidates >= left) & (candidates <= right)])
        ax_orient.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax_orient.set_xlim(lo / 1.15, hi * 1.25)
    ax_orient.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax_orient.set_xlabel(
        "value / original" + ("  (log scale)" if log_x else "")
        + "     — the black line at 1.0 is the original matrix",
        fontsize=9,
    )

    for ax, conditions in (
        (ax_controls, CONTROL_CONDITIONS),
        (ax_orient, ORIENTATION_CONDITIONS),
    ):
        # Identity is never carried by colour alone: every panel keeps its legend.
        ax.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=6,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                )
                for _, label, color in conditions
            ],
            loc="lower right",
            fontsize=8.5,
            framealpha=0.9,
        )

    # Stamp the run identity when the scan recorded it. Older results files predate
    # the stamp, and simply go without.
    meta_raw = results["meta_json"] if "meta_json" in results else None
    if meta_raw is not None:
        meta = json.loads(np.asarray(meta_raw).item())
        fields = [
            os.path.basename(str(meta.get("weights_path", "?"))),
            f"min_species={meta.get('min_species')}",
            f"smooth_sigma={meta.get('smooth_sigma')}",
            f"n_random={meta.get('n_random')}",
            f"n_shuffle={meta.get('n_shuffle')}",
            f"l1_lambda={meta.get('l1_lambda')}",
            f"ortho_lambda={meta.get('ortho_lambda')}",
            f"pool_length={meta.get('pool_length')}",
        ]
        fig.suptitle("  ".join(fields), fontsize=9, y=0.998)

    # Leave the suptitle its own band: without the rect it lands on the top panel's
    # own title.
    fig.tight_layout(rect=(0, 0, 1, 0.975 if meta_raw is not None else 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote control summary over {len(metrics)} metrics to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default=os.path.join(_HERE, "control_results.npz"),
        help="Control results npz written by control_sr_deviation.py.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "control_summary.png"),
        help="Output PNG.",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Linear ratio axis instead of the default log one, on which a halving "
        "and a doubling sit the same distance from the reference line.",
    )
    args = parser.parse_args()

    render(np.load(args.npz), args.out, log_x=not args.linear_x)


if __name__ == "__main__":
    main()
