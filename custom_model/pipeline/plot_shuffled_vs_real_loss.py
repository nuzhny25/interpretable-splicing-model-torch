"""Loss of the row-shuffled MALAT1 control against the real MALAT1 runs.

    python pipeline/plot_shuffled_vs_real_loss.py

Reads the history.npz of every real run ``runs/multiz100_f<F>_k<K>_s<seed>`` and every
shuffled run ``runs_shuffled/multiz100_shuffled_f<F>_k<K>_s0_shuf<k>`` (see
pipeline/shuffled_control.txt) and draws a 2x2 figure:

    top row      aad_loss (the scale-free cross-species abs-dev term, ``sd_loss`` in the
                 npz) over training, and each run's final value
    bottom row   total loss (aad_loss + ortho_lambda * mean|r|), the same two views

The two groups replicate different things. The real runs vary the initial weights on one
alignment. The shuffled runs share the seed-0 initial weights (the same as real s0) and
vary only the row permutation. aad_loss is the like-for-like comparison. The total loss
also carries the orthogonality term, which the shuffled runs measure on shuffled 18-mers,
so their mean|r|, and therefore their total, is not measured on the same sequence as the
real runs'.

Writes ``analysis/shuffled_vs_real_loss/shuffled_vs_real_loss.png`` and a CSV of the
final values, which serves as the figure's table view.
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

# Reference palette (dataviz skill): categorical slots 1 and 2, validated as a pair on
# the light surface. Chrome and ink are the palette's light-mode tokens.
REAL_COLOR, SHUF_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def load(pattern, group):
    rows, curves = [], []
    for path in sorted(glob.glob(pattern)):
        run = os.path.basename(os.path.dirname(path))
        d = np.load(path)
        shuf = re.search(r"_shuf(\d+)$", run)
        rows.append(
            {
                "group": group,
                "run": run,
                "init_seed": int(re.search(r"_s(\d+)(?:_shuf\d+)?$", run)[1]),
                "shuffle_seed": int(shuf[1]) if shuf else None,
                "final_aad_loss": float(d["sd_loss"][-1]),
                "final_total_loss": float(d["loss"][-1]),
                "final_mean_abs_r": float(d["mean_abs_r"][-1]),
            }
        )
        curves.append((run, d["epoch"], d["sd_loss"], d["loss"]))
    return rows, curves


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9, length=3)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-filters", type=int, default=20)
    parser.add_argument("--kernel-size", type=int, default=6)
    parser.add_argument(
        "--out-dir", default=os.path.join(_HERE, "analysis", "shuffled_vs_real_loss")
    )
    args = parser.parse_args()
    arch = f"f{args.num_filters}_k{args.kernel_size}"

    real_rows, real_curves = load(
        os.path.join(_HERE, "runs", f"multiz100_{arch}_s[0-9]*", "history.npz"), "real"
    )
    shuf_rows, shuf_curves = load(
        os.path.join(
            _HERE, "runs_shuffled", f"multiz100_shuffled_{arch}_s*_shuf*", "history.npz"
        ),
        "shuffled",
    )
    if not real_rows or not shuf_rows:
        raise SystemExit(f"Need both real and shuffled {arch} runs; found "
                         f"{len(real_rows)} real, {len(shuf_rows)} shuffled.")
    df = pd.DataFrame(real_rows + shuf_rows)
    df["shuffle_seed"] = df["shuffle_seed"].astype("Int64")  # blank for real runs

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8.2), sharey="row",
        gridspec_kw={"width_ratios": [3, 1.15], "wspace": 0.06, "hspace": 0.32},
    )
    fig.patch.set_facecolor(SURFACE)

    real_label = f"Real MALAT1 · {len(real_rows)} init seeds"
    shuf_label = f"Row-shuffled · {len(shuf_rows)} shuffles, init seed 0"
    metrics = [
        ("aad_loss", 2, "final_aad_loss", "aad_loss\n(cross-species abs-dev / within-species scale)"),
        ("total loss", 3, "final_total_loss", "total loss\n(aad_loss + 0.5 · mean|r|)"),
    ]

    for row, (name, idx, col, ylabel) in enumerate(metrics):
        ax_c, ax_d = axes[row]
        style(ax_c)
        style(ax_d)

        # ── Training curves: one thin line per run, colored by group ──
        for curves, color in ((real_curves, REAL_COLOR), (shuf_curves, SHUF_COLOR)):
            for c in curves:
                ax_c.plot(c[1], c[idx], color=color, lw=1.5, alpha=0.8,
                          solid_capstyle="round", solid_joinstyle="round")
        ax_c.set_xlim(0, 500)
        ax_c.set_ylabel(ylabel, color=INK_2, fontsize=10)
        ax_c.set_xlabel("epoch", color=INK_2, fontsize=10)
        ax_c.set_title(f"{name} during training", loc="left", color=INK,
                       fontsize=11, fontweight="semibold")

        # Direct labels in the empty band between the two bundles, in ink: real just above
        # its bundle, shuffled just below its own. Positioned off the curves' extent over
        # the label's x-span (epoch >= 300), not the final value, so a curve that is still
        # descending there cannot run through the text.
        span = 300
        real_top = max(c[idx][c[1] >= span].max() for c in real_curves)
        shuf_bottom = min(c[idx][c[1] >= span].min() for c in shuf_curves)
        ax_c.annotate(real_label, (495, real_top), xytext=(0, 5), textcoords="offset points",
                      ha="right", va="bottom", color=INK_2, fontsize=9)
        ax_c.annotate(shuf_label, (495, shuf_bottom), xytext=(0, -5),
                      textcoords="offset points", ha="right", va="top", color=INK_2,
                      fontsize=9)

        # ── Final values: one dot per run, group mean as a short ink tick ──
        for x, (group, color) in enumerate((("real", REAL_COLOR), ("shuffled", SHUF_COLOR))):
            sub = df[df.group == group].reset_index(drop=True)
            offsets = np.linspace(-0.16, 0.16, len(sub))
            ax_d.scatter(x + offsets, sub[col], s=64, color=color, edgecolor=SURFACE,
                         linewidth=1.5, zorder=3)
            mean = sub[col].mean()
            ax_d.plot([x - 0.26, x + 0.26], [mean, mean], color=INK, lw=1.2, zorder=4)
            ax_d.annotate(f"mean {mean:.3f}\nsd {sub[col].std(ddof=1):.3f}",
                          (x + 0.3, mean), xytext=(0, 0), textcoords="offset points",
                          va="center", ha="left", color=INK_2, fontsize=8.5)
        # Tag the real run that shares the shuffled runs' initial weights (the caption
        # says what the tag means). Above the dot: the band between the groups is empty.
        real = df[df.group == "real"].reset_index(drop=True)
        if (real.init_seed == 0).any():
            i = int(np.flatnonzero(real.init_seed.to_numpy() == 0)[0])
            x0 = np.linspace(-0.16, 0.16, len(real))[i]
            ax_d.annotate("s0", (x0, real[col][i]), xytext=(0, 7),
                          textcoords="offset points", ha="center", va="bottom",
                          color=MUTED, fontsize=8)
        gap = df[df.group == "shuffled"][col].mean() - df[df.group == "real"][col].mean()
        ax_d.set_title(f"final value (gap {gap:+.3f})", loc="left", color=INK,
                       fontsize=11, fontweight="semibold")
        ax_d.set_xticks([0, 1], ["real", "shuffled"])
        ax_d.set_xlim(-0.45, 1.95)
        ax_d.tick_params(axis="y", left=False)

    handles = [
        plt.Line2D([], [], color=REAL_COLOR, lw=2, label=real_label),
        plt.Line2D([], [], color=SHUF_COLOR, lw=2, label=shuf_label),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.955), ncol=2,
               frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=1.8)
    fig.suptitle(
        f"Row-shuffled vs real MALAT1 alignment — {args.num_filters}+{args.num_filters} "
        f"filters, kernel {args.kernel_size}, 500 epochs",
        x=0.075, y=0.995, ha="left", color=INK, fontsize=13, fontweight="semibold",
    )
    fig.text(
        0.075, 0.015,
        "Real runs vary the initial weights; shuffled runs all start from seed-0 weights "
        "(the same as the real run tagged s0) and vary only the row permutation.\n"
        "aad_loss is the like-for-like comparison. The total loss adds 0.5 · mean|r|, which "
        "the shuffled runs measure on shuffled 18-mers.",
        color=MUTED, fontsize=8.5, ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.1)

    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, "shuffled_vs_real_loss.png")
    csv = os.path.join(args.out_dir, "shuffled_vs_real_loss.csv")
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    df.to_csv(csv, index=False)
    print(f"Wrote {png}\nWrote {csv}")


if __name__ == "__main__":
    main()
