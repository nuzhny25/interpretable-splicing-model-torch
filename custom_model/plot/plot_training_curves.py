"""Training-curve readout for a run of custom_real_train_absolute_deviation.py.

Reads the per-epoch history npz that training writes next to its weights
(``weights/real_weights_aad_history.npz``) and renders two stacked panels:

  * Top — every component of the loss on one linear axis. The loss-space terms
    (``loss``, ``aad_loss``, ``ortho`` and ``l1`` where present) are solid; the
    quantities behind them are dashed — the ``aad_loss`` ratio's numerator
    ``raw_mean_col_aad`` and denominator ``within_species_scale``, and the unscaled
    ``mean_abs_r`` behind ``ortho``. The series differ by orders of magnitude, so a
    linear axis is set by the largest of them and the penalties run close to the
    floor; each legend entry therefore carries its own first-to-last change, which
    is where the smaller series' behaviour stays readable. Pass ``--log-y`` for a
    log axis instead, on which every series' relative change is directly visible.

  * Bottom — each term's share of the total loss. ``aad_loss`` is scale-free and so
    exerts no gradient along the overall-scale direction, which leaves the penalty
    terms as the only force there; this panel shows how much of the loss they
    actually account for as training proceeds.

``ortho`` and ``mean_abs_r`` are written only by the orthogonality trainers, whose
penalty is the mean |Pearson r| over the C(2F, 2) pairs of the model's 2F filter
channels (780 pairs for the original 20+20 bank). Both panels fall back to the
``aad_loss``/``l1`` pair when the npz lacks them, so a history from
custom_real_train_absolute_deviation.py plots exactly as it did before.

A term that is not part of the run's loss is left off both panels entirely: at
``l1_lambda = 0`` the recorded ``l1`` series is identically zero, so its line, its
stacked band, its two share annotations and its entry in the hyperparameter stamp are
all dropped rather than drawn flat against the axis.

Run after training:

    python plot/plot_training_curves.py
    python plot/plot_training_curves.py --history ../weights/real_weights_ortho_history.npz
    python plot/plot_training_curves.py --history ../weights/other_history.npz --log-x
    python plot/plot_training_curves.py --log-y
"""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(__file__)

# Shared between both panels so the stacked shares below read against the curves above.
COLOR_SD = "steelblue"
COLOR_L1 = "darkorange"
COLOR_LOSS = "black"
COLOR_SCALE = "seagreen"
COLOR_ORTHO = "rebeccapurple"


def change_label(name, values):
    """Legend label annotated with the series' first-to-last change.

    Reported as a fold change once the series has moved by more than 3x
    (``l1  (/245)``), and as a percentage otherwise (``aad_loss  (-48%)``). Small fold
    changes read poorly — a halving is clearer as "-53%" than as "/2" — so the
    percentage band is deliberately wider than a factor of two.
    """
    first, last = float(values[0]), float(values[-1])
    if first <= 0 or last <= 0:
        return name
    ratio = last / first
    if ratio < 1 / 3:
        change = f"/{first / last:,.0f}"
    elif ratio > 3:
        change = f"x{ratio:,.0f}"
    else:
        change = f"{(ratio - 1) * 100:+.0f}%"
    return f"{name}  ({change})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        default=os.path.join(_HERE, "..", "weights", "real_weights_aad_history.npz"),
        help="Per-epoch history npz written by custom_real_train_absolute_deviation.py.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "training_curves.png"),
        help="Output PNG.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Log-scale the epoch axis, which spreads out the early transient.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Log-scale the component axis, making each series' relative change "
        "readable even where they differ by orders of magnitude.",
    )
    args = parser.parse_args()

    data = np.load(args.history)
    epoch = data["epoch"]
    loss = data["loss"]
    sd_loss = data["sd_loss"]
    numerator = data["raw_mean_col_sd"]
    denominator = data["within_species_scale"]
    # The L1 term is optional. The orthogonality runs set l1_lambda = 0, which makes
    # the recorded l1 an identically-zero series: plotting it costs a flat line on the
    # axis, a zero-width band in the stackplot and two "l1 = 0.00%" callouts pointing at
    # nothing. Drop the term entirely when it was disabled (or never recorded), so the
    # figure shows only the terms that actually make up the loss.
    l1 = data["l1"] if "l1" in data.files else None
    if l1 is not None and np.allclose(l1, 0):
        l1 = None
    # Orthogonality terms, written only by custom_real_train_orthogonality.py. Absent
    # from every custom_real_train_absolute_deviation.py history, so everything below
    # that touches them is guarded and the older npzs plot unchanged.
    ortho = data["ortho"] if "ortho" in data.files else None
    mean_abs_r = data["mean_abs_r"] if "mean_abs_r" in data.files else None

    fig, (ax_terms, ax_share) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- All loss components together ---
    # Solid = terms of the loss sum; dashed = the two quantities whose ratio is aad_loss.
    ax_terms.plot(
        epoch, loss, color=COLOR_LOSS, lw=1.8, label=change_label("loss", loss)
    )
    ax_terms.plot(
        epoch, sd_loss, color=COLOR_SD, lw=1.5, label=change_label("aad_loss", sd_loss)
    )
    if l1 is not None:
        ax_terms.plot(epoch, l1, color=COLOR_L1, lw=1.5, label=change_label("l1", l1))
    if ortho is not None:
        ax_terms.plot(
            epoch, ortho, color=COLOR_ORTHO, lw=1.5, label=change_label("ortho", ortho)
        )
    if mean_abs_r is not None:
        # ortho = ortho_lambda * mean_abs_r, so this dashed line is the raw quantity
        # behind the solid one — directly comparable to the mean |r| that
        # filter_coactivation_covariance.py reports for the finished checkpoint.
        ax_terms.plot(
            epoch,
            mean_abs_r,
            color=COLOR_ORTHO,
            lw=1.2,
            ls="--",
            alpha=0.8,
            label=change_label("mean_abs_r  (unscaled)", mean_abs_r),
        )
    ax_terms.plot(
        epoch,
        numerator,
        color=COLOR_SD,
        lw=1.2,
        ls="--",
        alpha=0.8,
        label=change_label("raw_mean_col_aad  (numerator)", numerator),
    )
    ax_terms.plot(
        epoch,
        denominator,
        color=COLOR_SCALE,
        lw=1.2,
        ls="--",
        alpha=0.8,
        label=change_label("within_species_scale  (denominator)", denominator),
    )
    if args.log_y:
        ax_terms.set_yscale("log")
    ax_terms.set_ylabel("value (log scale)" if args.log_y else "value")
    ax_terms.set_title("Loss components during training")
    ax_terms.grid(True, which="both" if args.log_y else "major", alpha=0.2)
    ax_terms.legend(loc="best", fontsize=9, framealpha=0.9)

    # --- Share of the total loss ---
    # Guard against a zero total: shares are undefined there, so leave them at 0.
    safe_loss = np.where(loss > 0, loss, np.nan)
    sd_share = 100 * sd_loss / safe_loss
    l1_share = 100 * l1 / safe_loss if l1 is not None else None
    ortho_share = 100 * ortho / safe_loss if ortho is not None else None

    # ortho stacks between the other two, so the bands read in the same order as the
    # legend above. Without it the npz is an absolute-deviation history and the
    # aad_loss/l1 pair stacks exactly as it did before.
    stack_shares = [np.nan_to_num(sd_share)]
    stack_colors = [COLOR_SD]
    stack_labels = ["aad_loss"]
    if ortho_share is not None:
        stack_shares.append(np.nan_to_num(ortho_share))
        stack_colors.append(COLOR_ORTHO)
        stack_labels.append("ortho")
    if l1_share is not None:
        stack_shares.append(np.nan_to_num(l1_share))
        stack_colors.append(COLOR_L1)
        stack_labels.append("l1")

    ax_share.stackplot(
        epoch,
        *stack_shares,
        colors=stack_colors,
        labels=stack_labels,
        alpha=0.85,
    )
    ax_share.set_ylim(0, 100)
    ax_share.set_ylabel("share of total loss (%)")
    ax_share.set_xlabel("epoch")
    ax_share.legend(loc="center right", fontsize=9, framealpha=0.9)

    # The l1 band is the one that moves, and by the end it is far too thin to see.
    # Annotate its share at the first and last recorded epoch so the collapse is
    # legible even when the band itself has vanished against the top edge.
    for x, share, ha in (
        ()
        if l1_share is None
        else (
            (epoch[0], l1_share[0], "left"),
            (epoch[-1], l1_share[-1], "right"),
        )
    ):
        if np.isfinite(share):
            ax_share.annotate(
                f"l1 = {share:.2f}%",
                xy=(x, 100),
                xytext=(0, -34),
                textcoords="offset points",
                ha=ha,
                fontsize=9,
                color=COLOR_L1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_L1, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=COLOR_L1, lw=1),
            )

    # ortho is the other band that shrinks, and it ends up a sliver between the two.
    # It needs no leader line because it can be written straight into the middle of
    # its own band, which is where these labels sit.
    if ortho_share is not None:
        band_middle = np.nan_to_num(sd_share) + np.nan_to_num(ortho_share) / 2
        for i, ha in ((0, "left"), (-1, "right")):
            if np.isfinite(ortho_share[i]):
                ax_share.annotate(
                    f"ortho = {ortho_share[i]:.2f}%",
                    xy=(epoch[i], band_middle[i]),
                    ha=ha,
                    va="center",
                    fontsize=9,
                    color=COLOR_ORTHO,
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec=COLOR_ORTHO, alpha=0.9
                    ),
                )

    if args.log_x:
        ax_share.set_xscale("log")
    # Clamp to the data range (shared with the top panel): the default margins leave
    # the stacked area floating in white gutters at both ends of its own panel.
    ax_share.set_xlim(epoch[0], epoch[-1])

    # Stamp the run identity when training recorded it (the periodic mid-run flush
    # writes the arrays only, so a partial history has no hparams to report).
    if "hparams_json" in data.files:
        hp = json.loads(data["hparams_json"].item())
        # gene / num_filters / kernel_size are recorded only by pipeline runs; older
        # histories simply omit them and stamp the line they always did.
        fields = []
        gene = hp.get("gene_label") or hp.get("gene")
        if gene:
            fields.append(f"gene={gene}")
        if hp.get("num_filters") is not None:
            fields.append(f"filters={hp['num_filters']}+{hp['num_filters']}")
        if hp.get("kernel_size") is not None:
            fields.append(f"kernel={hp['kernel_size']}nt")
        fields += [f"seed={hp.get('seed')}", f"lr={hp.get('lr')}"]
        # ortho_lambda and pool_length are recorded only by the orthogonality run, so
        # an absolute-deviation history stamps the same line it always did.
        fields += [f"{k}={hp[k]}" for k in ("ortho_lambda", "pool_length") if k in hp]
        # Same rule as the curves above: a disabled L1 term is not part of this run's
        # loss, so naming its lambda in the stamp would describe a term the figure
        # (correctly) does not show.
        if l1 is not None:
            fields.append(f"l1_lambda={hp.get('l1_lambda')}")
        fields += [
            f"smooth_sigma={hp.get('smooth_sigma')}",
            f"{len(epoch)} points over {hp.get('num_epochs')} epochs",
        ]
        fig.suptitle("  ".join(fields), fontsize=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Wrote training curves over {len(epoch)} recorded epochs to {args.out}"
    )


if __name__ == "__main__":
    main()
