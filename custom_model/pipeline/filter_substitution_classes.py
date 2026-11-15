"""What the filters are *invariant to*: the substitution structure of a trained bank.

The logo readouts in ``run_pipeline.py`` ask which nucleotide each kernel position
prefers, and answer with an argmax consensus like ``CAGTTC``. That question presumes the
filter is a motif detector. When a bank instead learns *degenerate* positions -- ones
that score A and G alike, or C and G alike -- the argmax is close to meaningless: it
reports the winner of a tie the model never broke, and the logo shows a stack of similar
letters that reads as "mushy" rather than as the structured statement it actually is.

This script asks the complementary question. A filter position is a 4-vector of weights
``(w_A, w_C, w_G, w_T)``, and the input is one-hot with exactly one nucleotide per
column, so adding a constant to all four weights adds a constant to every score and is
absorbed by the bias. **The per-position gauge freedom is exactly the all-ones
direction**, and the meaningful part of the position is the mean-centered vector, which
lives in a 3-dimensional space. That space has a canonical orthogonal basis -- the three
ways to split ``{A,C,G,T}`` into two pairs, which are precisely the three standard
chemical partitions:

    h_RY = (+1, -1, +1, -1)/2    purine {A,G}  vs pyrimidine {C,T}
    h_SW = (-1, +1, +1, -1)/2    strong  {C,G}  vs weak       {A,T}
    h_KM = (-1, -1, +1, +1)/2    keto    {G,T}  vs amino      {A,C}

(the Klein four-group / Hadamard basis on the alphabet, the same one phylogenetics uses
for Hadamard conjugation). They are mutually orthogonal and unit-norm, so **every
gauge-fixed position decomposes into exactly these three components and their energies
sum to the position's total energy with nothing left over.** There is no residual term:
"how motif-like is this position" is not a fourth axis, it is the *combination* of the
three. A position that cares about one nucleotide alone (say A vs the rest) spreads its
energy evenly, 1/3 to each axis, so an even split is what "plain motif" looks like and
**1/3 is the null share for each axis** -- confirmed here by two independent nulls:
randomly initialized banks, and the ``k=1`` runs, which cannot form a motif at all.

Each axis names the substitutions the position is *blind* to. Loading only on ``h_RY``
means A and G score identically and C and T score identically, i.e. the position is
invariant to both transitions and sensitive only to transversions. So the energy split
across the three axes is a direct, complete statement of which nucleotide substitutions
the model decided do not matter -- the thing the argmax consensus throws away.

The same decomposition is reported in substitution space as well, because that is the
form comparable to evolutionary data. For each of the 6 unordered pairs ``x<->y`` the
**sensitivity** is the mean of ``(w_y - w_x)^2`` over every position of the bank: the
mean squared change in a filter's pre-activation score if that substitution were made.
The script pairs it with the **empirical substitution frequency** of the same pair,
counted directly off the gene's own alignment matrix (per aligned column, over all pairs
of species present), and correlates the two. A model that has learned the alignment's
substitution process as an invariance shows a *negative* correlation: the substitutions
that happen most are the ones it responds to least.

Two honest limits on that correlation. It is 6 points, and it is dominated by the
transition/transversion dichotomy -- ``--transversions-only`` reruns it within the four
transversions, where agreement is much weaker and inconsistent across genes. And the
column-permutation control in ``control_testing/shuffle_train.py`` is *not* a null for
any of this: permuting the column axis preserves each column's cross-species composition
exactly, which is the entire signal measured here, and only destroys motif ordering. It
therefore leaves the substitution structure intact (indeed sharpens it), which is a
result about what that control does and does not rule out, not evidence of an artifact.

Finally, for the R/Y-dominant filters the script prints the **R/Y word**: the sign of the
purine/pyrimidine component at each kernel position, read as a string like ``RYRRYR``.
This is the consensus motif the bank actually learned, written in the 2-letter alphabet
it is written in, and it is what should be compared across seeds and genes instead of the
4-letter argmax. Words are strand-collapsed against their R/Y reverse complement before
counting, so a motif and its reverse complement are one word.

**Channel order.** The analysis reads the alphabet each checkpoint was trained on
(``hparams["alphabet"]``, via ``custom_utils.resolve_alphabet``) and permutes the channels
back to A,C,G,T before anything else, so every number above is in true letters whatever
order the model saw. Runs over a reduced alphabet (R/Y) are skipped: they have no
within-class contrast to decompose. The shares are *also* computed on the raw slot order,
reading channel ``j`` as if it held ``ACGT[j]`` (``slot_share_*`` in ``per_run.csv``); for an
A/C/G/T run the two agree exactly.

That second view is what the C/G-swapped control (``run_pipeline_agct.py``) is read with.
When any analyzed run uses a reordered alphabet, it is paired by run name with the A/C/G/T
run of the same name in ``--reference-runs-dir`` and the script reports whether the
preference **follows the letters** (the dominant axis moves to a different slot-order axis
and stays put in true letters) or **follows the slots** (it stays put in slot order, and
so moves in true letters). Under the A/G/C/T order, true R/Y appears as slot-order K/M and
S/W is unmoved. The slot view is a relabelling of the true-letter view, not a second
independent test; it is reported so the swap is visible.

    python pipeline/filter_substitution_classes.py
    python pipeline/filter_substitution_classes.py --arch f20_k6 --transversions-only
    python pipeline/filter_substitution_classes.py --runs-dir /path/to/shuffled/runs
    python pipeline/filter_substitution_classes.py --runs-dir pipeline/runs_agct \
        --out-dir pipeline/analysis/filter_substitution_classes_agct

Outputs land in ``pipeline/analysis/filter_substitution_classes/``:

    per_run.csv         one row per run: alphabet, axis energy shares (true letters and
                        slot order), dominant-axis filter counts, the 6 substitution
                        sensitivities, tv/ti ratio
    per_filter.csv      one row per filter: energy, the three shares, dominant axis, R/Y word
    substitution_match.csv  per gene: alignment pair frequency vs model sensitivity + r
    ry_words.csv        recurrent R/Y words per (gene, arch) with seed support
    summary.txt         the same, formatted, including the random-init null
    axis_energy.png     axis shares vs kernel size against the 1/3 null, and the
                        sensitivity-vs-frequency scatter

and, only when reordered-alphabet runs are present:

    alphabet_control.csv  per paired seed: reference shares, control slot-order and
                          true-letter shares, dominant axis of each, outcome
    alphabet_control.png  reference vs control shares per axis, seed dots, 1/3 null
"""

import argparse
import csv
import collections
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
from custom_utils import ALPHABETS, resolve_alphabet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HERE = _HERE
DEFAULT_RUNS_DIR = os.path.join(HERE, "runs")
DEFAULT_OUT_DIR = os.path.join(HERE, "analysis", "filter_substitution_classes")

RUN_NAME_RE = re.compile(r"^(?P<gene>.+?)_f(?P<f>\d+)_k(?P<k>\d+)_s(?P<s>\d+)$")

NT = "ACGT"
# The three 2|2 partitions of the alphabet, as unit-norm contrast vectors over (A,C,G,T).
# Mutually orthogonal and jointly spanning the 3-dim mean-centered space, so the energies
# of a position's three projections sum to its total energy exactly.
AXES = {
    "R/Y": np.array([1.0, -1.0, 1.0, -1.0]) / 2.0,  # purine {A,G} / pyrimidine {C,T}
    "S/W": np.array([-1.0, 1.0, 1.0, -1.0]) / 2.0,  # strong {C,G} / weak {A,T}
    "K/M": np.array([-1.0, -1.0, 1.0, 1.0]) / 2.0,  # keto {G,T} / amino {A,C}
}
AXIS_NAMES = list(AXES)
NULL_SHARE = 1.0 / 3.0  # a single-nucleotide preference splits evenly across the axes

PAIRS = list(combinations(range(4), 2))
PAIR_LABELS = [f"{NT[a]}<->{NT[b]}" for a, b in PAIRS]
TRANSITIONS = {(0, 2), (1, 3)}  # A<->G, C<->T
IS_TRANSVERSION = np.array([p not in TRANSITIONS for p in PAIRS])

# Categorical slots 1-3 of the validated default palette, matching filter_bank_distance.py.
AXIS_STYLE = {"R/Y": "#2a78d6", "S/W": "#eb6834", "K/M": "#1baf7a"}
GRID_COLOR = "#d8d7d2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def gauge_fix(weights):
    """Mean-center each kernel position over the 4 nucleotides.

    The one-hot input puts exactly one nucleotide in every column, so a constant added
    down a column shifts every score equally and is absorbed by the bias. Mean-centering
    is the orthogonal projection that removes precisely that direction, which is what
    makes the three axis energies below sum to the total. (``plot_filters.py`` centers on
    the *median* instead, a robust choice for display; it is not an orthogonal projection,
    so it is deliberately not used here.)

    Args:
        weights: array of shape ``(n_filters, 4, kernel_size)``.

    Returns:
        The same shape, centered along the nucleotide axis.
    """
    return weights - weights.mean(axis=1, keepdims=True)


def axis_energies(centered):
    """Per-position energy on each of the three partition axes.

    Args:
        centered: gauge-fixed weights, ``(n_filters, 4, kernel_size)``.

    Returns:
        ``(coeffs, energies)`` where ``coeffs[axis]`` is ``(n_filters, kernel_size)``
        signed projections and ``energies[axis]`` is their squares.
    """
    v = centered.transpose(0, 2, 1)  # (n_filters, kernel_size, 4)
    coeffs = {name: v @ h for name, h in AXES.items()}
    return coeffs, {name: c**2 for name, c in coeffs.items()}


def shares(energies):
    """Fraction of total gauge-fixed energy carried by each axis (sums to 1)."""
    total = sum(e.sum() for e in energies.values())
    return {name: float(e.sum() / total) if total > 0 else 0.0 for name, e in energies.items()}


def sensitivity(centered):
    """Mean squared score change per unordered substitution, over every position.

    Args:
        centered: gauge-fixed weights, ``(n_filters, 4, kernel_size)``.

    Returns:
        Length-6 array aligned with ``PAIRS`` / ``PAIR_LABELS``.
    """
    v = centered.transpose(0, 2, 1).reshape(-1, 4)
    return np.array([float(((v[:, b] - v[:, a]) ** 2).mean()) for a, b in PAIRS])


def ry_word(ry_coeffs):
    """Read one filter's purine/pyrimidine component as an R/Y string."""
    return "".join("R" if c > 0 else "Y" for c in ry_coeffs)


def strand_collapse(word):
    """Fold an R/Y word together with its reverse complement (R and Y swap)."""
    rc = "".join("Y" if c == "R" else "R" for c in reversed(word))
    return min(word, rc)


def alignment_pair_frequencies(matrix_path):
    """Empirical unordered-pair frequencies from an alignment matrix.

    For each aligned column, every pair of species present contributes one count to the
    pair of nucleotides they carry; identical pairs land on the diagonal and are dropped.
    This is a raw pairwise-difference spectrum, not a phylogenetically corrected rate
    matrix -- deep alignments are saturated and the species are not independent -- so it
    is used only for the rank/linear comparison against model sensitivity.

    Args:
        matrix_path: path to ``alignment_matrix.npy``, ``(n_aligned, n_species)`` of
            single characters (gaps ``-``/``N``, possibly soft-masked lowercase).

    Returns:
        Length-6 array aligned with ``PAIRS``, normalized to sum to 1.
    """
    matrix = np.char.upper(np.load(matrix_path).astype(str))
    codes = np.full(matrix.shape, -1, dtype=np.int8)
    for i, c in enumerate(NT):
        codes[matrix == c] = i
    counts = np.zeros((4, 4))
    for col in range(codes.shape[0]):
        row = codes[col]
        row = row[row >= 0]
        if row.size < 2:
            continue
        b = np.bincount(row, minlength=4).astype(float)
        counts += np.outer(b, b)
    np.fill_diagonal(counts, 0.0)
    freq = np.array([counts[a, b] for a, b in PAIRS])
    return freq / freq.sum() if freq.sum() > 0 else freq


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def spearman(a, b):
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        return r

    return pearson(rank(a), rank(b))


def load_bank(run_dir):
    """Incl+skip weights from a run's checkpoint, in the checkpoint's own channel order.

    Returns:
        ``(weights, n_incl, alphabet_name, letters)``: weights ``(2F, n_chan, K)`` with
        channel ``j`` holding ``letters[j]``. The alphabet is the stamped
        ``hparams["alphabet"]`` when present, else inferred from the channel count --
        the unstamped A/C/G/T checkpoints from run_pipeline.py resolve to ``"acgt"``.
    """
    ckpt = torch.load(os.path.join(run_dir, "weights.pt"), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    incl, skip = sd["conv_incl.weight"].numpy(), sd["conv_skip.weight"].numpy()
    hparams = (ckpt.get("metadata") or {}).get("hparams", {})
    alphabet_name, letters = resolve_alphabet(incl.shape[1], hparams.get("alphabet"))
    return np.concatenate([incl, skip], axis=0), incl.shape[0], alphabet_name, letters


def slot_axis(letters, axis_name):
    """The slot-order axis a true-letter axis shows up as under a channel order.

    Reading channel ``j`` as ``ACGT[j]`` when it really holds ``letters[j]`` relabels the
    alphabet, and any relabelling maps the three 2|2 partitions onto one another (up to
    sign). Under ``"AGCT"``: R/Y <-> K/M, S/W unmoved. Under ``"ACGT"``: identity.
    """
    h_slot = np.array([AXES[axis_name][NT.index(b)] for b in letters])
    for name, h in AXES.items():
        if np.isclose(abs(float(h_slot @ h)), 1.0):
            return name
    raise ValueError(f"{letters!r} is not a permutation of {NT}")


def random_init_null(num_filters, kernel_size, seeds):
    """Axis shares of untrained banks: the empirical check that the null share is 1/3."""
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = PNASModel(num_filters=num_filters, kernel_size=kernel_size)
        w = np.concatenate(
            [
                model.conv_incl.weight.detach().numpy(),
                model.conv_skip.weight.detach().numpy(),
            ],
            axis=0,
        )
        _, energies = axis_energies(gauge_fix(w))
        out.append(shares(energies))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory of trained run folders."
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help="Where to write the outputs."
    )
    parser.add_argument(
        "--reference-runs-dir",
        default=DEFAULT_RUNS_DIR,
        help="A/C/G/T runs that reordered-alphabet runs are paired with by run name, "
        "for the channel-order control readout.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Restrict to one architecture, written as f<F>_k<K> (e.g. f20_k6).",
    )
    parser.add_argument(
        "--gene", default=None, help="Restrict to one gene directory name (e.g. multiz100)."
    )
    parser.add_argument(
        "--dominant-threshold",
        type=float,
        default=0.5,
        help="Share of a filter's energy on one axis before it is called dominant.",
    )
    parser.add_argument(
        "--word-threshold",
        type=float,
        default=0.6,
        help="R/Y share required before a filter contributes an R/Y word.",
    )
    parser.add_argument(
        "--transversions-only",
        action="store_true",
        help="Also report the sensitivity/frequency correlation within transversions, "
        "where the transition-vs-transversion dichotomy cannot carry it.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for name in sorted(os.listdir(args.runs_dir)):
        m = RUN_NAME_RE.match(name)
        if not m or not os.path.exists(os.path.join(args.runs_dir, name, "weights.pt")):
            continue
        gene, f, k, s = m["gene"], int(m["f"]), int(m["k"]), int(m["s"])
        if args.arch and f"f{f}_k{k}" != args.arch:
            continue
        if args.gene and gene != args.gene:
            continue
        runs.append((name, gene, f, k, s))
    if not runs:
        raise SystemExit(f"No runs matched under {args.runs_dir}.")
    logger.info(f"{len(runs)} runs matched under {args.runs_dir}")

    run_rows, filter_rows = [], []
    ry_words = collections.defaultdict(collections.Counter)
    ry_word_seeds = collections.defaultdict(lambda: collections.defaultdict(set))

    for name, gene, f, k, s in runs:
        weights, n_incl, alphabet_name, letters = load_bank(
            os.path.join(args.runs_dir, name)
        )
        # The decomposition needs base identity over 4 channels. A reduced alphabet (R/Y)
        # has no within-class contrast left to decompose.
        if sorted(letters) != sorted(NT):
            logger.info(f"  {name}: {'/'.join(letters)} alphabet, skipping")
            continue
        # Shares in the raw channel order, reading slot j as if it held NT[j]. For an
        # A/C/G/T run this is already the true-letter answer; for a reordered control it
        # is the view in which a slot-driven preference would stay put.
        _, slot_energies = axis_energies(gauge_fix(weights))
        slot_share = shares(slot_energies)
        # Everything below is in true letters: permute the channels back to A,C,G,T.
        weights = weights[:, [letters.index(b) for b in NT], :]
        centered = gauge_fix(weights)
        coeffs, energies = axis_energies(centered)
        run_share = shares(energies)
        sens = sensitivity(centered)
        ti = float(sens[~IS_TRANSVERSION].mean())
        tv = float(sens[IS_TRANSVERSION].mean())

        dom_counts = collections.Counter()
        for fi in range(weights.shape[0]):
            per = {a: float(energies[a][fi].sum()) for a in AXIS_NAMES}
            total = sum(per.values())
            sh = {a: per[a] / total if total > 0 else 0.0 for a in AXIS_NAMES}
            best = max(sh, key=sh.get)
            dominant = best if sh[best] >= args.dominant_threshold else "mixed"
            dom_counts[dominant] += 1
            word = ry_word(coeffs["R/Y"][fi])
            filter_rows.append(
                {
                    "run": name, "gene": gene, "num_filters": f, "kernel": k, "seed": s,
                    "bank": "incl" if fi < n_incl else "skip", "index": fi % n_incl,
                    "abs_w_centered": round(float(np.abs(centered[fi]).sum()), 4),
                    **{f"share_{a.replace('/', '')}": round(sh[a], 4) for a in AXIS_NAMES},
                    "dominant_axis": dominant, "ry_word": word,
                }
            )
            if sh["R/Y"] >= args.word_threshold:
                key = (gene, f"f{f}_k{k}")
                collapsed = strand_collapse(word)
                ry_words[key][collapsed] += 1
                ry_word_seeds[key][collapsed].add(s)

        run_rows.append(
            {
                "run": name, "gene": gene, "num_filters": f, "kernel": k, "seed": s,
                "alphabet": alphabet_name,
                **{f"share_{a.replace('/', '')}": round(run_share[a], 4) for a in AXIS_NAMES},
                **{
                    f"slot_share_{a.replace('/', '')}": round(slot_share[a], 4)
                    for a in AXIS_NAMES
                },
                **{f"n_dominant_{a.replace('/', '')}": dom_counts[a] for a in AXIS_NAMES},
                "n_mixed": dom_counts["mixed"],
                **{f"sens_{lab.replace('<->', '')}": round(float(v), 6)
                   for lab, v in zip(PAIR_LABELS, sens)},
                "sens_transition": round(ti, 6), "sens_transversion": round(tv, 6),
                "tv_over_ti": round(tv / ti, 4) if ti > 0 else float("nan"),
            }
        )

    # ── Per-gene comparison against the alignment's own substitution spectrum ──
    match_rows = []
    by_gene = collections.defaultdict(list)
    for row in run_rows:
        by_gene[row["gene"]].append(row)
    for gene, rows in sorted(by_gene.items()):
        matrix_path = os.path.join(ROOT, "data", gene, "alignment_matrix.npy")
        if not os.path.exists(matrix_path):
            logger.info(f"  no alignment matrix for {gene}, skipping frequency match")
            continue
        freq = alignment_pair_frequencies(matrix_path)
        # One architecture at a time: sensitivity magnitude is not comparable across
        # kernel widths, and averaging seeds within an arch is the meaningful pooling.
        for arch in sorted({f"f{r['num_filters']}_k{r['kernel']}" for r in rows}):
            sel = [r for r in rows if f"f{r['num_filters']}_k{r['kernel']}" == arch]
            sens = np.array(
                [[r[f"sens_{lab.replace('<->', '')}"] for lab in PAIR_LABELS] for r in sel]
            ).mean(axis=0)
            sens_n = sens / sens.sum()
            row = {
                "gene": gene, "arch": arch, "n_seeds": len(sel),
                "pearson_all6": round(pearson(freq, sens_n), 4),
                "spearman_all6": round(spearman(freq, sens_n), 4),
                "pearson_transversions": round(
                    pearson(freq[IS_TRANSVERSION], sens_n[IS_TRANSVERSION]), 4
                ),
            }
            for lab, fr, sn in zip(PAIR_LABELS, freq, sens_n):
                tag = lab.replace("<->", "")
                row[f"freq_{tag}"] = round(float(fr), 4)
                row[f"sens_{tag}"] = round(float(sn), 4)
            match_rows.append(row)

    # ── Write CSVs ──
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"wrote {path}")

    write_csv(os.path.join(args.out_dir, "per_run.csv"), run_rows)
    write_csv(os.path.join(args.out_dir, "per_filter.csv"), filter_rows)
    write_csv(os.path.join(args.out_dir, "substitution_match.csv"), match_rows)

    word_rows = []
    for (gene, arch), counter in sorted(ry_words.items()):
        for word, n in counter.most_common():
            word_rows.append(
                {
                    "gene": gene, "arch": arch, "ry_word": word, "n_filters": n,
                    "n_seeds": len(ry_word_seeds[(gene, arch)][word]),
                }
            )
    write_csv(os.path.join(args.out_dir, "ry_words.csv"), word_rows)

    # ── Channel-order control: pair reordered-alphabet runs with their A/C/G/T twins ──
    # run_pipeline_agct.py writes untagged names to its own tree, so a control run shares
    # its name with the A/C/G/T run of the same gene, architecture and seed -- and at the
    # same seed the two started from the same initial weight tensor.
    control_rows = []
    for r in run_rows:
        if r["alphabet"] == "acgt":
            continue
        ref_dir = os.path.join(args.reference_runs_dir, r["run"])
        if not os.path.exists(os.path.join(ref_dir, "weights.pt")):
            logger.info(
                f"  {r['run']}: no run of that name in {args.reference_runs_dir}, unpaired"
            )
            continue
        ref_weights, _, ref_alphabet, _ = load_bank(ref_dir)
        if ref_alphabet != "acgt":
            logger.info(f"  {r['run']}: reference run is {ref_alphabet}, not acgt, unpaired")
            continue
        _, ref_energies = axis_energies(gauge_fix(ref_weights))
        ref_share = shares(ref_energies)
        true_share = {a: r[f"share_{a.replace('/', '')}"] for a in AXIS_NAMES}
        slot_sh = {a: r[f"slot_share_{a.replace('/', '')}"] for a in AXIS_NAMES}
        ref_dom = max(ref_share, key=ref_share.get)
        true_dom = max(true_share, key=true_share.get)
        slot_dom = max(slot_sh, key=slot_sh.get)
        moved_to = slot_axis(ALPHABETS[r["alphabet"]], ref_dom)
        if moved_to == ref_dom:
            outcome = "uninformative"  # this channel order leaves the reference axis put
        elif true_dom == ref_dom:
            outcome = "letters"
        elif slot_dom == ref_dom:
            outcome = "slots"
        else:
            outcome = "other"
        control_rows.append(
            {
                "run": r["run"], "gene": r["gene"], "num_filters": r["num_filters"],
                "kernel": r["kernel"], "seed": r["seed"], "control_alphabet": r["alphabet"],
                **{f"ref_share_{a.replace('/', '')}": round(ref_share[a], 4)
                   for a in AXIS_NAMES},
                **{f"slot_share_{a.replace('/', '')}": slot_sh[a] for a in AXIS_NAMES},
                **{f"true_share_{a.replace('/', '')}": true_share[a] for a in AXIS_NAMES},
                "reference_dominant": ref_dom,
                "slot_dominant": slot_dom,
                "true_dominant": true_dom,
                "letters_predicts_slot_axis": moved_to,
                "outcome": outcome,
            }
        )
    write_csv(os.path.join(args.out_dir, "alphabet_control.csv"), control_rows)

    # One verdict per (gene, arch, control alphabet), shared by summary.txt and the figure.
    control_groups = collections.defaultdict(list)
    for r in control_rows:
        key = (r["gene"], f"f{r['num_filters']}_k{r['kernel']}", r["control_alphabet"])
        control_groups[key].append(r)
    control_verdicts = {}
    for key, rows in control_groups.items():
        letters = ALPHABETS[key[2]]
        ref_mean = {
            a: np.mean([r[f"ref_share_{a.replace('/', '')}"] for r in rows]) for a in AXIS_NAMES
        }
        ref_dom = max(ref_mean, key=ref_mean.get)
        moved_to = slot_axis(letters, ref_dom)
        # The true-letter axis that would read as ref_dom in slot order: where a
        # slot-driven preference ends up once relabelled.
        slots_true = next(a for a in AXIS_NAMES if slot_axis(letters, a) == ref_dom)
        seeds_by = collections.defaultdict(list)
        for r in sorted(rows, key=lambda r: r["seed"]):
            seeds_by[r["outcome"]].append(r["seed"])
        n = len(rows)
        if len(seeds_by["letters"]) == n:
            short = "FOLLOWS LETTERS"
            text = (
                f"FOLLOWS LETTERS in all {n} seeds: the dominant axis reads as {moved_to} in "
                f"slot order and stays {ref_dom} in true letters. The preference is carried "
                "by the nucleotides, not by the channel slots."
            )
        elif len(seeds_by["slots"]) == n:
            short = "FOLLOWS SLOTS"
            text = (
                f"FOLLOWS SLOTS in all {n} seeds: the dominant axis stays {ref_dom} in slot "
                f"order, i.e. becomes {slots_true} in true letters. The preference is an "
                "encoding or analysis artifact, not a property of the nucleotides."
            )
        elif len(seeds_by["uninformative"]) == n:
            short = "UNINFORMATIVE"
            text = (
                f"UNINFORMATIVE: channel order {'/'.join(letters)} does not move {ref_dom}, "
                "so it cannot separate letters from slots for this axis."
            )
        else:
            short = "MIXED"
            text = "MIXED: " + "; ".join(
                f"{outcome} in seeds {seeds}" for outcome, seeds in sorted(seeds_by.items())
            )
        control_verdicts[key] = {
            "short": short, "text": text, "ref_dom": ref_dom, "moved_to": moved_to,
        }

    # ── summary.txt ──
    null_arch = collections.Counter((r["num_filters"], r["kernel"]) for r in run_rows)
    (nf, nk), _ = null_arch.most_common(1)[0]
    nulls = random_init_null(nf, nk, sorted({r["seed"] for r in run_rows}))
    null_mean = {a: float(np.mean([n[a] for n in nulls])) for a in AXIS_NAMES}

    lines = []
    lines.append("Substitution-class decomposition of the trained filter banks")
    lines.append("=" * 74)
    lines.append("")
    lines.append(
        "Each gauge-fixed kernel position decomposes exactly onto the three 2|2 "
        "partitions\nof {A,C,G,T}; the shares sum to 1 and 1/3 is the null (a "
        "single-nucleotide preference\nspreads evenly). An axis above 1/3 means the "
        "model grouped those nucleotides together,\ni.e. made itself blind to the "
        "substitutions inside each group."
    )
    lines.append("")
    lines.append(
        f"NULL, random init (f{nf} k{nk}, {len(nulls)} seeds): "
        + "  ".join(f"{a}={null_mean[a]:.3f}" for a in AXIS_NAMES)
    )
    lines.append("")
    lines.append("--- mean axis share over seeds, by gene x kernel ---")
    lines.append(
        f"{'gene':12s} {'k':>3s} {'n':>3s} " + " ".join(f"{a:>7s}" for a in AXIS_NAMES)
        + f" {'tv/ti':>7s}"
    )
    grouped = collections.defaultdict(list)
    for r in run_rows:
        grouped[(r["gene"], r["kernel"])].append(r)
    for (gene, k) in sorted(grouped):
        rows = grouped[(gene, k)]
        vals = [
            np.mean([r[f"share_{a.replace('/', '')}"] for r in rows]) for a in AXIS_NAMES
        ]
        tvti = np.mean([r["tv_over_ti"] for r in rows])
        lines.append(
            f"{gene:12s} {k:3d} {len(rows):3d} "
            + " ".join(f"{v:7.3f}" for v in vals)
            + f" {tvti:7.2f}"
        )
    lines.append("")
    lines.append("--- model sensitivity vs alignment substitution frequency ---")
    lines.append(
        "A model that absorbed the alignment's substitution process shows r < 0: the\n"
        "substitutions that happen most are the ones the filters respond to least.\n"
        "Six points, and the all-6 value is carried by the transition/transversion\n"
        "split -- the transversion-only column is the honest residual test."
    )
    lines.append("")
    lines.append(
        f"{'gene':12s} {'arch':10s} {'r(all 6)':>9s} {'rho(all 6)':>11s} {'r(tv only)':>11s}"
    )
    for r in match_rows:
        lines.append(
            f"{r['gene']:12s} {r['arch']:10s} {r['pearson_all6']:9.2f} "
            f"{r['spearman_all6']:11.2f} {r['pearson_transversions']:11.2f}"
        )
    lines.append("")
    lines.append("--- recurrent R/Y words (strand-collapsed, R/Y-dominant filters) ---")
    for (gene, arch), counter in sorted(ry_words.items()):
        top = counter.most_common(6)
        lines.append(
            f"{gene:12s} {arch:10s} "
            + "  ".join(
                f"{w}(n={n},seeds={len(ry_word_seeds[(gene, arch)][w])})" for w, n in top
            )
        )
    if control_rows:

        def mean_sd(values):
            values = np.asarray(values, float)
            sd = values.std(ddof=1) if values.size > 1 else 0.0
            return f"{values.mean():.3f} ± {sd:.3f}"

        lines.append("")
        lines.append("--- channel-order control: does the preference follow letters or slots? ---")
        lines.append(
            "Each control run is paired by name with the A/C/G/T run of the same gene,\n"
            "architecture and seed (identical initial weights, assigned to different\n"
            "letters). The slot-order row reads the control's channels as if they were\n"
            "A/C/G/T; it is the true-letter row relabelled, shown so the swap is visible,\n"
            "not a second independent test. Shares are mean ± SD over paired seeds."
        )
        for key in sorted(control_groups):
            gene, arch, alph = key
            rows = sorted(control_groups[key], key=lambda r: r["seed"])
            verdict = control_verdicts[key]
            letters = ALPHABETS[alph]
            lines.append("")
            lines.append(
                f"{gene} {arch}: channel order {'/'.join(letters)} vs A/C/G/T, "
                f"{len(rows)} paired seeds"
            )
            lines.append(
                f"  under this order true {verdict['ref_dom']} reads as slot-order "
                f"{verdict['moved_to']}"
            )
            tags = {a: a.replace("/", "") for a in AXIS_NAMES}
            lines.append(f"  {'':24s}" + "".join(f"{a:>16s}" for a in AXIS_NAMES))
            for label, prefix in (
                ("reference, true letters", "ref_share"),
                ("control, slot order", "slot_share"),
                ("control, true letters", "true_share"),
            ):
                cells = [
                    mean_sd([r[f"{prefix}_{tags[a]}"] for r in rows]) for a in AXIS_NAMES
                ]
                lines.append(f"  {label:24s}" + "".join(f"{c:>16s}" for c in cells))
            diffs = [
                mean_sd([r[f"true_share_{tags[a]}"] - r[f"ref_share_{tags[a]}"] for r in rows])
                for a in AXIS_NAMES
            ]
            lines.append(f"  {'control - reference':24s}" + "".join(f"{c:>16s}" for c in diffs))
            lines.append("  per seed (reference -> control slot order / true letters):")
            for r in rows:
                lines.append(
                    f"    s{r['seed']}: {r['reference_dominant']} -> {r['slot_dominant']} / "
                    f"{r['true_dominant']}   {r['outcome']}"
                )
            lines.append(f"  VERDICT: {verdict['text']}")

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"wrote {summary_path}")
    print("\n".join(lines))

    # ── Figure: axis share vs kernel width, and sensitivity vs frequency ──
    fig, axes_ = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes_[0]
    kernels = sorted({r["kernel"] for r in run_rows})
    for a in AXIS_NAMES:
        means = [
            np.mean([r[f"share_{a.replace('/', '')}"] for r in run_rows if r["kernel"] == k])
            for k in kernels
        ]
        sds = [
            np.std([r[f"share_{a.replace('/', '')}"] for r in run_rows if r["kernel"] == k])
            for k in kernels
        ]
        ax.errorbar(
            kernels, means, yerr=sds, marker="o", capsize=3, color=AXIS_STYLE[a], label=a
        )
    ax.axhline(NULL_SHARE, color=TEXT_SECONDARY, ls="--", lw=1)
    ax.text(
        kernels[-1], NULL_SHARE, " null = 1/3", va="bottom", ha="right",
        color=TEXT_SECONDARY, fontsize=9,
    )
    ax.set_xlabel("kernel size (nt)")
    ax.set_ylabel("share of gauge-fixed filter energy")
    ax.set_title("Which substitutions the bank is blind to", color=TEXT_PRIMARY)
    ax.set_xticks(kernels)
    ax.grid(alpha=0.4, color=GRID_COLOR)
    ax.legend(frameon=False)

    # Transitions and transversions are colored apart because the two clouds are the
    # whole point: the strong all-6 correlation is the gap *between* them, and the
    # spread *within* the transversion cloud is how little else the model resolved.
    ax = axes_[1]
    for is_tv, color, label in (
        (False, "#eb6834", "transition (A<->G, C<->T)"),
        (True, "#2a78d6", "transversion"),
    ):
        xs, ys = [], []
        for r in match_rows:
            for lab, tv in zip(PAIR_LABELS, IS_TRANSVERSION):
                if bool(tv) != is_tv:
                    continue
                tag = lab.replace("<->", "")
                xs.append(r[f"freq_{tag}"])
                ys.append(r[f"sens_{tag}"])
        ax.scatter(xs, ys, s=22, alpha=0.7, color=color, label=label)
    if match_rows:
        r0 = match_rows[0]
        for lab in PAIR_LABELS:
            tag = lab.replace("<->", "")
            ax.annotate(
                lab, (r0[f"freq_{tag}"], r0[f"sens_{tag}"]),
                fontsize=8, color=TEXT_SECONDARY,
                xytext=(5, 5), textcoords="offset points",
            )
    ax.set_xlabel("alignment pair frequency")
    ax.set_ylabel("model sensitivity (normalized)")
    ax.set_title("Frequent substitutions move the score least", color=TEXT_PRIMARY)
    ax.grid(alpha=0.4, color=GRID_COLOR)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "axis_energy.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {fig_path}")

    # ── Figure: channel-order control ──
    # Color stays bound to the axis (same hues as axis_energy.png) and the x groups are the
    # three views, so the swap shows up as the tallest bar changing color in the middle
    # group and changing back in the right-hand one. Aqua sits under 3:1 contrast on the
    # light surface, so every bar carries its value as a visible label.
    if control_rows:
        conditions = (
            ("reference\nA/C/G/T, true letters", "ref_share"),
            ("control\nslot order", "slot_share"),
            ("control\ntrue letters", "true_share"),
        )
        width = 0.26
        keys = sorted(control_groups)
        fig, axs = plt.subplots(
            1, len(keys), figsize=(7.5 * len(keys), 5.6), squeeze=False
        )
        for ax, key in zip(axs[0], keys):
            gene, arch, alph = key
            rows = control_groups[key]
            top = 0.0
            for ci, (_, prefix) in enumerate(conditions):
                for ai, a in enumerate(AXIS_NAMES):
                    vals = np.array([r[f"{prefix}_{a.replace('/', '')}"] for r in rows])
                    x = ci + (ai - 1) * width
                    mean = float(vals.mean())
                    # White edge = the 2px surface gap between adjacent bars.
                    ax.bar(
                        x, mean, width=width, color=AXIS_STYLE[a], edgecolor="white",
                        linewidth=2, label=a if ci == 0 else None, zorder=2,
                    )
                    jitter = (
                        np.linspace(-0.06, 0.06, vals.size) if vals.size > 1 else np.zeros(1)
                    )
                    ax.scatter(
                        x + jitter, vals, s=14, color=TEXT_PRIMARY, alpha=0.55,
                        edgecolors="white", linewidths=0.6, zorder=3,
                    )
                    label_y = max(mean, float(vals.max())) + 0.012
                    ax.text(
                        x, label_y, f"{mean:.2f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT_PRIMARY, zorder=4,
                    )
                    top = max(top, label_y)
            ax.axhline(NULL_SHARE, color=TEXT_SECONDARY, ls="--", lw=1, zorder=1)
            ax.text(
                -0.55, NULL_SHARE + 0.008, "null = 1/3", ha="left", va="bottom",
                fontsize=8, color=TEXT_SECONDARY,
            )
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels([label for label, _ in conditions], fontsize=9)
            ax.set_xlim(-0.6, len(conditions) - 0.4)
            ax.set_ylim(0, max(0.75, top + 0.06))
            ax.set_ylabel("share of gauge-fixed filter energy")
            ax.grid(axis="y", alpha=0.4, color=GRID_COLOR, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            verdict = control_verdicts[key]
            ax.set_title(
                f"{gene} {arch}, channel order {'/'.join(ALPHABETS[alph])}: "
                f"{verdict['short']}\n"
                f"true {verdict['ref_dom']} should read as slot-order {verdict['moved_to']} "
                f"if the preference follows the letters  ({len(rows)} seeds, dots)",
                color=TEXT_PRIMARY, fontsize=10, pad=28,
            )
            ax.legend(
                ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                frameon=False, fontsize=9,
            )
        fig.tight_layout()
        control_fig_path = os.path.join(args.out_dir, "alphabet_control.png")
        fig.savefig(control_fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"wrote {control_fig_path}")


if __name__ == "__main__":
    main()
