"""Head-to-head comparison of the real-trained and column-shuffle-trained models.

``control_sr_deviation.py`` scores *one* model (the real-trained one) on many
matrices. That leaves the two training runs compared on different data: the real
model's loss comes from the real alignment, the control model's from the shuffled
one, so the difference confounds "what the model learned" with "which matrix it
was scored on".

This script fills in the missing cells of the 2x2 design by scoring *both*
checkpoints on *both* kinds of matrix with a single fixed metric:

                          | real matrix | shuffled matrix
    real-trained          |      x      |       x
    shuffle-trained       |      x      |       x

The decisive cell is *shuffle-trained on the real matrix*: both models are then
scored on identical data, so any gap is attributable purely to what training saw.
If the shuffle-trained filters score as well as the real-trained ones there, the
real run learned nothing conservation-specific.

Matrices evaluated:
  * ``real``           — the original MALAT1 multiz alignment.
  * ``train_shuffle``  — the *exact* permutation ``shuffle_train.py`` trained on
                         (``np.random.default_rng(0).permutation(n_aligned)``), i.e.
                         the control model's own training data.
  * ``shuffled``       — ``N_SHUFFLE`` independent permutations, giving each model a
                         spread over shuffles rather than a single draw.

The metric is imported directly from ``control_sr_deviation`` so it is bit-for-bit
the one already in use (same Gaussian smoothing, same scatter-to-column, same
NaN-aware numerator/denominator). Its ``MIN_SPECIES`` gate is set here to 30 to
match what *both* training runs used; the metric is also re-run at the module's
own default of 5 for continuity with ``control_results.npz``.

Beyond the table, the per-column deviation vectors on the real matrix are saved
for both models and compared *paired by column* — the same 7k columns scored twice
— via a bootstrap CI on the mean difference and a sign test, which says whether the
real-trained advantage is a broad shift or a few outlying columns.
"""

import os
import sys

import numpy as np
import torch

# Same import dance as control_sr_deviation.py: custom_model/ first on sys.path so
# the regular module custom_model/custom_model.py wins over the namespace package,
# then this directory so the control module itself is importable.
HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, CUSTOM_DIR)
sys.path.insert(0, HERE)
import control_sr_deviation as csd  # noqa: E402
from custom_model import PNASModel  # noqa: E402

MATRIX_PATH = os.path.join(CUSTOM_DIR, "..", "data", "multiz100", "alignment_matrix.npy")
OUTPUT_PATH = os.path.join(HERE, "cross_model_results.npz")

MODELS = {
    "real_trained": os.path.join(CUSTOM_DIR, "weights", "real_weights_aad.pt"),
    "shuffle_trained": os.path.join(HERE, "shuffle_weights.pt"),
    # The same 2x2 for the orthogonality-penalized loss. Kept alongside the AAD pair
    # rather than replacing it so the original comparison stays reproducible; each pair
    # must be read together, since a real/shuffle contrast is only meaningful between
    # two models that trained the same objective.
    "real_trained_ortho": os.path.join(CUSTOM_DIR, "weights", "real_weights_ortho.pt"),
    "shuffle_trained_ortho": os.path.join(HERE, "shuffle_weights_ortho.pt"),
}

N_SHUFFLE = 100  # independent permutations, for the per-model spread over shuffles
SHUFFLE_SEED = 0  # must equal shuffle_train.py's SHUFFLE_SEED to rebuild its matrix
EVAL_SEED = 1  # separate stream for the N_SHUFFLE independent permutations
# Evaluation threshold, not a training one. It matches the AAD pair, which both trained
# at 30; the ortho pair trained at 31, so for those two this is one species looser than
# their own loss used. Left at 30 deliberately — every model is then scored on the same
# column set, and the existing cross_model_results.npz stays comparable.
MIN_SPECIES = 30
LEGACY_MIN_SPECIES = 5  # control_sr_deviation.py's own default, for continuity
SMOOTH_SIGMA = 1.5  # matches the loss
N_BOOTSTRAP = 10000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = np.load(MATRIX_PATH)
    n_aligned, n_species = matrix.shape
    print(f"Alignment {matrix.shape} (aligned x species) on {device}")

    # The control model's own training matrix: same permutation shuffle_train.py used.
    train_shuffle = matrix[np.random.default_rng(SHUFFLE_SEED).permutation(n_aligned)]

    models = {}
    for name, path in MODELS.items():
        model = PNASModel().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
        models[name] = model
        print(f"Loaded {name:16s} <- {os.path.normpath(path)}")

    gauss_kernel, smooth_radius = csd.build_smoothing_kernel(SMOOTH_SIGMA, device)

    def evaluate(mat, model, min_species):
        # compute_metric reads MIN_SPECIES off its own module, so set it explicitly
        # rather than editing control_sr_deviation.py.
        csd.MIN_SPECIES = min_species
        return csd.compute_metric(mat, model, gauss_kernel, smooth_radius, device)

    results = {}

    # ── 1. The 2x2 table on fixed matrices ───────────────────────────────────
    print(f"\n=== 2x2 table (MIN_SPECIES={MIN_SPECIES}, matching both training runs) ===")
    print(f"{'model':<16s} {'matrix':<15s} {'norm_loss':>10s} {'mean_col_ad':>12s} {'within_scale':>13s}")
    for mname, model in models.items():
        for dname, mat in (("real", matrix), ("train_shuffle", train_shuffle)):
            m = evaluate(mat, model, MIN_SPECIES)
            col_ad, wss = m["col_abs_dev"], m["within_species_scale"]
            mean_ad, nloss = m["mean_col_abs_dev"], m["norm_loss"]
            results[f"{mname}__{dname}__col_abs_dev"] = col_ad
            results[f"{mname}__{dname}__within_species_scale"] = np.float64(wss)
            results[f"{mname}__{dname}__mean_col_abs_dev"] = np.float64(mean_ad)
            results[f"{mname}__{dname}__norm_loss"] = np.float64(nloss)
            print(f"{mname:<16s} {dname:<15s} {nloss:>10.4f} {mean_ad:>12.6f} {wss:>13.6f}")

    # Same four cells at the legacy gate, so these numbers can be read next to the
    # existing control_results.npz (which was produced with MIN_SPECIES=5).
    print(f"\n--- same cells at the legacy gate (MIN_SPECIES={LEGACY_MIN_SPECIES}) ---")
    for mname, model in models.items():
        for dname, mat in (("real", matrix), ("train_shuffle", train_shuffle)):
            m = evaluate(mat, model, LEGACY_MIN_SPECIES)
            mean_ad, nloss = m["mean_col_abs_dev"], m["norm_loss"]
            results[f"{mname}__{dname}__norm_loss_min{LEGACY_MIN_SPECIES}"] = np.float64(nloss)
            print(f"{mname:<16s} {dname:<15s} norm_loss={nloss:.4f}  mean_col_ad={mean_ad:.6f}")

    # ── 2. Spread over N_SHUFFLE independent permutations, per model ─────────
    print(f"\n=== {N_SHUFFLE} independent shuffles (MIN_SPECIES={MIN_SPECIES}) ===")
    rng = np.random.default_rng(EVAL_SEED)
    shuffles = [matrix[rng.permutation(n_aligned)] for _ in range(N_SHUFFLE)]
    for mname, model in models.items():
        nloss = np.empty(N_SHUFFLE)
        mean_ad = np.empty(N_SHUFFLE)
        for i, mat in enumerate(shuffles):
            m = evaluate(mat, model, MIN_SPECIES)
            mean_ad[i], nloss[i] = m["mean_col_abs_dev"], m["norm_loss"]
        results[f"{mname}__shuffled__norm_loss_all"] = nloss
        results[f"{mname}__shuffled__mean_col_abs_dev_all"] = mean_ad
        print(
            f"{mname:<16s} norm_loss over shuffles = {nloss.mean():.4f} +/- {nloss.std(ddof=1):.4f} "
            f"[min {nloss.min():.4f}, max {nloss.max():.4f}]"
        )

    # ── 3. Paired-by-column comparison on the REAL matrix ────────────────────
    # Per-column contribution to norm_loss = col_abs_dev / that model's own
    # within-species scale, so the two models are compared on the loss's own units.
    a = results["real_trained__real__col_abs_dev"] / results["real_trained__real__within_species_scale"]
    b = results["shuffle_trained__real__col_abs_dev"] / results["shuffle_trained__real__within_species_scale"]
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = b[both] - a[both]  # > 0 means the real-trained model is better at that column
    results["paired_real_matrix_diff"] = diff

    boot_rng = np.random.default_rng(EVAL_SEED)
    idx = boot_rng.integers(0, diff.size, size=(N_BOOTSTRAP, diff.size))
    boot = diff[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac = float((diff > 0).mean())
    # Sign test, normal approximation (no scipy in this env).
    z = (frac - 0.5) * 2 * np.sqrt(diff.size)
    results["paired_ci_lo"] = np.float64(lo)
    results["paired_ci_hi"] = np.float64(hi)
    results["paired_frac_real_better"] = np.float64(frac)

    print(f"\n=== Paired by column on the REAL matrix ({diff.size} columns valid in both) ===")
    print(f"mean per-column advantage of real-trained = {diff.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"median advantage = {np.median(diff):.4f}   IQR [{np.percentile(diff, 25):.4f}, {np.percentile(diff, 75):.4f}]")
    print(f"columns where real-trained is lower: {frac * 100:.1f}%  (sign-test z = {z:.1f})")

    np.savez(OUTPUT_PATH, **results)
    print(f"\nSaved to {os.path.normpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
