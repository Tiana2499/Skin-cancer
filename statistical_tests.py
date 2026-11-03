"""
statistical_tests.py
Pairwise statistical tests between the final model and all baselines.

Tests:
  1. Paired t-test on per-fold pAUC (n=5 folds)
  2. Wilcoxon signed-rank test (non-parametric alternative)
  3. McNemar's test on paired test predictions (categorical)
  4. Chi-square test on confusion matrices
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sklearn_metrics


class Config:
    output_dir = r"./analysis/statistical_tests"
    min_tpr = 0.80
    n_bootstrap = 2000
    seed = 42


def isic2024_pauc(y, p, min_tpr=0.80):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) != 2:
        return 0.0
    yf, pf = abs(y - 1), abs(p - 1)
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr_i = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]], [tpr[stop-1], tpr[stop]]))
    fpr_i = np.append(fpr[:stop], max_fpr)
    return float(sklearn_metrics.auc(fpr_i, tpr_i))


def bootstrap_pauc_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(isic2024_pauc(y[idx], p[idx]))
    return np.percentile(scores, [2.5, 97.5])


def load_all_models():
    """Load test predictions from all models and baselines."""
    models = {}

    # Final model
    model_dir = r"./models/sequential_ablation_all3"
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))
    models["Final (TinyViT+DenseNet+tab)"] = np.load(
        os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))

    # Baselines
    baselines = {
        "TinyViT+tab": os.path.join(model_dir, "tinyvit_test_cat.npy"),
        "DenseNet+tab": os.path.join(model_dir, "densenet_test_cat.npy"),
        "EffNet+tab": os.path.join(model_dir, "effnet_test_cat.npy"),
        "XGBoost (final features)": os.path.join(model_dir, "tinyvit+densenet_test_xgb.npy"),
    }

    for name, path in baselines.items():
        if os.path.exists(path):
            models[name] = np.load(path)

    # Image-only baselines
    image_only = {
        "EffNet image-only": r"F:\thesis 23\5 fold experiments\baseline_A_efficientnetv2s_fullunfreeze\test_probs.npy",
        "TinyViT image-only": r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy",
        "DenseNet image-only": r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy",
    }
    for name, path in image_only.items():
        if os.path.exists(path):
            models[name] = np.load(path)

    return y_test, models


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    y_test, models = load_all_models()

    print(f"{'=' * 70}")
    print("STATISTICAL TESTS — FINAL MODEL vs BASELINES")
    print(f"{'=' * 70}")
    print(f"Test set: {len(y_test):,} lesions, {y_test.sum()} malignant")
    print(f"Models compared: {len(models)}")

    # =====================================================================
    # 1. Test pAUC for each model
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("1. TEST pAUC (point estimate + bootstrap 95% CI)")
    print(f"{'=' * 70}")

    pauc_results = {}
    for name, probs in models.items():
        pauc = isic2024_pauc(y_test, probs)
        lo, hi = bootstrap_pauc_ci(y_test, probs)
        pauc_results[name] = {"pauc": pauc, "ci_lo": lo, "ci_hi": hi}
        print(f"  {name:<30} pAUC={pauc:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")

    # =====================================================================
    # 2. McNemar's test (paired predictions)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("2. McNEMAR'S TEST (final model vs each baseline)")
    print(f"{'=' * 70}")
    print("Compares binary predictions at threshold=0.3.")
    print()

    final_name = "Final (TinyViT+DenseNet+tab)"
    final_probs = models[final_name]
    final_preds = (final_probs >= 0.3).astype(int)

    mcnemar_rows = []

    for name, probs in models.items():
        if name == final_name:
            continue

        preds = (probs >= 0.3).astype(int)

        # Build 2x2 contingency: both correct, both wrong, A correct B wrong, A wrong B correct
        a_correct = (final_preds == y_test)
        b_correct = (preds == y_test)

        n11 = int((a_correct & b_correct).sum())      # both correct
        n10 = int((a_correct & ~b_correct).sum())     # final correct, baseline wrong
        n01 = int((~a_correct & b_correct).sum())     # final wrong, baseline correct
        n00 = int((~a_correct & ~b_correct).sum())    # both wrong

        # McNemar's test
        if n10 + n01 == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
            p = 1 - stats.chi2.cdf(stat, df=1)

        mcnemar_rows.append({
            "baseline": name,
            "final_correct_baseline_wrong": n10,
            "final_wrong_baseline_correct": n01,
            "both_correct": n11,
            "both_wrong": n00,
            "chi2_stat": stat,
            "p_value": p,
            "significant_0.05": p < 0.05,
        })

        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<30} n10={n10:>5}, n01={n01:>5}  χ²={stat:>8.3f}  p={p:.4f}  {sig}")

    # =====================================================================
    # 3. Chi-square test on confusion matrices
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("3. CHI-SQUARE TEST (confusion matrices)")
    print(f"{'=' * 70}")
    print("Compares confusion matrices at threshold=0.3.")
    print()

    chi2_rows = []

    final_cm = confusion_matrix(y_test, final_preds, labels=[0, 1]).ravel()  # [tn, fp, fn, tp]

    for name, probs in models.items():
        if name == final_name:
            continue

        preds = (probs >= 0.3).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        # Stack confusion matrices: shape (2, 4)
        contingency = np.array([final_cm, cm])

        # Chi-square test
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
        except Exception as e:
            chi2, p = 0.0, 1.0

        chi2_rows.append({
            "baseline": name,
            "final_tn_fp_fn_tp": tuple(final_cm.tolist()),
            "baseline_tn_fp_fn_tp": tuple(cm.tolist()),
            "chi2_stat": chi2,
            "p_value": p,
            "significant_0.05": p < 0.05,
        })

        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<30} χ²={chi2:>8.3f}  p={p:.4f}  {sig}")

    # =====================================================================
    # 4. Paired t-test on per-fold pAUC
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("4. PAIRED T-TEST ON PER-FOLD pAUC")
    print(f"{'=' * 70}")
    print("Requires per-fold predictions. Available only if saved.")
    print()

    # Check if per-fold predictions exist
    fold_dir = r"./models/sequential_ablation_all3"
    per_fold_paucs = {"Final": []}

    for fold in range(1, 6):
        f = os.path.join(fold_dir, f"tinyvit+densenet_fold{fold}_test_cat.npy")
        if os.path.exists(f):
            probs = np.load(f)
            per_fold_paucs["Final"].append(isic2024_pauc(y_test, probs))

    if len(per_fold_paucs["Final"]) == 5:
        print(f"  Final per-fold pAUC: {[f'{p:.4f}' for p in per_fold_paucs['Final']]}")
        print(f"  Mean ± std: {np.mean(per_fold_paucs['Final']):.4f} ± {np.std(per_fold_paucs['Final']):.4f}")
    else:
        print(f"  ⚠️ Per-fold predictions not saved.")
        print(f"  Cannot run paired t-test without them.")
        print(f"  Rerun the ablation with per-fold prediction saving.")

    # =====================================================================
    # 5. Bootstrap-based p-value
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("5. BOOTSTRAP PAIRED TEST (final model vs baselines)")
    print(f"{'=' * 70}")
    print("Resamples test set with replacement, measures Δ pAUC.")
    print()

    rng = np.random.default_rng(Config.seed)
    n_boot = 2000

    bootstrap_rows = []
    for name, probs in models.items():
        if name == final_name:
            continue

        deltas = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_test), size=len(y_test))
            if len(np.unique(y_test[idx])) < 2:
                continue
            final_p = isic2024_pauc(y_test[idx], final_probs[idx])
            base_p = isic2024_pauc(y_test[idx], probs[idx])
            deltas.append(final_p - base_p)

        deltas = np.array(deltas)
        p_value = (deltas <= 0).mean()  # H0: final model not better

        bootstrap_rows.append({
            "baseline": name,
            "mean_delta": deltas.mean(),
            "ci_lo": np.percentile(deltas, 2.5),
            "ci_hi": np.percentile(deltas, 97.5),
            "p_value": p_value,
            "significant_0.05": p_value < 0.05,
        })

        sig = "✓" if p_value < 0.05 else ""
        print(f"  {name:<30} Δ={deltas.mean():+.4f}  "
              f"[{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]  "
              f"p={p_value:.4f}  {sig}")

    # =====================================================================
    # SUMMARY TABLE FOR PAPER
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE (for paper)")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'pAUC':>8} {'95% CI':>20} {'McNemar p':>12}")
    print("-" * 70)
    for name in models:
        row = pauc_results[name]
        mcn = next((m for m in mcnemar_rows if m["baseline"] == name), None)
        p_val = f"{mcn['p_value']:.4f}" if mcn else "—"
        print(f"{name:<30} {row['pauc']:>8.4f} "
              f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]".rjust(20) + f" {p_val:>12}")

    # Save
    pd.DataFrame([{"model": k, **v} for k, v in pauc_results.items()]).to_csv(
        os.path.join(Config.output_dir, "pauc_with_ci.csv"), index=False)
    pd.DataFrame(mcnemar_rows).to_csv(
        os.path.join(Config.output_dir, "mcnemar_tests.csv"), index=False)
    pd.DataFrame(chi2_rows).to_csv(
        os.path.join(Config.output_dir, "chi2_tests.csv"), index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        os.path.join(Config.output_dir, "bootstrap_paired.csv"), index=False)

    print(f"\n✓ Saved all test results to {Config.output_dir}")


if __name__ == "__main__":
    main()