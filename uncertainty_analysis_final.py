"""
uncertainty_analysis_final.py
Ensemble uncertainty for the final model (TinyViT + DenseNet + 38 tabular).
Uses per-fold CatBoost predictions.
"""

import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pointbiserialr, spearmanr
import sklearn.metrics as sklearn_metrics


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    output_dir = r"./analysis/uncertainty_final"
    min_tpr = 0.80
    threshold = 0.3


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


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load test labels and meta
    y_test = np.load(os.path.join(Config.model_dir, "y_test.npy"))
    meta_test = pd.read_parquet(os.path.join(Config.model_dir, "meta_test.parquet"))

    # Load per-fold CatBoost test predictions
    fold_probs = []
    for fold in range(1, 6):
        model = joblib.load(os.path.join(
            Config.model_dir, f"tinyvit+densenet_cat_fold{fold}.pkl"))
        X_test = np.load(os.path.join(Config.model_dir, "final_X_test.npy"))
        fold_probs.append(model.predict_proba(X_test)[:, 1])

    fold_probs = np.array(fold_probs)  # (5, N)
    print(f"Fold predictions shape: {fold_probs.shape}")

    # Uncertainty metrics
    mean_prob = fold_probs.mean(axis=0)
    std_prob = fold_probs.std(axis=0)
    var_prob = fold_probs.var(axis=0)
    eps = 1e-7
    entropy = -(mean_prob * np.log(mean_prob + eps)
                + (1 - mean_prob) * np.log(1 - mean_prob + eps))

    print(f"\nUncertainty statistics:")
    print(f"  Std(prob):  mean={std_prob.mean():.4f}  max={std_prob.max():.4f}")
    print(f"  Var(prob):  mean={var_prob.mean():.6f}")
    print(f"  Entropy:    mean={entropy.mean():.4f}")

    # =====================================================================
    # Q1: Does uncertainty predict errors?
    # =====================================================================
    preds = (mean_prob >= Config.threshold).astype(int)
    errors = (preds != y_test).astype(int)

    correct_unc = std_prob[errors == 0]
    error_unc = std_prob[errors == 1]

    print(f"\n{'=' * 70}")
    print("Q1: Does uncertainty predict errors?")
    print(f"{'=' * 70}")
    print(f"  Correct: n={len(correct_unc):,}  mean_std={correct_unc.mean():.4f}")
    print(f"  Error:   n={len(error_unc):,}  mean_std={error_unc.mean():.4f}")

    corr, pval = pointbiserialr(errors, std_prob)
    print(f"  Point-biserial corr: r={corr:.4f}  p={pval:.2e}")

    # =====================================================================
    # Q2: Uncertainty vs marginal predictions
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Q2: Uncertainty vs marginal predictions")
    print(f"{'=' * 70}")
    marginal = np.abs(mean_prob - 0.5)
    rho, p_rho = spearmanr(std_prob, -marginal)
    print(f"  Spearman corr (std, closeness to 0.5): rho={rho:.4f}  p={p_rho:.2e}")

    # =====================================================================
    # Q3: Uncertainty at 80% sensitivity operating point
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Q3: Uncertainty at 80% sensitivity")
    print(f"{'=' * 70}")

    order = np.argsort(-mean_prob)
    y_sorted = y_test[order]
    cumsum_tp = np.cumsum(y_sorted)
    total_pos = y_test.sum()
    target_tp = int(np.ceil(0.80 * total_pos))
    idx_80 = np.searchsorted(cumsum_tp, target_tp)
    threshold_80 = mean_prob[order[idx_80]]
    preds_80 = (mean_prob >= threshold_80).astype(int)

    tp_mask = (y_test == 1) & (preds_80 == 1)
    fn_mask = (y_test == 1) & (preds_80 == 0)
    fp_mask = (y_test == 0) & (preds_80 == 1)
    tn_mask = (y_test == 0) & (preds_80 == 0)

    print(f"  Threshold @ 80% sens: {threshold_80:.4f}")
    print(f"  True Positives  (n={tp_mask.sum():>3}): mean_std={std_prob[tp_mask].mean():.4f}")
    print(f"  False Negatives (n={fn_mask.sum():>3}): mean_std={std_prob[fn_mask].mean():.4f}")
    print(f"  False Positives (n={fp_mask.sum():>5}): mean_std={std_prob[fp_mask].mean():.4f}")
    print(f"  True Negatives  (n={tn_mask.sum():>5}): mean_std={std_prob[tn_mask].mean():.4f}")

    # =====================================================================
    # Q4: Selective prediction
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("Q4: Selective prediction (keep N% most confident)")
    print(f"{'=' * 70}")

    order_unc = np.argsort(std_prob)
    for frac in [1.0, 0.9, 0.75, 0.5]:
        n_keep = int(frac * len(y_test))
        keep_idx = order_unc[:n_keep]
        y_keep = y_test[keep_idx]
        p_keep = mean_prob[keep_idx]
        if len(np.unique(y_keep)) < 2:
            continue
        pauc_keep = isic2024_pauc(y_keep, p_keep)
        print(f"  Keep {frac*100:>3.0f}%: pAUC={pauc_keep:.4f}  (n_pos={int(y_keep.sum())})")

    # =====================================================================
    # Save
    # =====================================================================
    out = pd.DataFrame({
        "isic_id": meta_test["isic_id"],
        "patient_id": meta_test["patient_id"],
        "target": y_test,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "entropy": entropy,
    })
    out.to_parquet(os.path.join(Config.output_dir, "uncertainty.parquet"), index=False)
    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()