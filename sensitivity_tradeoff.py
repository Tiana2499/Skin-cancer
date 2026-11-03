"""
sensitivity_tradeoff.py
Report sensitivity at fixed specificity for all models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
import sklearn.metrics as sklearn_metrics


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


def sensitivity_at_specificity(y, p, target_specificity):
    """Find threshold that yields the target specificity, return (threshold, sensitivity, threshold_used)."""
    fpr, tpr, thresholds = roc_curve(y, p)
    specificity = 1 - fpr

    # Find closest specificity to target
    idx = np.argmin(np.abs(specificity - target_specificity))
    return thresholds[idx], tpr[idx], specificity[idx]


def full_metrics_at_threshold(y, p, threshold):
    preds = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"threshold": threshold, "sensitivity": sens, "specificity": spec,
            "precision": prec, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    model_dir = r"./models/sequential_ablation_all3"
    y_test = np.load(os.path.join(model_dir, "y_test.npy"))

    # All models to compare
    models = {
        "TinyViT only":           r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN\test_probs.npy",
        "DenseNet only":          r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold\test_probs.npy",
        "Tabular only":           r"./models/tabular_only/test_probs.npy",
        "TinyViT + DenseNet":     os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"),
        "EffNet + TinyViT + DenseNet": os.path.join(model_dir, "effnet+tinyvit+densenet_test_cat.npy"),
    }

    print(f"{'=' * 90}")
    print("SENSITIVITY AT FIXED SPECIFICITY")
    print(f"{'=' * 90}")
    print(f"{'Model':<40} {'Spec 0.90':>12} {'Spec 0.95':>12} {'Spec 0.99':>12}")
    print("-" * 90)

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"{name:<40} SKIP (missing file)")
            continue
        probs = np.load(path)
        if len(probs) != len(y_test):
            print(f"{name:<40} SKIP (length mismatch: {len(probs)} vs {len(y_test)})")
            continue

        sens_90 = full_metrics_at_threshold(y_test, probs,
            sensitivity_at_specificity(y_test, probs, 0.90)[0])["sensitivity"]
        sens_95 = full_metrics_at_threshold(y_test, probs,
            sensitivity_at_specificity(y_test, probs, 0.95)[0])["sensitivity"]
        sens_99 = full_metrics_at_threshold(y_test, probs,
            sensitivity_at_specificity(y_test, probs, 0.99)[0])["sensitivity"]

        print(f"{name:<40} {sens_90:>12.4f} {sens_95:>12.4f} {sens_99:>12.4f}")

    # =========================================================================
    # HIGH-SENSITIVITY MODELS
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("SENSITIVITY AT FIXED SPECIFICITY (the trade-off)")
    print(f"{'=' * 90}")

    # Best model deep dive
    final_probs = np.load(os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))

    print(f"\nFinal Model: TinyViT + DenseNet + 38 tabular")
    print(f"{'Target Spec':<15} {'Threshold':>12} {'Sensitivity':>12} {'Precision':>12} {'TP':>5} {'FP':>6} {'FN':>5}")
    print("-" * 90)
    for target_spec in [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]:
        _, _, actual_spec = sensitivity_at_specificity(y_test, final_probs, target_spec)
        # Recompute using the returned specificity's threshold
        thr, _, _ = sensitivity_at_specificity(y_test, final_probs, target_spec)
        m = full_metrics_at_threshold(y_test, final_probs, thr)
        print(f"{target_spec:<15.2f} {m['threshold']:>12.4f} {m['sensitivity']:>12.4f} "
              f"{m['precision']:>12.4f} {m['tp']:>5d} {m['fp']:>6d} {m['fn']:>5d}")


if __name__ == "__main__":
    main()