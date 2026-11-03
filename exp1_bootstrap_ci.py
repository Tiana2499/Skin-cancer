"""
exp1_bootstrap_ci.py
"""

import os
import numpy as np
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


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(isic2024_pauc(y[idx], p[idx]))
    return np.percentile(scores, [2.5, 97.5])


model_dir = r"./models/sequential_ablation_all3"
cat_probs = np.load(os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

point = isic2024_pauc(y_test, cat_probs)
lo, hi = bootstrap_ci(y_test, cat_probs)

print(f"Final Model: TinyViT + DenseNet + 38 tabular → CatBoost")
print(f"Test pAUC:  {point:.4f}")
print(f"95% CI:     [{lo:.4f}, {hi:.4f}]")
print(f"Test AUROC: {sklearn_metrics.roc_auc_score(y_test, cat_probs):.4f}")