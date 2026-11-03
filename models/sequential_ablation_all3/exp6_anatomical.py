"""
exp6_anatomical.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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


model_dir = r"./models/sequential_ablation_all3"
cat_probs = np.load(os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

# Merge with test manifest for site column
test_manifest = pd.read_csv(r"F:\thesis 23\new exp segment\cohort_manifests\test_manifest_with_site.csv")
meta_test = pd.read_parquet(os.path.join(model_dir, "meta_test.parquet"))

meta_test = meta_test.merge(
    test_manifest[["isic_id", "anatom_site_general"]],
    on="isic_id", how="left")

for site in meta_test["anatom_site_general"].dropna().unique():
    mask = meta_test["anatom_site_general"] == site
    y_site = y_test[mask]
    p_site = cat_probs[mask]
    if len(np.unique(y_site)) < 2:
        continue
    pauc = isic2024_pauc(y_site, p_site)
    print(f"{site:<20s} n={mask.sum():>6d} n_pos={y_site.sum():>3d} pAUC={pauc:.4f}")