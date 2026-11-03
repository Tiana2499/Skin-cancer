"""
exp2_nnt.py
"""

import os
import numpy as np


def compute_nnt(y, p, target_sens):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    total_pos = y.sum()
    order = np.argsort(-p)
    y_sorted = y[order]
    cumsum_tp = np.cumsum(y_sorted)
    target_tp = int(np.ceil(target_sens * total_pos))
    idx = np.searchsorted(cumsum_tp, target_tp)
    n_flagged = idx + 1
    tp = int(cumsum_tp[idx])
    return n_flagged, tp, n_flagged / tp if tp > 0 else np.inf


model_dir = r"./models/sequential_ablation_all3"
cat_probs = np.load(os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

for sens in [0.80, 0.90]:
    flagged, tp, nnt = compute_nnt(y_test, cat_probs, sens)
    print(f"At {sens*100:.0f}% sensitivity: flagged={flagged}, TP={tp}, NNT={nnt:.1f}")

print(f"\nReference: ISIC 2024 winner NNT@80% = 51.57")