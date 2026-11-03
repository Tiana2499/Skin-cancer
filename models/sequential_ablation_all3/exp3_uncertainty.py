"""
exp3_uncertainty.py
Uses per-fold CatBoost predictions for the best config.
"""

import os
import numpy as np
import joblib
import sklearn.metrics as sklearn_metrics


model_dir = r"./models/sequential_ablation_all3"
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

# Per-fold predictions were not saved — regenerate by reloading models.
# The ablation script does NOT save per-fold models.
# Option: Use OOF variance as proxy. Recompute from OOF preds.

# Alternative: load the two image OOF probs and tabular features, retrain 5 folds
# and save per-fold test predictions.

# Simplest: use test predictions std across the 5 folds.
# You need per-fold test predictions from the ablation.

# Check if saved
for fold in range(1, 6):
    f = os.path.join(model_dir, f"tinyvit+densenet_fold{fold}_test_cat.npy")
    if not os.path.exists(f):
        print(f"Missing per-fold: {f}")