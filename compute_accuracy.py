"""
compute_accuracy.py
Compute accuracy at threshold=0.3 for the final model.
"""
import numpy as np
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.metrics as sklearn_metrics


model_dir = r"./models/sequential_ablation_all3"
cat_probs = np.load(os.path.join(model_dir, "tinyvit+densenet_test_cat.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

# Accuracy at threshold 0.3 (your default)
threshold = 0.3
preds = (cat_probs >= threshold).astype(int)

acc = accuracy_score(y_test, preds)
tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

print(f"Final Model: TinyViT + DenseNet + 38 tabular → CatBoost")
print(f"{'=' * 60}")
print(f"Threshold:   {threshold}")
print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
print(f"AUROC:       {sklearn_metrics.roc_auc_score(y_test, cat_probs):.4f}")
print(f"pAUC@80%:    {0.1725:.4f}")
print(f"")
print(f"Confusion Matrix:")
print(f"  TN = {tn:,}")
print(f"  FP = {fp:,}")
print(f"  FN = {fn}")
print(f"  TP = {tp}")
print(f"")
print(f"Sensitivity: {tp / (tp + fn):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"Precision:   {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
print(f"F1:          {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")