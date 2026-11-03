"""
anatomical_permutation_test.py
Permutation test comparing torso vs head/neck pAUC for the final model.
"""

import os
import numpy as np
import pandas as pd
import joblib
import sklearn.metrics as sklearn_metrics


class Config:
    model_dir = r"./models/sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    n_permutations = 5000
    seed = 42
    min_tpr = 0.80


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
    # Load final model predictions
    cat_probs = np.load(os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy"))
    y_test = np.load(os.path.join(Config.model_dir, "y_test.npy"))
    meta_test = pd.read_parquet(os.path.join(Config.model_dir, "meta_test.parquet"))

    # Load site column
    manifest = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest_with_site.csv"))
    meta_test = meta_test.merge(
        manifest[["isic_id", "anatom_site_general"]], on="isic_id", how="left")

    print(f"Total test samples: {len(meta_test)}")
    print(f"Positives: {y_test.sum()}")

    # Define groups
    head_neck_mask = meta_test["anatom_site_general"] == "head/neck"
    torso_mask = meta_test["anatom_site_general"].isin(["anterior torso", "posterior torso"])

    print(f"\nHead/Neck lesions: {head_neck_mask.sum()}  (malignant: {y_test[head_neck_mask].sum()})")
    print(f"Torso lesions:     {torso_mask.sum()}  (malignant: {y_test[torso_mask].sum()})")

    y_hn = y_test[head_neck_mask]
    p_hn = cat_probs[head_neck_mask]
    y_t = y_test[torso_mask]
    p_t = cat_probs[torso_mask]

    pauc_hn = isic2024_pauc(y_hn, p_hn)
    pauc_t = isic2024_pauc(y_t, p_t)
    observed_diff = pauc_t - pauc_hn

    print(f"\nHead/Neck pAUC: {pauc_hn:.4f}")
    print(f"Torso pAUC:     {pauc_t:.4f}")
    print(f"Observed difference (torso - head/neck): {observed_diff:+.4f}")

    # Permutation test
    print(f"\nRunning permutation test (n={Config.n_permutations})...")

    combined_y = np.concatenate([y_hn, y_t])
    combined_p = np.concatenate([p_hn, p_t])
    n_hn = len(y_hn)

    rng = np.random.default_rng(Config.seed)
    null_diffs = []

    for _ in range(Config.n_permutations):
        idx = rng.permutation(len(combined_y))
        y_perm = combined_y[idx]
        p_perm = combined_p[idx]

        y_hn_perm = y_perm[:n_hn]
        p_hn_perm = p_perm[:n_hn]
        y_t_perm = y_perm[n_hn:]
        p_t_perm = p_perm[n_hn:]

        if len(np.unique(y_hn_perm)) < 2 or len(np.unique(y_t_perm)) < 2:
            continue

        null_diffs.append(isic2024_pauc(y_t_perm, p_t_perm) - isic2024_pauc(y_hn_perm, p_hn_perm))

    null_diffs = np.array(null_diffs)
    p_value = (null_diffs >= observed_diff).mean()

    print(f"\nNull distribution: mean={null_diffs.mean():.4f}, std={null_diffs.std():.4f}")
    print(f"95th percentile of null: {np.percentile(null_diffs, 95):.4f}")
    print(f"Observed difference:     {observed_diff:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n✓ Disparity is statistically significant (p < 0.05)")
    else:
        print(f"\n✗ Disparity is NOT statistically significant (p >= 0.05)")


if __name__ == "__main__":
    main()