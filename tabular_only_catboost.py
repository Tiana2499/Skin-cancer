"""
tabular_only_catboost.py
Train CatBoost on 38 engineered tabular features only (no image).
This is the control baseline: how much signal is in the tabular features?
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


class Config:
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    tabular_path = r"./features/tabular/trainval_tabular.parquet"
    tabular_test_path = r"./features/tabular/test_tabular.parquet"
    output_dir = r"./models/tabular_only"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]


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


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load tabular features (already aligned to OOF order in the ablation script)
    tab_full = pd.read_parquet(Config.tabular_path)
    tab_test = pd.read_parquet(Config.tabular_test_path)

    # Drop banned columns
    drop_cols = [c for c in Config.banned_features if c in tab_full.columns]
    if drop_cols:
        print(f"Dropping banned: {drop_cols}")
        tab_full = tab_full.drop(columns=drop_cols)
        tab_test = tab_test.drop(columns=drop_cols, errors="ignore")

    # Feature columns
    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    X = tab_full[tab_cols].values.astype(np.float32)
    X_test = tab_test[tab_cols].values.astype(np.float32)
    y = tab_full["target"].values.astype(int)
    y_test = tab_test["target"].values.astype(int)
    groups = tab_full["patient_id"].values

    print(f"Tabular features: {len(tab_cols)}")
    print(f"X shape: {X.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Positives (train+val): {y.sum()}")
    print(f"Positives (test):      {y_test.sum()}")

    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # 5-fold GroupKFold
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    oof_cat = np.zeros(len(y))
    test_cat = []

    for fold, (tr, va) in enumerate(splits, 1):
        X_tr, y_tr = X[tr], y[tr]
        X_va = X[va]

        cat_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_seed=Config.seed, verbose=0, thread_count=-1)
        cat_model.fit(X_tr, y_tr)

        oof_cat[va] = cat_model.predict_proba(X_va)[:, 1]
        test_cat.append(cat_model.predict_proba(X_test)[:, 1])

        fold_pauc = isic2024_pauc(y[va], oof_cat[va])
        print(f"Fold {fold}: val pAUC = {fold_pauc:.4f}")

    # Ensemble test predictions
    test_probs = np.mean(test_cat, axis=0)

    # =====================================================================
    # RESULTS
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("TABULAR-ONLY RESULTS (CatBoost, 38 engineered features)")
    print(f"{'=' * 70}")

    oof_pauc = isic2024_pauc(y, oof_cat)
    test_pauc = isic2024_pauc(y_test, test_probs)
    oof_auroc = roc_auc_score(y, oof_cat)
    test_auroc = roc_auc_score(y_test, test_probs)
    lo, hi = bootstrap_ci(y_test, test_probs)

    print(f"Pooled OOF pAUC:  {oof_pauc:.4f}")
    print(f"Pooled OOF AUROC: {oof_auroc:.4f}")
    print(f"\nTest pAUC:        {test_pauc:.4f}")
    print(f"Test pAUC 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"Test AUROC:       {test_auroc:.4f}")

    # Save
    np.save(os.path.join(Config.output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(Config.output_dir, "test_targets.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "oof_probs.npy"), oof_cat)
    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()