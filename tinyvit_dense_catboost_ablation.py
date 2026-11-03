"""
tinyvit_densenet_catboost_ablation.py

Isolated ablation:
    TinyViT + DenseNet121 OOF probabilities + tabular features
    -> CatBoost only

OOF predictions are aligned by ISIC ID.
Patient-level grouping is used for 5-fold StratifiedGroupKFold.
The held-out test set is evaluated once using the mean of the five
fold-specific CatBoost test predictions.
"""

import os
import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


class Config:
    backbone_dirs = {
        "densenet": r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold",
        "tinyvit": r"F:\thesis 23\new experiment\tinyvitFULLUNFROZEN",
    }

    tabular_path = r"./features/tabular/trainval_tabular.parquet"
    tabular_test_path = r"./features/tabular/test_tabular.parquet"

    output_dir = r"./models/tinyvit_densenet_catboost_only"

    n_splits = 5
    seed = 42
    min_tpr = 0.80

    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]


def isic2024_pauc(y, p, min_tpr=0.80):
    """
    Preserve the pAUC calculation from the supplied script.
    It calculates ROC area up to max_fpr = 1 - min_tpr,
    after reversing the binary labels and scores.
    """
    y = np.asarray(y)
    p = np.asarray(p)

    if len(np.unique(y)) != 2:
        return 0.0

    yf = abs(y - 1)
    pf = abs(p - 1)

    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)

    stop = np.searchsorted(fpr, max_fpr, side="right")

    if stop == 0:
        return 0.0

    if stop >= len(fpr):
        return float(sklearn_metrics.auc(fpr, tpr))

    tpr_i = np.append(
        tpr[:stop],
        np.interp(
            max_fpr,
            [fpr[stop - 1], fpr[stop]],
            [tpr[stop - 1], tpr[stop]],
        ),
    )
    fpr_i = np.append(fpr[:stop], max_fpr)

    return float(sklearn_metrics.auc(fpr_i, tpr_i))


def load_backbones():
    required = ["tinyvit", "densenet"]

    oof_data = {}
    oof_ids_data = {}
    test_data = {}

    for name in required:
        path = Config.backbone_dirs[name]

        oof_path = os.path.join(path, "oof_probs.npy")
        ids_path = os.path.join(path, "oof_ids.npy")
        test_path = os.path.join(path, "test_probs.npy")

        missing = [
            p for p in [oof_path, ids_path, test_path]
            if not os.path.exists(p)
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing required {name} files:\n" + "\n".join(missing)
            )

        oof_data[name] = np.load(oof_path)
        oof_ids_data[name] = np.load(ids_path, allow_pickle=True)
        test_data[name] = np.load(test_path)

        print(
            f"{name}: {len(oof_data[name]):,} OOF predictions, "
            f"{len(test_data[name]):,} test predictions"
        )

    return oof_data, oof_ids_data, test_data


def align_oof(oof_data, oof_ids_data):
    reference_ids = np.asarray(oof_ids_data["tinyvit"])
    reference_sorted = np.sort(reference_ids)

    aligned = {}

    for name in ["tinyvit", "densenet"]:
        ids = np.asarray(oof_ids_data[name])
        probs = np.asarray(oof_data[name])

        if len(ids) != len(reference_ids):
            raise ValueError(f"{name}: OOF ID count does not match TinyViT.")

        if set(ids.tolist()) != set(reference_ids.tolist()):
            raise ValueError(f"{name}: OOF ISIC ID set does not match TinyViT.")

        order = np.argsort(ids)
        sorted_ids = ids[order]

        if not np.array_equal(sorted_ids, reference_sorted):
            raise ValueError(f"{name}: OOF IDs could not be aligned.")

        aligned[name] = probs[order]

    return aligned, reference_sorted


def load_tabular(reference_ids):
    tab_full = pd.read_parquet(Config.tabular_path)
    tab_test = pd.read_parquet(Config.tabular_test_path)

    if "isic_id" not in tab_full.columns:
        raise KeyError("isic_id is missing from the train/validation parquet.")
    if "isic_id" not in tab_test.columns:
        raise KeyError("isic_id is missing from the test parquet.")

    tab_full = (
        tab_full.set_index("isic_id")
        .loc[reference_ids]
        .reset_index()
    )

    drop_cols = [
        col for col in Config.banned_features
        if col in tab_full.columns
    ]

    if drop_cols:
        print(f"Dropping banned features: {drop_cols}")
        tab_full = tab_full.drop(columns=drop_cols)
        tab_test = tab_test.drop(columns=drop_cols, errors="ignore")

    return tab_full, tab_test


def build_features(tab_full, tab_test, oof_aligned, test_data):
    excluded = {"isic_id", "patient_id", "target"}
    tab_cols = [c for c in tab_full.columns if c not in excluded]

    missing_test_cols = [c for c in tab_cols if c not in tab_test.columns]
    if missing_test_cols:
        raise KeyError(
            "These train/validation features are missing from test data: "
            + ", ".join(missing_test_cols)
        )

    # Two image-model probabilities: TinyViT and DenseNet121.
    oof_image = np.column_stack([
        oof_aligned["tinyvit"],
        oof_aligned["densenet"],
    ])

    test_image = np.column_stack([
        test_data["tinyvit"],
        test_data["densenet"],
    ])

    tab_train = tab_full[tab_cols].to_numpy(dtype=np.float32)
    tab_test_values = tab_test[tab_cols].to_numpy(dtype=np.float32)

    X_trainval = np.concatenate([oof_image, tab_train], axis=1)
    X_test = np.concatenate([test_image, tab_test_values], axis=1)

    return X_trainval, X_test, tab_cols


def train_and_evaluate(X, y, groups, X_test, y_test, tab_full, tab_test):
    os.makedirs(Config.output_dir, exist_ok=True)

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    splitter = StratifiedGroupKFold(
        n_splits=Config.n_splits,
        shuffle=True,
        random_state=Config.seed,
    )

    splits = list(splitter.split(X, y, groups))

    oof_cat = np.zeros(len(y), dtype=np.float64)
    test_fold_predictions = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_seed=Config.seed,
            verbose=100,
            thread_count=-1,
            loss_function="Logloss",
            eval_metric="AUC",
        )

        model.fit(X_train, y_train)

        oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
        test_fold_predictions.append(model.predict_proba(X_test)[:, 1])

        model_path = os.path.join(
            Config.output_dir,
            f"tinyvit_densenet_catboost_fold{fold}.cbm",
        )
        model.save_model(model_path)

        print(f"Saved fold {fold} model: {model_path}")

    test_cat = np.mean(test_fold_predictions, axis=0)

    # Metrics on the held-out test set.
    test_pauc = isic2024_pauc(y_test, test_cat, Config.min_tpr)
    test_auroc = roc_auc_score(y_test, test_cat)

    # Cross-validated OOF metrics on train/validation.
    oof_pauc = isic2024_pauc(y, oof_cat, Config.min_tpr)
    oof_auroc = roc_auc_score(y, oof_cat)

    print("\n" + "=" * 72)
    print("TINYViT + DENSENET121 | CATBOOST ONLY")
    print("=" * 72)
    print(f"OOF AUROC:  {oof_auroc:.6f}")
    print(f"OOF pAUC:   {oof_pauc:.6f}")
    print(f"Test AUROC: {test_auroc:.6f}")
    print(f"Test pAUC:  {test_pauc:.6f}")
    print(f"Features:   {X.shape[1]} total")
    print("=" * 72)

    # Save predictions and metadata.
    np.save(
        os.path.join(Config.output_dir, "tinyvit_densenet_test_cat.npy"),
        test_cat,
    )
    np.save(
        os.path.join(Config.output_dir, "tinyvit_densenet_oof_cat.npy"),
        oof_cat,
    )
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)

    results = {
        "configuration": "tinyvit+densenet",
        "classifier": "CatBoost",
        "backbones": "tinyvit+densenet",
        "n_image_models": 2,
        "n_features": int(X.shape[1]),
        "n_tabular_features": int(X.shape[1] - 2),
        "n_splits": Config.n_splits,
        "min_tpr": Config.min_tpr,
        "oof_auroc": float(oof_auroc),
        "oof_pauc": float(oof_pauc),
        "test_auroc": float(test_auroc),
        "test_pauc": float(test_pauc),
    }

    with open(
        os.path.join(Config.output_dir, "ablation_summary.json"),
        "w",
    ) as f:
        json.dump(results, f, indent=2)

    pd.DataFrame([results]).to_csv(
        os.path.join(Config.output_dir, "ablation_results.csv"),
        index=False,
    )

    tab_test[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_test.parquet"),
        index=False,
    )
    tab_full[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_trainval.parquet"),
        index=False,
    )

    print(f"\nAll outputs saved to: {Config.output_dir}")


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    print("=" * 72)
    print("ISIC 2024: TinyViT + DenseNet121 CatBoost Ablation")
    print("=" * 72)

    print("\n1. Loading TinyViT and DenseNet121 predictions...")
    oof_data, oof_ids_data, test_data = load_backbones()

    print("\n2. Aligning OOF predictions by ISIC ID...")
    oof_aligned, reference_ids = align_oof(oof_data, oof_ids_data)
    print(f"Aligned samples: {len(reference_ids):,}")

    print("\n3. Loading tabular features...")
    tab_full, tab_test = load_tabular(reference_ids)

    y = tab_full["target"].to_numpy(dtype=np.int32)
    y_test = tab_test["target"].to_numpy(dtype=np.int32)
    groups = tab_full["patient_id"].to_numpy()

    print(f"Train+validation: {len(y):,} samples | positives: {y.sum():,}")
    print(f"Test:             {len(y_test):,} samples | positives: {y_test.sum():,}")

    print("\n4. Building TinyViT + DenseNet121 + tabular inputs...")
    X, X_test, tab_cols = build_features(
        tab_full,
        tab_test,
        oof_aligned,
        test_data,
    )

    print(f"X train/validation shape: {X.shape}")
    print(f"X test shape:             {X_test.shape}")
    print(f"Tabular feature count:    {len(tab_cols)}")

    print("\n5. Training CatBoost with patient-grouped 5-fold CV...")
    train_and_evaluate(X, y, groups, X_test, y_test, tab_full, tab_test)


if __name__ == "__main__":
    main()
