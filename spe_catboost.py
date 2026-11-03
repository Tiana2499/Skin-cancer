
"""
spe_catboost.py

CatBoost + Self-Paced Ensemble (SPE)
Using TinyViT + DenseNet121 OOF predictions + tabular features.

Keeps the same:
- Patient-wise StratifiedGroupKFold(n_splits=5), shuffle=False
- CatBoost configuration as sequential_ablation_eff_tiny_dense.py
- Custom ISIC 2024 pAUC calculation
- Banned feature removal
- Test feature row order from test_tabular.parquet

Includes:
- OOF ID alignment validation
- Test prediction length validation
- Test metadata alignment validation
- SPE fold validation
- Comparison with standard CatBoost predictions
"""

import os
import json
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from imbens.ensemble import SelfPacedEnsembleClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


# ================================================================
# CONFIGURATION
# ================================================================

class Config:

    tinyvit_dir = (
        r"F:\thesis 23\new experiment\tinyvitFULLUNFROZEN"
    )

    densenet_dir = (
        r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"
    )

    tabular_path = (
        r"./features/tabular/trainval_tabular.parquet"
    )

    tabular_test_path = (
        r"./features/tabular/test_tabular.parquet"
    )

    # Standard CatBoost predictions from your sequential script
    standard_cat_path = (
        r"./models/sequential_ablation_all3/"
        r"tinyvit+densenet_test_cat.npy"
    )

    # Metadata saved by your sequential script
    standard_meta_test_path = (
        r"./models/sequential_ablation_all3/meta_test.parquet"
    )

    output_dir = (
        r"./analysis/spe_catboost"
    )

    n_splits = 5
    seed = 42
    min_tpr = 0.80

    # SPE
    spe_n_estimators = 15
    spe_k_bins = 5

    # Match standard CatBoost configuration
    catboost_iterations = 500
    catboost_depth = 6
    catboost_learning_rate = 0.05

    banned_features = [
        "mel_thick_mm_dev",
        "mel_thick_mm"
    ]


# ================================================================
# CUSTOM ISIC pAUC
# ================================================================

def isic2024_pauc(y, p, min_tpr=0.80):

    y = np.asarray(y)
    p = np.asarray(p)

    if len(np.unique(y)) != 2:
        return 0.0

    yf = abs(y - 1)
    pf = abs(p - 1)

    max_fpr = 1 - min_tpr

    fpr, tpr, _ = sklearn_metrics.roc_curve(
        yf,
        pf
    )

    stop = np.searchsorted(
        fpr,
        max_fpr,
        "right"
    )

    # Guard against interpolation indexing errors
    if stop == 0:
        tpr_i = np.array([
            np.interp(max_fpr, fpr, tpr)
        ])
        fpr_i = np.array([max_fpr])

    elif stop >= len(fpr):
        tpr_i = tpr
        fpr_i = fpr

    else:
        tpr_i = np.append(
            tpr[:stop],
            np.interp(
                max_fpr,
                [fpr[stop - 1], fpr[stop]],
                [tpr[stop - 1], tpr[stop]]
            )
        )

        fpr_i = np.append(
            fpr[:stop],
            max_fpr
        )

    return float(
        sklearn_metrics.auc(fpr_i, tpr_i)
    )


# ================================================================
# LOAD AND ALIGN OOF PREDICTIONS
# ================================================================

def load_oof_predictions():

    tinyvit_oof_path = os.path.join(
        Config.tinyvit_dir,
        "oof_probs.npy"
    )

    tinyvit_ids_path = os.path.join(
        Config.tinyvit_dir,
        "oof_ids.npy"
    )

    dense_oof_path = os.path.join(
        Config.densenet_dir,
        "oof_probs.npy"
    )

    dense_ids_path = os.path.join(
        Config.densenet_dir,
        "oof_ids.npy"
    )

    required = [
        tinyvit_oof_path,
        tinyvit_ids_path,
        dense_oof_path,
        dense_ids_path
    ]

    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required OOF file not found: {path}"
            )

    tiny_probs = np.asarray(
        np.load(tinyvit_oof_path)
    ).reshape(-1)

    tiny_ids = np.asarray(
        np.load(tinyvit_ids_path, allow_pickle=True)
    ).reshape(-1)

    dense_probs = np.asarray(
        np.load(dense_oof_path)
    ).reshape(-1)

    dense_ids = np.asarray(
        np.load(dense_ids_path, allow_pickle=True)
    ).reshape(-1)

    # Length validation
    assert len(tiny_probs) == len(tiny_ids), (
        "TinyViT OOF prediction and ID lengths differ."
    )

    assert len(dense_probs) == len(dense_ids), (
        "DenseNet OOF prediction and ID lengths differ."
    )

    assert len(tiny_ids) == len(dense_ids), (
        "TinyViT and DenseNet OOF sample counts differ."
    )

    # Independently sort both backbones by ISIC ID
    tiny_order = np.argsort(tiny_ids)
    dense_order = np.argsort(dense_ids)

    tiny_ids_sorted = tiny_ids[tiny_order]
    dense_ids_sorted = dense_ids[dense_order]

    tiny_probs_sorted = tiny_probs[tiny_order]
    dense_probs_sorted = dense_probs[dense_order]

    # Critical ID equality validation
    assert np.array_equal(
        tiny_ids_sorted,
        dense_ids_sorted
    ), (
        "ERROR: TinyViT and DenseNet OOF IDs do not match. "
        "Do not combine these predictions."
    )

    print(
        f"✓ TinyViT OOF: {len(tiny_probs_sorted):,}"
    )

    print(
        f"✓ DenseNet OOF: {len(dense_probs_sorted):,}"
    )

    print(
        "✓ TinyViT and DenseNet OOF IDs match exactly."
    )

    return (
        tiny_probs_sorted,
        dense_probs_sorted,
        tiny_ids_sorted
    )


# ================================================================
# LOAD TABULAR DATA AND BUILD FEATURES
# ================================================================

def load_and_build_features():

    print("\nLoading OOF predictions...")

    (
        tiny_oof,
        dense_oof,
        reference_ids
    ) = load_oof_predictions()

    print("\nLoading tabular data...")

    tab_full = pd.read_parquet(
        Config.tabular_path
    )

    tab_test = pd.read_parquet(
        Config.tabular_test_path
    )

    # ------------------------------------------------------------
    # Check duplicate IDs
    # ------------------------------------------------------------

    if tab_full["isic_id"].duplicated().any():
        raise ValueError(
            "Duplicate isic_id found in trainval_tabular.parquet."
        )

    if tab_test["isic_id"].duplicated().any():
        raise ValueError(
            "Duplicate isic_id found in test_tabular.parquet."
        )

    # ------------------------------------------------------------
    # Align trainval tabular rows with sorted OOF IDs
    # ------------------------------------------------------------

    tab_full = (
        tab_full
        .set_index("isic_id")
        .loc[reference_ids]
        .reset_index()
    )

    assert np.array_equal(
        tab_full["isic_id"].values,
        reference_ids
    ), "Trainval tabular ID alignment failed."

    # ------------------------------------------------------------
    # Remove banned features
    # ------------------------------------------------------------

    drop_cols = [
        c for c in Config.banned_features
        if c in tab_full.columns
    ]

    if drop_cols:

        print(
            f"Dropping banned features: {drop_cols}"
        )

        tab_full = tab_full.drop(
            columns=drop_cols
        )

        tab_test = tab_test.drop(
            columns=drop_cols,
            errors="ignore"
        )

    # ------------------------------------------------------------
    # Validate tabular columns
    # ------------------------------------------------------------

    META = [
        "isic_id",
        "patient_id",
        "target"
    ]

    tab_cols = [
        c for c in tab_full.columns
        if c not in META
    ]

    missing_test_cols = [
        c for c in tab_cols
        if c not in tab_test.columns
    ]

    if missing_test_cols:
        raise ValueError(
            f"Missing test tabular columns: {missing_test_cols}"
        )

    # ------------------------------------------------------------
    # Load test predictions
    # ------------------------------------------------------------

    tiny_test_path = os.path.join(
        Config.tinyvit_dir,
        "test_probs.npy"
    )

    dense_test_path = os.path.join(
        Config.densenet_dir,
        "test_probs.npy"
    )

    for path in [
        tiny_test_path,
        dense_test_path
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Test prediction file not found: {path}"
            )

    tiny_test = np.asarray(
        np.load(tiny_test_path)
    ).reshape(-1)

    dense_test = np.asarray(
        np.load(dense_test_path)
    ).reshape(-1)

    # ------------------------------------------------------------
    # Preserve sequential script test row order
    # ------------------------------------------------------------

    n_test = len(tab_test)

    assert len(tiny_test) == n_test, (
        f"TinyViT test predictions: {len(tiny_test)}, "
        f"tabular test rows: {n_test}"
    )

    assert len(dense_test) == n_test, (
        f"DenseNet test predictions: {len(dense_test)}, "
        f"tabular test rows: {n_test}"
    )

    print(
        "✓ Test prediction lengths match test_tabular.parquet."
    )

    # ------------------------------------------------------------
    # Build feature matrices
    # ------------------------------------------------------------

    y = tab_full["target"].values.astype(int)

    y_test = tab_test["target"].values.astype(int)

    groups = tab_full["patient_id"].values

    # Two image prediction features:
    # TinyViT + DenseNet121

    X_image = np.column_stack([
        tiny_oof,
        dense_oof
    ]).astype(np.float32)

    X_test_image = np.column_stack([
        tiny_test,
        dense_test
    ]).astype(np.float32)

    X_tab = tab_full[
        tab_cols
    ].values.astype(np.float32)

    X_test_tab = tab_test[
        tab_cols
    ].values.astype(np.float32)

    X = np.concatenate(
        [X_image, X_tab],
        axis=1
    )

    X_test = np.concatenate(
        [X_test_image, X_test_tab],
        axis=1
    )

    # ------------------------------------------------------------
    # Validate final matrices
    # ------------------------------------------------------------

    assert len(X) == len(y)
    assert len(X_test) == len(y_test)
    assert len(groups) == len(y)

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found in training features."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Non-finite values found in test features."
        )

    print("\nFeature construction complete.")

    print(
        f"Trainval: {X.shape}"
    )

    print(
        f"Test: {X_test.shape}"
    )

    print(
        f"Image features: 2"
    )

    print(
        f"Tabular features: {len(tab_cols)}"
    )

    print(
        f"Trainval positives: {y.sum():,}"
    )

    print(
        f"Test positives: {y_test.sum():,}"
    )

    return (
        X,
        y,
        groups,
        X_test,
        y_test,
        tab_test["isic_id"].values,
        tab_cols,
        reference_ids
    )


# ================================================================
# COMPARE TEST ORDER WITH STANDARD CATBOOST METADATA
# ================================================================

def validate_standard_test_order(test_ids):

    if not os.path.exists(
        Config.standard_meta_test_path
    ):

        print(
            "\nWARNING: Standard meta_test.parquet not found. "
            "Cannot independently verify standard CatBoost test order."
        )

        return

    standard_meta = pd.read_parquet(
        Config.standard_meta_test_path
    )

    standard_ids = standard_meta[
        "isic_id"
    ].values

    if not np.array_equal(
        test_ids,
        standard_ids
    ):

        raise ValueError(
            "Test ID order differs from standard CatBoost metadata. "
            "Cannot safely compare the two prediction arrays."
        )

    print(
        "✓ Test IDs match standard CatBoost metadata exactly."
    )


# ================================================================
# TRAIN AND EVALUATE SPE + CATBOOST
# ================================================================

def train_spe_catboost(
    X,
    y,
    groups,
    X_test,
    y_test
):

    print("\nStarting CatBoost + SPE...")

    # Same CV configuration as sequential ablation
    sgkf = StratifiedGroupKFold(
        n_splits=Config.n_splits
    )

    splits = list(
        sgkf.split(X, y, groups)
    )

    oof_spe = np.zeros(
        len(y),
        dtype=np.float64
    )

    test_predictions = []

    fold_results = []

    for fold, (tr, va) in enumerate(
        splits,
        start=1
    ):

        print(
            f"\n{'=' * 65}"
        )

        print(
            f"SPE Fold {fold}/{Config.n_splits}"
        )

        print(
            f"{'=' * 65}"
        )

        X_tr = X[tr]
        y_tr = y[tr]

        X_va = X[va]
        y_va = y[va]

        # Check patient separation
        train_groups = set(groups[tr])
        val_groups = set(groups[va])

        assert train_groups.isdisjoint(
            val_groups
        ), (
            f"Patient overlap detected in fold {fold}."
        )

        print(
            f"Training samples: {len(tr):,}"
        )

        print(
            f"Validation samples: {len(va):,}"
        )

        print(
            f"Training positives: {y_tr.sum():,}"
        )

        print(
            f"Validation positives: {y_va.sum():,}"
        )

        # --------------------------------------------------------
        # CatBoost base estimator
        # --------------------------------------------------------

        cat_model = CatBoostClassifier(
            iterations=Config.catboost_iterations,
            depth=Config.catboost_depth,
            learning_rate=Config.catboost_learning_rate,
            verbose=0,
            thread_count=-1,
            random_seed=Config.seed
        )

        # --------------------------------------------------------
        # Self-Paced Ensemble
        # --------------------------------------------------------

        spe_model = SelfPacedEnsembleClassifier(
            estimator=cat_model,
            n_estimators=Config.spe_n_estimators,
            k_bins=Config.spe_k_bins,
            random_state=Config.seed,
            n_jobs=-1
        )

        spe_model.fit(
            X_tr,
            y_tr
        )

        # --------------------------------------------------------
        # Validation predictions
        # --------------------------------------------------------

        val_probs = spe_model.predict_proba(
            X_va
        )[:, 1]

        oof_spe[va] = val_probs

        val_pauc = isic2024_pauc(
            y_va,
            val_probs,
            Config.min_tpr
        )

        val_auroc = roc_auc_score(
            y_va,
            val_probs
        )

        # --------------------------------------------------------
        # Test predictions
        # --------------------------------------------------------

        fold_test_probs = spe_model.predict_proba(
            X_test
        )[:, 1]

        test_predictions.append(
            fold_test_probs
        )

        fold_results.append({
            "fold": fold,
            "validation_pauc": float(val_pauc),
            "validation_auroc": float(val_auroc),
            "train_samples": int(len(tr)),
            "validation_samples": int(len(va)),
            "train_positives": int(y_tr.sum()),
            "validation_positives": int(y_va.sum())
        })

        print(
            f"Fold {fold} validation AUROC: "
            f"{val_auroc:.6f}"
        )

        print(
            f"Fold {fold} validation pAUC: "
            f"{val_pauc:.6f}"
        )

        print(
            f"Fold {fold} completed."
        )

    # ============================================================
    # OOF METRICS
    # ============================================================

    oof_auroc = roc_auc_score(
        y,
        oof_spe
    )

    oof_pauc = isic2024_pauc(
        y,
        oof_spe,
        Config.min_tpr
    )

    # ============================================================
    # TEST ENSEMBLE
    # ============================================================

    test_probs = np.mean(
        test_predictions,
        axis=0
    )

    test_auroc = roc_auc_score(
        y_test,
        test_probs
    )

    test_pauc = isic2024_pauc(
        y_test,
        test_probs,
        Config.min_tpr
    )

    val_paucs = [
        r["validation_pauc"]
        for r in fold_results
    ]

    val_aurocs = [
        r["validation_auroc"]
        for r in fold_results
    ]

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print(
        f"\n{'=' * 75}"
    )

    print(
        "CATBOOST + SELF-PACED ENSEMBLE RESULTS"
    )

    print(
        f"{'=' * 75}"
    )

    print(
        f"OOF AUROC:       {oof_auroc:.6f}"
    )

    print(
        f"OOF pAUC:        {oof_pauc:.6f}"
    )

    print(
        f"Test AUROC:      {test_auroc:.6f}"
    )

    print(
        f"Test pAUC:       {test_pauc:.6f}"
    )

    print(
        f"Mean Val pAUC:   {np.mean(val_paucs):.6f}"
    )

    print(
        f"Std Val pAUC:    {np.std(val_paucs):.6f}"
    )

    print(
        f"Mean Val AUROC:  {np.mean(val_aurocs):.6f}"
    )

    print(
        f"Std Val AUROC:   {np.std(val_aurocs):.6f}"
    )

    # ============================================================
    # SAVE PREDICTIONS AND RESULTS
    # ============================================================

    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "spe_test_probs.npy"
        ),
        test_probs
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "spe_oof_probs.npy"
        ),
        oof_spe
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "spe_test_ids.npy"
        ),
        test_ids
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "spe_oof_ids.npy"
        ),
        trainval_ids
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "y_test.npy"
        ),
        y_test
    )

    np.save(
        os.path.join(
            Config.output_dir,
            "y_trainval.npy"
        ),
        y
    )

    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            Config.output_dir,
            "spe_fold_results.csv"
        ),
        index=False
    )

    results = {
        "method": "CatBoost + Self-Paced Ensemble",
        "configuration": "TinyViT + DenseNet121 + Tabular",
        "n_splits": Config.n_splits,
        "shuffle": False,
        "spe_n_estimators": Config.spe_n_estimators,
        "spe_k_bins": Config.spe_k_bins,
        "catboost_iterations": Config.catboost_iterations,
        "catboost_depth": Config.catboost_depth,
        "catboost_learning_rate": Config.catboost_learning_rate,
        "oof_auroc": float(oof_auroc),
        "oof_pauc": float(oof_pauc),
        "test_auroc": float(test_auroc),
        "test_pauc": float(test_pauc),
        "mean_validation_pauc": float(np.mean(val_paucs)),
        "std_validation_pauc": float(np.std(val_paucs)),
        "mean_validation_auroc": float(np.mean(val_aurocs)),
        "std_validation_auroc": float(np.std(val_aurocs)),
        "fold_results": fold_results
    }

    with open(
        os.path.join(
            Config.output_dir,
            "spe_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSPE results saved to: {Config.output_dir}"
    )

    return test_probs, results


# ================================================================
# COMPARE WITH STANDARD CATBOOST
# ================================================================

def compare_standard_catboost(
    spe_probs,
    y_test
):

    print(
        f"\n{'=' * 75}"
    )

    print(
        "STANDARD CATBOOST vs CATBOOST + SPE"
    )

    print(
        f"{'=' * 75}"
    )

    if not os.path.exists(
        Config.standard_cat_path
    ):

        print(
            "Standard CatBoost prediction file not found. "
            "Skipping direct comparison."
        )

        return

    standard_probs = np.asarray(
        np.load(Config.standard_cat_path)
    ).reshape(-1)

    if len(standard_probs) != len(y_test):

        raise ValueError(
            "Standard CatBoost prediction count does not "
            "match the test labels."
        )

    standard_auroc = roc_auc_score(
        y_test,
        standard_probs
    )

    standard_pauc = isic2024_pauc(
        y_test,
        standard_probs,
        Config.min_tpr
    )

    spe_auroc = roc_auc_score(
        y_test,
        spe_probs
    )

    spe_pauc = isic2024_pauc(
        y_test,
        spe_probs,
        Config.min_tpr
    )

    print(
        f"{'Metric':<20}"
        f"{'Standard CAT':>18}"
        f"{'CAT + SPE':>18}"
    )

    print(
        "-" * 56
    )

    print(
        f"{'Test AUROC':<20}"
        f"{standard_auroc:>18.6f}"
        f"{spe_auroc:>18.6f}"
    )

    print(
        f"{'Test pAUC':<20}"
        f"{standard_pauc:>18.6f}"
        f"{spe_pauc:>18.6f}"
    )

    comparison = pd.DataFrame([
        {
            "method": "Standard CatBoost",
            "test_auroc": standard_auroc,
            "test_pauc": standard_pauc
        },
        {
            "method": "CatBoost + SPE",
            "test_auroc": spe_auroc,
            "test_pauc": spe_pauc
        }
    ])

    comparison.to_csv(
        os.path.join(
            Config.output_dir,
            "catboost_vs_spe_comparison.csv"
        ),
        index=False
    )


# ================================================================
# MAIN
# ================================================================

def main():

    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    print(
        "=" * 75
    )

    print(
        "CATBOOST + SELF-PACED ENSEMBLE"
    )

    print(
        "TinyViT + DenseNet121 + Tabular Features"
    )

    print(
        "=" * 75
    )

    (
        X,
        y,
        groups,
        X_test,
        y_test,
        test_ids,
        tab_cols,
        trainval_ids
    ) = load_and_build_features()

    # Validate test ordering before comparing predictions
    validate_standard_test_order(
        test_ids
    )

    # Save the metadata corresponding to this run
    pd.DataFrame({
        "isic_id": test_ids,
        "target": y_test
    }).to_parquet(
        os.path.join(
            Config.output_dir,
            "spe_test_metadata.parquet"
        ),
        index=False
    )

    print(
        "\nStarting five-fold grouped training..."
    )

   

    spe_probs, results = train_spe_catboost(
        X,
        y,
        groups,
        X_test,
        y_test
    )

    compare_standard_catboost(
        spe_probs,
        y_test
    )

    print(
        "\nExperiment completed."
    )


if __name__ == "__main__":
    main()
