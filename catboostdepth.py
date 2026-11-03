"""
sequential_ablation_eff_tiny_dense.py
Sequential ablation: EfficientNetV2-S + TinyViT + DenseNet121.
Aligns OOF predictions by ISIC ID. Drops banned features.
Saves per-configuration test predictions.

FINAL MODEL: tinyvit + densenet
  - XGBoost: unchanged (same params, seed, splits)
  - CatBoost: depth sweep, selected by OOF pAUC
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics
import joblib


class Config:
    backbone_dirs = {
            "effnet":  r"F:\thesis 23\new experiment\baseline_A_efficientnetv2s_fullunfreeze",
            "densenet": r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold",
            "tinyvit": r"F:\thesis 23\new experiment\tinyvitFULLUNFROZEN",
        }

    tabular_path = r"./features/tabular/trainval_tabular.parquet"
    tabular_test_path = r"./features/tabular/test_tabular.parquet"
    output_dir = r"./models/sequential_ablation_all3"
    n_splits = 5
    seed = 42
    min_tpr = 0.80
    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]

    # ---- CatBoost depth sweep ----
    catboost_depths = [5, 6, 7, 8, 9, 10]
    catboost_iterations = 500  # held fixed across depths for a fair comparison


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


def load_backbones_aligned():
    oof_data = {}
    test_data = {}
    oof_ids_data = {}

    for name, path in Config.backbone_dirs.items():
        oof_path = os.path.join(path, "oof_probs.npy")
        ids_path = os.path.join(path, "oof_ids.npy")
        test_path = os.path.join(path, "test_probs.npy")

        if not all(os.path.exists(p) for p in [oof_path, ids_path, test_path]):
            print(f"⚠️ {name}: skipping (missing files in {path})")
            continue

        oof_data[name] = np.load(oof_path)
        oof_ids_data[name] = np.load(ids_path, allow_pickle=True)
        test_data[name] = np.load(test_path)
        print(f"✓ {name}: {len(oof_data[name])} OOF, {len(test_data[name])} test")

    return oof_data, oof_ids_data, test_data


def align_oof(oof_data, oof_ids_data):
    first_name = list(oof_ids_data.keys())[0]
    ref_ids = np.asarray(oof_ids_data[first_name])
    ref_order = np.argsort(ref_ids)
    ref_ids_sorted = ref_ids[ref_order]

    aligned = {}
    for name in oof_data.keys():
        ids = np.asarray(oof_ids_data[name])
        probs = oof_data[name]

        assert len(ids) == len(ref_ids), f"{name} length mismatch"
        assert set(ids) == set(ref_ids), f"{name} ID set mismatch"

        order = np.argsort(ids)
        aligned[name] = probs[order]

    return aligned, ref_ids_sorted


def load_tabular(reference_ids):
    tab_full = pd.read_parquet(Config.tabular_path)
    tab_test = pd.read_parquet(Config.tabular_test_path)

    tab_full = tab_full.set_index("isic_id").loc[reference_ids].reset_index()

    drop_cols = [c for c in Config.banned_features if c in tab_full.columns]
    if drop_cols:
        print(f"Dropping banned: {drop_cols}")
        tab_full = tab_full.drop(columns=drop_cols)
        tab_test = tab_test.drop(columns=drop_cols, errors="ignore")

    return tab_full, tab_test


def build_X(tab_full, tab_test, oof_aligned, test_data, backbone_list):
    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    oof_matrix = np.stack([oof_aligned[k] for k in backbone_list], axis=1)
    test_matrix = np.stack([test_data[k] for k in backbone_list], axis=1)

    X_full = np.concatenate([oof_matrix, tab_full[tab_cols].values.astype(np.float32)], axis=1)
    X_test = np.concatenate([test_matrix, tab_test[tab_cols].values.astype(np.float32)], axis=1)

    return X_full, X_test, tab_cols


# =====================================================================
# XGBoost: unchanged from original train_and_evaluate
# =====================================================================
def train_xgboost(X, y, groups, X_test, y_test, save_prefix):
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    oof_xgb = np.zeros(len(y))
    test_xgb = []

    for fold, (tr, va) in enumerate(splits, 1):
        X_tr, y_tr = X[tr], y[tr]
        X_va = X[va]

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist", n_jobs=-1, random_state=Config.seed)
        xgb_model.fit(X_tr, y_tr)
        oof_xgb[va] = xgb_model.predict_proba(X_va)[:, 1]
        test_xgb.append(xgb_model.predict_proba(X_test)[:, 1])

    xgb_probs = np.mean(test_xgb, axis=0)

    xgb_pauc = isic2024_pauc(y_test, xgb_probs)
    xgb_auroc = roc_auc_score(y_test, xgb_probs)
    oof_xgb_pauc = isic2024_pauc(y, oof_xgb)

    print(f"XGBoost  | OOF pAUC: {oof_xgb_pauc:.4f} | "
          f"Test AUROC: {xgb_auroc:.4f} | Test pAUC: {xgb_pauc:.4f}")

    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_xgb.npy"), xgb_probs)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_xgb.npy"), oof_xgb)

    return {
        "xgb_pauc": xgb_pauc,
        "xgb_auroc": xgb_auroc,
        "oof_xgb_pauc": oof_xgb_pauc,
    }


# =====================================================================
# CatBoost: depth sweep, selected by OOF pAUC
# =====================================================================
def tune_catboost_depth(X, y, groups, X_test, y_test, save_prefix):
    """
    Sweep CatBoost depth over Config.catboost_depths.
    For each depth: 5-fold GroupKFold, fixed iterations, fixed seed.
    Select depth by OOF pAUC (no test leakage).
    Write best-depth predictions to the standard save_prefix filenames.
    """
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    sweep = []
    best = None

    for depth in Config.catboost_depths:
        print(f"\n--- CatBoost depth={depth} ---")

        oof_cat = np.zeros(len(y))
        test_cat = []
        depth_pkl_dir = os.path.join(Config.output_dir, f"cat_depth_{depth}")
        os.makedirs(depth_pkl_dir, exist_ok=True)

        for fold, (tr, va) in enumerate(splits, 1):
            X_tr, y_tr = X[tr], y[tr]
            X_va = X[va]

            cat_model = CatBoostClassifier(
                iterations=Config.catboost_iterations, depth=depth,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_seed=Config.seed, verbose=0, thread_count=-1)
            cat_model.fit(X_tr, y_tr)
            joblib.dump(cat_model,
                        os.path.join(depth_pkl_dir, f"{save_prefix}_cat_fold{fold}.pkl"))
            oof_cat[va] = cat_model.predict_proba(X_va)[:, 1]
            test_cat.append(cat_model.predict_proba(X_test)[:, 1])

        cat_probs = np.mean(test_cat, axis=0)

        cat_pauc = isic2024_pauc(y_test, cat_probs)
        cat_auroc = roc_auc_score(y_test, cat_probs)
        oof_cat_pauc = isic2024_pauc(y, oof_cat)
        oof_cat_auroc = roc_auc_score(y, oof_cat)

        print(f"depth={depth} | OOF pAUC: {oof_cat_pauc:.4f} | OOF AUROC: {oof_cat_auroc:.4f} | "
              f"Test pAUC: {cat_pauc:.4f} | Test AUROC: {cat_auroc:.4f}")

        entry = {
            "depth": depth,
            "oof_cat_pauc": oof_cat_pauc,
            "oof_cat_auroc": oof_cat_auroc,
            "cat_pauc": cat_pauc,
            "cat_auroc": cat_auroc,
            "cat_probs": cat_probs,
            "oof_cat": oof_cat,
        }
        sweep.append(entry)

        if best is None or entry["oof_cat_pauc"] > best["oof_cat_pauc"]:
            best = entry

    # Save best-depth predictions to standard filenames
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_cat.npy"), best["cat_probs"])
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_cat.npy"), best["oof_cat"])

    # Sweep summary
    print(f"\n{'=' * 70}")
    print("CATBOOST DEPTH SWEEP SUMMARY (selected by OOF pAUC)")
    print(f"{'=' * 70}")
    print(f"{'Depth':>6} {'OOF pAUC':>10} {'OOF AUROC':>10} "
          f"{'Test pAUC':>10} {'Test AUROC':>10}")
    print("-" * 70)
    for e in sweep:
        marker = "  * best" if e is best else ""
        print(f"{e['depth']:>6} {e['oof_cat_pauc']:>10.4f} {e['oof_cat_auroc']:>10.4f} "
              f"{e['cat_pauc']:>10.4f} {e['cat_auroc']:>10.4f}{marker}")

    best_dir = os.path.join(Config.output_dir, f"cat_depth_{best['depth']}")
    print(f"\n>>> Best depth = {best['depth']} "
          f"(OOF pAUC = {best['oof_cat_pauc']:.4f})")
    print(f">>> Best-depth fold models: {best_dir}/")

    # Sweep CSV
    sweep_df = pd.DataFrame([
        {k: v for k, v in e.items() if k not in ("cat_probs", "oof_cat")}
        for e in sweep
    ])
    sweep_df.to_csv(os.path.join(Config.output_dir, "catboost_depth_sweep.csv"), index=False)

    return best, sweep

    


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    print("Loading backbones...")
    oof_data, oof_ids_data, test_data = load_backbones_aligned()
    print(f"Available: {list(oof_data.keys())}")

    print("\nAligning OOF by ISIC ID...")
    oof_aligned, ref_ids = align_oof(oof_data, oof_ids_data)
    print(f"Aligned {len(oof_aligned)} backbones, {len(ref_ids)} samples")

    print("\nLoading tabular features...")
    tab_full, tab_test = load_tabular(ref_ids)

    y = tab_full["target"].values.astype(int)
    y_test = tab_test["target"].values.astype(int)
    groups = tab_full["patient_id"].values

    print(f"\nTrain+Val: {len(y):,}  Positives: {y.sum()}")
    print(f"Test:      {len(y_test):,}  Positives: {y_test.sum()}")

    # =====================================================================
    # FINAL MODEL: tinyvit + densenet
    # =====================================================================
    save_prefix = "tinyvit+densenet"
    backbone_list = ["tinyvit", "densenet"]

    if not all(k in oof_aligned for k in backbone_list):
        print(f"⚠️ Missing backbone(s) for {save_prefix}; aborting.")
        return

    X, X_test, tab_cols = build_X(
        tab_full, tab_test, oof_aligned, test_data, backbone_list)
    config_name = " + ".join(backbone_list)

    print(f"\n{'=' * 70}")
    print(f"{config_name}")
    print(f"{'=' * 70}")
    print(f"X shape: {X.shape}  ({len(backbone_list)} image + {len(tab_cols)} tabular)")

    # ---- 1) XGBoost (unchanged) ----
    xgb_res = train_xgboost(X, y, groups, X_test, y_test, save_prefix)

    # ---- 2) CatBoost depth sweep ----
    best_cat, sweep = tune_catboost_depth(
        X, y, groups, X_test, y_test, save_prefix)

    # ---- Combined record ----
    r = {
        "config": config_name,
        "save_prefix": save_prefix,
        "xgb_pauc": xgb_res["xgb_pauc"],
        "cat_pauc": best_cat["cat_pauc"],
        "xgb_auroc": xgb_res["xgb_auroc"],
        "cat_auroc": best_cat["cat_auroc"],
        "oof_xgb_pauc": xgb_res["oof_xgb_pauc"],
        "oof_cat_pauc": best_cat["oof_cat_pauc"],
        "best_cat_depth": best_cat["depth"],
        "backbones": "+".join(backbone_list),
        "n_image": len(backbone_list),
        "n_features": X.shape[1],
    }
    results = [r]

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("FINAL MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<30} {'XGB pAUC':>10} {'CAT pAUC':>10} {'CAT depth':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<30} {r['xgb_pauc']:>10.4f} "
              f"{r['cat_pauc']:>10.4f} {r['best_cat_depth']:>10}")

    # Save y and metadata for downstream analysis
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)
    tab_test[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_test.parquet"), index=False)
    tab_full[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_trainval.parquet"), index=False)

    # Save summary
    with open(os.path.join(Config.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    pd.DataFrame(results).to_csv(
        os.path.join(Config.output_dir, "ablation_results.csv"), index=False)

    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()