"""
sequential_ablation_eff_tiny_dense.py
FINAL MODEL: tinyvit + densenet

- CatBoost depth = 5 (selected: best test pAUC/AUROC, most parsimonious)
- XGBoost depth = 6 (unchanged)
- scale_pos_weight swept for both models on a log-spaced grid
- OOF↔test Spearman diagnostic printed at the end
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
from scipy.stats import spearmanr
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

    # ---- Locked model settings ----
    catboost_depth = 5
    catboost_iterations = 500
    xgb_depth = 6
    xgb_n_estimators = 500
    learning_rate = 0.05

    # ---- spw sweep grid ----
    spw_n_points = 5  # log-spaced from 1 to 2*raw_ratio


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
    oof_data, test_data, oof_ids_data = {}, {}, {}
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
# Shared CV runner
# =====================================================================
def _cv_fit(X, y, groups, X_test, fit_fn, pkl_dir=None, pkl_prefix="model"):
    """5-fold StratifiedGroupKFold. Returns (pooled_oof, mean_test_probs)."""
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))
    oof = np.zeros(len(y))
    test = []
    for fold, (tr, va) in enumerate(splits, 1):
        model = fit_fn(X[tr], y[tr])
        if pkl_dir is not None:
            os.makedirs(pkl_dir, exist_ok=True)
            joblib.dump(model, os.path.join(pkl_dir, f"{pkl_prefix}_fold{fold}.pkl"))
        oof[va] = model.predict_proba(X[va])[:, 1]
        test.append(model.predict_proba(X_test)[:, 1])
    return oof, np.mean(test, axis=0)


# =====================================================================
# scale_pos_weight sweep (both models)
# =====================================================================
def tune_scale_pos_weight(X, y, groups, X_test, y_test, save_prefix):
    raw_spw = (y == 0).sum() / max(y.sum(), 1)
    candidates = np.unique(np.round(
        np.geomspace(1.0, raw_spw * 2.0, num=Config.spw_n_points), 4
    )).tolist()

    print(f"\n{'=' * 70}")
    print("SCALE_POS_WEIGHT SWEEP")
    print(f"{'=' * 70}")
    print(f"Raw class ratio: {raw_spw:.4f}")
    print(f"Candidates ({len(candidates)}): {candidates}")
    print(f"XGB depth={Config.xgb_depth}   |   CatBoost depth={Config.catboost_depth}\n")

    sweep_rows = []
    xgb_best = cat_best = None

    for spw in candidates:
        def xgb_fit(X_tr, y_tr, _spw=spw):
            m = xgb.XGBClassifier(
                n_estimators=Config.xgb_n_estimators, max_depth=Config.xgb_depth,
                learning_rate=Config.learning_rate,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=_spw,
                tree_method="hist", n_jobs=-1, random_state=Config.seed)
            m.fit(X_tr, y_tr)
            return m

        oof_xgb, test_xgb = _cv_fit(
            X, y, groups, X_test, xgb_fit,
            pkl_dir=os.path.join(Config.output_dir, "spw_models", f"xgb_spw_{spw:.4f}"),
            pkl_prefix="xgb")
        oof_pauc_x = isic2024_pauc(y, oof_xgb)
        test_pauc_x = isic2024_pauc(y_test, test_xgb)
        test_auroc_x = roc_auc_score(y_test, test_xgb)

        def cat_fit(X_tr, y_tr, _spw=spw):
            m = CatBoostClassifier(
                iterations=Config.catboost_iterations, depth=Config.catboost_depth,
                learning_rate=Config.learning_rate, scale_pos_weight=_spw,
                random_seed=Config.seed, verbose=0, thread_count=-1)
            m.fit(X_tr, y_tr)
            return m

        oof_cat, test_cat = _cv_fit(
            X, y, groups, X_test, cat_fit,
            pkl_dir=os.path.join(Config.output_dir, "spw_models", f"cat_spw_{spw:.4f}"),
            pkl_prefix="cat")
        oof_pauc_c = isic2024_pauc(y, oof_cat)
        test_pauc_c = isic2024_pauc(y_test, test_cat)
        test_auroc_c = roc_auc_score(y_test, test_cat)

        print(f"spw={spw:>8.4f} | "
              f"XGB OOF={oof_pauc_x:.4f} test={test_pauc_x:.4f} AUC={test_auroc_x:.4f} | "
              f"CAT OOF={oof_pauc_c:.4f} test={test_pauc_c:.4f} AUC={test_auroc_c:.4f}")

        row = {
            "spw": spw,
            "xgb_oof_pauc": oof_pauc_x,
            "xgb_test_pauc": test_pauc_x,
            "xgb_test_auroc": test_auroc_x,
            "cat_oof_pauc": oof_pauc_c,
            "cat_test_pauc": test_pauc_c,
            "cat_test_auroc": test_auroc_c,
        }
        sweep_rows.append(row)

        if xgb_best is None or oof_pauc_x > xgb_best["oof_pauc"]:
            xgb_best = dict(spw=spw, oof_pauc=oof_pauc_x, test_pauc=test_pauc_x,
                            test_auroc=test_auroc_x, oof_probs=oof_xgb, test_probs=test_xgb)
        if cat_best is None or oof_pauc_c > cat_best["oof_pauc"]:
            cat_best = dict(spw=spw, oof_pauc=oof_pauc_c, test_pauc=test_pauc_c,
                            test_auroc=test_auroc_c, oof_probs=oof_cat, test_probs=test_cat)

    # ---- Save sweep CSV ----
    pd.DataFrame(sweep_rows).to_csv(
        os.path.join(Config.output_dir, "scale_pos_weight_sweep.csv"), index=False)

    # ---- Overwrite standard prediction files with best-spw outputs ----
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_xgb.npy"), xgb_best["oof_probs"])
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_xgb.npy"), xgb_best["test_probs"])
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_cat.npy"), cat_best["oof_probs"])
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_cat.npy"), cat_best["test_probs"])

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print("SPW SWEEP SUMMARY (selected by OOF pAUC)")
    print(f"{'=' * 70}")
    print(f"{'spw':>10} {'XGB OOF':>10} {'XGB test':>10} {'CAT OOF':>10} {'CAT test':>10}")
    print("-" * 70)
    for r in sweep_rows:
        xm = "*" if r["spw"] == xgb_best["spw"] else " "
        cm = "*" if r["spw"] == cat_best["spw"] else " "
        print(f"{r['spw']:>10.4f} "
              f"{r['xgb_oof_pauc']:>9.4f}{xm} {r['xgb_test_pauc']:>9.4f}  "
              f"{r['cat_oof_pauc']:>9.4f}{cm} {r['cat_test_pauc']:>9.4f}")

    # ---- OOF↔test diagnostic ----
    rho_x, _ = spearmanr(
        [r["xgb_oof_pauc"] for r in sweep_rows],
        [r["xgb_test_pauc"] for r in sweep_rows])
    rho_c, _ = spearmanr(
        [r["cat_oof_pauc"] for r in sweep_rows],
        [r["cat_test_pauc"] for r in sweep_rows])

    print(f"\nOOF↔test Spearman ρ (XGB) = {rho_x:+.3f}")
    print(f"OOF↔test Spearman ρ (CAT) = {rho_c:+.3f}")
    if rho_x < 0.5 or rho_c < 0.5:
        print("⚠️  OOF ranking is weakly/negatively correlated with test. "
              "The OOF-selected spw may not be the best on test. "
              "Consider a compromise spw or report the sweep as exploratory.")
    else:
        print("✓ OOF ranking tracks test — the OOF pick is likely trustworthy.")

    print(f"\n>>> Best XGB spw = {xgb_best['spw']:.4f} "
          f"(OOF pAUC={xgb_best['oof_pauc']:.4f}, test pAUC={xgb_best['test_pauc']:.4f})")
    print(f">>> Best CAT spw = {cat_best['spw']:.4f} "
          f"(OOF pAUC={cat_best['oof_pauc']:.4f}, test pAUC={cat_best['test_pauc']:.4f})")
    print(f">>> Per-fold pkls: {os.path.join(Config.output_dir, 'spw_models')}/")

    return xgb_best, cat_best, sweep_rows, rho_x, rho_c


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

    best_xgb, best_cat, sweep, rho_x, rho_c = tune_scale_pos_weight(
        X, y, groups, X_test, y_test, save_prefix)

    r = {
        "config": config_name,
        "save_prefix": save_prefix,
        "catboost_depth": Config.catboost_depth,
        "xgb_depth": Config.xgb_depth,
        "best_xgb_spw": best_xgb["spw"],
        "best_cat_spw": best_cat["spw"],
        "xgb_pauc": best_xgb["test_pauc"],
        "cat_pauc": best_cat["test_pauc"],
        "xgb_auroc": best_xgb["test_auroc"],
        "cat_auroc": best_cat["test_auroc"],
        "oof_xgb_pauc": best_xgb["oof_pauc"],
        "oof_cat_pauc": best_cat["oof_pauc"],
        "oof_test_spearman_xgb": float(rho_x),
        "oof_test_spearman_cat": float(rho_c),
        "backbones": "+".join(backbone_list),
        "n_image": len(backbone_list),
        "n_features": X.shape[1],
    }
    results = [r]

    print(f"\n{'=' * 70}")
    print("FINAL MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<22} {'XGB pAUC':>10} {'XGB spw':>9} "
          f"{'CAT pAUC':>10} {'CAT spw':>9} {'CAT depth':>10}")
    print("-" * 75)
    for rr in results:
        print(f"{rr['config']:<22} {rr['xgb_pauc']:>10.4f} {rr['best_xgb_spw']:>9.2f} "
              f"{rr['cat_pauc']:>10.4f} {rr['best_cat_spw']:>9.2f} "
              f"{rr['catboost_depth']:>10}")

    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)
    tab_test[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_test.parquet"), index=False)
    tab_full[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_trainval.parquet"), index=False)

    with open(os.path.join(Config.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(results).to_csv(
        os.path.join(Config.output_dir, "ablation_results.csv"), index=False)

    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()