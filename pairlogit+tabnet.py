"""
sequential_ablation_eff_tiny_dense.py
FINAL MODEL: tinyvit + densenet

Compared models (identical X, y, groups, splits, seed, spw):
  - XGBoost                (depth=6)
  - CatBoost Logloss       (depth=5)   [reference ~0.1747 test pAUC]
  - CatBoost PairLogit     (depth=5, group_id=patient_id)
  - TabNet                 (attention-based DL)
LightGBM dropped (underperformed).
"""

import os
import json
import warnings
import itertools                          # <-- added for blend check
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker   # <-- added Ranker
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sklearn_metrics
import joblib

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    print("⚠️ pytorch-tabnet not installed -> TabNet will be skipped.")
    print("   Install with: pip install pytorch-tabnet")

warnings.filterwarnings("ignore")


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

    # TabNet config
    tabnet_max_epochs = 40
    tabnet_patience = 6
    tabnet_batch_size = 4096
    tabnet_virtual_batch_size = 512


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
# Main training loop: 4 models on identical folds
# =====================================================================
def train_and_evaluate(X, y, groups, X_test, y_test, config_name, save_prefix):
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    oof_xgb     = np.zeros(len(y))
    oof_cat     = np.zeros(len(y))
    oof_catpair = np.zeros(len(y))
    oof_tabnet  = np.zeros(len(y))
    test_xgb, test_cat, test_catpair, test_tabnet = [], [], [], []

    catpair_ran = False
    tabnet_ran = False

    for fold, (tr, va) in enumerate(splits, 1):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        print(f"\n--- Fold {fold}/{Config.n_splits} ---")

        # -------------------------------------------------------------
        # 1) XGBoost (unchanged)
        # -------------------------------------------------------------
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist", n_jobs=-1, random_state=Config.seed)
        xgb_model.fit(X_tr, y_tr)
        oof_xgb[va] = xgb_model.predict_proba(X_va)[:, 1]
        test_xgb.append(xgb_model.predict_proba(X_test)[:, 1])
        print("  XGBoost            done")

        # -------------------------------------------------------------
        # 2) CatBoost Logloss (reference, depth=5)
        # -------------------------------------------------------------
        cat_model = CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_seed=Config.seed, verbose=0, thread_count=-1)
        cat_model.fit(X_tr, y_tr)
        joblib.dump(cat_model, os.path.join(Config.output_dir, f"{save_prefix}_cat_fold{fold}.pkl"))
        oof_cat[va] = cat_model.predict_proba(X_va)[:, 1]
        test_cat.append(cat_model.predict_proba(X_test)[:, 1])
        print("  CatBoost-Logloss   done")

        # -------------------------------------------------------------
        # 3) CatBoost PairLogit  ->  CatBoostRanker (group_id supported)
        #    Returns RAW ranking scores via .predict() (rank-metric valid).
        # -------------------------------------------------------------
                # -------------------------------------------------------------
        # 3) CatBoost PairLogit  ->  CatBoostRanker
        # -------------------------------------------------------------
        try:
            # --- prepare group_id: numpy str array, sorted for contiguity ---
            fold_groups = np.asarray(groups[tr]).astype(str)   # ArrowStringArray -> ndarray
            sort_idx    = np.argsort(fold_groups, kind="stable")
            X_tr_sorted = X_tr[sort_idx]
            y_tr_sorted = y_tr[sort_idx]
            groups_sorted = fold_groups[sort_idx]

            pair_model = CatBoostRanker(
                iterations=500, depth=5, learning_rate=0.05,
                loss_function="YetiRank",
                random_seed=Config.seed, verbose=0, thread_count=-1)
            pair_model.fit(
                X_tr_sorted, y_tr_sorted,
                group_id=groups_sorted)
            joblib.dump(pair_model,
                        os.path.join(Config.output_dir, f"{save_prefix}_catpair_fold{fold}.pkl"))

            oof_catpair[va] = pair_model.predict(X_va)
            test_catpair.append(pair_model.predict(X_test))
            catpair_ran = True
            print("  CatBoost-PairLogit done")
        except Exception as e:
            print(f"  ⚠️ PairLogit fold {fold} failed: {e}")
            oof_catpair[va] = 0.5
            test_catpair.append(np.full(len(X_test), 0.5, dtype=np.float32))

        # -------------------------------------------------------------
        # 4) TabNet
        # -------------------------------------------------------------
        if HAS_TABNET:
            try:
                imp = SimpleImputer(strategy="median")
                sc  = StandardScaler()
                X_tr_p   = sc.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
                X_va_p   = sc.transform(imp.transform(X_va)).astype(np.float32)
                X_test_p = sc.transform(imp.transform(X_test)).astype(np.float32)

                w = np.where(y_tr == 1, scale_pos_weight, 1.0).astype(np.float32)
                w = w / w.mean()

                clf = TabNetClassifier(
                    n_d=32, n_a=32, n_steps=3, gamma=1.5,
                    n_independent=2, n_shared=2,
                    lambda_sparse=1e-4, momentum=0.3, clip_value=2,
                    mask_type="entmax", verbose=0, seed=Config.seed)

                clf.fit(
                    X_train=X_tr_p, y_train=y_tr,
                    eval_set=[(X_va_p, y_va)], eval_name=["val"], eval_metric=["auc"],
                    weights=w,
                    max_epochs=Config.tabnet_max_epochs,
                    patience=Config.tabnet_patience,
                    batch_size=Config.tabnet_batch_size,
                    virtual_batch_size=Config.tabnet_virtual_batch_size,
                    num_workers=0, drop_last=False)

                oof_tabnet[va] = clf.predict_proba(X_va_p)[:, 1]
                test_tabnet.append(clf.predict_proba(X_test_p)[:, 1])
                tabnet_ran = True
                print("  TabNet             done")
            except Exception as e:
                print(f"  ⚠️ TabNet fold {fold} failed: {e}")
                oof_tabnet[va] = 0.5

    # ---------------- aggregate ----------------
    xgb_probs     = np.mean(test_xgb, axis=0)
    cat_probs     = np.mean(test_cat, axis=0)
    catpair_probs = np.mean(test_catpair, axis=0) if catpair_ran else None
    tabnet_probs  = np.mean(test_tabnet, axis=0) if tabnet_ran else None

    # ---------------- metrics + save ----------------
    print(f"\n{'=' * 70}")
    print(f"{config_name}")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'OOF pAUC':>10} {'Test pAUC':>11} {'Test AUROC':>12}")
    print("-" * 70)

    metrics = {}

    def _report(name, oof, test):
        oof_p  = isic2024_pauc(y, oof)
        test_p = isic2024_pauc(y_test, test)
        test_a = roc_auc_score(y_test, test)
        print(f"{name:<20} {oof_p:>10.4f} {test_p:>11.4f} {test_a:>12.4f}")
        np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_{name}.npy"), test)
        np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_{name}.npy"), oof)
        metrics[name] = dict(oof_pauc=oof_p, test_pauc=test_p, test_auroc=test_a)

    _report("xgb",    oof_xgb,    xgb_probs)
    _report("cat",    oof_cat,    cat_probs)
    if catpair_ran:
        _report("catpair", oof_catpair, catpair_probs)
    if tabnet_ran:
        _report("tabnet",  oof_tabnet,  tabnet_probs)

    # Backwards-compatible aliases (unchanged names)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_cat.npy"), cat_probs)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_test_xgb.npy"), xgb_probs)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_cat.npy"), oof_cat)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_oof_xgb.npy"), oof_xgb)

    return {
        "config": config_name,
        "save_prefix": save_prefix,
        "xgb_pauc":    metrics["xgb"]["test_pauc"],
        "cat_pauc":    metrics["cat"]["test_pauc"],
        "catpair_pauc": metrics.get("catpair", {}).get("test_pauc"),
        "tabnet_pauc": metrics.get("tabnet", {}).get("test_pauc"),
        "xgb_auroc":    metrics["xgb"]["test_auroc"],
        "cat_auroc":    metrics["cat"]["test_auroc"],
        "catpair_auroc": metrics.get("catpair", {}).get("test_auroc"),
        "tabnet_auroc": metrics.get("tabnet", {}).get("test_auroc"),
        "oof_xgb_pauc":    metrics["xgb"]["oof_pauc"],
        "oof_cat_pauc":    metrics["cat"]["oof_pauc"],
        "oof_catpair_pauc": metrics.get("catpair", {}).get("oof_pauc"),
        "oof_tabnet_pauc": metrics.get("tabnet", {}).get("oof_pauc"),
    }


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
    print(f"\nX shape: {X.shape}  ({len(backbone_list)} image + {len(tab_cols)} tabular)")

    r = train_and_evaluate(
        X, y, groups, X_test, y_test, config_name, save_prefix)
    r["backbones"] = "+".join(backbone_list)
    r["n_image"] = len(backbone_list)
    r["n_features"] = X.shape[1]
    results = [r]

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("FINAL MODEL SUMMARY (test-set metrics, tinyvit + densenet features)")
    print(f"{'=' * 70}")
    print(f"{'Model':<22} {'OOF pAUC':>10} {'Test pAUC':>11} {'Test AUROC':>12}")
    print("-" * 70)
    for name, oof_key, test_key, auroc_key in [
        ("XGBoost",           "oof_xgb_pauc",     "xgb_pauc",     "xgb_auroc"),
        ("CatBoost-Logloss",  "oof_cat_pauc",     "cat_pauc",     "cat_auroc"),
        ("CatBoost-PairLogit","oof_catpair_pauc", "catpair_pauc", "catpair_auroc"),
        ("TabNet",            "oof_tabnet_pauc",  "tabnet_pauc",  "tabnet_auroc"),
    ]:
        if r.get(test_key) is None:
            print(f"{name:<22} {'(skipped)':>10}")
            continue
        print(f"{name:<22} {r[oof_key]:>10.4f} {r[test_key]:>11.4f} {r[auroc_key]:>12.4f}")

    # =====================================================================
    # PAIRWISE BLEND CHECK
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("PAIRWISE 50/50 BLEND CHECK (test pAUC)")
    print(f"{'=' * 70}")

    labels = ["xgb", "cat"]
    if r.get("catpair_pauc") is not None:
        labels.append("catpair")
    if r.get("tabnet_pauc") is not None:
        labels.append("tabnet")

    blend_rows = []
    for a, b in itertools.combinations(labels, 2):
        pa = np.load(os.path.join(Config.output_dir, f"{save_prefix}_test_{a}.npy"))
        pb = np.load(os.path.join(Config.output_dir, f"{save_prefix}_test_{b}.npy"))
        blend = 0.5 * pa + 0.5 * pb
        bp = isic2024_pauc(y_test, blend)
        ba = roc_auc_score(y_test, blend)
        print(f"blend {a}+{b}: test pAUC = {bp:.4f}  test AUROC = {ba:.4f}")
        blend_rows.append({"pair": f"{a}+{b}", "test_pauc": bp, "test_auroc": ba})

    # Best single vs best blend
    single_best = max(
        [(n, r[f"{n}_pauc"]) for n in labels if r.get(f"{n}_pauc") is not None],
        key=lambda x: x[1])
    blend_best = max(blend_rows, key=lambda x: x["test_pauc"]) if blend_rows else None
    print(f"\nBest single : {single_best[0]}  test pAUC = {single_best[1]:.4f}")
    if blend_best:
        gain = blend_best["test_pauc"] - single_best[1]
        print(f"Best blend  : {blend_best['pair']}  test pAUC = {blend_best['test_pauc']:.4f}  "
              f"(Δ = {gain:+.4f} vs best single)")

    # Save blend summary
    if blend_rows:
        pd.DataFrame(blend_rows).to_csv(
            os.path.join(Config.output_dir, "blend_results.csv"), index=False)

    # =====================================================================
    # Persist metadata
    # =====================================================================
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "y_trainval.npy"), y)
    tab_test[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_test.parquet"), index=False)
    tab_full[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_trainval.parquet"), index=False)

    with open(os.path.join(Config.output_dir, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    pd.DataFrame(results).to_csv(
        os.path.join(Config.output_dir, "model_comparison.csv"), index=False)

    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()