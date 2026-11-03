"""
sequential_ablation_eff_tiny.py
Sequential ablation with automatic alignment of OOF predictions by ISIC ID.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklearn_metrics


class Config:
    backbone_dirs = {
        "effnet":  r"F:\thesis 23\new exp segment\baseline_A_efficientnetv2s_5fold_fixed",
        "tinyvit": r"F:\thesis 23\new exp segment\tinyvitFULLUNFROZEN",
    }
    tabular_path = r"./features/tabular/trainval_tabular.parquet"
    tabular_test_path = r"./features/tabular/test_tabular.parquet"
    output_dir = r"./models/sequential_ablation_eff_tiny"
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


def load_backbones_aligned():
    """
    Load OOF and test predictions from both backbones.
    Aligns OOF by sorting ISIC IDs. Test predictions aligned by their own IDs.
    """
    oof_data = {}
    test_data = {}
    oof_ids_data = {}
    test_ids_data = {}

    for name, path in Config.backbone_dirs.items():
        oof_probs = np.load(os.path.join(path, "oof_probs.npy"))
        oof_ids = np.load(os.path.join(path, "oof_ids.npy"), allow_pickle=True)

        oof_data[name] = oof_probs
        oof_ids_data[name] = oof_ids

        # Test predictions
        test_path = os.path.join(path, "test_probs.npy")
        test_ids_path = os.path.join(path, "test_ids.npy")

        if os.path.exists(test_path):
            test_data[name] = np.load(test_path)
            if os.path.exists(test_ids_path):
                test_ids_data[name] = np.load(test_ids_path, allow_pickle=True)
            else:
                test_ids_data[name] = None
        else:
            print(f"⚠️ {name}: test_probs.npy not found")

    return oof_data, oof_ids_data, test_data, test_ids_data


def align_oof(oof_data, oof_ids_data):
    """
    Align OOF predictions from all backbones by ISIC ID.
    Returns: aligned_probs dict, common_ids array
    """
    # Get reference IDs from first backbone
    first_name = list(oof_ids_data.keys())[0]
    ref_ids = np.asarray(oof_ids_data[first_name])

    # Sort reference IDs
    ref_order = np.argsort(ref_ids)
    ref_ids_sorted = ref_ids[ref_order]

    aligned = {}
    for name in oof_data.keys():
        ids = np.asarray(oof_ids_data[name])
        probs = oof_data[name]

        if len(ids) != len(ref_ids):
            raise ValueError(
                f"{name}: {len(ids)} OOF vs {len(ref_ids)} reference. "
                f"Different sample counts."
            )

        # Check if same set of IDs
        if set(ids) != set(ref_ids):
            missing = set(ref_ids) - set(ids)
            extra = set(ids) - set(ref_ids)
            raise ValueError(
                f"{name}: ID set mismatch. "
                f"Missing from {name}: {len(missing)}; Extra: {len(extra)}"
            )

        # Sort this backbone's IDs and reindex its predictions
        order = np.argsort(ids)
        ids_sorted = ids[order]
        probs_sorted = probs[order]

        # Verify alignment
        assert (ids_sorted == ref_ids_sorted).all(), \
            f"{name}: sorted IDs still misaligned"

        aligned[name] = probs_sorted

    return aligned, ref_ids_sorted


def load_tabular(reference_ids):
    """Load tabular features, align to reference IDs, drop banned columns."""
    tab_full = pd.read_parquet(Config.tabular_path)
    tab_test = pd.read_parquet(Config.tabular_test_path)

    # Align train+val to OOF order
    tab_full = tab_full.set_index("isic_id").loc[reference_ids].reset_index()

    # Drop banned
    drop_cols = [c for c in Config.banned_features if c in tab_full.columns]
    if drop_cols:
        print(f"Dropping banned features: {drop_cols}")
        tab_full = tab_full.drop(columns=drop_cols)
        tab_test = tab_test.drop(columns=drop_cols, errors="ignore")

    return tab_full, tab_test


def build_X(tab_full, tab_test, oof_aligned, test_data, backbone_list):
    """Build feature matrices."""
    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    oof_matrix = np.stack([oof_aligned[k] for k in backbone_list], axis=1)
    test_matrix = np.stack([test_data[k] for k in backbone_list], axis=1)

    X_full = np.concatenate([oof_matrix, tab_full[tab_cols].values.astype(np.float32)], axis=1)
    X_test = np.concatenate([test_matrix, tab_test[tab_cols].values.astype(np.float32)], axis=1)

    return X_full, X_test, tab_cols


def train_and_evaluate(X, y, groups, X_test, y_test, config_name):
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))

    test_xgb = []
    test_cat = []

    for fold, (tr, va) in enumerate(splits, 1):
        X_tr, y_tr = X[tr], y[tr]

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist", n_jobs=-1, random_state=Config.seed)
        xgb_model.fit(X_tr, y_tr)
        test_xgb.append(xgb_model.predict_proba(X_test)[:, 1])

        cat_model = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_seed=Config.seed, verbose=0, thread_count=-1)
        cat_model.fit(X_tr, y_tr)
        test_cat.append(cat_model.predict_proba(X_test)[:, 1])

    xgb_probs = np.mean(test_xgb, axis=0)
    cat_probs = np.mean(test_cat, axis=0)

    xgb_pauc = isic2024_pauc(y_test, xgb_probs)
    cat_pauc = isic2024_pauc(y_test, cat_probs)
    xgb_auroc = roc_auc_score(y_test, xgb_probs)
    cat_auroc = roc_auc_score(y_test, cat_probs)

    print(f"\n{'=' * 70}")
    print(f"{config_name}")
    print(f"{'=' * 70}")
    print(f"XGBoost  | AUROC: {xgb_auroc:.4f} | pAUC: {xgb_pauc:.4f}")
    print(f"CatBoost | AUROC: {cat_auroc:.4f} | pAUC: {cat_pauc:.4f}")

    return {
        "config": config_name,
        "xgb_pauc": xgb_pauc,
        "cat_pauc": cat_pauc,
        "xgb_auroc": xgb_auroc,
        "cat_auroc": cat_auroc,
    }

    np.save(os.path.join(Config.output_dir, "test_probs_eff_tiny.npy"), cat_probs)
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(Config.output_dir, "oof_probs_eff_tiny.npy"), oof_cat_full)
def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    print("Loading backbones...")
    oof_data, oof_ids_data, test_data, test_ids_data = load_backbones_aligned()

    print("\nAligning OOF predictions by ISIC ID...")
    oof_aligned, ref_ids = align_oof(oof_data, oof_ids_data)
    print(f"Aligned {len(oof_aligned)} backbones, {len(ref_ids)} samples")

    print("\nLoading tabular features...")
    tab_full, tab_test = load_tabular(ref_ids)

    y = tab_full["target"].values.astype(int)
    y_test = tab_test["target"].values.astype(int)
    groups = tab_full["patient_id"].values

    print(f"\nTrain+Val: {len(y):,}  Positives: {y.sum()}")
    print(f"Test:      {len(y_test):,}  Positives: {y_test.sum()}")

    results = []
    cumulative = []

    for name in oof_aligned.keys():
        cumulative.append(name)
        X, X_test, tab_cols = build_X(
            tab_full, tab_test, oof_aligned, test_data, cumulative)
        config_name = f"Cumulative: {' + '.join(cumulative)}"
        print(f"\nX shape: {X.shape}  ({len(cumulative)} image + {len(tab_cols)} tabular)")

        r = train_and_evaluate(X, y, groups, X_test, y_test, config_name)
        r["backbones"] = "+".join(cumulative)
        r["n_features"] = X.shape[1]
        results.append(r)

    print(f"\n{'=' * 70}")
    print("SEQUENTIAL ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<40} {'XGB pAUC':>10} {'CAT pAUC':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<40} {r['xgb_pauc']:>10.4f} {r['cat_pauc']:>10.4f}")

    print(f"\n{'=' * 70}")
    print("MARGINAL GAINS (CatBoost pAUC)")
    print(f"{'=' * 70}")
    prev = 0.0
    for r in results:
        gain = r["cat_pauc"] - prev if prev > 0 else r["cat_pauc"]
        print(f"{r['backbones']:<30} pAUC={r['cat_pauc']:.4f}  (Δ {gain:+.4f})")
        prev = r["cat_pauc"]

    pd.DataFrame(results).to_csv(
        os.path.join(Config.output_dir, "ablation_results.csv"), index=False)
    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()