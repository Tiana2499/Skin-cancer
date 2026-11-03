"""
multi_model_ensemble.py
Train CatBoost / LightGBM / XGBoost / ExtraTrees / LogReg on identical folds.
Rank-average greedy blend, selected by OOF pAUC.
"""

import os
import json
import warnings
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import sklearn.metrics as sklearn_metrics
import joblib

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

    cat_depth = 5
    lgb_depth = 5
    xgb_depth = 5

    bag_seeds = [42, 123, 2024]

    # greedy blend: min OOF pAUC gain to keep a model
    blend_min_gain = 0.0005


def isic2024_pauc(y, p, min_tpr=0.80):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) != 2:
        return 0.0
    yf, pf = abs(y - 1), abs(p - 1)
    max_fpr = 1 - min_tpr
    fpr, tpr, _ = sklearn_metrics.roc_curve(yf, pf)
    stop = np.searchsorted(fpr, max_fpr, "right")
    tpr_i = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]],
                                            [tpr[stop-1], tpr[stop]]))
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
    ref_ids_sorted = ref_ids[np.argsort(ref_ids)]
    aligned = {}
    for name in oof_data.keys():
        ids = np.asarray(oof_ids_data[name])
        order = np.argsort(ids)
        aligned[name] = oof_data[name][order]
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
# Model runners — all share identical splits
# =====================================================================
def cv_fit(X, y, groups, X_test, fit_fn, preprocess_fn=None):
    sgkf = StratifiedGroupKFold(n_splits=Config.n_splits)
    splits = list(sgkf.split(X, y, groups))
    oof = np.zeros(len(y))
    test = []
    for tr, va in splits:
        X_tr, X_va = X[tr], X[va]
        if preprocess_fn is not None:
            X_tr_p, X_va_p, X_te_p = preprocess_fn(X_tr, X_va, X_test)
        else:
            X_tr_p, X_va_p, X_te_p = X_tr, X_va, X_test
        oof_va, test_fold = fit_fn(X_tr_p, y[tr], X_va_p, X_te_p)
        oof[va] = oof_va
        test.append(test_fold)
    return oof, np.mean(test, axis=0)


def run_catboost(X, y, groups, X_test, spw):
    def fit(X_tr, y_tr, X_va, X_test_p):
        bag_oof, bag_test = [], []
        for s in Config.bag_seeds:
            m = CatBoostClassifier(
                iterations=500, depth=Config.cat_depth, learning_rate=0.05,
                scale_pos_weight=spw, random_seed=s, verbose=0, thread_count=-1)
            m.fit(X_tr, y_tr)
            bag_oof.append(m.predict_proba(X_va)[:, 1])
            bag_test.append(m.predict_proba(X_test_p)[:, 1])
        return np.mean(bag_oof, axis=0), np.mean(bag_test, axis=0)
    return cv_fit(X, y, groups, X_test, fit)


def run_lightgbm(X, y, groups, X_test, spw):
    def fit(X_tr, y_tr, X_va, X_test_p):
        bag_oof, bag_test = [], []
        for s in Config.bag_seeds:
            m = lgb.LGBMClassifier(
                n_estimators=500, max_depth=Config.lgb_depth,
                num_leaves=2 ** Config.lgb_depth - 1,
                learning_rate=0.05, scale_pos_weight=spw,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                random_state=s, n_jobs=-1, verbose=-1)
            m.fit(X_tr, y_tr)
            bag_oof.append(m.predict_proba(X_va)[:, 1])
            bag_test.append(m.predict_proba(X_test_p)[:, 1])
        return np.mean(bag_oof, axis=0), np.mean(bag_test, axis=0)
    return cv_fit(X, y, groups, X_test, fit)


def run_xgboost(X, y, groups, X_test, spw):
    def fit(X_tr, y_tr, X_va, X_test_p):
        bag_oof, bag_test = [], []
        for s in Config.bag_seeds:
            m = xgb.XGBClassifier(
                n_estimators=500, max_depth=Config.xgb_depth,
                learning_rate=0.05, scale_pos_weight=spw,
                subsample=0.7, colsample_bytree=0.7,
                tree_method="hist", n_jobs=-1, random_state=s)
            m.fit(X_tr, y_tr)
            bag_oof.append(m.predict_proba(X_va)[:, 1])
            bag_test.append(m.predict_proba(X_test_p)[:, 1])
        return np.mean(bag_oof, axis=0), np.mean(bag_test, axis=0)
    return cv_fit(X, y, groups, X_test, fit)


def run_extratrees(X, y, groups, X_test, spw):
    def preprocess(X_tr, X_va, X_te):
        imp = SimpleImputer(strategy="median")
        return (imp.fit_transform(X_tr),
                imp.transform(X_va),
                imp.transform(X_te))

    def fit(X_tr, y_tr, X_va, X_test_p):
        m = ExtraTreesClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1, random_state=Config.seed)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test_p)[:, 1]

    return cv_fit(X, y, groups, X_test, fit, preprocess_fn=preprocess)


def run_logreg(X, y, groups, X_test, spw):
    def preprocess(X_tr, X_va, X_te):
        # rank-normalize features so LogReg is invariant to scaling
        n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
        stacked = np.vstack([X_tr, X_va, X_te])
        imp = SimpleImputer(strategy="median")
        stacked = imp.fit_transform(stacked)
        n_cols = stacked.shape[1]
        ranked = np.zeros_like(stacked)
        for c in range(n_cols):
            ranked[:, c] = rankdata(stacked[:, c]) / len(stacked)
        return (ranked[:n_tr], ranked[n_tr:n_tr+n_va], ranked[n_tr+n_va:])

    def fit(X_tr, y_tr, X_va, X_test_p):
        m = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=2000, solver="lbfgs", n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test_p)[:, 1]

    return cv_fit(X, y, groups, X_test, fit, preprocess_fn=preprocess)


# =====================================================================
# Rank-average blending with greedy forward selection
# =====================================================================
def rank01(x):
    return rankdata(x) / len(x)


def rank_average(preds_dict, y, subset):
    """Equal-weight rank-average over a subset of models. Returns (blend_oof, blend_test)."""
    n_oof = len(next(iter(preds_dict.values()))[0])
    n_test = len(next(iter(preds_dict.values()))[1])
    oof = np.zeros(n_oof)
    test = np.zeros(n_test)
    for name in subset:
        o, t = preds_dict[name]
        oof += rank01(o) / len(subset)
        test += rank01(t) / len(subset)
    return oof, test


def greedy_forward_blend(preds_dict, y):
    """Add models one-by-one if OOF pAUC improves by at least blend_min_gain."""
    remaining = list(preds_dict.keys())
    selected = []
    current_pauc = 0.0
    history = []

    print(f"\nGreedy rank-average forward selection (min_gain={Config.blend_min_gain}):")
    while remaining:
        cands = []
        for cand in remaining:
            trial = selected + [cand]
            oof_b, _ = rank_average(preds_dict, y, trial)
            p = isic2024_pauc(y, oof_b)
            cands.append((cand, p, p - current_pauc))
        cands.sort(key=lambda x: x[1], reverse=True)
        cand, p, gain = cands[0]

        if selected and gain < Config.blend_min_gain:
            print(f"  STOP — best available '{cand}' would add only {gain:+.4f}")
            break

        selected.append(cand)
        current_pauc = p
        remaining.remove(cand)
        history.append({"step": len(selected), "added": cand,
                        "blend_oof_pauc": p, "gain": gain})
        print(f"  + {cand:<12}  OOF pAUC = {p:.4f}  (gain {gain:+.4f})")

    return selected, history


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    print("Loading backbones...")
    oof_data, oof_ids_data, test_data = load_backbones_aligned()
    oof_aligned, ref_ids = align_oof(oof_data, oof_ids_data)

    print("\nLoading tabular...")
    tab_full, tab_test = load_tabular(ref_ids)

    y = tab_full["target"].values.astype(int)
    y_test = tab_test["target"].values.astype(int)
    groups = tab_full["patient_id"].values
    spw = (y == 0).sum() / max(y.sum(), 1)
    print(f"Train+Val: {len(y):,} (pos={y.sum()}) | Test: {len(y_test):,} (pos={y_test.sum()}) | spw={spw:.2f}")

    save_prefix = "tinyvit+densenet"
    backbone_list = ["tinyvit", "densenet"]
    X, X_test, tab_cols = build_X(
        tab_full, tab_test, oof_aligned, test_data, backbone_list)
    print(f"X shape: {X.shape}")

    # ---------------- train all models ----------------
    preds = {}
    metrics = {}

    def _train(name, fn):
        print(f"\n=== {name} ===")
        oof, test = fn(X, y, groups, X_test, spw)
        preds[name] = (oof, test)
        oof_p = isic2024_pauc(y, oof)
        test_p = isic2024_pauc(y_test, test)
        test_a = roc_auc_score(y_test, test)
        metrics[name] = dict(oof_pauc=oof_p, test_pauc=test_p, test_auroc=test_a)
        print(f"  {name:<12}  OOF pAUC={oof_p:.4f}  test pAUC={test_p:.4f}  AUROC={test_a:.4f}")
        np.save(os.path.join(Config.output_dir, f"{save_prefix}_{name}_oof.npy"), oof)
        np.save(os.path.join(Config.output_dir, f"{save_prefix}_{name}_test.npy"), test)

    _train("cat",         run_catboost)
    _train("lgb",         run_lightgbm)
    _train("xgb",         run_xgboost)
    _train("extratrees",  run_extratrees)
    _train("logreg",      run_logreg)

    # ---------------- single-model summary ----------------
    print(f"\n{'=' * 70}")
    print("SINGLE-MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<14} {'OOF pAUC':>10} {'Test pAUC':>11} {'Test AUROC':>12}")
    print("-" * 70)
    for name, m in metrics.items():
        print(f"{name:<14} {m['oof_pauc']:>10.4f} {m['test_pauc']:>11.4f} {m['test_auroc']:>12.4f}")

    best_single = max(metrics.items(), key=lambda kv: kv[1]["test_pauc"])
    print(f"\nBest single: {best_single[0]}  test pAUC = {best_single[1]['test_pauc']:.4f}")

    # ---------------- diversity check ----------------
    print(f"\n{'=' * 70}")
    print("PAIRWISE SPEARMAN CORRELATION (test predictions)")
    print("(lower = more diverse, better ensemble candidate)")
    print(f"{'=' * 70}")
    names = list(preds.keys())
    from scipy.stats import spearmanr
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in itertools.product(names, names):
        corr.loc[a, b] = spearmanr(preds[a][1], preds[b][1]).correlation
    print(corr.round(3).to_string())

    # ---------------- greedy blend ----------------
    selected, history = greedy_forward_blend(preds, y)

    blend_oof, blend_test = rank_average(preds, y, selected)
    blend_oof_p = isic2024_pauc(y, blend_oof)
    blend_test_p = isic2024_pauc(y_test, blend_test)
    blend_test_a = roc_auc_score(y_test, blend_test)

    print(f"\n{'=' * 70}")
    print("BLEND RESULT")
    print(f"{'=' * 70}")
    print(f"Selected models: {selected}")
    print(f"Blend OOF pAUC : {blend_oof_p:.4f}")
    print(f"Blend test pAUC: {blend_test_p:.4f}  "
          f"(Δ vs best single {best_single[0]}: "
          f"{blend_test_p - best_single[1]['test_pauc']:+.4f})")
    print(f"Blend test AUROC: {blend_test_a:.4f}  "
          f"(Δ vs best single: {blend_test_a - best_single[1]['test_auroc']:+.4f})")

    # persist
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_blend_oof.npy"), blend_oof)
    np.save(os.path.join(Config.output_dir, f"{save_prefix}_blend_test.npy"), blend_test)

    summary = {
        "single_models": metrics,
        "best_single": {"name": best_single[0], **best_single[1]},
        "blend": {
            "selected": selected,
            "history": history,
            "oof_pauc": blend_oof_p,
            "test_pauc": blend_test_p,
            "test_auroc": blend_test_a,
            "delta_vs_best_single": blend_test_p - best_single[1]["test_pauc"],
        },
        "pairwise_spearman_test": corr.round(4).to_dict(),
    }
    with open(os.path.join(Config.output_dir, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nSaved to {Config.output_dir}")


if __name__ == "__main__":
    main()