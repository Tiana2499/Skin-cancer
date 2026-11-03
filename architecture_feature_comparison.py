"""
architecture_feature_comparison.py
Compare SHAP feature importance across three model configurations:
  - TinyViT only + tabular
  - DenseNet only + tabular
  - TinyViT + DenseNet + tabular
"""

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt


class Config:
    model_dir = r"F:\thesis 23\tinyvit experiment\models\sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    tinyvit_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"
    output_dir = r"./analysis/architecture_shap"
    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]


def load_features(backbone_names):
    """Build feature matrix for the given backbones + tabular."""
    tab_full = pd.read_parquet(os.path.join(Config.feature_dir, "trainval_tabular.parquet"))
    tab_full = tab_full.drop(columns=[c for c in Config.banned_features if c in tab_full.columns])

    backbone_data = []
    for name in backbone_names:
        if name == "tinyvit":
            d = Config.tinyvit_dir
        elif name == "densenet":
            d = Config.densenet_dir
        else:
            raise ValueError(name)

        oof = np.load(os.path.join(d, "oof_probs.npy"))
        ids = np.load(os.path.join(d, "oof_ids.npy"), allow_pickle=True)
        order = np.argsort(ids)
        backbone_data.append((name, oof[order], ids[order]))

    # Align by ISIC ID
    ref_ids = backbone_data[0][2]
    tab_full = tab_full.set_index("isic_id").loc[ref_ids].reset_index()

    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    X = np.concatenate(
        [bd[1][:, None] for bd in backbone_data] +
        [tab_full[tab_cols].values.astype(np.float32)],
        axis=1
    )

    feature_names = [f"{bd[0]}_OOF" for bd in backbone_data] + tab_cols
    y = tab_full["target"].values.astype(int)

    return X, y, feature_names


def compute_shap_importance(model_path, X, y, feature_names, label):
    """Load model, compute SHAP, return per-feature mean |SHAP|."""
    cat_model = joblib.load(model_path)

    malignant_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0][:500]
    sample_idx = np.concatenate([malignant_idx, benign_idx])
    X_sample = X[sample_idx]

    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)

    print(f"\n{label}")
    sorted_idx = np.argsort(-mean_abs)
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {feature_names[idx]:<40s} {mean_abs[idx]:.4f}")

    return mean_abs


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # =====================================================================
    # Configuration A: TinyViT only + tabular
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("CONFIGURATION A: TinyViT only + tabular")
    print(f"{'=' * 70}")
    X_tiny, y_tiny, feat_tiny = load_features(["tinyvit"])
    imp_tiny = compute_shap_importance(
        os.path.join(Config.model_dir, "tinyvit_cat_fold1.pkl"),
        X_tiny, y_tiny, feat_tiny, "TinyViT + tabular")

    # =====================================================================
    # Configuration B: DenseNet only + tabular
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("CONFIGURATION B: DenseNet only + tabular")
    print(f"{'=' * 70}")
    X_dense, y_dense, feat_dense = load_features(["densenet"])
    imp_dense = compute_shap_importance(
        os.path.join(Config.model_dir, "densenet_cat_fold1.pkl"),
        X_dense, y_dense, feat_dense, "DenseNet + tabular")

    # =====================================================================
    # Configuration C: TinyViT + DenseNet + tabular
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("CONFIGURATION C: TinyViT + DenseNet + tabular")
    print(f"{'=' * 70}")
    X_both, y_both, feat_both = load_features(["tinyvit", "densenet"])
    imp_both = compute_shap_importance(
        os.path.join(Config.model_dir, "tinyvit+densenet_cat_fold1.pkl"),
        X_both, y_both, feat_both, "TinyViT + DenseNet + tabular")

    # =====================================================================
    # Compare tabular feature importance across configurations
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("TABULAR FEATURE IMPORTANCE COMPARISON")
    print(f"{'=' * 70}")

    # Get tabular columns only
    tab_cols = [c for c in feat_both if c not in ["tinyvit_OOF", "densenet_OOF"]]

    # Map each config's SHAP to the tabular feature names
    def get_tabular_importance(feature_names, importance):
        name_to_imp = dict(zip(feature_names, importance))
        return np.array([name_to_imp.get(c, 0.0) for c in tab_cols])

    tab_imp_tiny = get_tabular_importance(feat_tiny, imp_tiny)
    tab_imp_dense = get_tabular_importance(feat_dense, imp_dense)
    tab_imp_both = get_tabular_importance(feat_both, imp_both)

    # Rank correlation between configs
    from scipy.stats import spearmanr

    rho_td, _ = spearmanr(tab_imp_tiny, tab_imp_dense)
    rho_tb, _ = spearmanr(tab_imp_tiny, tab_imp_both)
    rho_db, _ = spearmanr(tab_imp_dense, tab_imp_both)

    print(f"Spearman rank correlation of tabular feature importance:")
    print(f"  TinyViT vs DenseNet:    rho = {rho_td:.4f}")
    print(f"  TinyViT vs Both:        rho = {rho_tb:.4f}")
    print(f"  DenseNet vs Both:       rho = {rho_db:.4f}")

    # Top-5 comparison
    print(f"\nTop-5 tabular features per configuration:")
    print(f"{'Rank':<6} {'TinyViT only':<30} {'DenseNet only':<30} {'Both':<30}")
    print("-" * 96)

    tiny_rank = np.argsort(-tab_imp_tiny)
    dense_rank = np.argsort(-tab_imp_dense)
    both_rank = np.argsort(-tab_imp_both)

    for i in range(5):
        print(f"{i+1:<6} "
              f"{tab_cols[tiny_rank[i]]:<30} "
              f"{tab_cols[dense_rank[i]]:<30} "
              f"{tab_cols[both_rank[i]]:<30}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tab_cols))
    width = 0.3
    ax.bar(x - width, tab_imp_tiny, width, label="TinyViT only")
    ax.bar(x, tab_imp_dense, width, label="DenseNet only")
    ax.bar(x + width, tab_imp_both, width, label="Both")
    ax.set_xticks(x)
    ax.set_xticklabels(tab_cols, rotation=90, fontsize=7)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Tabular Feature Importance Across Configurations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, "architecture_shap_comparison.png"),
                dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()