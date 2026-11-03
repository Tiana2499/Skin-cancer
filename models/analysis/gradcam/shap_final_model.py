"""
shap_final_model.py
SHAP analysis on the final TinyViT + DenseNet + 38 tabular CatBoost model.
"""

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold


class Config:
    model_dir = r"F:\thesis 23\tinyvit experiment\models\sequential_ablation_all3"
    feature_dir = r"./features/tabular"
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    tinyvit_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"
    output_dir = r"./analysis/shap_final"
    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]
    seed = 42
    n_splits = 5


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Rebuild the exact feature matrix used by the final model
    # ---------------------------------------------------------------------
    tab_full = pd.read_parquet(os.path.join(Config.feature_dir, "trainval_tabular.parquet"))
    tab_full = tab_full.drop(columns=[c for c in Config.banned_features if c in tab_full.columns])

    # Load and align OOF predictions from both backbones
    tiny_oof = np.load(os.path.join(Config.tinyvit_dir, "oof_probs.npy"))
    tiny_ids = np.load(os.path.join(Config.tinyvit_dir, "oof_ids.npy"), allow_pickle=True)
    dense_oof = np.load(os.path.join(Config.densenet_dir, "oof_probs.npy"))
    dense_ids = np.load(os.path.join(Config.densenet_dir, "oof_ids.npy"), allow_pickle=True)

    # Sort both by ISIC ID
    tiny_order = np.argsort(tiny_ids)
    dense_order = np.argsort(dense_ids)
    tiny_oof_s = tiny_oof[tiny_order]
    dense_oof_s = dense_oof[dense_order]
    tiny_ids_s = tiny_ids[tiny_order]

    # Align tabular to ISIC ID
    tab_full = tab_full.set_index("isic_id").loc[tiny_ids_s].reset_index()

    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_full.columns if c not in META]

    # Build the 40-feature matrix
    X_full = np.concatenate([
        tiny_oof_s[:, None],
        dense_oof_s[:, None],
        tab_full[tab_cols].values.astype(np.float32),
    ], axis=1)

    feature_names = ["TinyViT_OOF", "DenseNet_OOF"] + tab_cols

    y_full = tab_full["target"].values.astype(int)
    groups = tab_full["patient_id"].values

    print(f"X_full: {X_full.shape}")
    print(f"Feature names: {len(feature_names)}")

    # ---------------------------------------------------------------------
    # 2. Load the saved fold models, use fold 1 as the reference model
    # ---------------------------------------------------------------------
    cat_fold1 = joblib.load(os.path.join(Config.model_dir, "tinyvit+densenet_cat_fold1.pkl"))
    print(f"Loaded CatBoost fold 1 model")

    # ---------------------------------------------------------------------
    # 3. SHAP on a stratified sample
    # ---------------------------------------------------------------------
    malignant_idx = np.where(y_full == 1)[0]
    benign_idx = np.where(y_full == 0)[0][:500]
    sample_idx = np.concatenate([malignant_idx, benign_idx])
    X_sample = X_full[sample_idx]

    print(f"Computing SHAP on {len(sample_idx)} samples (363 malignant + 500 benign)")

    explainer = shap.TreeExplainer(cat_fold1)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ---------------------------------------------------------------------
    # 4. Global feature importance
    # ---------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("GLOBAL FEATURE IMPORTANCE (Mean |SHAP|)")
    print(f"{'=' * 70}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(-mean_abs_shap)

    for rank, idx in enumerate(sorted_idx, 1):
        if rank > 20:
            break
        print(f"  {rank:2d}. {feature_names[idx]:<40s} {mean_abs_shap[idx]:.4f}")

    # ---------------------------------------------------------------------
    # 5. SHAP summary plot
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, max_display=20,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, "shap_summary_final.png"),
                dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: shap_summary_final.png")

    # ---------------------------------------------------------------------
    # 6. Category aggregation
    # ---------------------------------------------------------------------
    categories = {
        "Image OOF": ["TinyViT_OOF", "DenseNet_OOF"],
        "Size": [c for c in tab_cols if "size" in c.lower() or "area" in c.lower() or "diam" in c.lower() or "perimeter" in c.lower() or "minoraxis" in c.lower()],
        "Color": [c for c in tab_cols if "_a_" in c or "_b_" in c or "_l_" in c or "_h_" in c or "color" in c.lower() or "delta" in c.lower() or "hue" in c.lower()],
        "Position": [c for c in tab_cols if c.endswith("_x_dev") or c.endswith("_y_dev") or c.endswith("_z_dev")],
        "Shape/Border": [c for c in tab_cols if "border" in c.lower() or "symm" in c.lower() or "eccentric" in c.lower() or "norm" in c.lower()],
        "Patient Context": ["lesions_per_patient", "patient_mean_age", "patient_mean_size", "training_support_reliability"],
        "Nevus/DNN Confidence": [c for c in tab_cols if "nevi_confidence" in c.lower() or "dnn" in c.lower()],
    }

    print(f"\n{'=' * 70}")
    print("CATEGORY-LEVEL IMPORTANCE")
    print(f"{'=' * 70}")

    for cat_name, cat_features in categories.items():
        idxs = [i for i, f in enumerate(feature_names) if f in cat_features]
        if not idxs:
            continue
        cat_shap = mean_abs_shap[idxs].sum()
        print(f"  {cat_name:<30s} {cat_shap:.4f}  ({len(idxs)} features)")

    # ---------------------------------------------------------------------
    # 7. Feature direction analysis for top features
    # ---------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("TOP FEATURE DIRECTIONS")
    print(f"{'=' * 70}")

    for rank, idx in enumerate(sorted_idx[:10], 1):
        feat_name = feature_names[idx]
        feat_values = X_sample[:, idx]
        shap_vals = shap_values[:, idx]

        # Correlation between feature value and SHAP value
        corr = np.corrcoef(feat_values, shap_vals)[0, 1]
        direction = "Higher → malignant" if corr > 0.1 else "Higher → benign" if corr < -0.1 else "Mixed"

        print(f"  {rank:2d}. {feat_name:<40s} corr={corr:+.3f}  {direction}")

    print(f"\n✓ Saved to {Config.output_dir}")


if __name__ == "__main__":
    main()