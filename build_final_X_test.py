"""
build_final_X_test.py
Reconstruct final_X_test.npy without rerunning the ablation.
Order: [TinyViT_OOF, DenseNet_OOF, 38 tabular features].
"""

import os
import numpy as np
import pandas as pd


class Config:
    tinyvit_dir = r"F:\thesis 23\tinyvit experiment\tinyvitFULLUNFROZEN"
    densenet_dir = r"F:\thesis 23\5 fold experiments\baseline_B_densenet121_5fold"
    tabular_test = r"./features/tabular/test_tabular.parquet"
    manifest_dir = r"F:\thesis 23\tinyvit experiment\cohort_manifests"
    output_dir = r"./models/sequential_ablation_all3"
    banned_features = ["mel_thick_mm_dev", "mel_thick_mm"]


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    # Load test tabular
    tab_test = pd.read_parquet(Config.tabular_test)
    tab_test = tab_test.drop(columns=[c for c in Config.banned_features if c in tab_test.columns])

    # Load test manifest to get the correct order
    test_manifest = pd.read_csv(os.path.join(Config.manifest_dir, "test_manifest.csv"))

    # Align tabular to test manifest order
    tab_test = tab_test.set_index("isic_id").loc[test_manifest["isic_id"].values].reset_index()

    # Load image test predictions (already in test manifest order)
    tiny_test = np.load(os.path.join(Config.tinyvit_dir, "test_probs.npy"))
    dense_test = np.load(os.path.join(Config.densenet_dir, "test_probs.npy"))

    print(f"TinyViT test_probs:  {tiny_test.shape}")
    print(f"DenseNet test_probs: {dense_test.shape}")
    print(f"tab_test:            {tab_test.shape}")

    assert len(tiny_test) == len(tab_test), "Length mismatch"
    assert len(dense_test) == len(tab_test), "Length mismatch"

    META = ["isic_id", "patient_id", "target"]
    tab_cols = [c for c in tab_test.columns if c not in META]
    print(f"Tabular features: {len(tab_cols)}")

    # Build the final feature matrix in the same order as the ablation used
    X_test = np.concatenate([
        tiny_test[:, None].astype(np.float32),
        dense_test[:, None].astype(np.float32),
        tab_test[tab_cols].values.astype(np.float32),
    ], axis=1)

    print(f"\nX_test shape: {X_test.shape}")
    print(f"Expected:     ({len(tab_test)}, 40)")

    np.save(os.path.join(Config.output_dir, "final_X_test.npy"), X_test)

    # Also save y_test and meta_test for consistency
    y_test = tab_test["target"].values.astype(int)
    np.save(os.path.join(Config.output_dir, "y_test.npy"), y_test)
    tab_test[["isic_id", "patient_id", "target"]].to_parquet(
        os.path.join(Config.output_dir, "meta_test.parquet"), index=False)

    print(f"\n✓ Saved final_X_test.npy, y_test.npy, meta_test.parquet to {Config.output_dir}")


if __name__ == "__main__":
    main()