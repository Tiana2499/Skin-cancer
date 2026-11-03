"""
nnt_analysis_final.py
Number-Needed-to-Triage at 80% and 90% sensitivity for the final model.
"""

import os
import numpy as np
import pandas as pd


class Config:
    model_dir = r"./models/sequential_ablation_all3"


def compute_nnt(y, p, target_sens):
    """Returns (n_flagged, tp, nnt)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    total_pos = y.sum()
    order = np.argsort(-p)
    y_sorted = y[order]
    cumsum_tp = np.cumsum(y_sorted)
    target_tp = int(np.ceil(target_sens * total_pos))
    idx = np.searchsorted(cumsum_tp, target_tp)
    n_flagged = idx + 1
    tp = int(cumsum_tp[idx])
    return n_flagged, tp, n_flagged / tp if tp > 0 else np.inf


def main():
    cat_probs = np.load(os.path.join(Config.model_dir, "tinyvit+densenet_test_cat.npy"))
    y_test = np.load(os.path.join(Config.model_dir, "y_test.npy"))

    print(f"{'=' * 70}")
    print("NNT ANALYSIS — FINAL MODEL")
    print(f"{'=' * 70}")
    print(f"Model: TinyViT + DenseNet + 38 tabular → CatBoost")
    print(f"Test lesions: {len(y_test):,}")
    print(f"Malignant:    {y_test.sum()}")

    results = []
    for sens in [0.80, 0.90]:
        flagged, tp, nnt = compute_nnt(y_test, cat_probs, sens)
        fp = flagged - tp
        fn = int(y_test.sum()) - tp
        tn = len(y_test) - y_test.sum() - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        print(f"\nAt {sens*100:.0f}% sensitivity:")
        print(f"  Flagged:       {flagged:,}")
        print(f"  TP / FP:       {tp} / {fp}")
        print(f"  FN / TN:       {fn} / {tn}")
        print(f"  Specificity:   {specificity:.4f}")
        print(f"  Precision:     {precision:.4f}")
        print(f"  NNT:           {nnt:.1f}")

        results.append({
            "sensitivity": sens,
            "flagged": flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "specificity": specificity,
            "precision": precision,
            "nnt": nnt,
        })

    # Reference
    print(f"\n{'=' * 70}")
    print("REFERENCE")
    print(f"{'=' * 70}")
    print(f"ISIC 2024 winner: NNT@80% = 51.57")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Sensitivity':<15} {'Flagged':>10} {'TP':>5} {'NNT':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['sensitivity']*100:<15.0f} {r['flagged']:>10,} {r['tp']:>5} {r['nnt']:>10.1f}")

    pd.DataFrame(results).to_csv(
        os.path.join(Config.model_dir, "nnt_final.csv"), index=False)
    print(f"\n✓ Saved to {Config.model_dir}/nnt_final.csv")


if __name__ == "__main__":
    main()