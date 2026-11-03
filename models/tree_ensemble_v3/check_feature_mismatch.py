"""
check_feature_mismatch.py
"""

import pandas as pd
import numpy as np

# Load test manifest with site columns
test_manifest = pd.read_csv("./cohort_manifests/test_manifest_with_site.csv")

# Load corrected features
X_test = np.load("./features/corrected/X_test.npy")
print(f"X_test shape: {X_test.shape}")

# Reconstruct feature list
META_COLS = ["isic_id", "patient_id", "target",
             "anatom_site_general", "tbp_lv_location",
             "tbp_lv_location_simple", "sex"]
BANNED = ["mel_thick_mm_dev", "mel_thick_mm"]

tab_cols = [c for c in test_manifest.columns
            if c not in META_COLS and c not in BANNED]

print(f"Tabular features from manifest: {len(tab_cols)}")
print(f"Expected total (1 OOF + {len(tab_cols)}): {1 + len(tab_cols)}")
print(f"Actual X_test columns: {X_test.shape[1]}")

missing = X_test.shape[1] - (1 + len(tab_cols))
print(f"Missing: {missing} feature name(s)")

# List all manifest columns
print(f"\nAll manifest columns ({len(test_manifest.columns)}):")
for c in test_manifest.columns:
    tag = ""
    if c in META_COLS: tag = " [META]"
    elif c in BANNED: tag = " [BANNED]"
    print(f"  {c}{tag}")