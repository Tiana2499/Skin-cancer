"""
check_banned_columns.py
Verify mel_thick_mm_dev is actually absent from the tabular parquet files.
"""

import pandas as pd

for split in ["trainval", "test"]:
    path = f"./features/tabular/{split}_tabular.parquet"
    df = pd.read_parquet(path)
    has_banned = [c for c in ["mel_thick_mm_dev", "mel_thick_mm"] if c in df.columns]
    print(f"{split}: shape={df.shape}, banned_cols={has_banned}")