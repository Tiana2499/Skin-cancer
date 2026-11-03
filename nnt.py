import pandas as pd

tab = pd.read_parquet("./features/tabular/trainval_tabular.parquet")
print(f"Columns in parquet: {len(tab.columns)}")
print(f"mel_thick_mm_dev present: {'mel_thick_mm_dev' in tab.columns}")
print(f"\nAll columns:")
for c in sorted(tab.columns):
    print(f"  {c}")