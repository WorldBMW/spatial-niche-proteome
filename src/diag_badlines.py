"""精确验证：不用 on_bad_lines 读 lusc_prot 看第一列值"""
import pandas as pd

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

# 不设 on_bad_lines
print("=== lusc_proteomics, no on_bad_lines ===")
try:
    df = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", encoding="latin-1")
    print(f"  shape: {df.shape}")
    print(f"  first col dtype: {df.iloc[:,0].dtype}")
    print(f"  first col head: {df.iloc[:5, 0].astype(str).tolist()}")
except Exception as e:
    print(f"  ERROR: {e}")

# 设 on_bad_lines='skip'
print("\n=== lusc_proteomics, on_bad_lines='skip' ===")
df2 = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", encoding="latin-1", on_bad_lines="skip")
print(f"  shape: {df2.shape}")
print(f"  first col dtype: {df2.iloc[:,0].dtype}")
print(f"  first col head: {df2.iloc[:5, 0].astype(str).tolist()}")

# 比较行数差
print(f"\n  row count diff (no skip - skip): 212 - {df2.shape[0]} = {212 - df2.shape[0]}")
