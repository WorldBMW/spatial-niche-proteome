"""数每行的字段数，找 lusc_prot 列数不匹配的原因"""
import pandas as pd

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

# 用 csv 模块数字段数
import csv

for name in ["luad_proteomics.csv", "lusc_proteomics.csv"]:
    path = f"{OMICS}/{name}"
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f)
        row0 = next(reader)
        row1 = next(reader)
    print(f"\n=== {name} ===")
    print(f"  header 字段数: {len(row0)}")
    print(f"  row1 字段数: {len(row1)}")
    print(f"  header 前 3: {row0[:3]}")
    print(f"  row1 前 3: {row1[:3]}")

# 更精确：lusc 到底哪个列是 patient_id 值
print("\n=== lusc_proteomics: 找哪一列有 C3L/C3N 值 ===")
import pandas as pd
df = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", encoding="latin-1", header=None, nrows=5)
print(f"  shape (no header): {df.shape}")
# 扫描所有列，哪列的值看起来像 patient_id
for col in range(df.shape[1]):
    vals = df[col].astype(str).tolist()
    if any('C3L' in v or 'C3N' in v or '11LU' in v for v in vals):
        print(f"  col {col}: {vals[:5]}")
        break
