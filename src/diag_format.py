import pandas as pd
import numpy as np

# proteomics 可能是宽表 (行=蛋白, 列=样本), 不是长表
# 试两种读方式

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

# 试 1: header=0 (默认, 行=样本, 列=蛋白)
print("=== 方式 1: 默认 header=0 ===")
luad1 = pd.read_csv(f"{OMICS}/luad_proteomics.csv", encoding="latin-1", on_bad_lines="skip")
print(f"  shape: {luad1.shape}")
print(f"  col0 head: {luad1.columns[0]}, head values: {luad1.iloc[:3,0].tolist()}")

# 试 2: 不设 header, 然后 set_index 第一列
print("\n=== 方式 2: header=None, 然后 set_index ===")
luad2 = pd.read_csv(f"{OMICS}/luad_proteomics.csv", encoding="latin-1", on_bad_lines="skip", header=None)
print(f"  shape: {luad2.shape}")
print(f"  row0 head (列名行): {luad2.iloc[0, :5].tolist()}")
print(f"  col0 head (patient_id 列): {luad2.iloc[:5, 0].tolist()}")

# 试 3: 用第一列作 index (宽表: 行=蛋白, 列=样本)
print("\n=== 方式 3: index_col=0 (宽表转置) ===")
luad3 = pd.read_csv(f"{OMICS}/luad_proteomics.csv", encoding="latin-1", on_bad_lines="skip", index_col=0)
print(f"  shape: {luad3.shape} (行=蛋白, 列=样本)")
print(f"  col head (样本ID): {luad3.columns[:5].tolist()}")
print(f"  row head (蛋白): {luad3.index[:5].tolist()}")

# 对 lusc 也试方式 3
print("\n=== lusc_prot index_col=0 ===")
lusc3 = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", encoding="latin-1", on_bad_lines="skip", index_col=0)
print(f"  shape: {lusc3.shape}")
print(f"  col head (样本): {lusc3.columns[:5].tolist()}")
print(f"  row head (蛋白): {lusc3.index[:5].tolist()}")

# 和 features case_id 匹配
print("\n=== 用方式 3 的列名当 patient_id 试交集 ===")
feat = pd.read_csv(r"f:\ai_paper\experiment2_fm_time\results_vitl14_jbi\all_features_omics_aligned.csv")
feat_ids = set(feat["case_id"].astype(str))

luad_ids = set(luad3.columns.astype(str))
lusc_ids = set(lusc3.columns.astype(str))
all_prot_ids = luad_ids | lusc_ids

common = feat_ids & all_prot_ids
print(f"  feat case_ids: {len(feat_ids)}")
print(f"  prot (宽表 col) case_ids: {len(all_prot_ids)}")
print(f"  交集: {len(common)}")
print(f"  feat only: {list(feat_ids - all_prot_ids)[:5]}")
print(f"  prot only: {list(all_prot_ids - feat_ids)[:5]}")

# 看交集的 case_id 长什么样
print(f"\n  common case_ids head: {list(common)[:10]}")
