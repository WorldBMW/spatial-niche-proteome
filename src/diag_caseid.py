"""诊断 case_id 匹配 + 为什么 n_valid=0"""
import pandas as pd
import numpy as np

FEAT = r"f:\ai_paper\experiment2_fm_time\results_vitl14_jbi\all_features_omics_aligned.csv"
OMICS_DIR = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

print("=== 1. features case_id ===")
feat = pd.read_csv(FEAT)
print(f"  shape: {feat.shape}")
print(f"  case_id dtype: {feat['case_id'].dtype}")
print(f"  case_id head: {feat['case_id'].head(5).tolist()}")
print(f"  case_id unique: {feat['case_id'].nunique()}")

print("\n=== 2. proteomics case_id ===")
luad_p = pd.read_csv(f"{OMICS_DIR}/luad_proteomics.csv", encoding="latin-1", on_bad_lines="skip")
lusc_p = pd.read_csv(f"{OMICS_DIR}/lusc_proteomics.csv", encoding="latin-1", on_bad_lines="skip")
print(f"  luad_prot shape: {luad_p.shape}")
print(f"  lusc_prot shape: {lusc_p.shape}")

# 找 case_id 列
for df in [luad_p, lusc_p]:
    id_cols = [c for c in df.columns if any(k in c.lower() for k in ["patient", "case", "sample", "id"])]
    print(f"  possible id cols: {id_cols[:5]}")

# 用 run_cptac 产出的 roster 对齐
roster = pd.read_csv(f"{OMICS_DIR}/patient_roster.csv", encoding="latin-1", on_bad_lines="skip")
print(f"\n  roster shape: {roster.shape}")
print(f"  roster head:\n{roster.head()}")

print("\n=== 3. case_id 交集 ===")
# proteomics 的 case_id 列
luad_pid = luad_p.columns[0]  # 假设第一列
lusc_pid = lusc_p.columns[0]
print(f"  using id cols: luad={luad_pid}, lusc={lusc_pid}")

luad_ids = set(luad_p[luad_pid].astype(str))
lusc_ids = set(lusc_p[lusc_pid].astype(str))
prot_ids = luad_ids | lusc_ids
feat_ids = set(feat["case_id"].astype(str))

print(f"  feat case_ids: {len(feat_ids)}")
print(f"  prot case_ids (LUAD+LUSC union): {len(prot_ids)}")
print(f"  intersection: {len(feat_ids & prot_ids)}")
print(f"  feat only: {len(feat_ids - prot_ids)}")
print(f"  prot only: {len(prot_ids - feat_ids)}")

# 看几个 feat only 长什么样
only_feat = feat_ids - prot_ids
print(f"\n  feat only examples: {list(only_feat)[:10]}")
print(f"  prot only examples: {list(prot_ids - feat_ids)[:10]}")

print("\n=== 4. proteomics target 非 NaN 率 ===")
prot = pd.concat([luad_p, lusc_p], ignore_index=True)
prot = prot.rename(columns={luad_pid: "case_id"})
prot["case_id"] = prot["case_id"].astype(str)

# 取交集
common_ids = feat_ids & prot_ids
prot_common = prot[prot["case_id"].isin(common_ids)]
print(f"  prot common shape: {prot_common.shape}")

# 随机抽几个 target 看非 NaN 率
target_cols = [c for c in prot_common.columns if c != "case_id"]
print(f"  total targets: {len(target_cols)}")

non_nan_rates = prot_common[target_cols].notna().mean().sort_values(ascending=False)
print(f"  target 非 NaN rate top5: {non_nan_rates.head().tolist()}")
print(f"  target 非 NaN rate bot5: {non_nan_rates.tail().tolist()}")
print(f"  target 非 NaN rate mean: {non_nan_rates.mean():.3f}")
print(f"  target 非 NaN rate <0.5 的数: {(non_nan_rates < 0.5).sum()}")
