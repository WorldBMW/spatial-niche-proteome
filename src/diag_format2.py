"""验证正确的 omics 读入方式 + case_id 匹配"""
import pandas as pd
import numpy as np

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

# === proteomics ===
luad = pd.read_csv(f"{OMICS}/luad_proteomics.csv", encoding="latin-1", on_bad_lines="skip")
lusc = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", encoding="latin-1", on_bad_lines="skip")

# 找 patient_id 列
def get_id_col(df):
    for c in df.columns:
        if c.lower() in ("patient_id", "case_id", "sample_id") or "patient" in c.lower():
            return c
    return df.columns[0]

luad_id = get_id_col(luad)
lusc_id = get_id_col(lusc)
print(f"luad: id_col={luad_id}, shape={luad.shape}")
print(f"  luad[{luad_id}] head: {luad[luad_id].astype(str).tolist()[:5]}")
print(f"lusc: id_col={lusc_id}, shape={lusc.shape}")
print(f"  lusc[{lusc_id}] head: {lusc[lusc_id].astype(str).tolist()[:5]}")

# 统一 + 拼接
luad = luad.rename(columns={luad_id: "case_id"})
lusc = lusc.rename(columns={lusc_id: "case_id"})

# 去掉 lusc 里的 'patient_id' 列（如果 lusc 第一列就是 patient_id 值而不是列名？）
# 看 lusc 的列名第一个是什么
print(f"\nlusc columns[:3]: {lusc.columns[:3].tolist()}")

prot = pd.concat([luad, lusc], ignore_index=True)
prot["case_id"] = prot["case_id"].astype(str).str.strip()
print(f"\nproteomics combined shape: {prot.shape}")
print(f"case_id 唯一数: {prot['case_id'].nunique()}")

# === features ===
feat = pd.read_csv(r"f:\ai_paper\experiment2_fm_time\results_vitl14_jbi\all_features_omics_aligned.csv")
feat_ids = set(feat["case_id"].astype(str).str.strip())
prot_ids = set(prot["case_id"])

common = feat_ids & prot_ids
print(f"\nfeat ids: {len(feat_ids)}")
print(f"prot ids: {len(prot_ids)}")
print(f"交集: {len(common)}")

# 如果交集还是少，看哪些 feat_ids 在 prot 里找不到
only_feat = list(feat_ids - prot_ids)[:10]
only_prot = list(prot_ids - feat_ids)[:10]
print(f"feat only examples: {only_feat}")
print(f"prot only examples: {only_prot}")

# 检查 case_id 格式差异
if only_feat and only_prot:
    # CPTAC 有时 patient_id 带 .T/.N 后缀
    prot_ids_no_suffix = {x.split('.')[0] for x in prot_ids}
    common2 = feat_ids & prot_ids_no_suffix
    print(f"\nprot ids 去后缀后交集: {len(common2)}")
    
    # 用 roster 查
    roster = pd.read_csv(f"{OMICS}/patient_roster.csv", encoding="latin-1", on_bad_lines="skip")
    print(f"\nroster shape: {roster.shape}")
    print(f"roster patient_id head: {roster['patient_id'].astype(str).tolist()[:5]}")
    
    roster_ids = set(roster["patient_id"].astype(str))
    print(f"feat ∩ roster: {len(feat_ids & roster_ids)}")
    print(f"prot ∩ roster: {len(prot_ids & roster_ids)}")
    print(f"prot_no_suffix ∩ roster: {len(prot_ids_no_suffix & roster_ids)}")
