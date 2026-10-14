"""精确差集: 本地 .svs patients vs clinical vs 服务器已 embed 的 126"""
import os, glob, re, pandas as pd
from pathlib import Path

# 本地 .svs case_id
svs_dirs = [
    r"e:\ai_paper\experiment2_fm_time\data\wsi",
    r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\svs",
    r"e:\ai_paper\experiment1\data\wsi",
    r"e:\ai_paper\experiment1\data\raw\TCGA\svs",
    r"e:\ai_paper\raw\TCGA\svs",
]
local_svs = {}
for d in svs_dirs:
    if not os.path.exists(d):
        continue
    for f in glob.glob(os.path.join(d, "**", "*.svs"), recursive=True):
        m = re.match(r"(TCGA-\d{2}-\d{4})", os.path.basename(f))
        if m:
            cid = m.group(1)
            if cid not in local_svs:
                local_svs[cid] = []
            local_svs[cid].append(f)
local_ids = set(local_svs.keys())
print(f"本地 .svs unique case_id: {len(local_ids)}")

# clinical 里的 case_id
inst_csv = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_with_institution.csv"
clinical = pd.read_csv(inst_csv)
print(f"clinical: {clinical.shape}")
clin_ids = set(clinical["case_id_x"].astype(str))
print(f"clinical case_id: {len(clin_ids)}")

# 交集: 本地 .svs ∩ clinical (这些是能跑实验的)
usable = local_ids & clin_ids
print(f"\n⭐ 本地 .svs ∩ clinical = {len(usable)} patients 能跑实验")

# split 分布
usable_df = clinical[clinical["case_id_x"].isin(usable)]
print(f"  train: {(usable_df['split_institution']=='train').sum()}")
print(f"  test:  {(usable_df['split_institution']=='test').sum()}")

# 服务器已有 126 从哪来？下游用 clinical ∩ features
# 看本地 dinov2 features CSV 里的 case_id
feat_csv = r"e:\ai_paper\experiment2_fm_time\data\features\dinov2_vitl14_features\all_features.csv"
feat_df = pd.read_csv(feat_csv) if os.path.exists(feat_csv) else pd.DataFrame()
print(f"\n本地 features CSV (dinov2): {feat_df.shape}")

# 服务器 embed 了 126 个 → 假设这些是 usable 的子集 (本地有 svs + clinical 有 label)
# 那要补的 = usable - 126
need_embed = usable - set()  # 暂时不知道服务器具体哪 126

# 算 usable 总大小
total_mb = sum(os.path.getsize(p)/1e6 for cid in usable for p in local_svs[cid])
print(f"\nusable 总大小: {total_mb/1024:.1f} GB")

# 按机构分
print(f"\nusable 按 institution_code top 10:")
print(usable_df["institution_code"].value_counts().head(10))

print("""
──────────────────────────────────────────────────────
服务器已 embed 126 (clinical ∩ features)
usable = 本地 .svs ∩ clinical = ? patients

要补的 = usable - 126 = ? patients
这些 .svs 需要上传服务器 + 补 embed

但不知道服务器上具体是哪 126 个 case_id → 
最快方案: 本地所有 usable 的 .svs 全传到服务器, 
服务器重新全部 embed 一遍 (GPU 跑 ~10h for 222 patients)
──────────────────────────────────────────────────────
""")
