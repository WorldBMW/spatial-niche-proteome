"""诊断: 本地 .svs vs 服务器已 embed 的 case_id 差集"""
import os, glob, re, pandas as pd
from pathlib import Path

# 1. 本地所有 .svs → case_id
print("=" * 60)
print("1. 本地 .svs 扫描")
print("=" * 60)
svs_dirs = [
    r"e:\ai_paper\experiment2_fm_time\data\wsi",
    r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\svs",
    r"e:\ai_paper\experiment1\data\wsi",
    r"e:\ai_paper\experiment1\data\raw\TCGA\svs",
    r"e:\ai_paper\raw\TCGA\svs",
]
all_svs = {}  # case_id → list of paths
for d in svs_dirs:
    if not os.path.exists(d):
        continue
    for f in glob.glob(os.path.join(d, "**", "*.svs"), recursive=True):
        fname = os.path.basename(f)
        # TCGA filename 通常: TCGA-XX-XXXX-01Z-00-DX1.XXXXXX.svs
        m = re.match(r"(TCGA-\d{2}-\d{4})", fname)
        if m:
            cid = m.group(1)
            if cid not in all_svs:
                all_svs[cid] = []
            all_svs[cid].append(f)

print(f"  本地 unique case_id: {len(all_svs)}")
total_svs = sum(len(v) for v in all_svs.values())
print(f"  总 .svs 文件: {total_svs}")

# 2. 服务器已 embed 的 (看本地 features CSV 里的 case_id)
print()
print("=" * 60)
print("2. 服务器已 embed patients (features CSV)")
print("=" * 60)
features_dir = r"e:\ai_paper\experiment2_fm_time\data\features"
emb_case_ids = set()
for bb in ["resnet50", "vit_b16", "dinov2_vitb14", "dinov2_vitl14"]:
    csv = os.path.join(features_dir, f"{bb}_features", "all_features.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        col = [c for c in df.columns if "case" in c.lower() or "patient" in c.lower() or "id" in c.lower()]
        print(f"  {bb}: {df.shape[0]} rows, id cols={col}")
        if col:
            emb_case_ids.update(df[col[0]].astype(str).tolist())
    else:
        print(f"  {bb}: CSV 不存在")
print(f"  服务器已 embed unique case_id: {len(emb_case_ids)}")

# 3. 差集
print()
print("=" * 60)
print("3. 差集")
print("=" * 60)
local_ids = set(all_svs.keys())
missing = local_ids - emb_case_ids
overlap = local_ids & emb_case_ids
extra_server = emb_case_ids - local_ids

print(f"  本地有 & 服务器已 embed: {len(overlap)}")
print(f"  本地有 & 服务器没 embed: {len(missing)}  ← 要补的!")
print(f"  服务器有 & 本地没有: {len(extra_server)}")

if missing:
    # 估算要传多少 GB
    total_mb = 0
    for cid in missing:
        for p in all_svs[cid]:
            total_mb += os.path.getsize(p) / 1e6
    print(f"\n  补 embed 需要传: {len(missing)} patients, {total_mb/1024:.1f} GB")
    
    # 按机构分 train/test 看 missing 分布
    inst_csv = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_with_institution.csv"
    if os.path.exists(inst_csv):
        inst_df = pd.read_csv(inst_csv)
        miss_df = inst_df[inst_df["case_id"].isin(missing)] if "case_id" in inst_df.columns else pd.DataFrame()
        if not miss_df.empty:
            print(f"\n  missing patients 中 train/test split:")
            print(miss_df["split_institution"].value_counts().to_dict())
            
            # 关键: missing 里有多少是 test set (机构 split 外部验证必须的)
            test_miss = miss_df[miss_df["split_institution"] == "test"]
            print(f"\n  ⭐ 外部验证 test set 里缺 {len(test_miss)} patients")

# 4. 列出前 20 个 missing case_id
print()
print("=" * 60)
print("4. Missing case_id 样本")
print("=" * 60)
print(f"  {sorted(missing)[:20]}")
if len(missing) > 20:
    print(f"  ... 还有 {len(missing)-20} 个")
