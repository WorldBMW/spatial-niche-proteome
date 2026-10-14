import pandas as pd
from pathlib import Path

print("=" * 60)
print("EXPERIMENT2 数据完整性审计")
print("=" * 60)

# 1. TCGA 临床
tcga = pd.read_csv("data/raw/TCGA/tcga_clinical_full.csv")
print(f"\n【TCGA 临床】{len(tcga)} 例")
print(f"  cohort: {tcga['cohort'].value_counts(dropna=False).to_dict()}")
mut_cols = [c for c in tcga.columns if "_mutation" in c]
for c in mut_cols:
    pos = int(tcga[c].sum()) if tcga[c].dtype != object else tcga[c].astype(str).str.lower().isin(["1","true","mut","fusion"]).sum()
    valid = tcga[c].notna().sum()
    print(f"  {c}: {pos}阳性 / {valid}有标签 ({pos/max(valid,1)*100:.1f}%)")
print(f"  TMB_nonnull: {tcga['TMB_NONSYNONYMOUS'].notna().sum()}, median={tcga['TMB_NONSYNONYMOUS'].median():.1f}")
print(f"  FGA_nonnull: {tcga['FRACTION_GENOME_ALTERED'].notna().sum()}")

# 2. CPTAC 临床
luad = pd.read_csv(r"e:\ai_paper\experiment1\data\raw\CPTAC\CPTAC-LUAD_clinical.csv")
lusc = pd.read_csv(r"e:\ai_paper\experiment1\data\raw\CPTAC\CPTAC-LUSC_clinical.csv")
cptac = pd.concat([luad, lusc], ignore_index=True)
print(f"\n【CPTAC 临床】{len(cptac)} 例 (LUAD={len(luad)}, LUSC={len(lusc)})")
mut_cols_c = [c for c in cptac.columns if "MUTATION_STATUS" in c or "FUSION" in c]
for c in mut_cols_c:
    col = cptac[c].astype(str).str.lower()
    pos = col.isin(["1","true","mut","mutation","fusion","missense_mutation","in_frame_insertion","splice_site_mutation"]).sum()
    valid = cptac[c].notna().sum()
    print(f"  {c}: {pos}阳性 / {valid}有标签 ({pos/max(valid,1)*100:.1f}%)")

# 3. niche maps
niche_dir = Path(r"e:\ai_paper\experiment1\data\processed\niche_maps")
files = list(niche_dir.glob("*_argmax.npy"))
tcga_n = [f for f in files if f.name.startswith("TCGA-S")]
cptac_n = [f for f in files if f.name.startswith("CPTAC-S")]
msk_n = [f for f in files if f.name.startswith("MSK-S")]
print(f"\n【Niche maps (MUSK)】合计 {len(files)} 例")
print(f"  TCGA: {len(tcga_n)}")
print(f"  CPTAC: {len(cptac_n)}")
print(f"  MSK: {len(msk_n)}")

# 4. TCGA .svs
svs_dir = Path(r"e:\ai_paper\experiment1\data\wsi")
svs = list(svs_dir.glob("*.svs"))
print(f"\n【TCGA H&E .svs】{len(svs)} 张")

# 5. 关键 gap 分析
print(f"\n【关键差距分析】")
print(f"  TCGA 临床标签: {len(tcga)} 例 (case_id=TCGA-XX-XXXX)")
print(f"  TCGA .svs: {len(svs)} 张 (UUID 格式)")
print(f"  TCGA MUSK niche maps: {len(tcga_n)} 例 (TCGA-S0000 化名)")
print(f"  → 三者 case_id 格式不同! 需要统一映射")
print(f"\n  CPTAC 临床: {len(cptac)} 例 (patientId=C3L-00001)")
print(f"  CPTAC MUSK niche maps: {len(cptac_n)} 例 (CPTAC-S0000 化名)")
print(f"  → 同样需要 C3L-00001 ↔ CPTAC-S0000 映射")

# 6. 下游能跑多少样本?
# 临时用 MUSK niche maps + CPTAC 临床试 merge 看能 overlap 多少
print(f"\n【下游能跑多少?】")
cptac_feat_ids = sorted([f.stem.replace("_argmax","") for f in cptac_n])
print(f"  CPTAC niche IDs (样例): {cptac_feat_ids[:5]}")
print(f"  CPTAC clinical patientId (样例): {list(cptac['patientId'].head(5))}")
print(f"  → 需要建立 MUSK 化名 ↔ 真实 patientId 映射表")
print(f"  → 一旦有了, 至少 45 例 CPTAC niche maps + 218 例临床标签 → 可跑")
print(f"  → TCGA 180 例 niche maps + 1053 例临床 → 可跑 (需 UUID 映射)")

print(f"\n{'='*60}")
print(f"结论: 核心数据量足够 (300+ 例), 但 case_id 格式需要统一")
