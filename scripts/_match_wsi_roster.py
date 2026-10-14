"""PathDB 清单 x 组学 roster 交集: 统计 216 例中实际可下载切片数。"""
import pandas as pd

META = r"data/raw/CPTAC/cptac_pathdb_metadata.csv"
ROSTER = r"data/raw/CPTAC/omics/patient_roster.csv"

df = pd.read_csv(META, dtype=str)
lung = df[df["Tumor"].isin(["LUAD", "LSCC"])].copy()
print(f"PathDB 全部玻片行: {len(df)}; LUAD+LSCC 行: {len(lung)}")
print(f"LUAD/LSCC 不同病例 Case_ID: {lung['Case_ID'].nunique()}")
print(f"  其中 LUAD 病例: {lung[lung.Tumor=='LUAD']['Case_ID'].nunique()}, "
      f"LSCC 病例: {lung[lung.Tumor=='LSCC']['Case_ID'].nunique()}")

print("\n按 Specimen_Type x Embedding_Medium 玻片数:")
print(lung.groupby(["Tumor", "Specimen_Type", "Embedding_Medium"]).size())

print("\nPathology 列(下载标识)非空率:", lung["Pathology"].notna().mean().round(3))
print("Pathology 示例:", lung["Pathology"].dropna().unique()[:3].tolist())

# 组学完整 5 模态 roster
ros = pd.read_csv(ROSTER, dtype=str)
mods = ["has_clinical", "has_proteomics", "has_phosphoproteomics",
        "has_transcriptomics", "has_somatic_mutation"]
ros["n_mods"] = ros[mods].astype(int).sum(axis=1)
complete = ros[ros.n_mods == 5].copy()
print(f"\n组学 5 模态完整病例: {len(complete)} "
      f"(LUAD {sum(complete.cohort=='luad')}, LUSC {sum(complete.cohort=='lusc')})")

# join: PathDB 玻片清单 ∩ 完整病例
m = lung.merge(complete[["patient_id", "cohort"]],
               left_on="Case_ID", right_on="patient_id", how="inner")
print(f"\n===== 交集: 落在 216 完整病例上的玻片行 = {len(m)} =====")
print(f"覆盖病例数: {m['Case_ID'].nunique()} / {len(complete)}")
miss = set(complete.patient_id) - set(m.Case_ID)
print(f"无任何 PathDB 玻片的完整病例: {len(miss)} -> {sorted(miss)}")

# 仅肿瘤组织 + FFPE(H&E 常规诊断)
tumor_ffpe = m[(m.Specimen_Type == "tumor_tissue") & (m.Embedding_Medium == "FFPE")]
print(f"\n肿瘤组织 FFPE 玻片: {len(tumor_ffpe)} 行, 覆盖 {tumor_ffpe['Case_ID'].nunique()} 例")
tumor_any = m[m.Specimen_Type == "tumor_tissue"]
print(f"肿瘤组织(任意包埋)玻片: {len(tumor_any)} 行, 覆盖 {tumor_any['Case_ID'].nunique()} 例")

print("\n按癌种统计 (肿瘤组织 FFPE):")
for c, g in tumor_ffpe.groupby("Tumor"):
    per = g.groupby("Case_ID").size()
    print(f"  {c}: {g['Case_ID'].nunique()} 例, {len(g)} 张, "
          f"每例 {per.min()}-{per.max()} 张, 均值 {per.mean():.2f}")

# 每例玻片数分布(肿瘤FFPE)
print("\n每例肿瘤FFPE玻片数分布:")
print(tumor_ffpe.groupby("Case_ID").size().value_counts().sort_index())

# 保存可下载清单
out = tumor_ffpe[["Case_ID", "Tumor", "Specimen_ID", "Slide_ID", "Pathology",
                  "Embedding_Medium", "Specimen_Type", "Percent_Tumor_Nuclei",
                  "Tumor_Segment_Acceptable"]].copy()
out.to_csv(r"data/raw/CPTAC/cptac_wsi_tumor_ffpe_manifest.csv", index=False)
print(f"\n可下载肿瘤FFPE清单 -> data/raw/CPTAC/cptac_wsi_tumor_ffpe_manifest.csv ({len(out)} 行)")
