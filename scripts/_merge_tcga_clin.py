"""把 TMB/FGA 临床表 + 突变宽表 merge 成完整 tcga_clinical_full.csv"""
import pandas as pd

clin = pd.read_csv(r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_full.csv")
mut = pd.read_csv(r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_mutations_wide.csv")

print(f"临床表: {len(clin)} patients, case_id 样例: {list(clin['case_id'].head(3))}")
print(f"突变表: {len(mut)} patients, case_id 样例: {list(mut['case_id'].head(3))}")

merged = clin.merge(mut, on="case_id", how="outer")
print(f"\nmerge 后: {len(merged)} patients")

# 加生存数据: 从 experiment1 现有 clinical.csv 用 uuid→case_id 映射补 os_days + event
# experiment1 clinical.csv 里 case_id 是 UUID，需要先找到 UUID → TCGA-XX-XXXX 的映射
exp1_clin = pd.read_csv(r"e:\ai_paper\experiment1\data\raw\TCGA\tcga_clinical.csv")
exp1_clin = exp1_clin.rename(columns={"case_id": "uuid"})

# 读 WSI manifest 拿 UUID → TCGA case_id 映射
manifest = pd.read_csv(r"e:\ai_paper\experiment1\data\processed\tcga_wsi_manifest.csv")
print(f"\nmanifest: {len(manifest)} entries")
if "case_id" in manifest.columns:
    print(f"  manifest case_id 样例: {list(manifest['case_id'].head(3))}")

# manifest case_id 是 GDC UUID，experiment1 patch_file 也是 UUID.npy
# 我们需要找到 UUID → TCGA-XX-XXXX 真实 patientId 映射
# 让我从 manifest 直接去查 GDC
import requests
manifest = manifest.head(5)
uuid_map = {}
for _, row in manifest.iterrows():
    uid = row["case_id"]
    r = requests.get(f"https://api.gdc.cancer.gov/files/{uid}", timeout=10)
    if r.status_code == 200:
        d = r.json()
        # 拿 associated_entities → cases → submitter_id
        cases = d.get("associated_entities", [{}])[0].get("case_id", "")
        # 或者直接 cases
        data = d.get("data", {})
        cases2 = data.get("cases", [{}])
        if cases2:
            pid = cases2[0].get("submitter_id", "")
            uuid_map[uid] = pid
            print(f"  {uid[:8]}... → {pid}")

print(f"\nUUID map (sample): {list(uuid_map.items())[:3]}")

# merge survival 等 UUID 字段
exp1_clin["tcga_case_id"] = exp1_clin["uuid"].map(uuid_map)
print(f"\nexp1_clin 映射后 tcga_case_id 非空: {exp1_clin['tcga_case_id'].notna().sum()}")

final = merged.merge(exp1_clin[["tcga_case_id", "os_days", "event", "vital_status"]].rename(
    columns={"tcga_case_id": "case_id"}), on="case_id", how="left")

out = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_full.csv"
final.to_csv(out, index=False)
print(f"\n[SAVED] {len(final)} patients → {out}")
print(f"  列 ({len(final.columns)}): {list(final.columns)}")

mut_cols = [c for c in final.columns if "_mutation" in c]
print(f"\n突变列 ({len(mut_cols)}): {mut_cols}")
for c in mut_cols[:5]:
    print(f"  {c}: {final[c].sum()} pos / {final[c].notna().sum()} valid")

print(f"\nTMB_high: {final['TMB_high'].sum()} / {final['TMB_high'].notna().sum()}")
print(f"os_days valid: {final['os_days'].notna().sum()} / {len(final)}")
print(f"event 非空: {final['event'].notna().sum()}")
