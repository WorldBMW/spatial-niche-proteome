"""
把 cBioPortal 的 clinical OS 数据 + 已有的 TMB/FGA/突变 合成完整 TCGA 标签。
直接查 OS_MONTHS 和 OS_STATUS clinical attributes。
"""
import requests, time, pandas as pd, numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE = "https://www.cbioportal.org/api"
STUDIES = [
    ("TCGA-LUAD", "luad_tcga_pan_can_atlas_2018"),
    ("TCGA-LUSC", "lusc_tcga_pan_can_atlas_2018"),
]


def fetch_clinical_attrs(study_id: str, attrs: list) -> pd.DataFrame:
    """拉指定 clinical attributes。"""
    rows = []
    for attr in attrs:
        r = requests.get(
            f"{BASE}/studies/{study_id}/clinical-data",
            params={"attributeId": attr, "projection": "DETAILED"},
            timeout=20,
        )
        print(f"  {attr}: {r.status_code}", end="")
        if r.status_code == 200:
            data = r.json()
            print(f"  records={len(data)}")
            for d in data:
                pid = d.get("patientId") or d.get("caseListId") or d.get("value", "")
                rows.append({"case_id": pid, "attribute": attr, "value": d.get("value", "")})
        else:
            print(f"  {r.text[:100]}")
        time.sleep(0.3)
    return pd.DataFrame(rows)


# Step 1: 查有哪些可用 clinical attributes 里跟 OS 相关的
print("=== 先查可用 OS 相关 attributes ===")
r = requests.get(
    f"{BASE}/studies/luad_tcga_pan_can_atlas_2018/clinical-attributes",
    timeout=15)
if r.status_code == 200:
    attrs = r.json()
    os_attrs = [a for a in attrs if any(k in a.get("clinicalAttributeId", "").upper() for k in ["OS", "SURVIV", "DAYS", "STATUS", "VITAL"])]
    print(f"OS 相关 ({len(os_attrs)}):")
    for a in os_attrs:
        print(f"  {a.get('clinicalAttributeId')}: {a.get('description', '')[:60]}")
else:
    print(f"status={r.status_code}")
    os_attrs = []

# Step 2: 直接查常用的 survival attrs
target_attrs = [
    "OS_MONTHS", "OS_STATUS", "OS_DAYS",
    "VITAL_STATUS", "SURVIVAL_DAYS",
    "DS_MONTHS", "DS_STATUS",
    "PFS_MONTHS", "PFS_STATUS",
]

all_clin = []
for cohort, sid in STUDIES:
    print(f"\n=== {cohort} ===")
    df = fetch_clinical_attrs(sid, target_attrs)
    df["cohort"] = cohort
    all_clin.append(df)

clin_long = pd.concat(all_clin, ignore_index=True)
print(f"\n合计 {len(clin_long)} clinical rows")

# 长表 → 宽表
wide = clin_long.pivot_table(index="case_id", columns="attribute", values="value", aggfunc="first").reset_index()
print(f"宽表 shape: {wide.shape}")
print(f"columns: {list(wide.columns)}")

# 数值转换
for c in wide.columns:
    if c != "case_id":
        wide[c] = pd.to_numeric(wide[c], errors="ignore")

# 生存处理
if "OS_DAYS" in wide.columns:
    wide["os_days"] = wide["OS_DAYS"]
elif "OS_MONTHS" in wide.columns:
    wide["os_days"] = wide["OS_MONTHS"] * 30.44

if "OS_STATUS" in wide.columns:
    wide["event"] = wide["OS_STATUS"].astype(str).map(
        lambda x: 1 if "DEAD" in x.upper() else (0 if "ALIVE" in x.upper() else np.nan)
    )
elif "VITAL_STATUS" in wide.columns:
    wide["event"] = wide["VITAL_STATUS"].astype(str).map(
        lambda x: 1 if "DEAD" in x.upper() else (0 if "ALIVE" in x.upper() else np.nan)
    )

print(f"\nos_days non-null: {wide['os_days'].notna().sum() if 'os_days' in wide.columns else 'N/A'}")
print(f"event non-null: {wide['event'].notna().sum() if 'event' in wide.columns else 'N/A'}")
print(f"event value_counts: {wide['event'].value_counts(dropna=False).to_dict() if 'event' in wide.columns else 'N/A'}")

# 保存
os_out = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_survival_only.csv"
wide.to_csv(os_out, index=False)
print(f"\n[SAVED] survival → {os_out}")

# Step 3: merge with 已有的 clinical+mutations
prev = pd.read_csv(r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_full.csv")
print(f"\nprev clinical+mut: {len(prev)} patients")

final = prev.merge(wide[["case_id", "os_days", "event"]].drop_duplicates("case_id"), on="case_id", how="outer")
print(f"final merged: {len(final)}")

out = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_clinical_full.csv"
final.to_csv(out, index=False)
print(f"[SAVED] {out}")
print(f"  os_days valid: {final['os_days'].notna().sum()}")
print(f"  event valid: {final['event'].notna().sum()}")
