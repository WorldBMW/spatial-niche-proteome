"""正式拉取 TCGA-LUAD/LUSC 的全部突变、TMB、CIMP 标签"""
import sys, os, json, time, warnings
from pathlib import Path
import requests
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "TCGA"
DATA_RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://www.cbioportal.org/api"
GENES = ["EGFR", "KRAS", "TP53", "ALK", "RET", "ROS1", "BRAF", "STK11", "KEAP1", "MET", "ARAF", "MAP2K1", "PIK3CA"]
STUDIES = [
    ("TCGA-LUAD", "luad_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018_mutations"),
    ("TCGA-LUSC", "lusc_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018_mutations"),
]


def api(path: str, params: dict = None, retries: int = 3):
    url = f"{BASE}{path}"
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code != 429:
                print(f"  [{r.status_code}] {path} {str(params)[:80]}")
        except Exception as e:
            print(f"  [RETRY {i+1}] {e}")
        time.sleep(1.5 * (2 ** i))
    return None


def fetch_all_mutations(study_id: str, mut_profile_id: str, genes: list) -> pd.DataFrame:
    """拉全部 mutation（不用 gene 过滤，避免 sampleListId 参数问题）。"""
    print(f"  fetching ALL mutations from {study_id}...")
    # 用 POST mutations/fetch 或直接 mutations endpoint 不带 gene filter
    all_muts = []
    for gene in genes:
        data = api(
            f"/molecular-profiles/{mut_profile_id}/mutations",
            params={"hugoGeneSymbol": gene, "projection": "DETAILED"}
        )
        if data:
            for m in data:
                all_muts.append({
                    "case_id": m.get("patientId", ""),
                    "sample_id": m.get("sampleId", ""),
                    "gene": gene,
                    "mutation_type": m.get("mutationType", ""),
                    "variant": m.get("variantAllele", ""),
                    "protein_change": m.get("proteinChange", ""),
                })
        time.sleep(0.4)
    df = pd.DataFrame(all_muts)
    print(f"    got {len(df)} mutation records")
    return df


def fetch_clinical_all(study_id: str) -> pd.DataFrame:
    """拉 clinical: PATIENT_ID, SEX, TMB, OS, 甲基化 cluster 等"""
    print(f"  fetching clinical from {study_id}...")
    # 先查 available attributes
    attrs_needed = [
        "PATIENT_ID", "SEX", "OS_MONTHS", "OS_STATUS", "VITAL_STATUS",
        "TMB_NONSYNONYMOUS", "FRACTION_GENOME_ALTERED", "CSR_METHYLATION",
        "CIMP_STATUS", "METHYLATION_CLUSTER", "ESTIMATE_ESTIMATESCORE",
        "ESTIMATE_IMMUNESCORE", "ESTIMATE_STROMALSCORE",
    ]
    rows = []
    for attr in attrs_needed:
        data = api(f"/studies/{study_id}/clinical-data", params={"attributeId": attr})
        if data:
            for r in data:
                pid = r.get("patientId") or r.get("value", "")
                rows.append({
                    "case_id": pid,
                    "attribute": attr,
                    "value": r.get("value", ""),
                })
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    # 长表 → 宽表
    wide = df.pivot_table(index="case_id", columns="attribute", values="value", aggfunc="first").reset_index()
    print(f"    got {len(wide)} patients, {len(wide.columns)-1} attributes")
    return wide


def process_study(study_name: str, study_id: str, mut_profile_id: str) -> tuple:
    print(f"\n{'='*60}")
    print(f"[{study_name}]")
    print(f"{'='*60}")
    mut_df = fetch_all_mutations(study_id, mut_profile_id, GENES)
    clin_df = fetch_clinical_all(study_id)
    return mut_df, clin_df


def merge_mutations_to_wide(mut_df: pd.DataFrame) -> pd.DataFrame:
    """case_id × gene → 1/0 mutation 宽表"""
    if mut_df.empty:
        return pd.DataFrame(columns=["case_id"] + GENES)
    wide = mut_df.groupby(["case_id", "gene"]).size().unstack(fill_value=0)
    wide = (wide > 0).astype(int)
    wide.columns = [f"{c}_mutation" for c in wide.columns]
    wide = wide.reset_index()
    return wide


def main():
    out_path = DATA_RAW / "tcga_clinical_full.csv"
    all_muts, all_clins = [], []

    for name, sid, mp in STUDIES:
        mut, clin = process_study(name, sid, mp)
        mut["cohort"] = name
        clin["cohort"] = name
        all_muts.append(mut)
        all_clins.append(clin)

    mut_df = pd.concat(all_muts, ignore_index=True) if all_muts else pd.DataFrame()
    clin_df = pd.concat(all_clins, ignore_index=True) if all_clins else pd.DataFrame()

    # 突变宽表
    mut_wide = merge_mutations_to_wide(mut_df)

    # TMB 二值化
    if "TMB_NONSYNONYMOUS" in clin_df.columns:
        clin_df["TMB_NONSYNONYMOUS"] = pd.to_numeric(clin_df["TMB_NONSYNONYMOUS"], errors="coerce")
        med = clin_df["TMB_NONSYNONYMOUS"].median()
        clin_df["TMB_high"] = (clin_df["TMB_NONSYNONYMOUS"] > med).astype("Int64")
        print(f"\nTMB median={med:.1f}  high={clin_df['TMB_high'].sum()}")

    # CIMP 二值化
    for col in ["CSR_METHYLATION", "CIMP_STATUS", "METHYLATION_CLUSTER"]:
        if col in clin_df.columns:
            clin_df["CIMP_high"] = clin_df[col].astype(str).map(
                lambda x: 1 if any(k in x.lower() for k in ["cimp+", "cimp high", "hypermethylated"])
                else (0 if any(k in x.lower() for k in ["cimp-", "cimp low", "hypomethylated", "cpg"]) else np.nan)
            )
            print(f"CIMP_status from {col}: {clin_df['CIMP_high'].value_counts(dropna=False).to_dict()}")
            break

    # OS 转换
    if "OS_MONTHS" in clin_df.columns:
        clin_df["os_days"] = pd.to_numeric(clin_df["OS_MONTHS"], errors="coerce") * 30.44
        clin_df["event"] = clin_df["OS_STATUS"].astype(str).map(
            lambda x: 1 if "DEAD" in x.upper() else (0 if "ALIVE" in x.upper() else np.nan)
        )

    # merge
    final = clin_df.merge(mut_wide, on="case_id", how="left")

    # 补 experiment1 已有的 200 例 survival（先标 uuid，等真实 UUID 映射）
    exp1_surv = pd.read_csv(ROOT.parent / "experiment1" / "data" / "raw" / "TCGA" / "tcga_clinical.csv")
    exp1_surv = exp1_surv.rename(columns={"case_id": "uuid"})

    final.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"[SAVED] {len(final)} patients → {out_path}")
    print(f"  columns ({len(final.columns)}): {list(final.columns[:15])}...")
    print(f"  突变列: {[c for c in final.columns if '_mutation' in c]}")
    print(f"  OS 可用: {final['os_days'].notna().sum()}")
    print(f"  TMB 可用: {final['TMB_high'].notna().sum()}")


if __name__ == "__main__":
    main()
