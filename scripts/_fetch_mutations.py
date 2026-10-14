"""
拉 TCGA-LUAD/LUSC 全部 mutation data
cBioPortal mutations endpoint 需要用 POST /mutations/fetch
"""
import requests, json, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

BASE = "https://www.cbioportal.org/api"
STUDIES = [
    ("TCGA-LUAD", "luad_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018_mutations"),
    ("TCGA-LUSC", "lusc_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018_mutations"),
]

GENES = ["EGFR", "KRAS", "TP53", "ALK", "RET", "ROS1", "BRAF", "STK11", "KEAP1", "MET", "ARAF", "MAP2K1", "PIK3CA"]


def fetch_all_mutations_post(mut_profile_id: str) -> pd.DataFrame:
    """用 POST /mutations/fetch 拉全部 mutation records (比逐 gene 更可靠)。"""
    url = f"{BASE}/molecular-profiles/{mut_profile_id}/mutations/fetch"
    body = {
        "entrezGeneIds": [],  # empty = all genes
        "hugoGeneSymbols": GENES,
        "uniqueSampleKeys": [],
        "projection": "DETAILED",
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(3):
        r = requests.post(url, json=body, headers=headers, timeout=60)
        print(f"  POST status={r.status_code} len={len(r.text) if r.text else 0}")
        if r.status_code == 200:
            return pd.DataFrame(r.json())
        if r.status_code == 400:
            # 试不带 genes filter
            body2 = {"hugoGeneSymbols": GENES, "projection": "DETAILED"}
            r2 = requests.post(url, json=body2, headers=headers, timeout=60)
            print(f"  retry simpler body: status={r2.status_code}")
            if r2.status_code == 200:
                return pd.DataFrame(r2.json())
        print(f"  [RETRY {attempt+1}] sleep 2s...")
        time.sleep(2)
    return pd.DataFrame()


def process():
    all_muts = []
    for cohort_name, study_id, mut_profile_id in STUDIES:
        print(f"\n=== {cohort_name} ===")
        mut_df = fetch_all_mutations_post(mut_profile_id)
        print(f"  拿到 {len(mut_df)} mutation records")
        if not mut_df.empty:
            # 看看列名
            print(f"  columns: {list(mut_df.columns[:15])}")
            all_muts.append(mut_df)

    if not all_muts:
        print("\n[WARN] POST fetch 失败, 用 GET + 不同参数试...")
        return None

    df = pd.concat(all_muts, ignore_index=True)
    # 提取 case_id 和 gene
    if "patientId" in df.columns and "hugoGeneSymbol" in df.columns:
        df = df[["patientId", "hugoGeneSymbol", "mutationType", "variantAllele", "proteinChange"]]
        df.columns = ["case_id", "gene", "mutation_type", "variant", "protein_change"]
    elif "sampleId" in df.columns:
        df = df.rename(columns={"sampleId": "case_id"})[["case_id", "hugoGeneSymbol", "mutationType"]]
        df.columns = ["case_id", "gene", "mutation_type"]

    print(f"\n合计: {len(df)} mutations, {df['case_id'].nunique()} cases")

    # 宽表
    wide = df.groupby(["case_id", "gene"]).size().unstack(fill_value=0)
    wide = (wide > 0).astype(int)
    print(f"宽表 shape: {wide.shape}")
    print(f"突变基因: {list(wide.columns)}")

    out = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_mutations_wide.csv"
    wide.to_csv(out)
    print(f"\n[SAVED] {out}")
    return wide


if __name__ == "__main__":
    process()
