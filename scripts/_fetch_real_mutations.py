import requests, json, time, pandas as pd, numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE = "https://www.cbioportal.org/api"

# Gene → entrezGeneId (NCBI)
GENES = {
    "EGFR": 1956, "KRAS": 3845, "TP53": 7157, "ALK": 238, "RET": 5979,
    "ROS1": 6098, "BRAF": 673, "STK11": 6794, "KEAP1": 9817,
    "MET": 4233, "ARAF": 367, "MAP2K1": 5604, "PIK3CA": 5290,
}

STUDIES = [
    ("TCGA-LUAD", "luad_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018_mutations", "luad_tcga_pan_can_atlas_2018_all"),
    ("TCGA-LUSC", "lusc_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018_mutations", "lusc_tcga_pan_can_atlas_2018_all"),
]


def fetch_mutations(cohort, study_id, mut_profile_id, sample_list_id):
    all_rows = []
    for gene, entrez_id in GENES.items():
        r = requests.get(
            f"{BASE}/molecular-profiles/{mut_profile_id}/mutations",
            params={
                "sampleListId": sample_list_id,
                "entrezGeneId": entrez_id,
                "projection": "DETAILED",
            },
            timeout=15,
        )
        if r.status_code == 200:
            for m in r.json():
                all_rows.append({
                    "case_id": m.get("patientId", ""),
                    "gene": gene,
                    "mutation_type": m.get("mutationType", ""),
                    "protein_change": m.get("proteinChange", ""),
                    "variant": m.get("variantAllele", ""),
                    "cohort": cohort,
                })
        else:
            print(f"  [{cohort}] {gene}: {r.status_code} {r.text[:80]}")
        time.sleep(0.3)
    return all_rows


def main():
    all_data = []
    for cohort, sid, mid, slid in STUDIES:
        print(f"\n=== {cohort} ===")
        rows = fetch_mutations(cohort, sid, mid, slid)
        print(f"  拿到 {len(rows)} mutation records")
        all_data.extend(rows)

    df = pd.DataFrame(all_data)
    if df.empty:
        print("[FATAL] 没拿到任何数据")
        return

    print(f"\n合计 {len(df)} mutations, {df['case_id'].nunique()} cases")
    print(df.groupby(["cohort", "gene"]).size().unstack(fill_value=0).to_string())

    # 宽表
    wide = df.groupby(["case_id", "gene"]).size().unstack(fill_value=0)
    wide = (wide > 0).astype(int)
    wide.columns = [f"{c}_mutation" for c in wide.columns]
    wide = wide.reset_index()
    print(f"\n宽表 shape: {wide.shape}")

    out = r"e:\ai_paper\experiment2_fm_time\data\raw\TCGA\tcga_mutations_wide.csv"
    wide.to_csv(out, index=False)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
