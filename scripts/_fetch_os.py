import requests, pandas as pd, numpy as np, time
BASE = "https://www.cbioportal.org/api"

for cohort, sid in [("TCGA-LUAD", "luad_tcga_pan_can_atlas_2018"),
                    ("TCGA-LUSC", "lusc_tcga_pan_can_atlas_2018")]:
    print(f"\n=== {cohort} ===")
    # 试 clinical-patient-data endpoint
    r = requests.get(f"{BASE}/studies/{sid}/clinical-patient-data",
                     params={"projection": "DETAILED"}, timeout=30)
    print(f"  clinical-patient-data: status={r.status_code}", end="")
    if r.status_code == 200:
        d = r.json()
        print(f"  records={len(d)}")
        if d:
            print(f"  样例 keys: {list(d[0].keys())[:15]}")
            # 提取 OS_MONTHS / OS_STATUS
            rows = []
            for rec in d:
                pid = rec.get("patientId", "")
                os_m = rec.get("OS_MONTHS", rec.get("osMonths"))
                os_s = rec.get("OS_STATUS", rec.get("osStatus"))
                if os_m is not None or os_s is not None:
                    rows.append({"case_id": pid, "OS_MONTHS": os_m, "OS_STATUS": os_s})
            print(f"  有 OS 数据的: {len(rows)}")
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(f"e:/ai_paper/experiment2_fm_time/data/raw/TCGA/{cohort}_os.csv", index=False)
                print(f"  → saved {cohort}_os.csv")
                print(f"  OS_STATUS 分布: {df['OS_STATUS'].value_counts(dropna=False).to_dict()}")
                print(f"  OS_MONTHS 非空: {df['OS_MONTHS'].notna().sum()}")
    else:
        print(f"  {r.text[:150]}")
    time.sleep(1)
