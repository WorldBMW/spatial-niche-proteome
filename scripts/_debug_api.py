import requests, json
BASE = "https://www.cbioportal.org/api"

# 1. 试 GET /studies/{sid}/mutations
r = requests.get(f"{BASE}/studies/luad_tcga_pan_can_atlas_2018/mutations", timeout=15)
print(f"[1] GET /studies/luad/mutations: status={r.status_code} body={r.text[:300]}")
print()

# 2. 试 profile mutations + SUMMARY
r2 = requests.get(
    f"{BASE}/molecular-profiles/luad_tcga_pan_can_atlas_2018_mutations/mutations",
    params={"projection": "SUMMARY", "pageSize": 5}, timeout=15)
print(f"[2] profile mutations SUMMARY: status={r2.status_code} body={r2.text[:300]}")
print()

# 3. sample lists
r3 = requests.get(f"{BASE}/studies/luad_tcga_pan_can_atlas_2018/sample-lists", timeout=15)
print(f"[3] sample lists: status={r3.status_code}")
if r3.status_code == 200:
    sls = r3.json()
    print(f"  count={len(sls)}")
    for sl in sls[:5]:
        sid = sl.get("sampleListId", "?")
        name = sl.get("name", "?")[:50]
        cnt = sl.get("sampleCount", "?")
        print(f"    {sid} | {name} | {cnt}")
print()

# 4. 用 sampleListId + gene 试
if r3.status_code == 200:
    all_sl = [s for s in r3.json() if "ALL" in s.get("sampleListId", "").upper()][0]
    sl_id = all_sl.get("sampleListId")
    r4 = requests.get(
        f"{BASE}/molecular-profiles/luad_tcga_pan_can_atlas_2018_mutations/mutations",
        params={"sampleListId": sl_id, "hugoGeneSymbol": "EGFR", "projection": "DETAILED"},
        timeout=15)
    print(f"[4] EGFR with sampleListId={sl_id}: status={r4.status_code}")
    if r4.status_code == 200:
        d = r4.json()
        print(f"  records={len(d)}")
        if d:
            m = d[0]
            print(f"  样例: patientId={m.get('patientId')}, gene={m.get('hugoGeneSymbol')}, type={m.get('mutationType')}")
    else:
        print(f"  body={r4.text[:200]}")
