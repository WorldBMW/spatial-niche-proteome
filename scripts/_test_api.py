import requests, time

print("=" * 50)
print("连通性测试")
print("=" * 50)

# 1. cBioPortal
try:
    r = requests.get("https://www.cbioportal.org/api/studies/luad_tcga_pan_can_atlas_2018", timeout=15)
    print(f"[cBioPortal] status={r.status_code}  ", end="")
    if r.status_code == 200:
        d = r.json()
        print(f"name={d.get('name','?')[:50]}")
    else:
        print(r.text[:100])
except Exception as e:
    print(f"[cBioPortal] FAIL: {e}")

# 2. GDC
try:
    r = requests.get("https://api.gdc.cancer.gov/projects", timeout=15,
                     headers={"Content-Type": "application/json"})
    print(f"[GDC API] status={r.status_code}")
except Exception as e:
    print(f"[GDC API] FAIL: {e}")

# 3. cBioPortal molecular-profiles (突变数据 availability)
for study in ["luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"]:
    try:
        r = requests.get(f"https://www.cbioportal.org/api/studies/{study}/molecular-profiles", timeout=15)
        if r.status_code == 200:
            profiles = r.json()
            mut_profiles = [p for p in profiles if "mutation" in p.get("molecularAlterationType","").lower()]
            print(f"[{study}] profiles={len(profiles)}  mutation_profiles={len(mut_profiles)}")
            for mp in mut_profiles:
                print(f"    -> {mp.get('molecularProfileId')}")
        else:
            print(f"[{study}] status={r.status_code}")
    except Exception as e:
        print(f"[{study}] FAIL: {e}")

# 4. 试拉 LUAD 1 个患者的 EGFR 突变
print("\n--- 试拉 EGFR mutations (LUAD) ---")
try:
    r = requests.get(
        "https://www.cbioportal.org/api/molecular-profiles/luad_tcga_pan_can_atlas_2018_mutations/mutations",
        params={"hugoGeneSymbol": "EGFR", "projection": "DETAILED"},
        timeout=15
    )
    print(f"  status={r.status_code}  ", end="")
    if r.status_code == 200:
        muts = r.json()
        print(f"count={len(muts)}")
        if muts:
            print(f"  样例: patientId={muts[0].get('patientId')}, gene={muts[0].get('hugoGeneSymbol')}, type={muts[0].get('mutationType')}")
    else:
        print(r.text[:200])
except Exception as e:
    print(f"  FAIL: {e}")
