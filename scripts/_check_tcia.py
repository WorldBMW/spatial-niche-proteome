import requests, pandas as pd
TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
for col in ["CPTAC-LUAD", "CPTAC-LSCC"]:
    r = requests.get(f"{TCIA_BASE}/getSeries", params={"Collection": col, "format": "JSON"}, timeout=60)
    series = r.json()
    mods = {}
    for s in series:
        m = s.get("Modality", "?")
        mods[m] = mods.get(m, 0) + 1
    print(f"{col}: {len(series)} series, modalities={mods}")
    if series:
        print(f"  keys: {list(series[0].keys())[:20]}")
        pid = series[0].get("PatientID", "?")
        suid = series[0].get("SeriesInstanceUID", "?")
        print(f"  样例 PatientID={pid} SeriesUID={suid}")
