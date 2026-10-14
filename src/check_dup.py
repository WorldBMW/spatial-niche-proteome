import csv

def check_dup_cols(fp):
    with open(fp, encoding="latin-1") as f:
        reader = csv.reader(f)
        header = next(reader)
    seen = {}
    dups = []
    for i, h in enumerate(header):
        if h in seen:
            dups.append((seen[h], i, h))
        else:
            seen[h] = i
    return len(header), len(dups), dups[:5]

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"
for name in ["luad_proteomics", "lusc_proteomics", 
             "luad_phosphoproteomics", "lusc_phosphoproteomics",
             "luad_transcriptomics", "lusc_transcriptomics"]:
    fp = f"{OMICS}/{name}.csv"
    try:
        n, ndup, dups = check_dup_cols(fp)
        print(f"{name}: {n} 列, {ndup} 重复 → {dups[:3]}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
