"""重跑全部实验 (fold 内残差化 + fold 内标准化版本), 结果写入 results_vitl14_jbi_r2/"""
import subprocess, sys, time, os

FEAT = "results_vitl14_jbi/all_features_omics_aligned.csv"
BASE = "results_vitl14_jbi_r2"
PY = sys.executable

jobs = [
    ("ALL proteomics 2000", [PY, "src/downstream_omics.py", "--features", FEAT,
     "--cohort", "ALL", "--modalities", "proteomics", "--targets-top-k", "2000",
     "--seeds", "3", "--no-permutation", "--out-dir", f"{BASE}/dn_ALL_prot2000"]),
    ("LUAD proteomics 2000", [PY, "src/downstream_omics.py", "--features", FEAT,
     "--cohort", "LUAD", "--modalities", "proteomics", "--targets-top-k", "2000",
     "--seeds", "3", "--no-permutation", "--out-dir", f"{BASE}/dn_LUAD_prot2000"]),
    ("LUSC proteomics 2000", [PY, "src/downstream_omics.py", "--features", FEAT,
     "--cohort", "LUSC", "--modalities", "proteomics", "--targets-top-k", "2000",
     "--seeds", "3", "--no-permutation", "--out-dir", f"{BASE}/dn_LUSC_prot2000"]),
    ("ALL phospho 500", [PY, "src/downstream_omics.py", "--features", FEAT,
     "--cohort", "ALL", "--modalities", "phosphoproteomics", "--targets-top-k", "500",
     "--seeds", "3", "--no-permutation", "--out-dir", f"{BASE}/dn_ALL_phospho500"]),
    ("ALL rna 500", [PY, "src/downstream_omics.py", "--features", FEAT,
     "--cohort", "ALL", "--modalities", "transcriptomics", "--targets-top-k", "500",
     "--seeds", "3", "--no-permutation", "--out-dir", f"{BASE}/dn_ALL_rna500"]),
]

for name, cmd in jobs:
    t0 = time.time()
    print(f"\n===== {name} =====", flush=True)
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f"[FAIL] {name} rc={r.returncode}", flush=True)
    else:
        print(f"[OK] {name}  {(time.time()-t0)/60:.1f} min", flush=True)

# OS Cox (fold 内 PCA)
print("\n===== OS Cox =====", flush=True)
r = subprocess.run([PY, "scripts/_run_os_only.py"], capture_output=False)
print(f"[OS] rc={r.returncode}", flush=True)
print("\nALL DONE", flush=True)