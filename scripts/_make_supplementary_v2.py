"""补充表生成 — 严格版 r2。
输出: papers/20261014_JBI_submission/supplementary/
  - TableS2_sig_proteomics_ALL.csv   (96 显著蛋白，全列，含 n_valid)
  - TableS2b_sig_proteomics_LUAD.csv (63)
  - TableS2c_sig_proteomics_LUSC.csv (69)
  - TableS2d_sig_phospho.csv         (4)
  - TableS2e_sig_rna.csv             (39)
  - TableS4_kscan.csv
  - TableS5_baseline_comparison.csv  (生态位 vs 全局池化)
(TableS1 druggable 见 _drugbank_annot_v2.py；enrichment_strict.xlsx 见 _run_enrichment_v2.py)
"""
import os
import re
import pandas as pd

R2 = "results_vitl14_jbi_r2"
OUT = "papers/20261014_JBI_submission/supplementary"
os.makedirs(OUT, exist_ok=True)

SUFFIX = re.compile(r"__\d+$")

def read_csv(p):
    return pd.read_csv(p, encoding="latin-1")

def sig_table(path, out_name):
    df = read_csv(path)
    s = df[(df["q_value"] < 0.05) & df["pearson"].notna()].copy()
    s = s.sort_values("q_value").reset_index(drop=True)
    # 全列输出（含 n_valid，审稿人要求标注有效样本数）
    s.to_csv(f"{OUT}/{out_name}", index=False, encoding="utf-8-sig")
    print(f"{out_name}: {len(s)} rows")
    return s

FILES = [
    (f"{R2}/dn_ALL_prot2000/ALL_proteomics_results.csv", "TableS2_sig_proteomics_ALL.csv"),
    (f"{R2}/dn_LUAD_prot2000/LUAD_proteomics_results.csv", "TableS2b_sig_proteomics_LUAD.csv"),
    (f"{R2}/dn_LUSC_prot2000/LUSC_proteomics_results.csv", "TableS2c_sig_proteomics_LUSC.csv"),
    (f"{R2}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv", "TableS2d_sig_phospho.csv"),
    (f"{R2}/dn_ALL_rna500/ALL_transcriptomics_results.csv", "TableS2e_sig_rna.csv"),
]
for path, name in FILES:
    sig_table(path, name)

# ---- TableS4: k_scan 格式化 ----
ks = read_csv("results_vitl14_jbi/k_scan.csv")
ks_out = pd.DataFrame({
    "k": ks["k"].astype(int),
    "inertia": ks["inertia"].map(lambda x: f"{x:.4e}"),
    "silhouette": ks["silhouette"].round(4),
    "calinski_harabasz": ks["calinski_harabasz"].round(1),
})
ks_out["selected"] = ["yes" if k == 10 else "" for k in ks_out["k"]]
ks_out.to_csv(f"{OUT}/TableS4_kscan.csv", index=False, encoding="utf-8-sig")
print(f"TableS4_kscan.csv: {len(ks_out)} rows")

# ---- TableS5: 生态位 vs 全局池化 baseline ----
prot = read_csv(f"{R2}/dn_ALL_prot2000/ALL_proteomics_results.csv")
bl = read_csv(f"{R2}/dn_baseline_globalpool/ALL_proteomics_results.csv")
sc = set(prot[(prot["q_value"] < 0.05) & prot["pearson"].notna()]["target"])
sb = set(bl[(bl["q_value"] < 0.05) & bl["pearson"].notna()]["target"])
inter = sorted(sc & sb)
niche_uniq = sorted(sc - sb)
base_only = sorted(sb - sc)

rows = [
    {"comparison_item": "spatial-niche features (ALL, pan-tumor)", "n": len(sc),
     "targets": ";".join(sorted(sc))},
    {"comparison_item": "global-average-pooling baseline (ALL, pan-tumor)", "n": len(sb),
     "targets": ";".join(sorted(sb))},
    {"comparison_item": "intersection (both significant)", "n": len(inter),
     "targets": ";".join(inter)},
    {"comparison_item": "niche-unique significant", "n": len(niche_uniq),
     "targets": ";".join(niche_uniq)},
    {"comparison_item": "baseline-only significant", "n": len(base_only),
     "targets": ";".join(base_only)},
    {"comparison_item": "note: GLB1L3 in niche-unique set", "n": int("GLB1L3" in niche_uniq),
     "targets": "GLB1L3" if "GLB1L3" in niche_uniq else ""},
    {"comparison_item": "note: CISD3 in niche-unique set", "n": int("CISD3" in niche_uniq),
     "targets": "CISD3 not in ALL 2000-target list; significant in LUSC only (pearson=-0.99, q=0.0)"},
]
t5 = pd.DataFrame(rows, columns=["comparison_item", "n", "targets"])
t5.to_csv(f"{OUT}/TableS5_baseline_comparison.csv", index=False, encoding="utf-8-sig")
print(f"TableS5_baseline_comparison.csv: {len(t5)} rows "
      f"(sig {len(sc)} vs {len(sb)}, inter {len(inter)}, niche-unique {len(niche_uniq)}, baseline-only {len(base_only)})")
