"""统计已产出部分结果: LUAD proteomics 显著数 / k-scan / OS."""
import pandas as pd, numpy as np, os

B = "results_vitl14_jbi_r2"
print("=" * 60)

# 1. LUAD proteomics (fold 内预处理版)
luad = pd.read_csv(f"{B}/dn_LUAD_prot2000/LUAD_proteomics_results.csv", encoding="latin-1")
print("LUAD columns:", [c for c in luad.columns if "q" in c.lower() or "r_" in c.lower() or c.lower() in ("target","pearson_mean","n_valid")][:12])
qcol = [c for c in luad.columns if "q" in c.lower() and "fdr" not in c.lower()]
rcol = [c for c in luad.columns if "pearson_mean" in c]
print("qcol:", qcol, "rcol:", rcol)
tot = len(luad)
valid = luad.dropna(subset=rcol)
sig = luad[(luad[qcol[0]] < 0.05) if qcol else (pd.Series(False, index=luad.index))] if qcol else luad
# 更稳健: 直接筛
if qcol:
    sig = luad[luad[qcol[0]] < 0.05]
    sig_strong = sig[sig[rcol[0]].abs() > 0.3] if rcol else sig
else:
    sig = sig_strong = pd.DataFrame()
print(f"[LUAD_prot] total={tot} valid_r={len(valid)} sig(q<0.05)={len(sig)} sig|r|>0.3={len(sig_strong)}")

# 2. k-scan
ks = pd.read_csv("results_vitl14_jbi/k_scan.csv")
print("\n[k-scan]")
print(ks.to_string(index=False))

# 3. OS
osr = pd.read_csv(f"{B}/dn_ALL_rna500/ALL_os_cox.csv")
print("\n[OS]")
print(osr.to_string(index=False))
print("=" * 60)