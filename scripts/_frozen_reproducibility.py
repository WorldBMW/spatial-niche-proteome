import pandas as pd
import numpy as np

frozen = pd.read_csv("results_vitl14_jbi/dn_frozen_prot2000/ALL_proteomics_results.csv")
cptac = pd.read_csv("results_vitl14_jbi/dn_ALL_prot2000/ALL_proteomics_results.csv")

f_sig = set(frozen[(frozen["q_value"] < 0.05) & frozen["pearson"].notna()]["target"])
c_sig = set(cptac[(cptac["q_value"] < 0.05) & cptac["pearson"].notna()]["target"])
overlap = f_sig & c_sig
print(f"CPTAC self-fit sig: {len(c_sig)}")
print(f"Frozen (TCGA centroids) sig: {len(f_sig)}")
print(f"Overlap: {len(overlap)}")
print(f"Reproducibility rate = {len(overlap)/len(c_sig)*100:.1f}%")
print()
print(f"CPTAC valid: {cptac['pearson'].notna().sum()}, frozen valid: {frozen['pearson'].notna().sum()}")
print()
print("Frozen top 5 by |r|:")
top = frozen[(frozen["q_value"] < 0.05) & frozen["pearson"].notna()]
top = top.reindex(top["pearson"].abs().sort_values(ascending=False).index).head(5)
for _, r in top.iterrows():
    tname = str(r["target"])[:35].ljust(35)
    print(f"  {tname} r={r['pearson']:+.3f} q={r['q_value']:.2e}")
