"""提取所有显著 hits 的基因名 → 用于 DrugBank + GO/KEGG"""
import pandas as pd
import numpy as np
import re

runs = {
    "ALL_prot":    "results_vitl14_jbi/dn_ALL_prot2000/ALL_proteomics_results.csv",
    "LUAD_prot":   "results_vitl14_jbi/dn_LUAD_prot2000/LUAD_proteomics_results.csv",
    "LUSC_prot":   "results_vitl14_jbi/dn_LUSC_prot2000/LUSC_proteomics_results.csv",
    "ALL_phospho": "results_vitl14_jbi/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv",
    "ALL_rna":     "results_vitl14_jbi/dn_ALL_rna500/ALL_transcriptomics_results.csv",
}

all_genes = set()
for name, path in runs.items():
    df = pd.read_csv(path)
    sig = df[(df["q_value"] < 0.05) & (df["pearson"].notna())]
    for t in sig["target"]:
        # proteomics: "GENE__2" → "GENE"; phospho: "GENE|S123|..." → "GENE"; RNA: "GENE"
        gene = re.split(r'[|_]', str(t))[0]
        # 去掉 __2 后缀
        gene = re.sub(r'__\d+$', '', gene)
        all_genes.add(gene)

print(f"Total significant genes (unique): {len(all_genes)}")
print(f"Sample: {sorted(list(all_genes))[:30]}")

# 保存
with open("results_vitl14_jbi/sig_genes.txt", "w") as f:
    for g in sorted(all_genes):
        f.write(g + "\n")
print(f"\nSaved → results_vitl14_jbi/sig_genes.txt")
