import os
import pandas as pd
import numpy as np

R = "results_vitl14_jbi"
OUT = "papers/supplementary"
os.makedirs(OUT, exist_ok=True)

# ---- Table S1: druggable targets (36) ----
targets = open(f"{R}/druggable_targets.txt").read().split()
# 从 ALL proteomics 结果中取这些靶点的统计量
prot = pd.read_csv(f"{R}/dn_ALL_prot2000/ALL_proteomics_results.csv")
prot["gene"] = prot["target"].str.split("__").str[0]
sub = prot[prot["gene"].isin(targets)].copy()
# merge order
order = pd.DataFrame({"gene": targets})
s1 = order.merge(prot[["gene","target","pearson","q_value","n_valid"]], on="gene", how="left")
s1.columns = ["GeneSymbol", "Protein_ID", "Pearson_r", "q_value", "n_valid"]
s1["Pearson_r"] = s1["Pearson_r"].round(3)
s1["q_value"] = s1["q_value"].map(lambda x: f"{x:.2e}")
s1.to_csv(f"{OUT}/TableS1_druggable_targets.csv", index=False)

# ---- Table S2: full significant lists per modality ----
def sig_table(path, name):
    df = pd.read_csv(path)
    s = df[(df["q_value"] < 0.05) & df["pearson"].notna()].copy()
    s = s.sort_values("q_value")
    s = s[["target", "pearson", "q_value", "n_valid", "r2", "mae"]]
    s.columns = ["Target", "Pearson_r_CV", "q_value", "n_valid", "R2_CV", "MAE"]
    s["Pearson_r_CV"] = s["Pearson_r_CV"].round(3)
    s["q_value"] = s["q_value"].map(lambda x: f"{x:.2e}")
    s["R2_CV"] = s["R2_CV"].round(3)
    s["MAE"] = s["MAE"].round(3)
    s.to_csv(f"{OUT}/TableS2_{name}_significant.csv", index=False)
    print(f"{name}: {len(s)} rows")

sig_table(f"{R}/dn_ALL_prot2000/ALL_proteomics_results.csv", "proteomics_pan")
sig_table(f"{R}/dn_LUAD_prot2000/LUAD_proteomics_results.csv", "proteomics_LUAD")
sig_table(f"{R}/dn_LUSC_prot2000/LUSC_proteomics_results.csv", "proteomics_LUSC")
sig_table(f"{R}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv", "phospho_pan")
sig_table(f"{R}/dn_ALL_rna500/ALL_transcriptomics_results.csv", "rna_pan")

# ---- Table S3: enrichment (copy clean) ----
import shutil
shutil.copy(f"{R}/enrichment.xlsx", f"{OUT}/TableS3_enrichment.xlsx")
print("S3 enrichment copied")
print("done ->", OUT)