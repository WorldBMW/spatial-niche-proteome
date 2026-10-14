"""GO/KEGG 通路富集分析"""
import gseapy as gp
import pandas as pd

genes = []
with open("results_vitl14_jbi/sig_genes.txt") as f:
    for line in f:
        g = line.strip()
        if g and not g.startswith("AC") and not g.startswith("AL") and not g.startswith("AP0"):
            genes.append(g)

print(f"Input genes (filtered lncRNA): {len(genes)}")

libs = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
all_results = {}

for lib in libs:
    try:
        enr = gp.enrichr(gene_list=genes,
                         gene_sets=lib,
                         organism='human',
                         outdir=None,
                         cutoff=0.05)
        res = enr.results
        sig = res[res["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
        all_results[lib] = sig
        print(f"\n=== {lib}: {len(sig)} sig pathways ===")
        for _, r in sig.head(10).iterrows():
            print(f"  {r['Term'][:70]:<70}  adjP={r['Adjusted P-value']:.2e}  genes={r['Overlap']}")
    except Exception as e:
        print(f"{lib} failed: {e}")

# 保存
with pd.ExcelWriter("results_vitl14_jbi/enrichment.xlsx") as writer:
    for lib, df in all_results.items():
        short = lib.split("_")[0]
        df.to_excel(writer, sheet_name=short[:30], index=False)
print("\nSaved → results_vitl14_jbi/enrichment.xlsx")
