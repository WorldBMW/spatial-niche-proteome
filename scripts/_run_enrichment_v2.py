"""GO/KEGG/Reactome 富集分析 — 严格版 r2。
输入: results_vitl14_jbi_r2/dn_ALL_prot2000/ALL_proteomics_results.csv
  - 显著基因: q_value<0.05 且 pearson 非 NaN（96 个，target 去 "__N" 后缀去重）
  - 背景集:   该文件全部 2000 靶点（q 非 NaN），同样去后缀去重
输出: papers/20261014_JBI_submission/supplementary/
  - enrichment_strict.xlsx（3 个库各一个 sheet）
  - enrichment_summary_strict.json（显著通路数与名单，供 KEY_NUMBERS 合并）
"""
import json
import os
import re
import sys
import pandas as pd
import gseapy as gp

BASE = "results_vitl14_jbi_r2/dn_ALL_prot2000/ALL_proteomics_results.csv"
OUT = "papers/20261014_JBI_submission/supplementary"
os.makedirs(OUT, exist_ok=True)

SUFFIX = re.compile(r"__\d+$")

def to_gene(t):
    return SUFFIX.sub("", str(t))

df = pd.read_csv(BASE, encoding="latin-1")
bg_df = df[df["q_value"].notna()]
background = sorted({to_gene(t) for t in bg_df["target"]})

sig = df[(df["q_value"] < 0.05) & df["pearson"].notna()]
genes = sorted({to_gene(t) for t in sig["target"]})

print(f"significant gene list: {len(genes)} (from {len(sig)} targets)")
print(f"background gene list:  {len(background)} (from {len(bg_df)} targets)")
sys.stdout.flush()

LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
all_results, summary = {}, {}
os.makedirs("logs", exist_ok=True)

for lib in LIBS:
    try:
        enr = gp.enrichr(gene_list=genes,
                         gene_sets=lib,
                         organism="human",
                         background=background,
                         outdir=None,
                         cutoff=0.25,
                         no_plot=True,
                         verbose=True)
        res = enr.results.copy()
        # gseapy 1.3.1: 自定义 background 时无 Overlap 列，从 Genes 列计数
        if "Overlap" in res.columns:
            res["n_genes"] = res["Overlap"].str.split("/").str[0].astype(int)
        else:
            res["n_genes"] = (res["Genes"].astype(str).str.rstrip(";")
                              .str.split(";").map(len))
        res = res.sort_values("Adjusted P-value").reset_index(drop=True)
        all_results[lib] = res
        sigres = res[res["Adjusted P-value"] < 0.05]
        summary[lib] = {
            "n_sig_pathways": int(len(sigres)),
            "sig_terms": sigres["Term"].tolist(),
            "n_genes_input": len(genes),
            "n_background": len(background),
        }
        print(f"\n=== {lib}: {len(sigres)} sig pathways (adjP<0.05), "
              f"{len(res)} terms reported (adjP<0.25) ===")
        for _, r in sigres.head(10).iterrows():
            print(f"  {r['Term'][:75]:<75} adjP={r['Adjusted P-value']:.2e} genes={r['Overlap']}")
        sys.stdout.flush()
    except Exception as e:
        print(f"{lib} FAILED: {type(e).__name__}: {e}")
        sys.stdout.flush()

if not all_results:
    raise SystemExit("所有富集库均失败，请检查网络/Enrichr 服务。")

# ---- xlsx 输出 ----
short = {"GO_Biological_Process_2023": "GO_BP", "KEGG_2021_Human": "KEGG", "Reactome_2022": "Reactome"}
xlsx = f"{OUT}/enrichment_strict.xlsx"
with pd.ExcelWriter(xlsx) as writer:
    for lib, res in all_results.items():
        keep = ["Term", "Overlap", "n_genes", "P-value", "Adjusted P-value",
                "Old P-value", "Old Adjusted P-value", "Old adjusted P-value",
                "Odds Ratio", "Combined Score", "Genes"]
        keep = [c for c in keep if c in res.columns]
        # 去重保持顺序（大小写变体兼容 gseapy 1.3.1）
        seen = set(); ordered = []
        for c in keep:
            if c not in seen:
                seen.add(c); ordered.append(c)
        res[ordered].to_excel(writer, sheet_name=short[lib][:31], index=False)
print(f"\nSaved -> {xlsx}")

with open(f"{OUT}/enrichment_summary_strict.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Saved -> {OUT}/enrichment_summary_strict.json")

# 基因清单落盘（便于审稿复核）
with open(f"{OUT}/enrichment_input_genes_strict.txt", "w") as f:
    f.write("\n".join(genes) + "\n")
with open(f"{OUT}/enrichment_background_genes_strict.txt", "w") as f:
    f.write("\n".join(background) + "\n")
print("Saved gene list + background list txt")
