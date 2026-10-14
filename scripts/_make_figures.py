"""
Generate JBI manuscript figures from real results.
Outputs to papers/figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "results_vitl14_jbi"
OUT = "papers/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ---------- load ----------
prot = pd.read_csv(f"{R}/dn_ALL_prot2000/ALL_proteomics_results.csv")
luad = pd.read_csv(f"{R}/dn_LUAD_prot2000/LUAD_proteomics_results.csv")
lusc = pd.read_csv(f"{R}/dn_LUSC_prot2000/LUSC_proteomics_results.csv")
phos = pd.read_csv(f"{R}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv")
rna = pd.read_csv(f"{R}/dn_ALL_rna500/ALL_transcriptomics_results.csv")
frozen = pd.read_csv(f"{R}/dn_frozen_prot2000/ALL_proteomics_results.csv")
enr = pd.ExcelFile(f"{R}/enrichment.xlsx")

# ============================================================
# Figure 1 — workflow schematic
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")
boxes = [
    (0.02, 0.62, "CPTAC-LUAD\n(95 WSI)", "#cfe2f3"),
    (0.02, 0.15, "CPTAC-LUSC\n(108 WSI)", "#cfe2f3"),
    (0.20, 0.42, "DINOv2 ViT-L\npatch embeddings", "#d9ead3"),
    (0.40, 0.42, "KMeans (k=10)\nniche decomposition", "#d9ead3"),
    (0.60, 0.42, "230 spatial-ecology\nfeatures / WSI", "#ffe599"),
    (0.80, 0.42, "ElasticNet nested CV\n(5-fold x 3 seeds)", "#f4cccc"),
    (0.80, 0.15, "BH-FDR (q<0.05)\nsignificant hits", "#e6c1d9"),
]
for x, y, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 0.17, 0.28, boxstyle="round,pad=0.01",
                                linewidth=1, edgecolor="#444", facecolor=c))
    ax.text(x + 0.085, y + 0.14, t, ha="center", va="center", fontsize=8)
arrows = [(0.19, 0.70, 0.20, 0.62), (0.19, 0.28, 0.20, 0.36),
          (0.37, 0.56, 0.40, 0.56), (0.57, 0.56, 0.60, 0.56),
          (0.77, 0.56, 0.80, 0.56), (0.885, 0.42, 0.885, 0.43)]
for x1, y1, x2, y2 in arrows:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color="#555", lw=1.2))
ax.add_patch(FancyArrowPatch((0.885, 0.30), (0.885, 0.15), arrowstyle="-|>",
                             mutation_scale=12, color="#555", lw=1.2, linestyle="--"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Figure 1. Study design and analytical workflow", fontweight="bold", pad=8)
plt.savefig(f"{OUT}/fig1_workflow.png"); plt.close()

# ============================================================
# Figure 2 — volcano plot (ALL proteomics)
# ============================================================
d = prot.copy()
d["neglogq"] = -np.log10(d["q_value"].clip(lower=1e-300))
d = d[d["pearson"].notna()]
sig = d["q_value"] < 0.05
strong = (d["q_value"] < 0.05) & (d["pearson"].abs() > 0.3)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(d.loc[~sig, "pearson"], d.loc[~sig, "neglogq"], s=8, c="#bbbbbb", alpha=0.5, label="n.s.")
ax.scatter(d.loc[sig & ~strong, "pearson"], d.loc[sig & ~strong, "neglogq"], s=14, c="#3b82f6", alpha=0.85, label="q<0.05")
ax.scatter(d.loc[strong, "pearson"], d.loc[strong, "neglogq"], s=16, c="#dc2626", alpha=0.9, label="q<0.05 & |r|>0.3")
ax.axhline(-np.log10(0.05), ls="--", c="#333", lw=0.8)
ax.axvline(0, ls="-", c="#ccc", lw=0.6)
# label top hits
top = d.loc[sig].reindex(d.loc[sig, "pearson"].abs().sort_values(ascending=False).index).head(6)
for _, r in top.iterrows():
    ax.annotate(str(r["target"]).split("__")[0], (r["pearson"], r["neglogq"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points", color="#111")
ax.set_xlabel("Cross-validated Pearson r  (morphology -> protein abundance)")
ax.set_ylabel("-log10(BH-adjusted q)")
ax.set_title("Figure 2. Morphology-proteome associations (pan-tumor, %d proteins tested)" % len(d))
ax.legend(fontsize=7, loc="upper left", frameon=False)
plt.savefig(f"{OUT}/fig2_volcano_all_prot.png"); plt.close()

# ============================================================
# Figure 3 — significant hits bar chart by modality
# ============================================================
rows = []
def stat(name, df):
    v = df[df["pearson"].notna()]; s = v[v["q_value"] < 0.05]
    strong = s[s["pearson"].abs() > 0.3]
    return (name, len(v), len(s), len(strong))
for name, df in [("Proteomics\n(pan)", prot), ("Proteomics\nLUAD", luad),
                 ("Proteomics\nLUSC", lusc), ("Phospho-\nproteomics", phos),
                 ("Transcript-\nomics", rna)]:
    rows.append(stat(name, df))
dfb = pd.DataFrame(rows, columns=["name", "valid", "sig", "strong"])
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(dfb)); w = 0.38
ax.bar(x - w/2, dfb["sig"], w, color="#3b82f6", label="q<0.05")
ax.bar(x + w/2, dfb["strong"], w, color="#dc2626", label="q<0.05 & |r|>0.3")
for i, (s, st) in enumerate(zip(dfb["sig"], dfb["strong"])):
    ax.text(i - w/2, s + 3, str(s), ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, st + 3, str(st), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(dfb["name"], fontsize=8)
ax.set_ylabel("Number of significant associations")
ax.set_title("Figure 3. Significant morphology-omics associations by modality")
ax.legend(fontsize=8, frameon=False)
plt.savefig(f"{OUT}/fig3_sig_counts.png"); plt.close()

# ============================================================
# Figure 4 — enrichment dot plot
# ============================================================
df_go = enr.parse("GO"); df_k = enr.parse("KEGG"); df_r = enr.parse("Reactome")
df_go = df_go.assign(src="GO:BP"); df_k = df_k.assign(src="KEGG"); df_r = df_r.assign(src="Reactome")
comb = pd.concat([df_go, df_k, df_r], ignore_index=True)
comb = comb.sort_values("Adjusted P-value")
comb = comb.head(18)
comb["neglog10p"] = -np.log10(comb["Adjusted P-value"].clip(lower=1e-300))
comb["n_overlap"] = comb["Overlap"].str.split("/").str[0].astype(int)
comb = comb.sort_values("neglog10p")
src_colors = {"GO:BP": "#34d399", "KEGG": "#3b82f6", "Reactome": "#a855f7"}
fig, ax = plt.subplots(figsize=(9, 7))
for src in ["GO:BP", "KEGG", "Reactome"]:
    sub = comb[comb["src"] == src]
    ax.scatter(sub["neglog10p"], sub["Term"], s=sub["n_overlap"]*3.2,
               color=src_colors[src], alpha=0.82, label=src, edgecolor="#333", linewidth=0.4)
ax.set_xlabel("-log10(adjusted P-value)")
ax.set_ylabel("")
import matplotlib.patches as mp
handles = [mp.Patch(color=c, label=l) for l, c in src_colors.items()]
ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
ax.axvline(-np.log10(0.05), ls="--", c="#888", lw=0.8)
ax.set_title("Figure 4. Enriched pathways of significant proteomic hits")
plt.savefig(f"{OUT}/fig4_enrichment.png"); plt.close()

# ============================================================
# Figure 5 — OS Cox result (negative) c-index summary
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
c = 0.4856; sd = 0.0572
ax.barh([0], [c], xerr=[[sd]], color="#f59e0b", height=0.35, capsize=6)
ax.axvline(0.5, ls="--", c="#333", lw=1, label="Random (0.50)")
ax.axvline(0.62, ls=":", c="#dc2626", lw=1.2, label="Clinical utility (0.62)")
ax.set_xlim(0, 1)
ax.set_yticks([0]); ax.set_yticklabels(["Morphology-only\nCox model"], fontsize=9)
ax.set_xlabel("Concordance index (C-index)")
ax.set_title("Figure 5. Prognostic value of morphology features for overall survival")
ax.legend(fontsize=8, frameon=False)
plt.savefig(f"{OUT}/fig5_os_cox.png"); plt.close()

# ============================================================
# Figure 6 — frozen KMeans reproducibility
# ============================================================
c_sig = set(prot[(prot["q_value"] < 0.05) & prot["pearson"].notna()]["target"])
f_sig = set(frozen[(frozen["q_value"] < 0.05) & frozen["pearson"].notna()]["target"])
overlap = c_sig & f_sig
fig, ax = plt.subplots(figsize=(6.5, 4.8))
cats = ["CPTAC-fitted\nKMeans", "Shared\n(reproducible)", "TCGA-frozen\nKMeans"]
vals = [len(c_sig) - len(overlap), len(overlap), len(f_sig) - len(overlap)]
colors = ["#3b82f6", "#22c55e", "#a855f7"]
ax.bar(cats, vals, color=colors, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
rep_rate = len(overlap)/len(c_sig)*100
ax.set_ylabel("Number of significant proteins")
ax.set_title(f"Figure 6. Reproducibility across clustering models\n"
             f"(overlap rate = {rep_rate:.1f}%)")
plt.savefig(f"{OUT}/fig6_frozen_repro.png"); plt.close()

print("All figures written to", OUT)
import glob
for f in sorted(glob.glob(f"{OUT}/*.png")):
    print("  ", f)