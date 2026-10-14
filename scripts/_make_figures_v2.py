"""投稿图表重生成 — 严格版 r2（300 dpi, PNG+PDF）。
输出: papers/20261014_JBI_submission/figures/
  - fig1_workflow.png（复制原图）
  - fig2_volcano_all_prot.png/pdf
  - fig3_sig_counts.png/pdf
  - fig4_kscan.png/pdf
  - fig5_os_cox.png/pdf
  - FIGURE_DATA_MAP.txt
fig6 (frozen) 本次跳过：frozen 严格版数据未就绪。
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R2 = "results_vitl14_jbi_r2"
KS = "results_vitl14_jbi/k_scan.csv"
OUT = "papers/20261014_JBI_submission/figures"
shutil.os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def read_csv(p):
    return pd.read_csv(p, encoding="latin-1")

prot = read_csv(f"{R2}/dn_ALL_prot2000/ALL_proteomics_results.csv")
luad = read_csv(f"{R2}/dn_LUAD_prot2000/LUAD_proteomics_results.csv")
lusc = read_csv(f"{R2}/dn_LUSC_prot2000/LUSC_proteomics_results.csv")
phos = read_csv(f"{R2}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv")
rna = read_csv(f"{R2}/dn_ALL_rna500/ALL_transcriptomics_results.csv")
ks = read_csv(KS)
cox = read_csv(f"{R2}/dn_ALL_rna500/ALL_os_cox.csv")

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"saved {name}.png/.pdf")

# ---------- fig1: 复制原图 ----------
shutil.copy("papers/figures/fig1_workflow.png", f"{OUT}/fig1_workflow.png")
print("copied fig1_workflow.png")

# ---------- fig2: 火山图 (ALL proteomics, 严格版) ----------
d = prot.copy()
d["neglogq"] = -np.log10(d["q_value"].clip(lower=1e-300))
d = d[d["pearson"].notna()]
sig = (d["q_value"] < 0.05)
strong = sig & (d["pearson"].abs() > 0.3)
n_sig = int(sig.sum()); n_strong = int(strong.sum())

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(d.loc[~sig, "pearson"], d.loc[~sig, "neglogq"], s=8, c="#bbbbbb", alpha=0.5, label="n.s.")
ax.scatter(d.loc[sig & ~strong, "pearson"], d.loc[sig & ~strong, "neglogq"],
           s=14, c="#3b82f6", alpha=0.85, label=f"q<0.05 (n={n_sig - n_strong})")
ax.scatter(d.loc[strong, "pearson"], d.loc[strong, "neglogq"],
           s=16, c="#dc2626", alpha=0.9, label=f"q<0.05 & |r|>0.3 (n={n_strong})")
ax.axhline(-np.log10(0.05), ls="--", c="#333", lw=0.8)
ax.axvline(0, ls="-", c="#ccc", lw=0.6)
top = d.loc[sig].reindex(d.loc[sig, "pearson"].abs().sort_values(ascending=False).index).head(8)
for _, r in top.iterrows():
    ax.annotate(str(r["target"]).split("__")[0], (r["pearson"], r["neglogq"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points", color="#111")
ax.set_xlabel("Cross-validated Pearson r  (morphology -> protein abundance)")
ax.set_ylabel("-log10(BH-adjusted q)")
ax.set_title(f"Figure 2. Morphology-proteome associations (pan-tumor, {len(d)} proteins tested)")
ax.legend(fontsize=7, loc="upper left", frameon=False)
save(fig, "fig2_volcano_all_prot")

# ---------- fig3: 各模态显著数 ----------
def stat(df):
    v = df[df["pearson"].notna()]
    s = v[v["q_value"] < 0.05]
    return len(s), int((s["pearson"].abs() > 0.3).sum())

rows = []
for name, df in [("Proteomics\n(pan)", prot), ("Proteomics\nLUAD", luad),
                 ("Proteomics\nLUSC", lusc), ("Phospho-\nproteomics", phos),
                 ("Transcript-\nomics", rna)]:
    rows.append((name,) + stat(df))
dfb = pd.DataFrame(rows, columns=["name", "sig", "strong"])

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(dfb)); w = 0.38
ax.bar(x - w/2, dfb["sig"], w, color="#3b82f6", label="q<0.05")
ax.bar(x + w/2, dfb["strong"], w, color="#dc2626", label="q<0.05 & |r|>0.3")
for i, (s, st) in enumerate(zip(dfb["sig"], dfb["strong"])):
    ax.text(i - w/2, s + 3, str(s), ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, st + 3, str(st), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(dfb["name"], fontsize=8)
ax.set_ylabel("Number of significant associations")
ax.set_ylim(0, dfb["sig"].max() * 1.18)
ax.set_title("Figure 3. Significant morphology-omics associations by modality (strict rerun)")
ax.legend(fontsize=8, frameon=False)
save(fig, "fig3_sig_counts")

# ---------- fig4: KMeans k 扫描 ----------
k = ks["k"].values
sil = ks["silhouette"].values
ch = ks["calinski_harabasz"].values
k10 = int((k == 10).nonzero()[0][0])

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
l1, = ax1.plot(k, sil, "o-", color="#3b82f6", lw=1.8, ms=6, label="Silhouette")
l2, = ax2.plot(k, ch, "s--", color="#dc2626", lw=1.8, ms=6, label="Calinski-Harabasz")
ax1.axvline(10, ls=":", c="#555", lw=1.2)
ax1.annotate("selected k=10", xy=(10, sil[k10]), xytext=(10.4, sil[k10] + 0.04),
             fontsize=9, color="#333",
             arrowprops=dict(arrowstyle="->", color="#555", lw=0.9))
ax1.set_xlabel("Number of KMeans clusters (k)")
ax1.set_ylabel("Silhouette coefficient", color="#3b82f6")
ax2.set_ylabel("Calinski-Harabasz index", color="#dc2626")
ax1.tick_params(axis="y", labelcolor="#3b82f6")
ax2.tick_params(axis="y", labelcolor="#dc2626")
ax1.set_xticks(k.astype(int))
ax1.set_title("Figure 4. KMeans cluster-number scan on DINOv2 patch embeddings")
ax1.legend(handles=[l1, l2], fontsize=8, frameon=False, loc="center right")
save(fig, "fig4_kscan")

# ---------- fig5: OS Cox ----------
c = float(cox.loc[0, "cindex_mean"]); sd = float(cox.loc[0, "cindex_std"])
n = int(cox.loc[0, "n"]); er = float(cox.loc[0, "event_rate"])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.barh([0], [c], xerr=[[sd]], color="#f59e0b", height=0.35, capsize=6)
ax.axvline(0.5, ls="--", c="#333", lw=1, label="Random (0.50)")
ax.axvline(0.62, ls=":", c="#dc2626", lw=1.2, label="Clinical utility (0.62)")
ax.set_xlim(0, 1)
ax.set_yticks([0]); ax.set_yticklabels(["Morphology-only\nCox model"], fontsize=9)
ax.set_xlabel("Concordance index (C-index)")
ax.text(c + 0.02, 0.22, f"C-index = {c:.3f} $\\pm$ {sd:.3f}\n(n = {n}, event rate = {er:.1%})",
        fontsize=9, ha="left", va="bottom")
ax.set_title("Figure 5. Prognostic value of morphology features for overall survival")
ax.legend(fontsize=8, frameon=False, loc="upper left")
save(fig, "fig5_os_cox")

# ---------- FIGURE_DATA_MAP.txt ----------
map_txt = "\n".join([
    "FIGURE_DATA_MAP (2026-10-14, strict rerun results_vitl14_jbi_r2)",
    "每张图的数据来源：",
    "",
    "fig1_workflow.png",
    "  <- 复制自 papers/figures/fig1_workflow.png（上一版工作流示意图，内容与本次流程一致，未重做）",
    "",
    "fig2_volcano_all_prot.png / .pdf",
    "  <- results_vitl14_jbi_r2/dn_ALL_prot2000/ALL_proteomics_results.csv",
    "     (x=pearson, y=-log10(q_value)；灰=不显著，蓝=q<0.05，红=q<0.05 且 |r|>0.3；标注按 |pearson| 排序的 top8 基因)",
    "",
    "fig3_sig_counts.png / .pdf",
    "  <- results_vitl14_jbi_r2/dn_ALL_prot2000/ALL_proteomics_results.csv",
    "  <- results_vitl14_jbi_r2/dn_LUAD_prot2000/LUAD_proteomics_results.csv",
    "  <- results_vitl14_jbi_r2/dn_LUSC_prot2000/LUSC_proteomics_results.csv",
    "  <- results_vitl14_jbi_r2/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv",
    "  <- results_vitl14_jbi_r2/dn_ALL_rna500/ALL_transcriptomics_results.csv",
    "     (蓝柱=q<0.05 显著数，红柱=其中 |pearson|>0.3 的数目)",
    "",
    "fig4_kscan.png / .pdf",
    "  <- results_vitl14_jbi/k_scan.csv",
    "     (左轴 silhouette，右轴 Calinski-Harabasz，k=5..15，虚线标注 k=10)",
    "",
    "fig5_os_cox.png / .pdf",
    "  <- results_vitl14_jbi_r2/dn_ALL_rna500/ALL_os_cox.csv",
    "     (C-index = 0.4876 ± 0.0589, n=194, event rate=23.7%；参考线 0.50=随机, 0.62=临床效用阈值)",
    "",
    "fig6_frozen（本次跳过：frozen 严格版数据未就绪）",
])
with open(f"{OUT}/FIGURE_DATA_MAP.txt", "w", encoding="utf-8") as f:
    f.write(map_txt + "\n")
print(f"saved FIGURE_DATA_MAP.txt -> {OUT}")
