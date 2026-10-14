"""关键数字汇总（严格版 r2 重跑结果）。
计算 5 组 + baseline 的显著性统计、交集/独有基因名单、k_scan、OS Cox，
并合并富集 (enrichment_summary_strict.json) 与 DrugBank (drugbank_summary_strict.json) 结果，
输出 papers/20261014_JBI_submission/KEY_NUMBERS.json + KEY_NUMBERS.txt
可重复运行：富集/DrugBank 未完成时对应字段为 null，完成后重跑即可填充。
"""
import json
import os
import numpy as np
import pandas as pd

BASE = "results_vitl14_jbi_r2"
OUT = "papers/20261014_JBI_submission"
os.makedirs(OUT, exist_ok=True)

FILES = {
    "ALL_prot": f"{BASE}/dn_ALL_prot2000/ALL_proteomics_results.csv",
    "LUAD": f"{BASE}/dn_LUAD_prot2000/LUAD_proteomics_results.csv",
    "LUSC": f"{BASE}/dn_LUSC_prot2000/LUSC_proteomics_results.csv",
    "phospho": f"{BASE}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv",
    "rna": f"{BASE}/dn_ALL_rna500/ALL_transcriptomics_results.csv",
    "baseline_globalpool": f"{BASE}/dn_baseline_globalpool/ALL_proteomics_results.csv",
}

def load(path):
    return pd.read_csv(path, encoding="latin-1")

def sig_df(df):
    return df[(df["q_value"] < 0.05) & df["pearson"].notna()]

def gene_sym(target):
    return str(target).split("__")[0]

# ---------- 1. 每组统计 ----------
groups = {}
data = {}
for name, path in FILES.items():
    df = load(path)
    data[name] = df
    s = sig_df(df)
    strong = s[s["pearson"].abs() > 0.3]
    groups[name] = {
        "total_targets": int(len(df)),
        "n_valid_q_nonNaN": int(df["q_value"].notna().sum()),
        "n_tested_pearson_nonNaN": int(df["pearson"].notna().sum()),
        "sig_q05": int(len(s)),
        "sig_strong_abs_r_gt_0.3": int(len(strong)),
    }

# ---------- 2. ALL_prot 96 显著蛋白 n_valid 分布 ----------
s96 = sig_df(data["ALL_prot"]).copy()
nv = s96["n_valid"]
nvalid_stats = {
    "n_sig": int(len(s96)),
    "n_valid_min": float(nv.min()),
    "n_valid_median": float(nv.median()),
    "n_valid_max": float(nv.max()),
    "n_valid_ge100": int((nv >= 100).sum()),
    "n_valid_ge50": int((nv >= 50).sum()),
    "n_valid_lt50": int((nv < 50).sum()),
}

# ---------- 3. 每组 top10 显著（按 |pearson|） ----------
top10 = {}
for name in FILES:
    s = sig_df(data[name]).copy()
    s = s.reindex(s["pearson"].abs().sort_values(ascending=False).index).head(10)
    top10[name] = [
        {"target": r["target"], "gene": gene_sym(r["target"]),
         "pearson": round(float(r["pearson"]), 4),
         "q_value": float(r["q_value"]), "n_valid": int(r["n_valid"])}
        for _, r in s.iterrows()
    ]

# ---------- 4. LUAD ∩ LUSC ----------
sl = set(sig_df(data["LUAD"])["target"])
ss = set(sig_df(data["LUSC"])["target"])
luad_lusc = {
    "LUAD_sig": len(sl), "LUSC_sig": len(ss),
    "intersection": len(sl & ss), "LUAD_only": len(sl - ss), "LUSC_only": len(ss - sl),
    "intersection_genes": sorted(sl & ss),
    "LUAD_only_genes": sorted(sl - ss),
    "LUSC_only_genes": sorted(ss - sl),
}

# ---------- 5. 生态位 vs baseline ----------
sc = set(sig_df(data["ALL_prot"])["target"])
sb = set(sig_df(data["baseline_globalpool"])["target"])
inter, niche_uniq, base_only = sc & sb, sc - sb, sb - sc
niche_baseline = {
    "niche_sig": len(sc), "baseline_sig": len(sb),
    "intersection": len(inter), "niche_unique": len(niche_uniq),
    "baseline_only": len(base_only),
    "intersection_targets": sorted(inter),
    "niche_unique_targets": sorted(niche_uniq),
    "baseline_only_targets": sorted(base_only),
    "GLB1L3_in_niche_unique": "GLB1L3" in niche_uniq,
    "CISD3_in_niche_unique": "CISD3" in niche_uniq,
    "CISD3_note": ("CISD3 不在 ALL pan-tumor 2000 靶点列表中，因此不属于生态位 96 显著集合；"
                   "它仅在 LUSC 亚组中显著 (pearson=-0.99, q=0.0)。"),
}

# ---------- 6. k_scan ----------
ks = pd.read_csv("results_vitl14_jbi/k_scan.csv", encoding="latin-1")
k_scan = [
    {"k": int(r["k"]),
     "inertia": float(r["inertia"]),
     "silhouette": float(r["silhouette"]),
     "calinski_harabasz": float(r["calinski_harabasz"])}
    for _, r in ks.iterrows()
]

# ---------- 7. OS Cox ----------
cox = pd.read_csv(f"{BASE}/dn_ALL_rna500/ALL_os_cox.csv", encoding="latin-1")
os_cox = {
    "analysis": str(cox.loc[0, "analysis"]),
    "n": int(cox.loc[0, "n"]),
    "cindex_mean": float(cox.loc[0, "cindex_mean"]),
    "cindex_std": float(cox.loc[0, "cindex_std"]),
    "event_rate": float(cox.loc[0, "event_rate"]),
}

# ---------- 8. 合并富集 / DrugBank 摘要（若已完成） ----------
enrichment = None
enr_path = f"{OUT}/supplementary/enrichment_summary_strict.json"
if os.path.exists(enr_path):
    with open(enr_path, encoding="utf-8") as f:
        enrichment = json.load(f)

drugbank = None
db_path = f"{OUT}/supplementary/drugbank_summary_strict.json"
if os.path.exists(db_path):
    with open(db_path, encoding="utf-8") as f:
        drugbank = json.load(f)

key_numbers = {
    "description": ("CPTAC LUAD+LUSC DINOv2 ViT-L 形态特征-蛋白/磷酸化/RNA 关联，"
                    "fold 内预处理严格版 (results_vitl14_jbi_r2)。显著定义: q_value<0.05 且 pearson 非 NaN。"),
    "groups": groups,
    "ALL_prot_sig_n_valid_distribution": nvalid_stats,
    "top10_sig_per_group_by_abs_pearson": top10,
    "LUAD_LUSC_overlap": luad_lusc,
    "niche_vs_baseline_globalpool": niche_baseline,
    "k_scan": k_scan,
    "os_cox": os_cox,
    "enrichment_ALL_prot_sig96": enrichment,
    "drugbank_annotation_ALL_prot_sig96": drugbank,
}

with open(f"{OUT}/KEY_NUMBERS.json", "w", encoding="utf-8") as f:
    json.dump(key_numbers, f, ensure_ascii=False, indent=2)

# ---------- txt 版本 ----------
lines = []
lines.append("=" * 70)
lines.append("KEY NUMBERS — strict rerun (results_vitl14_jbi_r2)")
lines.append("显著定义: q_value < 0.05 且 pearson 非 NaN")
lines.append("=" * 70)
lines.append("")
lines.append("[每组统计]  (total / q非NaN / tested / sig / sig且|r|>0.3)")
for name, g in groups.items():
    lines.append(f"  {name:22s} total={g['total_targets']:4d}  q非NaN={g['n_valid_q_nonNaN']:4d}  "
                 f"tested={g['n_tested_pearson_nonNaN']:4d}  sig={g['sig_q05']:4d}  "
                 f"|r|>0.3={g['sig_strong_abs_r_gt_0.3']:4d}")
lines.append("")
lines.append("[ALL_prot 96 显著蛋白 n_valid 分布]")
for k, v in nvalid_stats.items():
    lines.append(f"  {k}: {v}")
lines.append("")
lines.append("[每组 top10 显著 (按 |pearson| 降序)]")
for name, items in top10.items():
    lines.append(f"  --- {name} ---")
    for it in items:
        lines.append(f"    {it['target']:15s} pearson={it['pearson']:+.4f}  q={it['q_value']:.3e}  n_valid={it['n_valid']}")
lines.append("")
lines.append("[LUAD vs LUSC]")
lines.append(f"  LUAD sig={luad_lusc['LUAD_sig']}, LUSC sig={luad_lusc['LUSC_sig']}, "
             f"交集={luad_lusc['intersection']}, LUAD独有={luad_lusc['LUAD_only']}, LUSC独有={luad_lusc['LUSC_only']}")
lines.append("")
lines.append("[生态位 vs 全局池化 baseline]")
lines.append(f"  生态位 sig={niche_baseline['niche_sig']}, baseline sig={niche_baseline['baseline_sig']}, "
             f"交集={niche_baseline['intersection']}, 生态位独有={niche_baseline['niche_unique']}, "
             f"baseline独有={niche_baseline['baseline_only']}")
lines.append(f"  GLB1L3 在生态位独有集合: {niche_baseline['GLB1L3_in_niche_unique']}")
lines.append(f"  CISD3 在生态位独有集合: {niche_baseline['CISD3_in_niche_unique']}")
lines.append(f"  注: {niche_baseline['CISD3_note']}")
lines.append(f"  交集名单 ({len(inter)}): {', '.join(sorted(inter))}")
lines.append(f"  生态位独有名单 ({len(niche_uniq)}): {', '.join(sorted(niche_uniq))}")
lines.append("")
lines.append("[k_scan (KMeans k=5-15, CPTAC pan-tumor patch embeddings)]")
lines.append("  k   inertia        silhouette   CH")
for row in k_scan:
    lines.append(f"  {row['k']:2d}  {row['inertia']:.4e}  {row['silhouette']:.4f}  {row['calinski_harabasz']:.1f}")
lines.append("")
lines.append("[OS Cox]")
lines.append(f"  {os_cox['analysis']}: n={os_cox['n']}, C-index={os_cox['cindex_mean']:.4f} ± {os_cox['cindex_std']:.4f}, "
             f"event_rate={os_cox['event_rate']:.4f}")
lines.append("")
lines.append("[富集分析 (96 显著蛋白, 背景=2000 靶点)]")
if enrichment:
    for lib, info in enrichment.items():
        lines.append(f"  {lib}: sig(adjP<0.05) 通路数 = {info['n_sig_pathways']}")
        for t in info["sig_terms"]:
            lines.append(f"    - {t}")
else:
    lines.append("  (尚未完成 — 运行 _run_enrichment_v2.py 后重跑本脚本)")
lines.append("")
lines.append("[DrugBank 可成药注释 (96 显著蛋白)]")
if drugbank:
    lines.append(f"  可成药基因数: {drugbank['n_druggable']}")
    lines.append(f"  基因: {', '.join(drugbank['druggable_genes'])}")
else:
    lines.append("  (尚未完成 — 运行 _drugbank_annot_v2.py 后重跑本脚本)")
lines.append("")

with open(f"{OUT}/KEY_NUMBERS.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"KEY_NUMBERS.json / .txt 已写入 {OUT}")
print(json.dumps({k: key_numbers[k] for k in ["groups", "ALL_prot_sig_n_valid_distribution"]},
                 ensure_ascii=False, indent=1))
