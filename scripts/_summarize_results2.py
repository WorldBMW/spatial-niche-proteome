import pandas as pd
import numpy as np

base = "results_vitl14_jbi"

def summarize(path, name):
    df = pd.read_csv(path)
    print(f"\n=== {name} ===")
    print(f"  rows={len(df)}")
    # q_value 分布
    q = df["q_value"]
    print(f"  q_value<0.05: {(q<0.05).sum()}, q<0.01: {(q<0.01).sum()}, q==1.0: {(q==1.0).sum()}")
    print(f"  q_value min/median: {q.min():.3g} / {q.median():.3g}")
    # 显著子集
    sig = df[(df["q_value"] < 0.05) & df["pearson"].notna()]
    valid = df[df["pearson"].notna()]
    strong = sig[sig["pearson"].abs() > 0.3]
    print(f"  valid pearson: {len(valid)}")
    print(f"  sig q<0.05: {len(sig)}")
    print(f"  sig & |pearson|>0.3: {len(strong)}")
    # 显著 top hits（按 |pearson|）
    top = sig.reindex(sig["pearson"].abs().sort_values(ascending=False).index).head(15)
    print("  TOP 15 significant hits (by |pearson|):")
    for _, row in top.iterrows():
        n = row["n_valid"]
        print(f"    {str(row['target'])[:34]:<34} r={row['pearson']:+.3f} q={row['q_value']:.2e} n={n}")

for p, n in [
    ("dn_ALL_prot2000/ALL_proteomics_results.csv", "ALL proteomics (2000)"),
    ("dn_LUAD_prot2000/LUAD_proteomics_results.csv", "LUAD proteomics (2000)"),
    ("dn_LUSC_prot2000/LUSC_proteomics_results.csv", "LUSC proteomics (2000)"),
    ("dn_ALL_phospho500/ALL_phosphoproteomics_results.csv", "ALL phospho (500)"),
    ("dn_ALL_rna500/ALL_transcriptomics_results.csv", "ALL RNA (500)"),
    ("dn_frozen_prot2000/ALL_proteomics_results.csv", "FROZEN proteomics (2000)"),
]:
    summarize(f"{base}/{p}", n)