import pandas as pd
import numpy as np

base = "results_vitl14_jbi"

def summarize(path, name):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[{name}] READ ERROR: {e}")
        return
    print(f"\n=== {name} ===")
    print(f"  columns: {list(df.columns)}")
    print(f"  rows: {len(df)}")
    # 找 q_value / p_value / pearson / target 列
    qcol = None
    pcol = None
    rcol = None
    tcol = None
    for c in df.columns:
        cl = c.lower()
        if 'q' in cl and ('val' in cl or 'q' == cl or 'fdr' in cl): qcol = c
        if cl in ('p_value', 'pvalue', 'p_val', 'p'): pcol = c
        if 'pearson' in cl or cl in ('r', 'corr', 'rho'): rcol = c
        if cl in ('target', 'gene', 'protein', 'probe', 'feature'): tcol = c
    print(f"  detected: qcol={qcol}, pcol={pcol}, rcol={rcol}, tcol={tcol}")
    if tcol is None:
        # 打印前几行看结构
        print(df.head(3).to_string())
        return
    # 显著计数
    if qcol and rcol:
        sub = df[df[rcol].notna()]
        sig = df[(df[qcol] < 0.05) & df[rcol].notna()]
        strong = sig[sig[rcol].abs() > 0.3]
        print(f"  valid pearson: {len(sub)}")
        print(f"  sig q<0.05: {len(sig)}")
        print(f"  sig & |r|>0.3: {len(strong)}")
    if pcol and rcol:
        sigp = df[(df[pcol] < 0.05) & df[rcol].notna()]
        print(f"  sig p<0.05: {len(sigp)}")
    # top hits
    if rcol:
        top = df[df[rcol].notna()].reindex(df[df[rcol].notna()][rcol].abs().sort_values(ascending=False).index).head(10)
        print("  top 10 by |r|:")
        for _, row in top.iterrows():
            qs = f"{row[qcol]:.2e}" if qcol else ""
            print(f"    {str(row[tcol])[:30]:<30} r={row[rcol]:+.3f} q={qs}")

summarize(f"{base}/dn_ALL_prot2000/ALL_proteomics_results.csv", "ALL proteomics (2000)")
summarize(f"{base}/dn_LUAD_prot2000/LUAD_proteomics_results.csv", "LUAD proteomics (2000)")
summarize(f"{base}/dn_LUSC_prot2000/LUSC_proteomics_results.csv", "LUSC proteomics (2000)")
summarize(f"{base}/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv", "ALL phospho (500)")
summarize(f"{base}/dn_ALL_rna500/ALL_transcriptomics_results.csv", "ALL RNA (500)")
summarize(f"{base}/dn_frozen_prot2000/ALL_proteomics_results.csv", "FROZEN proteomics (2000)")