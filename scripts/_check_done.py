import pandas as pd
import numpy as np

runs = {
    "LUAD_prot": "results_vitl14_jbi/dn_LUAD_prot2000/LUAD_proteomics_results.csv",
    "ALL_phospho": "results_vitl14_jbi/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv",
}

print(f"{'实验':<14} {'总数':>6} {'有效':>6} {'q<0.05':>8} {'|r|>0.3':>8} {'显著率':>8}  Top hit")
print("-" * 95)
for name, path in runs.items():
    df = pd.read_csv(path)
    n = len(df)
    valid = int(df["pearson"].notna().sum())
    sig = int(((df["q_value"] < 0.05) & (df["pearson"].notna())).sum())
    strong = int(((df["q_value"] < 0.05) & (df["pearson"].abs() > 0.3) & (df["pearson"].notna())).sum())
    if sig > 0:
        sig_df = df[(df["q_value"] < 0.05) & (df["pearson"].notna())]
        top = sig_df.reindex(sig_df["pearson"].abs().sort_values(ascending=False).index).iloc[0]
        tname = str(top["target"])[:40].ljust(40)
        top_str = f"{tname} r={top['pearson']:+.3f} q={top['q_value']:.2e}"
    else:
        top_str = "(none)"
    rate = sig / valid * 100 if valid else 0
    print(f"{name:<14} {n:>6} {valid:>6} {sig:>8} {strong:>8} {rate:>7.1f}%  {top_str}")
