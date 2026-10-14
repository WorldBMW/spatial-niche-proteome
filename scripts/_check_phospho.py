import pandas as pd
import numpy as np

df = pd.read_csv("results_vitl14_jbi/dn_ALL_phospho500/ALL_phosphoproteomics_results.csv")
valid = df["pearson"].notna()
sig = df[valid & (df["q_value"] < 0.05)]
strong = sig[sig["pearson"].abs() > 0.3]
print(f"phospho500: total={len(df)}, valid={valid.sum()}, q<0.05={len(sig)}, |r|>0.3={len(strong)}")
print(f"显著率: {len(sig)/valid.sum()*100:.1f}%")
print()
print("Top 10 by |r|:")
top = sig.reindex(sig["pearson"].abs().sort_values(ascending=False).index).head(10)
for _, r in top.iterrows():
    tname = str(r["target"])[:45].ljust(45)
    print(f"  {tname} r={r['pearson']:+.3f}  q={r['q_value']:.2e}  n={int(r['n_valid'])}")
