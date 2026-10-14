"""核实: 1) GLB1L3 在 ALL vs LUAD 的 r 是否真的相同; 2) GLB1L3 是否 niche-exclusive; 3) top hit 榜单."""
import pandas as pd

B = "results_vitl14_jbi_r2"
ENC = "latin-1"

allr = pd.read_csv(f"{B}/dn_ALL_prot2000/ALL_proteomics_results.csv", encoding=ENC)
luad = pd.read_csv(f"{B}/dn_LUAD_prot2000/LUAD_proteomics_results.csv", encoding=ENC)
lusc = pd.read_csv(f"{B}/dn_LUSC_prot2000/LUSC_proteomics_results.csv", encoding=ENC)
bs = pd.read_csv(f"{B}/dn_baseline_globalpool/ALL_proteomics_results.csv", encoding=ENC)

# 1. GLB1L3 行对比
for name, df in [("ALL", allr), ("LUAD", luad), ("LUSC", lusc), ("baseline", bs)]:
    row = df[df["target"].str.startswith("GLB1L3")]
    if len(row):
        r = row.iloc[0]
        print(f"{name:9s} GLB1L3: pearson={r['pearson']:.4f} q={r['q_value']:.3e} n_valid={r['n_valid']}")
    else:
        print(f"{name:9s} GLB1L3: NOT in target list")

# 2. niche-exclusive 集合的 top hits (按 |pearson| 排序前 10)
allr_sig = allr[allr["q_value"] < 0.05]
bs_sig = bs[bs["q_value"] < 0.05]
eco_s, bs_s = set(allr_sig["target"]), set(bs_sig["target"])
exclusive = allr_sig[allr_sig["target"].isin(eco_s - bs_s)].copy()
exclusive = exclusive.reindex(exclusive["pearson"].abs().sort_values(ascending=False).index)
print("\nniche-exclusive top 10 (by |r|):")
for _, r in exclusive.head(10).iterrows():
    print(f"  {r['target']:15s} r={r['pearson']:+.3f} q={r['q_value']:.2e} n={r['n_valid']}")

# 3. ALL/LUAD/LUSC 各自 top 5 (按 |pearson|, q<0.05)
for name, df in [("ALL", allr), ("LUAD", luad), ("LUSC", lusc)]:
    sig = df[df["q_value"] < 0.05].copy()
    sig = sig.reindex(sig["pearson"].abs().sort_values(ascending=False).index)
    print(f"\n{name} top 5:")
    for _, r in sig.head(5).iterrows():
        print(f"  {r['target']:15s} r={r['pearson']:+.3f} q={r['q_value']:.2e} n={r['n_valid']}")