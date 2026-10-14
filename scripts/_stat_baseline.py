"""生态位特征 vs 全局池化 baseline 对比 + 完整九项。"""
import pandas as pd, numpy as np

B = "results_vitl14_jbi_r2"
ENC = "latin-1"

def count_sig(path, qcol="q_value", rcol="pearson"):
    df = pd.read_csv(path, encoding=ENC)
    df = df[df[qcol].notna()]
    sig = df[df[qcol] < 0.05]
    n_r = len(sig[sig[rcol].abs() > 0.3]) if rcol in sig.columns else len(sig)
    return len(df), len(sig), n_r

print("=" * 70)
print("生态位特征 (fold 内预处理) vs 全局池化 baseline")
print("-" * 70)
for name, ecol in [("意味特征 ALL_prot", "dn_ALL_prot2000/ALL_proteomics_results.csv"),
                   ("baseline 全局池化", "dn_baseline_globalpool/ALL_proteomics_results.csv")]:
    n_tot, n_sig, n_r = count_sig(f"{B}/{ecol}" if name.startswith("意味") else f"{B}/{ecol}")
    print(f"{name:20s} total={n_tot:4d} sig={n_sig:4d} sig|r|>0.3={n_r}")

# 交集: baseline vs 生态位 显著蛋白
eco = pd.read_csv(f"{B}/dn_ALL_prot2000/ALL_proteomics_results.csv", encoding=ENC)
eco = eco[eco['q_value'].notna() & (eco['q_value'] < 0.05)]
bs = pd.read_csv(f"{B}/dn_baseline_globalpool/ALL_proteomics_results.csv", encoding=ENC)
bs = bs[bs['q_value'].notna() & (bs['q_value'] < 0.05)]
eco_s = set(eco['target']); bs_s = set(bs['target'])
inter = eco_s & bs_s
print("-" * 70)
print(f"生态位 sig={len(eco_s)}, baseline sig={len(bs_s)}, 交集={len(inter)}")
print(f"生态位独有={len(eco_s-bs_s)}, baseline独有={len(bs_s-eco_s)}")
print("=" * 70)