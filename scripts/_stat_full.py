"""fold 内预处理版完整统计: 5组显著数 + OS + k=10依据(简)。"""
import pandas as pd, numpy as np

B = "results_vitl14_jbi_r2"
ENC = "latin-1"

def count_sig(path, qcol="q_value", rcol="pearson"):
    df = pd.read_csv(path, encoding=ENC)
    df = df[df[qcol].notna()]
    n_tot = len(df)
    sig = df[df[qcol] < 0.05]
    n_sig = len(sig)
    n_r = len(sig[sig[rcol].abs() > 0.3]) if rcol in sig.columns else len(sig)
    top = df.reindex(df[rcol].abs().sort_values(ascending=False).index).iloc[0] if len(df) else None
    top_txt = f"top={top['target']} r={top[rcol]:.3f}" if top is not None else "none"
    return n_tot, n_sig, n_r, top_txt

print("=" * 70)
for name, f in [("ALL_prot","dn_ALL_prot2000/ALL_proteomics_results.csv"),
                ("LUAD_prot","dn_LUAD_prot2000/LUAD_proteomics_results.csv"),
                ("LUSC_prot","dn_LUSC_prot2000/LUSC_proteomics_results.csv"),
                ("ALL_phospho","dn_ALL_phospho500/ALL_phosphoproteomics_results.csv"),
                ("ALL_rna","dn_ALL_rna500/ALL_transcriptomics_results.csv")]:
    try:
        n_tot, n_sig, n_r, top = count_sig(f"{B}/{f}")
        print(f"{name:12s} total={n_tot:5d} sig(q<0.05)={n_sig:4d}  sig|r|>0.3={n_r:4d}  {top}")
    except Exception as e:
        print(f"{name:12s} ERROR {e}")

osr = pd.read_csv(f"{B}/dn_ALL_rna500/ALL_os_cox.csv", encoding=ENC)
print("OS:", osr.to_dict("records"))

# phospho/RNA 单独显著列名可能不同
for name, f, qc in [("ALL_phospho(c2)","dn_ALL_phospho500/ALL_phosphoproteomics_results.csv",None)]:
    df = pd.read_csv(f"{B}/{f}", encoding=ENC, nrows=3)
    print(name, "cols:", [c for c in df.columns])
print("=" * 70)