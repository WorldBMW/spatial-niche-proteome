"""直接看 lusc_proteomics.csv 的原始前 5 行，看真实表头结构"""
import pandas as pd

OMICS = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics"

# header=None 读前 5 行，看原始结构
lusc_raw = pd.read_csv(f"{OMICS}/lusc_proteomics.csv", header=None, nrows=5, encoding="latin-1", on_bad_lines="skip")
print("=== lusc_proteomics 原始前 5 行 ===")
for i in range(5):
    print(f"  row {i}: {lusc_raw.iloc[i, :8].tolist()}")

# luad 也看一下
luad_raw = pd.read_csv(f"{OMICS}/luad_proteomics.csv", header=None, nrows=5, encoding="latin-1", on_bad_lines="skip")
print("\n=== luad_proteomics 原始前 5 行 ===")
for i in range(5):
    print(f"  row {i}: {luad_raw.iloc[i, :8].tolist()}")
