"""快速测试 pandas 读 lusc_prot 的正确方式"""
import pandas as pd
import csv, time

FP = r"f:\ai_paper\experiment2_fm_time\data\raw\CPTAC\omics\lusc_proteomics.csv"

# 1. 用 csv 模块数 header 字段数
t0 = time.time()
with open(FP, encoding="latin-1") as f:
    reader = csv.reader(f)
    header = next(reader)
    # 验证 row 1 字段数
    row1 = next(reader)
print(f"header 字段数: {len(header)}, row1 字段数: {len(row1)}")
print(f"time: {time.time()-t0:.1f}s")

ncols = len(header)

# 2. 试 pandas header=0 + on_bad_lines='skip' + usecols
t0 = time.time()
df = pd.read_csv(FP, encoding="latin-1", header=0,
                 on_bad_lines="skip", usecols=range(ncols))
print(f"\n=== pandas header=0 + skip + usecols ===")
print(f"shape: {df.shape}")
print(f"patient_id head: {df['patient_id'].astype(str).tolist()[:5]}")
print(f"time: {time.time()-t0:.1f}s")
