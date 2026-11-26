# PDA 三臂消融 batch 提交 (composition-covariate, n≈61)
# 特征矩阵就绪后运行: python scripts/run_pda_ablation_batch.py
import subprocess, sys, time
from pathlib import Path

ARMS = ["only-trans", "only-abund", "matched-dim"]
LOG = Path("results_vitl14_jbi_r4/dn_r4_pda_ablation_log.txt")
LOG.parent.mkdir(parents=True, exist_ok=True)

with open(LOG, "w") as f:
    f.write(f"=== PDA ablation batch start {time.ctime()} ===\n")

for arm in ARMS:
    print(f"\n{'='*60}\n[RUN] {arm}\n{'='*60}", flush=True)
    with open(LOG, "a") as f:
        f.write(f"\n=== {arm} START {time.ctime()} ===\n")
    cmd = [sys.executable, "scripts/_pda_ablation.py", "--feature-subset", arm]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(LOG, "a") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
        f.write(f"\n=== {arm} DONE (exit={proc.returncode}) {time.ctime()} ===\n")
    print(f"[DONE] {arm} exit={proc.returncode}", flush=True)

print(f"\nAll done. Log: {LOG}")
