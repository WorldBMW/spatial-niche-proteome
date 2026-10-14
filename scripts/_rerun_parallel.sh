#!/bin/bash
# 并行跑 5 组主实验 + OS (替换串行 rerun)
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
FEAT=data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv
BASE=results_vitl14_jbi_r2
mkdir -p $BASE

# 停旧串行 rerun (模式带 [] 避免匹配自身)
pkill -f "_rerun_al[l]" 2>/dev/null || true
pkill -f "downstream_omics.p[y]" 2>/dev/null || true
sleep 2

run_one() {
  local name=$1; shift
  nohup $PY src/downstream_omics.py "$@" > logs_rerun_$name.log 2>&1 &
  echo "$name PID $!"
}

run_one ALL_prot    --features $FEAT --cohort ALL  --modalities proteomics        --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_prot2000
run_one LUAD_prot   --features $FEAT --cohort LUAD --modalities proteomics        --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_LUAD_prot2000
run_one LUSC_prot   --features $FEAT --cohort LUSC --modalities proteomics        --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_LUSC_prot2000
run_one ALL_phospho --features $FEAT --cohort ALL  --modalities phosphoproteomics --targets-top-k 500  --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_phospho500
run_one ALL_rna     --features $FEAT --cohort ALL  --modalities transcriptomics   --targets-top-k 500  --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_rna500

# OS (独立, 输出到 dn_ALL_rna500/ALL_os_cox.csv, 与 RNA 结果不同文件名不冲突)
nohup $PY scripts/_run_os_only.py > logs_rerun_OS.log 2>&1 &
echo "OS PID $!"

sleep 20
echo "--- ps ---"
ps aux | grep downstream_omics | grep -v grep | awk '{print $2, $12, $13, $14, $15}'
echo "--- ALL_prot ---"
tail -2 logs_rerun_ALL_prot.log
echo "--- OS ---"
tail -2 logs_rerun_OS.log
echo DONE