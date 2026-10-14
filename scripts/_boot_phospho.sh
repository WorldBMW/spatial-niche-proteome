#!/bin/bash
# 重启 phospho (修复聚合阶段 to_numeric 后)
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
FEAT=data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv
BASE=results_vitl14_jbi_r2
nohup $PY src/downstream_omics.py --features $FEAT --cohort ALL --modalities phosphoproteomics \
    --targets-top-k 500 --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_phospho500 \
    > logs_rerun_ALL_phospho.log 2>&1 &
PID=$!
sleep 15
echo "PHOSPHO_PID $PID"
echo "--- log ---"
tail -4 logs_rerun_ALL_phospho.log
echo DONE