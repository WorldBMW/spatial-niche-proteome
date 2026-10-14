#!/bin/bash
# 服务器端: 重跑全部主实验 (fold 内残差化 + 标准化版) + OS
set -e
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
FEAT=data/features/dinov2_vitl14_features_cptac/all_features_omics_aligned.csv
BASE=results_vitl14_jbi_r2
mkdir -p $BASE

log() { echo "===== $1 =====" | tee -a logs_rerun_server.log; }

log "ALL proteomics 2000"
$PY src/downstream_omics.py --features $FEAT --cohort ALL --modalities proteomics \
    --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_prot2000 \
    2>&1 | tee -a logs_rerun_server.log

log "LUAD proteomics 2000"
$PY src/downstream_omics.py --features $FEAT --cohort LUAD --modalities proteomics \
    --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_LUAD_prot2000 \
    2>&1 | tee -a logs_rerun_server.log

log "LUSC proteomics 2000"
$PY src/downstream_omics.py --features $FEAT --cohort LUSC --modalities proteomics \
    --targets-top-k 2000 --seeds 3 --no-permutation --out-dir $BASE/dn_LUSC_prot2000 \
    2>&1 | tee -a logs_rerun_server.log

log "ALL phospho 500"
$PY src/downstream_omics.py --features $FEAT --cohort ALL --modalities phosphoproteomics \
    --targets-top-k 500 --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_phospho500 \
    2>&1 | tee -a logs_rerun_server.log

log "ALL rna 500"
$PY src/downstream_omics.py --features $FEAT --cohort ALL --modalities transcriptomics \
    --targets-top-k 500 --seeds 3 --no-permutation --out-dir $BASE/dn_ALL_rna500 \
    2>&1 | tee -a logs_rerun_server.log

log "OS Cox (fold 内 PCA)"
$PY scripts/_run_os_only.py 2>&1 | tee -a logs_rerun_server.log

echo "ALL_DONE" | tee -a logs_rerun_server.log