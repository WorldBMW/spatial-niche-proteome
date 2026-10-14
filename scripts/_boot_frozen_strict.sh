#!/bin/bash
# strict 版 frozen KMeans 重跑 (fold 内预处理新代码)
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
FEAT=data/features/dinov2_vitl14_features_frozen/all_features_omics_aligned.csv
if [ ! -f "$FEAT" ]; then echo "FROZEN_FEAT_MISSING $FEAT"; ls data/features/; exit 1; fi
mkdir -p results_vitl14_jbi_r2/dn_frozen_prot2000
nohup $PY src/downstream_omics.py --features $FEAT --cohort ALL --modalities proteomics \
    --targets-top-k 2000 --seeds 3 --no-permutation \
    --out-dir results_vitl14_jbi_r2/dn_frozen_prot2000 \
    > logs_rerun_frozen.log 2>&1 &
PID=$!
sleep 15
echo "FROZEN_PID $PID"
echo "--- log ---"
tail -5 logs_rerun_frozen.log
echo DONE