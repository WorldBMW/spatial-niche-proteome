#!/bin/bash
# 重启 OS Cox (建输出目录 + nohup 后台)
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
mkdir -p results_vitl14_jbi_r2/dn_ALL_rna500
nohup $PY scripts/_run_os_only.py > logs_rerun_OS.log 2>&1 &
PID=$!
sleep 12
echo "OS_PID $PID"
echo "--- log ---"
tail -4 logs_rerun_OS.log
echo "DONE"