#!/bin/bash
# 启动 baseline (PCA 修复版), 短 sleep 后退出, 避免 exec 管道挂起
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python
pkill -f "baseline_global_poo[l].py" 2>/dev/null || true
sleep 1
nohup $PY scripts/_baseline_global_pool.py > logs_baseline2.log 2>&1 &
PID=$!
sleep 15
echo "BASELINE_PID $PID"
echo "--- ps ---"
ps aux | grep baseline_global | grep -v grep | awk '{print $2, $11, $12, $13}'
echo "--- log ---"
tail -5 logs_baseline2.log
echo "DONE"