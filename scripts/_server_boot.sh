#!/bin/bash
# 服务器启动脚本: 测试 import + 后台启动 baseline2 和 rerun
cd /root/autodl-tmp/ai_paper2/experiment2_fm_time
PY=/root/miniconda3/bin/python

echo "=== import test ==="
timeout 120 $PY -c "import sys; sys.path.insert(0,'src'); import downstream_omics; print('IMPORT OK')" 2>&1 | tail -3

echo "=== start baseline2 (nohup) ==="
nohup $PY scripts/_baseline_global_pool.py > logs_baseline2.log 2>&1 &
echo "baseline2 pid $!"

echo "=== start rerun (nohup) ==="
nohup bash scripts/_rerun_all.sh > logs_rerun_all.log 2>&1 &
echo "rerun pid $!"

sleep 8
echo "=== ps ==="
ps aux | grep -E "downstream_omics|baseline_global|rerun_all" | grep -v grep | awk '{print $2, $11, $12, $13}' | head

echo "=== baseline2 log ==="
tail -3 logs_baseline2.log 2>/dev/null
echo "=== rerun log ==="
tail -3 logs_rerun_all.log 2>/dev/null
echo "BOOT_DONE"