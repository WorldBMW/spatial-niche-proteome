"""性能预演: 用 4095 个随机坐标跑完整 morans_i (含 1000 次置换)，测时间"""
import sys
import time

import numpy as np

sys.path.insert(0, r"f:\ai_paper\experiment2_fm_time\scripts")
from slpi_spatial_analysis import morans_i

rng = np.random.default_rng(0)
n = 4095
coords = rng.uniform(0, 30000, size=(n, 2))
vals = np.sin(coords[:, 0] / 5000.0) + 0.1 * rng.normal(size=n)

t0 = time.time()
res = morans_i(vals, coords)
print("耗时 = %.2f s" % (time.time() - t0))
for k, v in res.items():
    print(" %-10s %s" % (k, v))
