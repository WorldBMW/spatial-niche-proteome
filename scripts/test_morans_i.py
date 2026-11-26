"""
Moran's I 自实现单元验证（合成数据）
- 平滑梯度场 → I 应接近 1
- 随机噪声 → I 应接近 E[I] = -1/(n-1)，p 不显著
- 两种写法（邻居均值法 vs 权重矩阵法）应一致
"""
import numpy as np
import sys
sys.path.insert(0, r"f:\ai_paper\experiment2_fm_time\scripts")
from slpi_spatial_analysis import morans_i

# 10x10 网格
g = np.arange(10)
xs, ys = np.meshgrid(g, g)
coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)

def i_matrix_form(v, coords, k=6):
    """权重矩阵法交叉验证"""
    from scipy.spatial import cKDTree
    n = len(v)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i, 1:]] = 1.0 / k
    z = v - v.mean()
    return (n / W.sum()) * (z @ W @ z) / (z @ z)

# 1) 平滑梯度
smooth = coords[:, 0] + coords[:, 1]
r1 = morans_i(smooth, coords, n_perm=200)
r1b = i_matrix_form(smooth, coords)
print(f"[平滑梯度]  I(邻域法)={r1['I']:.4f}  I(矩阵法)={r1b:.4f}  z={r1['z']:.2f}  p_perm={r1['p_perm']:.4f}")
assert abs(r1['I'] - r1b) < 1e-9, "两种 I 写法不一致"
assert r1['I'] > 0.8, "平滑场 I 应接近 1"

# 2) 随机噪声
rng = np.random.default_rng(0)
noise = rng.normal(size=len(coords))
r2 = morans_i(noise, coords, n_perm=200)
print(f"[随机噪声]  I={r2['I']:.4f}  E[I]={r2['EI']:.4f}  z={r2['z']:.2f}  p_perm={r2['p_perm']:.4f}")
assert abs(r2['I'] - r2['EI']) < 0.1, "噪声 I 应接近 E[I]"

# 3) 置换零分布合理性
print(f"[置换] 零分布 mean={r1['perm_mean']:.4f} sd={r1['perm_sd']:.5f} (应接近 E[I])")
assert abs(r1['perm_mean'] - r1['EI']) < 0.02

# 4) 边缘情形：全零向量
r3 = morans_i(np.zeros(20), coords[:20], n_perm=50)
print(f"[全零]  I={r3['I']} (应为 nan，且不报错)")

print("\n全部断言通过 ✔")
