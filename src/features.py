"""
features.py
===========
从 CANVAS 式 niche 软分配概率 (N_patches, 10) 中提取 262 维空间生态特征。

6 大类:
  1. 丰度占比        10 维
  2. Ripley's K/L   20 维
  3. 界面距离/长度   12 维
  4. CAF 纤维化评分  10 维
  5. 多样性/熵       10 维
  6. 转移概率/拓扑  200 维
合计 262 维（实际实现见下方）
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

N_NICHES = config.N_NICHES


# ==============================================================
# 1. 丰度
# ==============================================================
def abundance(probs: np.ndarray) -> np.ndarray:
    return probs.mean(axis=0)  # (10,)


# ==============================================================
# 2. Ripley's K (简化版：统计同 niche 邻居密度)
# ==============================================================
def ripley_k(probs: np.ndarray, radii=(1, 2, 3, 4), grid=20) -> np.ndarray:
    """返回 10 × len(radii) = 40 维 → 展平。"""
    n_patches = len(probs)
    if n_patches < grid * grid:
        return np.zeros(N_NICHES * len(radii))

    # 重排为 grid × grid 网格（忽略真实坐标，近似拓扑）
    k_per_niche_radius = []
    for k in range(N_NICHES):
        binary = (probs[:, k] > probs[:, k].mean()).astype(int)
        padded = np.zeros(grid * grid, dtype=int)
        padded[:min(n_patches, grid*grid)] = binary[:min(n_patches, grid*grid)]
        grid_2d = padded.reshape(grid, grid)
        coords = np.argwhere(grid_2d == 1)
        n = len(coords)
        if n < 2:
            k_per_niche_radius.extend([0.0] * len(radii))
            continue
        vals = []
        for r in radii:
            count = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d = np.abs(coords[i] - coords[j]).max()
                        if d <= r:
                            count += 1
            vals.append(count / (n * (n - 1)) if n > 1 else 0.0)
        k_per_niche_radius.extend(vals)
    return np.array(k_per_niche_radius, dtype=np.float32)


# ==============================================================
# 3. 界面（肿瘤核心 ↔ 免疫相关 niche 的邻接）
# ==============================================================
def interface_features(probs: np.ndarray) -> np.ndarray:
    """tumor_core 与各免疫 niche 的概率乘积均值 + 对比熵 = 12 维。"""
    tumor = probs[:, 0]                          # tumor_core
    immune_niches = [1, 4, 5, 6]                 # macrophage, neutrophil, t_cell, tumor_immune_interface
    feat = []
    for im in immune_niches:
        feat.append(float((tumor * probs[:, im]).mean()))         # 乘积均值
    # tumor vs stroma / vessel / necrosis 的分离度
    for other in [7, 8, 9]:
        p_t = probs[:, 0].mean()
        p_o = probs[:, other].mean()
        feat.append(float(np.abs(p_t - p_o)))                      # 丰度差
    # 熵类
    e = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    feat.append(float(e.mean()))                                    # 平均熵
    feat.append(float(e.std()))                                     # 熵的标准差
    feat.append(float((tumor > 0.5).mean()))                        # 肿瘤占比
    feat.append(float(((tumor + probs[:, 6]) > 0.5).mean()))        # 肿瘤+界面占比
    return np.array(feat, dtype=np.float32)


# ==============================================================
# 4. CAF 纤维化评分
# ==============================================================
def caf_score(probs: np.ndarray) -> np.ndarray:
    """fibroblast_CAF + stroma + vessel 的联合比例 → 10 维。"""
    caf = probs[:, 3]       # fibroblast_CAF
    stroma = probs[:, 7]
    vessel = probs[:, 9]
    necrosis = probs[:, 8]
    feat = [
        float(caf.mean()),
        float(stroma.mean()),
        float(vessel.mean()),
        float((caf + stroma).mean()),
        float((caf + stroma + vessel).mean()),
        float(caf.std()),
        float(stroma.std()),
        float((caf * stroma).mean()),
        float((caf * necrosis).mean()),
        float((1 - probs[:, 0]).mean()),   # non-tumor fraction
    ]
    return np.array(feat, dtype=np.float32)


# ==============================================================
# 5. 多样性（Shannon / Simpson）
# ==============================================================
def diversity_features(probs: np.ndarray) -> np.ndarray:
    shannon = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    simpson = 1 - np.sum(probs ** 2, axis=1)
    feat = [
        float(shannon.mean()), float(shannon.std()),
        float(simpson.mean()), float(simpson.std()),
    ]
    # 每个 niche 的相对熵
    for k in range(N_NICHES):
        p = probs[:, k].mean()
        feat.append(float(-p * np.log(p + 1e-9)))
    return np.array(feat, dtype=np.float32)  # 4 + 10 = 14 维


# ==============================================================
# 6. 拓扑：转移概率 (KMeans cluster k→l 的共现)
# ==============================================================
def topology_features(argmax: np.ndarray) -> np.ndarray:
    """基于 argmax 的邻接对转移矩阵，展平 = 100 维。"""
    n = len(argmax)
    if n < 2:
        return np.zeros(N_NICHES * N_NICHES, dtype=np.float32)
    # 相邻 patches 的转移计数
    t = np.zeros((N_NICHES, N_NICHES), dtype=np.float32)
    for i in range(n - 1):
        t[argmax[i], argmax[i+1]] += 1
    # 行归一化
    row_sum = t.sum(axis=1, keepdims=True) + 1e-6
    t_norm = t / row_sum
    return t_norm.flatten()


# ==============================================================
# 7. 两两 niche 概率相关（45 维）
# ==============================================================
def niche_correlations(probs: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(N_NICHES):
        for j in range(i + 1, N_NICHES):
            if probs[:, i].std() > 0 and probs[:, j].std() > 0:
                feats.append(float(np.corrcoef(probs[:, i], probs[:, j])[0, 1]))
            else:
                feats.append(0.0)
    return np.array(feats, dtype=np.float32)


# ==============================================================
# 主入口
# ==============================================================
FEATURE_NAMES = (
    [f"abund_{k}" for k in range(N_NICHES)] +              # 10
    [f"ripley_{k}_{r}" for k in range(N_NICHES) for r in range(1, 5)] +  # 40
    [f"intf_{i}" for i in range(11)] +                       # 11 (匹配 interface_features 实际输出)
    [f"caf_{i}" for i in range(10)] +                        # 10
    [f"div_{i}" for i in range(14)] +                        # 14
    [f"trans_{k}_{l}" for k in range(N_NICHES) for l in range(N_NICHES)] +  # 100
    [f"corr_{i}_{j}" for i in range(N_NICHES) for j in range(i+1, N_NICHES)]  # 45
)  # 合计 10+40+11+10+14+100+45 = 230 维


def extract_one(probs: np.ndarray, argmax: np.ndarray) -> np.ndarray:
    parts = [
        abundance(probs),       # 10
        ripley_k(probs),        # 40
        interface_features(probs),  # 12
        caf_score(probs),       # 10
        diversity_features(probs),  # 14
        topology_features(argmax),  # 100
        niche_correlations(probs),   # 45
    ]
    return np.concatenate(parts).astype(np.float32)


def run(backbone: str = None):
    backbone = backbone or config.DEFAULT_BACKBONE
    niche_dir = config.niche_cache_dir(backbone)
    out_dir = config.features_dir(backbone)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_files = sorted(niche_dir.glob("*_probs.npy"))
    rows = []
    for pf in tqdm(probs_files, desc=f"features[{backbone}]"):
        case_id = pf.name.replace("_probs.npy", "")
        probs = np.load(pf)
        argmax = np.load(niche_dir / f"{case_id}_argmax.npy")
        feat = extract_one(probs, argmax)
        rows.append([case_id] + feat.tolist())

    cols = ["case_id"] + FEATURE_NAMES
    df = pd.DataFrame(rows, columns=cols)
    out_csv = out_dir / "all_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"[features] saved {len(df)} cases × {len(FEATURE_NAMES)} features → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=config.DEFAULT_BACKBONE)
    args = ap.parse_args()
    run(args.backbone)
