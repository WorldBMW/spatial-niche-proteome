"""
spikein_gen.py
==============
合成 WSI niche 地图 + 信号注入 + 混杂因子，生成 spike-in 数据集。

用法:
    python scripts/spikein_gen.py --n 20 --m-tiles 400 --n-proteins 10 --out data/spikein_smoke
    python scripts/spikein_gen.py --n 203 --m-tiles 2000 --n-proteins 50 --out data/spikein_full

输出:
    {out}/niche_maps/{case_id}_probs.npy   (M, 10) 软分配
    {out}/niche_maps/{case_id}_argmax.npy  (M,)   硬分配
    {out}/features_raw.csv                 (N, 230) 真实管线提取的特征
    {out}/features_shuffled.csv            (N, 230) 空间置换后的特征 (A5 对照)
    {out}/clinical.csv                     (N, 4)   age/sex/subtype/batch
    {out}/proteomics.csv                   (N, P+1) 合成蛋白组 (case_id + P 蛋白)
    {out}/ground_truth.csv                 (P, 4)   每个蛋白的真实组(C/A/M/N)与信号通路
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import extract_one, FEATURE_NAMES

N_NICHES = 10
GRID = 20  # 与 features.py 的 ripley/topology 实现一致


def generate_niche_map(rng: np.random.RandomState, m_tiles: int,
                       subtype: int, temperature: float = 1.0) -> tuple:
    """
    生成单个 WSI 的 niche 软分配。
    subtype: 0=LUAD (偏向 niche 0,1,2), 1=LUSC (偏向 niche 3,4,5)
    temperature: Potts 温度, 低→强聚集, 高→随机
    返回 (probs (M,10), argmax (M,))
    """
    # 1. 组成频率 (Dirichlet)
    base_alpha = np.ones(N_NICHES)
    if subtype == 0:  # LUAD
        base_alpha[[0, 1, 2]] += 2.0
    else:  # LUSC
        base_alpha[[3, 4, 5]] += 2.0
    freq = rng.dirichlet(base_alpha)

    # 2. 空间排列 (简化 Potts: 在 GRIDxGRID 网格上做 Gibbs 采样)
    n_grid = GRID * GRID
    # 初始化: 按 freq 随机分配
    grid = rng.choice(N_NICHES, size=n_grid, p=freq)
    grid = grid.reshape(GRID, GRID)

    # Gibbs 采样 (足够迭代以建立空间结构)
    n_sweeps = 50
    for _ in range(n_sweeps):
        for i in range(GRID):
            for j in range(GRID):
                # 邻居 (4-邻接)
                neighbors = []
                if i > 0:
                    neighbors.append(grid[i-1, j])
                if i < GRID-1:
                    neighbors.append(grid[i+1, j])
                if j > 0:
                    neighbors.append(grid[i, j-1])
                if j < GRID-1:
                    neighbors.append(grid[i, j+1])
                # Potts 能量: 邻居同类别越多越有利
                logits = np.log(freq + 1e-9)
                for nb in neighbors:
                    logits[nb] += 1.0 / temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                grid[i, j] = rng.choice(N_NICHES, p=probs)

    argmax = grid.flatten()[:m_tiles]
    # 3. 软分配: 以 p=0.85 取 argmax, 其余随机
    probs = np.zeros((m_tiles, N_NICHES), dtype=np.float32)
    for t in range(m_tiles):
        k = argmax[t]
        p = rng.dirichlet(np.ones(N_NICHES) * 0.1)
        p[k] += 0.85
        p /= p.sum()
        probs[t] = p

    return probs, argmax


def inject_signals(features: np.ndarray, clinical: pd.DataFrame,
                   n_proteins: int, beta: float, rng: np.random.RandomState):
    """
    注入 C/A/M/N 信号。
    features: (N, 230)
    返回:
        y: (N, n_proteins)
        ground_truth: list of dict {protein, group, comp_feat, arr_feat, beta_comp, beta_arr}
    """
    N, D = features.shape
    # 特征索引: 前 10 维是 abundance
    abund_idx = list(range(10))
    # topology 特征: trans_* (100 维), 从第 10+40+11+10+14=85 开始
    trans_start = 10 + 40 + 11 + 10 + 14  # = 85
    trans_idx = list(range(trans_start, trans_start + 100))
    # ripley 特征: 10-49
    ripley_idx = list(range(10, 50))

    # 标准化 features (用于信号构造)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X = (features - features.mean(axis=0)) / feat_std

    y = np.zeros((N, n_proteins), dtype=np.float64)
    gt = []

    n_per_group = n_proteins // 4
    for p in range(n_proteins):
        if p < n_per_group:
            group = "C"  # 组成型
        elif p < 2 * n_per_group:
            group = "A"  # 排列型
        elif p < 3 * n_per_group:
            group = "M"  # 混合型
        else:
            group = "N"  # 噪声

        beta_comp = beta if group in ("C", "M") else 0.0
        beta_arr = beta if group in ("A", "M") else 0.0
        if group == "M":
            beta_comp = beta * 0.5
            beta_arr = beta * 0.5

        # 选择信号通路
        comp_feat = rng.choice(abund_idx)
        arr_feat = rng.choice(trans_idx)

        signal = beta_comp * X[:, comp_feat] + beta_arr * X[:, arr_feat]

        # 亚型混杂 (模拟真实 96% 衰减来源)
        subtype = clinical["subtype"].values.astype(float)
        subtype_effect = 0.3 * (subtype - subtype.mean())

        # 年龄/性别弱混杂
        age_z = (clinical["age"].values - clinical["age"].mean()) / clinical["age"].std()
        sex_effect = 0.05 * clinical["sex"].values.astype(float)

        noise = rng.normal(0, 1.0, N)
        y[:, p] = signal + subtype_effect + 0.1 * age_z + sex_effect + noise

        gt.append({
            "protein": f"prot_{p:03d}",
            "group": group,
            "comp_feat": int(comp_feat),
            "arr_feat": int(arr_feat),
            "beta_comp": float(beta_comp),
            "beta_arr": float(beta_arr),
        })

    return y, pd.DataFrame(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="患者数")
    ap.add_argument("--m-tiles", type=int, default=400, help="每例 tile 数 (建议 ≤ 400)")
    ap.add_argument("--n-proteins", type=int, default=10, help="蛋白数 (建议 10 的倍数)")
    ap.add_argument("--beta", type=float, default=0.8, help="信号强度")
    ap.add_argument("--temperature", type=float, default=1.0, help="Potts 温度")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="data/spikein_smoke")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    out = Path(args.out)
    niche_dir = out / "niche_maps"
    niche_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成临床协变量
    case_ids = [f"case_{i:03d}" for i in range(args.n)]
    subtypes = rng.choice([0, 1], size=args.n, p=[0.5, 0.5])  # LUAD/LUSC
    ages = rng.normal(60, 10, args.n).clip(30, 90)
    sexes = rng.choice([0, 1], size=args.n, p=[0.5, 0.5])
    batches = rng.choice([0, 1, 2], size=args.n)
    clinical = pd.DataFrame({
        "case_id": case_ids,
        "age": ages,
        "sex": sexes,
        "subtype": subtypes,
        "batch": batches,
    })

    # 2. 生成 niche 地图 + 提取特征
    features_raw = []
    features_shuffled = []
    for i, cid in enumerate(case_ids):
        probs, argmax = generate_niche_map(rng, args.m_tiles, int(subtypes[i]),
                                           temperature=args.temperature)
        np.save(niche_dir / f"{cid}_probs.npy", probs)
        np.save(niche_dir / f"{cid}_argmax.npy", argmax)

        # 真实管线特征提取
        feat_raw = extract_one(probs, argmax)
        features_raw.append(feat_raw)

        # 空间置换 (A5 对照): 同步行置换
        perm = rng.permutation(len(probs))
        feat_shuf = extract_one(probs[perm], argmax[perm])
        features_shuffled.append(feat_shuf)

        if (i + 1) % 10 == 0:
            print(f"[gen] {i+1}/{args.n} cases done", flush=True)

    X_raw = np.stack(features_raw)
    X_shuf = np.stack(features_shuffled)

    feat_df_raw = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    feat_df_raw.insert(0, "case_id", case_ids)
    feat_df_raw["subtype"] = subtypes

    feat_df_shuf = pd.DataFrame(X_shuf, columns=FEATURE_NAMES)
    feat_df_shuf.insert(0, "case_id", case_ids)
    feat_df_shuf["subtype"] = subtypes

    # 3. 注入信号
    y, gt_df = inject_signals(X_raw, clinical, args.n_proteins, args.beta, rng)
    prot_df = pd.DataFrame(y, columns=gt_df["protein"].tolist())
    prot_df.insert(0, "case_id", case_ids)

    # 4. 保存
    clinical.to_csv(out / "clinical.csv", index=False)
    feat_df_raw.to_csv(out / "features_raw.csv", index=False)
    feat_df_shuf.to_csv(out / "features_shuffled.csv", index=False)
    prot_df.to_csv(out / "proteomics.csv", index=False)
    gt_df.to_csv(out / "ground_truth.csv", index=False)

    print(f"\n[done] spike-in dataset saved to {out}")
    print(f"  cases: {args.n}, tiles: {args.m_tiles}, proteins: {args.n_proteins}")
    print(f"  feature dim: {X_raw.shape[1]}")
    print(f"  group counts: {gt_df['group'].value_counts().to_dict()}")
    print(f"  beta: {args.beta}, temperature: {args.temperature}")


if __name__ == "__main__":
    main()
