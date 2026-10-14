import sys, numpy as np
sys.path.insert(0, "src")
from downstream_omics import evaluate_enet_nested, residualize_fit_transform, run_os_analysis
import pandas as pd

rng = np.random.RandomState(0)
n, p = 120, 20
X = rng.randn(n, p)
cov = np.column_stack([rng.randn(n), rng.randint(0, 2, n).astype(float), np.r_[rng.randn(n - 10), np.full(10, np.nan)]])
y = X[:, 0] * 0.8 + cov[:, 0] * 0.3 + rng.randn(n) * 0.3

r = evaluate_enet_nested(X, y, cov=cov, n_seeds=1)
print("nestedCV w/ cov:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})

# 无 cov
r2 = evaluate_enet_nested(X, y, n_seeds=1)
print("nestedCV w/o cov:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r2.items()})

# 有 NaN 的 X
Xn = X.copy(); Xn[0, 3] = np.nan; Xn[5, 7] = np.nan
r3 = evaluate_enet_nested(Xn, y, cov=cov, n_seeds=1)
print("nestedCV w/ NaN:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r3.items()})

# residualize_fit_transform 直接测试
Xtr = rng.randn(80, 5); Xva = rng.randn(40, 5)
c1 = rng.randn(80, 2); c2 = rng.randn(40, 2)
Xtr_r, Xva_r = residualize_fit_transform(Xtr, Xva, c1, c2)
print("resid shapes:", Xtr_r.shape, Xva_r.shape)

# run_os_analysis smoke
clin = pd.DataFrame({
    "case_id": [f"C{i}" for i in range(n)],
    "Overall survival, days": rng.rand(n) * 2000 + 30,
    "Survival status (1, dead; 0, alive)": (rng.rand(n) < 0.4).astype(float),
})
os_df = run_os_analysis(X, clin, [f"C{i}" for i in range(n)], seed=2026)
print("OS result:", os_df.to_dict("records"))
print("SMOKE OK")