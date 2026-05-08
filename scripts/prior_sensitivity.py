"""
Prior sensitivity for the Λcos s_0 constraint (Table V).

Reweights the baseline post-burn chain (`results/lcos_post.csv`, flat
prior on s_0 ∈ [0.001, 0.99]) under two alternative priors and reports
the weighted median and 95% upper limit of s_0.

Run:
    python prior_sensitivity.py
Output:
    results/prior_sensitivity.csv

Reweighting rule: w_i ∝ π_new(s_0^(i)) / π_baseline(s_0^(i)). The
baseline prior on s_0 is uniform, so π_baseline is constant inside the
support and drops out (constants are absorbed into normalization).

    Flat in s_0²:        prior density on s_0 ∝ s_0       (Jacobian 2 s_0)
    Flat in log_10(s_0): prior density on s_0 ∝ 1 / s_0   (Jacobian 1/(s_0 ln 10))
"""

import numpy as np, pandas as pd

post = pd.read_csv("../results/lcos_post.csv")
s0 = post["s0"].values

def weighted_quantile(x, w, q):
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    c = np.cumsum(w) / w.sum()
    return np.interp(q, c, x)

priors = {
    "Flat in s_0 (baseline)":      np.ones_like(s0),
    "Flat in s_0^2 (more weight at larger s_0)": s0,
    "Flat in log_10(s_0) (scale-invariant)":     1.0 / s0,
}

rows = []
for name, w in priors.items():
    median = weighted_quantile(s0, w, 0.5)
    ul95   = weighted_quantile(s0, w, 0.95)
    rows.append({"prior": name, "s0_median": median, "s0_95UL": ul95})
    print(f"{name:50s}  median={median:.4f}  95% UL={ul95:.4f}")

pd.DataFrame(rows).to_csv("../results/prior_sensitivity.csv", index=False)
