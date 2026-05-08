"""
Savage-Dickey Bayes factor B_01 = ΛCDM vs Λcos at the s_0 prior boundary.

Λcos nests ΛCDM at s_0 = 0. The baseline prior on s_0 is uniform on
[0.001, 0.99]; the prior boundary 0.001 is the closest reachable point
to the ΛCDM limit. The Savage-Dickey ratio at that boundary is

    B_01 = π_posterior(s_0 = 0.001) / π_prior(s_0 = 0.001),

where B_01 > 1 indicates evidence for the nested model (ΛCDM).

The posterior density at the boundary is estimated by a
boundary-reflected KDE on `results/lcos_post.csv`. Reported with a
bandwidth scan to confirm stability.

Run:
    python bayes_factor.py
Output:
    results/bayes_factor.csv  (one row per bandwidth)
"""

import numpy as np, pandas as pd
from scipy.stats import gaussian_kde

s0 = pd.read_csv("../results/lcos_post.csv")["s0"].values
boundary = 0.001
prior_density = 1.0 / (0.99 - boundary)   # = 1.0102...

# Boundary-reflected sample: density at the boundary doubles after
# folding samples that fall just above 0.001 across the wall.
reflected = np.concatenate([s0, 2 * boundary - s0])

# emcee's chain has ~128k post-burn points; Scott's rule bandwidth is
# ≈ n^(-1/5) σ ≈ 0.012 for σ(s_0) ≈ 0.07. Scan across bandwidth values
# straddling that to verify stability.
bandwidths = [0.008, 0.010, 0.012, 0.015, 0.020]

rows = []
for bw in bandwidths:
    kde = gaussian_kde(reflected, bw_method=bw / reflected.std())
    post_density = 2.0 * kde(boundary)[0]   # ×2 for the reflection
    B01 = post_density / prior_density
    rows.append({"bandwidth": bw, "post_density": post_density, "B_01": B01})
    print(f"bw={bw:.3f}  π_post(0.001)={post_density:.4f}  B_01={B01:.3f}")

df = pd.DataFrame(rows)
df.to_csv("../results/bayes_factor.csv", index=False)

print(f"\nB_01 = {df.B_01.mean():.2f} ± {df.B_01.std():.2f} across bandwidths")
