"""
χ²_RSD consistency check for ΛCDM and Λcos against DESI DR1 ShapeFit+BAO
compressed growth amplitudes (Paper V, Appendix A, Eqs. A.13–A.24).

Why DR1 here when our BAO is DR2: DESI DR2 (March 2025) released BAO distances
only; the full-shape RSD products remain at DR1. Combining DR2 BAO with DR1 FS
follows the collaboration's own precedent (Lodha et al. 2025; Forero-Sánchez
et al. arXiv:2602.18761 for the rigorous joint treatment). We treat the two
likelihoods as independent here; cross-correlations at overlapping effective
redshifts are <~10% in published joint analyses and do not affect the
consistency-diagnostic conclusion at current precision.

ShapeFit's compressed fσ_s8 parameter is, strictly, only equivalent to the
physical fσ_8(z) at the fiducial cosmology of the compression. For Λcos at
data-allowed s_0 < 0.19 the deviation from ΛCDM is < 1% per bin, well within
ShapeFit's nonlinear shape-correction error budget. We compare fσ_8(z) model
predictions to fσ_s8 measurements directly.

Run:
    python compute_rsd_chi2.py

Outputs:
    results/rsd_chi2.csv          (model, chi2_RSD, dof, chi2_per_dof)
    results/rsd_residuals.csv     (z_eff, model, predicted, observed, residual, sigma)
"""

import numpy as np
import pandas as pd

DATA_PATH = "../data/desi_dr1_fs_fsigma8.csv"
OUT_DIR   = "../results/"

GROWTH_FILES = {
    "ΛCDM (Ω_m=0.315)":         "growth_LCDM_Om0p315.csv",
    "Λcos (s_0=0.076, median)": "growth_Lcos_s00p076.csv",
    "Λcos (s_0=0.185, 95% UL)": "growth_Lcos_s00p185.csv",
}

data = pd.read_csv(DATA_PATH, comment="#")
z_eff = data["z_eff"].values
fs8_obs = data["fsigmas8"].values
sigma   = data["sigma_fsigmas8"].values

chi2_rows = []
res_rows  = []

print(f"{'Model':<30}  {'χ²_RSD':>8}  {'dof':>4}  {'χ²/dof':>8}")
print("-" * 60)
for label, fname in GROWTH_FILES.items():
    g = pd.read_csv(OUT_DIR + fname)
    fs8_pred = np.interp(z_eff, g["z"].values, g["fsigma8"].values)
    delta = fs8_obs - fs8_pred
    chi2  = float(np.sum((delta / sigma) ** 2))
    dof   = len(data)
    chi2_rows.append({"model": label, "chi2_RSD": chi2, "dof": dof, "chi2_per_dof": chi2/dof})
    print(f"{label:<30}  {chi2:8.3f}  {dof:4d}  {chi2/dof:8.3f}")
    for i, z in enumerate(z_eff):
        res_rows.append({
            "model": label, "tracer": data["tracer"].iloc[i], "z_eff": z,
            "predicted": fs8_pred[i], "observed": fs8_obs[i],
            "residual": delta[i], "sigma": sigma[i],
            "pull": delta[i] / sigma[i],
        })

pd.DataFrame(chi2_rows).to_csv(OUT_DIR + "rsd_chi2.csv", index=False)
pd.DataFrame(res_rows).to_csv(OUT_DIR + "rsd_residuals.csv", index=False)

print("\nPer-tracer pulls (observed − predicted, in units of σ):")
res = pd.DataFrame(res_rows)
pivot = res.pivot(index="z_eff", columns="model", values="pull")
print(pivot.to_string(float_format=lambda x: f"{x:+.2f}σ"))

print("\nΛcos vs ΛCDM Δχ²_RSD:")
chi2_lcdm = chi2_rows[0]["chi2_RSD"]
for r in chi2_rows[1:]:
    print(f"  {r['model']:<30}  Δχ² = {r['chi2_RSD'] - chi2_lcdm:+.3f}")
