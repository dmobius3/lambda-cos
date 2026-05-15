"""
FIG 5: f σ_8(z) trajectories for ΛCDM and Λcos overlaid with DESI DR1 FS
(ShapeFit+BAO) compressed amplitudes. At s_0=0.076 (posterior median) the
Λcos curve is visually on top of ΛCDM; at s_0=0.185 (95% UL) the separation
becomes visible at high z but still sits inside the error bars. The point:
current data cannot distinguish the models, and the figure shows where the
separation grows.

FIG 6: Ω_m(z) diagnostic over z ∈ [0, 3]. The 3% Λcos–ΛCDM split at z=2.3
is the most distinctive signature available in the model and is visually
clear well beyond the current DR1 FS reach (z ≤ 1.49).

Run:
    python make_growth_figures.py
Outputs:
    results/fig5_fsigma8.pdf  (+ .png)
    results/fig6_omegam_z.pdf (+ .png)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from growth import make_E_lcdm, make_E_lcos

DATA_PATH = "../data/desi_dr1_fs_fsigma8.csv"
RES_DIR   = "../results/"

# ---------------- LOAD ----------------
data = pd.read_csv(DATA_PATH, comment="#")
lcdm     = pd.read_csv(RES_DIR + "growth_LCDM_Om0p315.csv")
lcos_med = pd.read_csv(RES_DIR + "growth_Lcos_s00p076.csv")
lcos_ul  = pd.read_csv(RES_DIR + "growth_Lcos_s00p185.csv")

# ---------------- FIG 5: f σ_8(z) ----------------
fig, ax = plt.subplots(figsize=(7.0, 5.0))

m = lambda df: df.z <= 2.0
ax.plot(lcdm.z[m(lcdm)], lcdm.fsigma8[m(lcdm)],
        "-",  color="black", lw=2.0, label=r"$\Lambda$CDM ($\Omega_m=0.315$)")
ax.plot(lcos_med.z[m(lcos_med)], lcos_med.fsigma8[m(lcos_med)],
        "--", color="C0", lw=1.6, label=r"$\Lambda$cos ($s_0=0.076$, posterior median)")
ax.plot(lcos_ul.z[m(lcos_ul)], lcos_ul.fsigma8[m(lcos_ul)],
        ":",  color="C3", lw=1.8, label=r"$\Lambda$cos ($s_0=0.185$, 95% UL)")

ax.errorbar(data.z_eff, data.fsigmas8, yerr=data.sigma_fsigmas8,
            fmt="o", color="C2", markersize=6, capsize=3, capthick=1.2,
            ecolor="C2", label=r"DESI DR1 FS (ShapeFit$+$BAO)")
for _, row in data.iterrows():
    ax.annotate(row.tracer, (row.z_eff, row.fsigmas8),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, color="dimgray")

ax.set_xlabel(r"$z$", fontsize=12)
ax.set_ylabel(r"$f\sigma_8(z)$", fontsize=12)
ax.set_xlim(0, 2.0)
ax.set_ylim(0.20, 0.70)
ax.legend(loc="lower left", frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RES_DIR + "fig5_fsigma8.pdf")
plt.savefig(RES_DIR + "fig5_fsigma8.png", dpi=200)
plt.close()

# ---------------- FIG 6: Ω_m(z) over z ∈ [0, 3] ----------------
z_grid = np.linspace(0.0, 3.0, 600)
Om0 = 0.315
E_l    = make_E_lcdm(Om0)
E_med  = make_E_lcos(0.076, OL=0.685)
E_ul   = make_E_lcos(0.185, OL=0.685)
Omz_l   = Om0 * (1+z_grid)**3 / E_l(z_grid)**2
Omz_med = Om0 * (1+z_grid)**3 / E_med(z_grid)**2
Omz_ul  = Om0 * (1+z_grid)**3 / E_ul(z_grid)**2

fig, ax = plt.subplots(figsize=(7.0, 5.0))
ax.plot(z_grid, Omz_l,   "-",  color="black", lw=2.0, label=r"$\Lambda$CDM ($\Omega_m=0.315$)")
ax.plot(z_grid, Omz_med, "--", color="C0",    lw=1.6, label=r"$\Lambda$cos ($s_0=0.076$)")
ax.plot(z_grid, Omz_ul,  ":",  color="C3",    lw=1.8, label=r"$\Lambda$cos ($s_0=0.185$, 95% UL)")

# Annotate the high-z split: ~3% at z = 2.3
delta_z23 = (Omz_ul[np.argmin(abs(z_grid - 2.3))] - Omz_l[np.argmin(abs(z_grid - 2.3))]) / Omz_l[np.argmin(abs(z_grid - 2.3))]
ax.axvline(2.3, color="gray", lw=0.6, ls="-", alpha=0.5)
ax.annotate(f"$\\Delta\\Omega_m / \\Omega_m = {100*delta_z23:+.1f}$% at $z=2.3$",
            xy=(2.3, Omz_ul[np.argmin(abs(z_grid-2.3))]),
            xytext=(1.6, 0.55), fontsize=9, color="C3",
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))

# Shade the region beyond current growth-data reach (DR1 FS max z=1.49)
ax.axvspan(1.49, 3.0, alpha=0.08, color="gray")
ax.text(2.25, 0.355, "no current growth data", fontsize=8, color="dimgray",
        ha="center", va="bottom", style="italic")

ax.set_xlabel(r"$z$", fontsize=12)
ax.set_ylabel(r"$\Omega_m(z)$", fontsize=12)
ax.set_xlim(0, 3.0)
ax.set_ylim(0.30, 1.00)
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RES_DIR + "fig6_omegam_z.pdf")
plt.savefig(RES_DIR + "fig6_omegam_z.png", dpi=200)
plt.close()

print(f"wrote {RES_DIR}fig5_fsigma8.pdf  + .png")
print(f"wrote {RES_DIR}fig6_omegam_z.pdf + .png")
print(f"\nΩ_m(z=2.3) split:  ΛCDM={Omz_l[np.argmin(abs(z_grid-2.3))]:.4f}  "
      f"Λcos(UL)={Omz_ul[np.argmin(abs(z_grid-2.3))]:.4f}  ({100*delta_z23:+.2f}%)")
