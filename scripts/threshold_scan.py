"""
CPL threshold scan (Fig. 2, §4.3 table).

Repeats the CPL-vs-Λcos mock fit across s0 ∈ [0.01, 0.40] in steps of
0.01, recording (w0, wa) at each point. Demonstrates that the formal
phantom crossing w0 < −1 persists at every s0 > 0, with the induced
distortion modest at the SN+BAO 95% CL upper limit (s0 < 0.18).

Run:
    python scripts/threshold_scan.py
Outputs:
    results/threshold_scan.csv   (s0, w0, wa, chi2) per row
    results/threshold_scan.png   Fig. 2 of the paper
    results/threshold_scan.pdf
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

# ---------------- CONFIG ----------------
ΩΛ = 0.685
Om = 1 - ΩΛ          # 0.315
H0 = 67.4            # km/s/Mpc
rd = 147.1           # Mpc
H0rd = H0 * rd       # 9914.54 km/s
c = 299792.458

s_min, s_max, s_step = 0.01, 0.40, 0.01
SN_BAO_95_UL = 0.18  # vertical CL line in Fig. 2

data_dir = "../data/"
out_dir = "../results/"

# ---------------- DATA ----------------
bao = pd.read_csv(data_dir+"desi_dr2_bao.csv")
Cbao = np.load(data_dir+"desi_dr2_bao_cov.npy")
Cbao_inv = np.linalg.inv(Cbao)

# ---------------- MODEL HELPERS ----------------
zgrid = np.linspace(0, 2.5, 4000)

def E_lcos(z, s0):
    return np.sqrt(Om/(1-s0**2)*(1+z)**3 - Om*s0**2/(1-s0**2)*(1+z) + ΩΛ)

def bao_obs(Ez_grid):
    I = cumulative_trapezoid(1/Ez_grid, zgrid, initial=0)
    DM = c/H0rd * np.interp(bao.z_eff, zgrid, I)
    DH = c/H0rd / np.interp(bao.z_eff, zgrid, Ez_grid)
    out = []
    i = 0
    for _, row in bao.iterrows():
        if row.observable == "DV_rd":
            out.append((row.z_eff*DM[i]**2*DH[i])**(1/3))
        elif row.observable == "DM_rd":
            out.append(DM[i])
        else:
            out.append(DH[i])
        i += 1
    return np.array(out)

def w_cpl(a, w0, wa):
    return w0 + wa*(1-a)

def E_cpl(zgrid, w0, wa):
    a_grid = 1/(1+zgrid)
    integrand = 3*(1 + w_cpl(a_grid, w0, wa))/(1+zgrid)
    integral = cumulative_trapezoid(integrand, zgrid, initial=0)
    rho_de = np.exp(integral)
    return np.sqrt(Om*(1+zgrid)**3 + (1-Om)*rho_de)

# ---------------- SCAN ----------------
svals = np.arange(s_min, s_max + s_step, s_step)
w0_list, wa_list, chi2_list = [], [], []

for s0 in svals:
    bao_mock = bao_obs(E_lcos(zgrid, s0))

    def chi2(params):
        w0, wa = params
        pred = bao_obs(E_cpl(zgrid, w0, wa))
        d = pred - bao_mock
        return d @ Cbao_inv @ d

    res = minimize(chi2, x0=[-1.0, 0.0],
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})
    w0_list.append(res.x[0])
    wa_list.append(res.x[1])
    chi2_list.append(res.fun)

df = pd.DataFrame({"s0": svals, "w0": w0_list, "wa": wa_list, "chi2": chi2_list})
df.to_csv(out_dir+"threshold_scan.csv", index=False)

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(svals, w0_list, "o-", color="C0", lw=1.5, markersize=5, label=r"$w_0$")
ax.plot(svals, wa_list, "s-", color="C1", lw=1.5, markersize=5, label=r"$w_a$")

ax.axhline(-1, color="black", lw=0.8, ls="--")
ax.axvline(SN_BAO_95_UL, color="black", lw=0.8, ls=":")
ax.text(SN_BAO_95_UL + 0.005, -0.4, "SN+BAO 95% CL",
        rotation=90, va="center", fontsize=10)

ax.set_xlabel(r"$s_0$")
ax.set_ylabel("Recovered CPL parameter")
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.savefig(out_dir+"threshold_scan.png", dpi=200)
plt.savefig(out_dir+"threshold_scan.pdf")
