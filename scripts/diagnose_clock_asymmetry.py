#!/usr/bin/env python3
"""
DIAGNOSIS, NOT AUTHORITY.

Post-run diagnostics for the clock-asymmetry (epsilon) result. The
frozen registration (mode-identity-theory clock-asymmetry-fit.md) was
executed from tag eps-family-v1.0 and scored on its own verdict table
(row 3: zero excluded at 95%). Nothing in this script can change that
scoring. These diagnostics answer the adjudicated follow-up questions,
in order:

  audit    Recompute the headline numbers independently from the
           committed raw outputs (tier1 curve, tier2 postburn) and
           compare to the committed summaries. Guards against a
           summary-layer bug.

  omegaL   The Omega_Lambda-free scan: profile (s0, H0rd, M_B,
           Omega_Lambda) over an epsilon grid. Question posed before
           running: does the epsilon valley survive when Omega_Lambda
           moves? Evaporation points to a fixed-Omega_Lambda tension
           absorber; survival makes the valley a real family feature.

  split    SN-only and BAO-only profiles over the same epsilon grid
           (SN profiles (s0, M_B); BAO profiles (s0, H0rd)). Question
           posed before running: do the two datasets independently
           lean negative (harder to dismiss), or disagree (probably
           not tick physics)?

  all      audit, then omegaL, then split.

Model functions and data handling are mirrored from the frozen
scripts; the only new freedom is the one each diagnostic names
(Omega_Lambda as a profiled parameter; dataset restriction). Grids:
epsilon in [-0.40, +0.20], step 0.02 for omegaL (cost), 0.01 for the
split (cheap). No MCMC here; everything is the deterministic profile.

Outputs (results/):
    clock_asymmetry_audit.json
    clock_asymmetry_diag_omegaL_curve.csv, _summary.json
    clock_asymmetry_diag_snonly_curve.csv
    clock_asymmetry_diag_baoonly_curve.csv
    clock_asymmetry_diag_split_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

OMEGA_LAMBDA_FIXED = 0.685
LCDM_CHI2_MIN = 1772.456
C_LIGHT = 299792.458
H0_SN = 70.0

EPS_GRID_OMEGAL = np.round(np.arange(-0.40, 0.20 + 1e-12, 0.02), 6)
EPS_GRID_SPLIT = np.round(np.arange(-0.40, 0.20 + 1e-12, 0.01), 6)
OL_BOUNDS = (0.55, 0.80)


def n_of_eps(eps: float) -> float:
    return -0.5 - eps


# ---- data (mirrored) ----
pantheon = pd.read_csv(DATA_DIR / "pantheon_plus.csv")
z_sn = pantheon["zHD"].to_numpy(float)
m_b_corr = pantheon["m_b_corr"].to_numpy(float)
cov_sn = np.load(DATA_DIR / "pantheon_plus_cov.npy")
cho_sn = cho_factor(cov_sn, lower=True, check_finite=False)

bao = pd.read_csv(DATA_DIR / "desi_dr2_bao.csv")
z_bao = bao["z_eff"].to_numpy(float)
bao_values = bao["value"].to_numpy(float)
inv_cov_bao = np.linalg.inv(np.load(DATA_DIR / "desi_dr2_bao_cov.npy"))

zmax = max(float(np.max(z_sn)), float(np.max(z_bao))) * 1.002
z_grid = np.linspace(0.0, zmax, 5000)


# ---- model with Omega_Lambda as an explicit argument ----
def e2_model(z, s0, n_exp, OL):
    z = np.asarray(z, dtype=float)
    if not (0.0 < s0 < 1.0):
        return np.full_like(z, np.nan)
    S = s0 / (1.0 + z)
    ratio = ((1.0 - S**2) * S**(2.0 * n_exp - 2.0)) / (
        (1.0 - s0**2) * s0**(2.0 * n_exp - 2.0)
    )
    return (1.0 - OL) * ratio + OL


def e_and_integral(s0, n_exp, OL):
    e2 = e2_model(z_grid, s0, n_exp, OL)
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        return None, None
    e = np.sqrt(e2)
    return e, cumulative_trapezoid(1.0 / e, z_grid, initial=0.0)


def sn_chi2(s0, n_exp, M_B, OL):
    e, integral = e_and_integral(s0, n_exp, OL)
    if e is None:
        return np.inf
    I = np.interp(z_sn, z_grid, integral)
    d_l = (1.0 + z_sn) * (C_LIGHT / H0_SN) * I
    if np.any(d_l <= 0) or np.any(~np.isfinite(d_l)):
        return np.inf
    delta = m_b_corr - M_B - (5.0 * np.log10(d_l) + 25.0)
    return float(delta @ cho_solve(cho_sn, delta, check_finite=False))


def bao_chi2(s0, H0rd, n_exp, OL):
    e, integral = e_and_integral(s0, n_exp, OL)
    if e is None:
        return np.inf
    I = np.interp(z_bao, z_grid, integral)
    E_bao = np.interp(z_bao, z_grid, e)
    D_M = C_LIGHT / H0rd * I
    D_H = C_LIGHT / H0rd / E_bao
    D_V = (z_bao * D_M**2 * D_H) ** (1.0 / 3.0)
    model = np.where(bao["observable"] == "DV_rd", D_V,
                     np.where(bao["observable"] == "DM_rd", D_M, D_H))
    delta = model - bao_values
    return float(delta @ inv_cov_bao @ delta)


def profile(fun, starts, bounds):
    best = None
    for start in starts:
        r1 = minimize(fun, start, method="Nelder-Mead",
                      options={"maxiter": 1600, "xatol": 1e-8, "fatol": 1e-6})
        r2 = minimize(fun, r1.x, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1600, "ftol": 1e-9})
        cand = r2 if r2.fun <= r1.fun else r1
        if best is None or cand.fun < best.fun:
            best = cand
    return np.array(best.x, dtype=float), float(best.fun)


# ---- audit ----
def run_audit():
    report = {"mode": "audit", "checks": []}

    curve = pd.read_csv(RESULTS_DIR / "clock_asymmetry_tier1_curve.csv")
    with open(RESULTS_DIR / "clock_asymmetry_tier1_summary.json") as f:
        t1 = json.load(f)
    i = curve["chi2_total"].idxmin()
    checks = [
        ("tier1 eps_hat", float(curve.loc[i, "eps"]), t1["eps_hat"]),
        ("tier1 chi2_min", float(curve.loc[i, "chi2_total"]), t1["chi2_min"]),
        ("tier1 chi2_eps0",
         float(curve.loc[np.isclose(curve.eps, 0.0), "chi2_total"].iloc[0]),
         t1["chi2_eps0"]),
    ]

    post = pd.read_csv(RESULTS_DIR / "clock_asymmetry_tier2_postburn.csv")
    with open(RESULTS_DIR / "clock_asymmetry_tier2_summary.json") as f:
        t2 = json.load(f)
    q = np.percentile(post["eps"], [2.5, 16, 50, 84, 97.5])
    checks += [
        ("tier2 eps_median", float(q[2]), t2["eps_median"]),
        ("tier2 eps_95_lo", float(q[0]), t2["eps_95"][0]),
        ("tier2 eps_95_hi", float(q[4]), t2["eps_95"][1]),
        ("tier2 corr_s0_eps",
         float(np.corrcoef(post["s0"], post["eps"])[0, 1]), t2["corr_s0_eps"]),
    ]

    all_ok = True
    for name, recomputed, committed in checks:
        ok = abs(recomputed - committed) <= max(1e-9, 1e-6 * abs(committed))
        all_ok &= ok
        report["checks"].append({"check": name, "recomputed": recomputed,
                                 "committed": committed, "pass": bool(ok)})
        print(f"  audit {name}: recomputed {recomputed:+.6f}, "
              f"committed {committed:+.6f} -> {'PASS' if ok else 'FAIL'}")
    report["all_pass"] = bool(all_ok)
    report["zero_excluded_95_tier1"] = bool(t1["interval_95"][1] < 0.0)
    report["zero_excluded_95_tier2"] = bool(t2["eps_95"][1] < 0.0)
    with open(RESULTS_DIR / "clock_asymmetry_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"AUDIT {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---- Omega_Lambda-free scan ----
def run_omegaL():
    rows = []
    for eps in EPS_GRID_OMEGAL:
        n = n_of_eps(float(eps))
        fun = lambda x: (sn_chi2(x[0], n, x[2], x[3])
                         + bao_chi2(x[0], x[1], n, x[3])
                         if (0.001 <= x[0] <= 0.99 and 8000 <= x[1] <= 12000
                             and -20 <= x[2] <= -18
                             and OL_BOUNDS[0] <= x[3] <= OL_BOUNDS[1])
                         else np.inf)
        starts = [
            np.array([0.05, 10000.0, -19.35, 0.685]),
            np.array([0.47, 9900.0, -19.34, 0.685]),
            np.array([0.47, 9900.0, -19.34, 0.65]),
            np.array([0.20, 10000.0, -19.35, 0.72]),
            np.array([0.70, 9600.0, -19.30, 0.685]),
        ]
        bounds = [(0.001, 0.99), (8000, 12000), (-20, -18), OL_BOUNDS]
        x, chi2 = profile(fun, starts, bounds)
        rows.append({"eps": float(eps), "chi2_total": chi2,
                     "delta_lcdm": chi2 - LCDM_CHI2_MIN,
                     "s0": x[0], "H0rd": x[1], "M_B": x[2], "OL": x[3]})
        print(f"  omegaL eps={eps:+.2f}: chi2={chi2:.3f} "
              f"(dLCDM {chi2 - LCDM_CHI2_MIN:+.2f}), s0={x[0]:.3f}, OL={x[3]:.4f}")

    curve = pd.DataFrame(rows)
    chi2_eps0 = float(curve.loc[np.isclose(curve.eps, 0.0), "chi2_total"].iloc[0])
    curve["delta_eps0"] = curve["chi2_total"] - chi2_eps0
    curve.to_csv(RESULTS_DIR / "clock_asymmetry_diag_omegaL_curve.csv", index=False)

    i = curve["chi2_total"].idxmin()
    summary = {
        "question": "does the epsilon valley survive when Omega_Lambda is free?",
        "eps_hat_OLfree": float(curve.loc[i, "eps"]),
        "valley_depth_vs_eps0_OLfree": float(curve.loc[i, "chi2_total"] - chi2_eps0),
        "OL_at_min": float(curve.loc[i, "OL"]),
        "s0_at_min": float(curve.loc[i, "s0"]),
        "delta_lcdm_at_min": float(curve.loc[i, "chi2_total"] - LCDM_CHI2_MIN),
        "fixed_OL_reference": {"eps_hat": -0.106, "valley_depth_vs_eps0": -7.85},
        "note": "diagnosis, not authority",
    }
    with open(RESULTS_DIR / "clock_asymmetry_diag_omegaL_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---- SN-only / BAO-only split ----
def run_split():
    sn_rows, bao_rows = [], []
    for eps in EPS_GRID_SPLIT:
        n = n_of_eps(float(eps))

        fun_sn = lambda x: (sn_chi2(x[0], n, x[1], OMEGA_LAMBDA_FIXED)
                            if (0.001 <= x[0] <= 0.99 and -20 <= x[1] <= -18)
                            else np.inf)
        x, chi2 = profile(fun_sn,
                          [np.array([0.05, -19.35]), np.array([0.39, -19.34]),
                           np.array([0.70, -19.30])],
                          [(0.001, 0.99), (-20, -18)])
        sn_rows.append({"eps": float(eps), "chi2_sn": chi2,
                        "s0": x[0], "M_B": x[1]})

        fun_bao = lambda x: (bao_chi2(x[0], x[1], n, OMEGA_LAMBDA_FIXED)
                             if (0.001 <= x[0] <= 0.99 and 8000 <= x[1] <= 12000)
                             else np.inf)
        x, chi2 = profile(fun_bao,
                          [np.array([0.05, 10000.0]), np.array([0.47, 9900.0]),
                           np.array([0.70, 9600.0])],
                          [(0.001, 0.99), (8000, 12000)])
        bao_rows.append({"eps": float(eps), "chi2_bao": chi2,
                         "s0": x[0], "H0rd": x[1]})

    sn = pd.DataFrame(sn_rows)
    sn["delta_eps0"] = sn["chi2_sn"] - float(
        sn.loc[np.isclose(sn.eps, 0.0), "chi2_sn"].iloc[0])
    sn.to_csv(RESULTS_DIR / "clock_asymmetry_diag_snonly_curve.csv", index=False)

    bb = pd.DataFrame(bao_rows)
    bb["delta_eps0"] = bb["chi2_bao"] - float(
        bb.loc[np.isclose(bb.eps, 0.0), "chi2_bao"].iloc[0])
    bb.to_csv(RESULTS_DIR / "clock_asymmetry_diag_baoonly_curve.csv", index=False)

    def edge68(df, col):
        d = df[col].to_numpy() - df[col].min()
        e = df["eps"].to_numpy()
        ihat = int(np.argmin(d))
        lo = hi = None
        for j in range(len(e) - 1):
            if (d[j] - 1.0) * (d[j + 1] - 1.0) < 0:
                x = e[j] + (1.0 - d[j]) * (e[j + 1] - e[j]) / (d[j + 1] - d[j])
                if e[j] < e[ihat]:
                    lo = x
                else:
                    hi = hi if hi is not None else x
        return float(e[ihat]), lo, hi

    sn_hat, sn_lo, sn_hi = edge68(sn, "chi2_sn")
    bao_hat, bao_lo, bao_hi = edge68(bb, "chi2_bao")
    summary = {
        "question": "do SN and BAO independently lean negative, or disagree?",
        "eps_SN": {"hat": sn_hat, "68_lo": sn_lo, "68_hi": sn_hi,
                   "depth_vs_eps0": float(sn["delta_eps0"].min())},
        "eps_BAO": {"hat": bao_hat, "68_lo": bao_lo, "68_hi": bao_hi,
                    "depth_vs_eps0": float(bb["delta_eps0"].min())},
        "eps_joint_reference": {"hat": -0.106, "68": [-0.139, -0.069]},
        "note": "diagnosis, not authority; grid edges at [-0.40, +0.20]",
    }
    with open(RESULTS_DIR / "clock_asymmetry_diag_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "audit":
        ok = run_audit()
        sys.exit(0 if ok else 1)
    elif mode == "omegaL":
        run_omegaL()
    elif mode == "split":
        run_split()
    elif mode == "all":
        if not run_audit():
            sys.exit("Audit failed; diagnostics halted. Technical, not physics.")
        run_omegaL()
        run_split()
    else:
        sys.exit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
