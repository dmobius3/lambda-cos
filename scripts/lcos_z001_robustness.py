#!/usr/bin/env python3
"""
Paper 1 robustness check: does the published Lambda-cos result survive
the standard z > 0.01 Pantheon+ cut?

Context: the gamma-gate run showed the shipped Pantheon+ file carries
111 light curves below z = 0.01, and that a separate one-parameter
deformation's apparent preference was low-redshift driven. Paper 1's
claim is different in kind: Model D+Lambda is statistically equivalent
to LCDM (published Delta chi2 = +0.11 with s0 constrained near the
LCDM limit, s0 < 0.19 at 95% CL under a flat prior). This script asks
whether that near-null status is stable under the standard cut.

Model (Paper 1's Model D+Lambda, Omega_Lambda = 0.685 fixed):
    E^2 = (1-OL)/(1-s0^2) [ (1+z)^3 - s0^2 (1+z) ] + OL
Parameters (s0, H0rd, M_B), priors as the published baseline
(s0 flat [0.001, 0.99]; H0rd [8000, 12000]; M_B [-20, -18]).

EXPECTATION, stated before running: the result remains near LCDM with
a similar s0 upper bound, because the published preferred s0 already
sits near the LCDM limit and the removed low-z SNe were driving a
different (matter-exponent) direction.

FROZEN BARS (Delta chi2 is Lambda-cos best fit minus the same-variant
free-Omega_m LCDM best fit, both on the cut data):
    PASS (Paper 1 stands cleanly, no action):
        Delta chi2 in [-2, +3] on z > 0.01 AND s0 95% UL <= 0.25.
    ROBUSTNESS NOTE (record, do not retract):
        Delta chi2 in (+3, +9] or s0 95% UL in (0.25, 0.45].
    REASSESS: anything beyond.

Outputs (results/): lcos_z001_robustness.json, lcos_z001_post.csv,
plus a transcript via the runner. Single execution, one results commit.
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

C_LIGHT = 299792.458
H0_SN = 70.0
OMEGA_LAMBDA = 0.685
NWALKERS, NSTEPS, BURN, RNG_SEED = 32, 5000, 1000, 12345

PUBLISHED = {"delta_chi2": 0.11, "s0_95UL": 0.19}
BARS = {"pass_dchi2": (-2.0, 3.0), "pass_UL": 0.25,
        "note_dchi2_hi": 9.0, "note_UL": 0.45}

_pan = pd.read_csv(DATA_DIR / "pantheon_plus.csv")
Z_ALL = _pan["zHD"].to_numpy(float)
MB_ALL = _pan["m_b_corr"].to_numpy(float)
COV_ALL = np.load(DATA_DIR / "pantheon_plus_cov.npy")

bao = pd.read_csv(DATA_DIR / "desi_dr2_bao.csv")
z_bao = bao["z_eff"].to_numpy(float)
bao_values = bao["value"].to_numpy(float)
bao_obs = bao["observable"].to_numpy()
INV_COV_BAO = np.linalg.inv(np.load(DATA_DIR / "desi_dr2_bao_cov.npy"))

Z_GRID = np.linspace(0.0, max(float(Z_ALL.max()), float(z_bao.max())) * 1.002, 5000)


class SN:
    def __init__(self, zmin):
        m = Z_ALL > zmin if zmin > 0 else np.ones_like(Z_ALL, bool)
        self.z, self.mb, self.n = Z_ALL[m], MB_ALL[m], int(m.sum())
        self.cho = cho_factor(COV_ALL[np.ix_(m, m)], lower=True, check_finite=False)


VAR = {"all": SN(0.0), "z001": SN(0.01), "z010": SN(0.10)}


def e2_lcos(z, s0):
    z = np.asarray(z, float)
    if not (0.0 < s0 < 1.0):
        return np.full_like(z, np.nan)
    A = (1 - OMEGA_LAMBDA) / (1 - s0 ** 2)
    return A * (1 + z) ** 3 - A * s0 ** 2 * (1 + z) + OMEGA_LAMBDA


def e2_lcdm(z, Om):
    z = np.asarray(z, float)
    return Om * (1 + z) ** 3 + (1 - Om)


def chi2(e2fun, shape, MB, H0rd, sn):
    e2 = e2fun(Z_GRID, shape)
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        return np.inf
    e = np.sqrt(e2)
    I = cumulative_trapezoid(1.0 / e, Z_GRID, initial=0.0)
    Isn = np.interp(sn.z, Z_GRID, I)
    dl = (1 + sn.z) * (C_LIGHT / H0_SN) * Isn
    if np.any(dl <= 0):
        return np.inf
    d = sn.mb - MB - (5 * np.log10(dl) + 25)
    c_sn = float(d @ cho_solve(sn.cho, d, check_finite=False))
    Ib = np.interp(z_bao, Z_GRID, I)
    Eb = np.interp(z_bao, Z_GRID, e)
    DM = C_LIGHT / H0rd * Ib
    DH = C_LIGHT / H0rd / Eb
    DV = (z_bao * DM ** 2 * DH) ** (1 / 3)
    model = np.where(bao_obs == "DV_rd", DV, np.where(bao_obs == "DM_rd", DM, DH))
    dd = model - bao_values
    return c_sn + float(dd @ INV_COV_BAO @ dd)


def fit(e2fun, shape_bounds, sn, starts):
    def fun(x):
        s, MB, H0rd = x[0], x[1], x[2]
        if not (shape_bounds[0] <= s <= shape_bounds[1]
                and -20 <= MB <= -18 and 8000 <= H0rd <= 12000):
            return np.inf
        return chi2(e2fun, s, MB, H0rd, sn)
    best = None
    for s0 in starts:
        x0 = np.array([s0, -19.35, 10000.0])
        r1 = minimize(fun, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-6})
        r2 = minimize(fun, r1.x, method="L-BFGS-B",
                      bounds=[shape_bounds, (-20, -18), (8000, 12000)],
                      options={"maxiter": 2000, "ftol": 1e-9})
        cand = r2 if r2.fun <= r1.fun else r1
        if best is None or cand.fun < best.fun:
            best = cand
    return np.array(best.x), float(best.fun)


def main():
    out = {"published": PUBLISHED, "bars": BARS,
           "counts": {k: v.n for k, v in VAR.items()}}
    for vname, sn in VAR.items():
        _, c_lcdm = fit(e2_lcdm, (0.0, 1.0), sn, [0.25, 0.30, 0.35])
        x, c_lcos = fit(e2_lcos, (0.001, 0.99), sn, [0.01, 0.10, 0.30, 0.50, 0.85])
        out[vname] = {"chi2_LCDM_freeOm": c_lcdm, "chi2_Lcos": c_lcos,
                      "delta_chi2": c_lcos - c_lcdm,
                      "s0_best": x[0], "H0rd_best": x[1], "M_B_best": x[2]}
        print(f"  [{vname}] N_SN={sn.n}  LCDM={c_lcdm:.3f}  Lcos={c_lcos:.3f}  "
              f"dchi2={c_lcos - c_lcdm:+.3f}  s0_best={x[0]:.4f}")

    import emcee
    sn = VAR["z001"]
    def logp(t):
        s0, MB, H0rd = t
        if not (0.001 <= s0 <= 0.99 and -20 <= MB <= -18 and 8000 <= H0rd <= 12000):
            return -np.inf
        c = chi2(e2_lcos, s0, MB, H0rd, sn)
        return -0.5 * c if np.isfinite(c) else -np.inf
    rng = np.random.default_rng(RNG_SEED)
    center = np.array([out["z001"]["s0_best"], out["z001"]["M_B_best"], out["z001"]["H0rd_best"]])
    center = np.array([max(center[0], 0.05), center[1], center[2]])
    p0 = center + np.array([0.03, 0.01, 25.0]) * rng.normal(size=(NWALKERS, 3))
    p0[:, 0] = np.clip(p0[:, 0], 0.002, 0.95)
    p0[:, 1] = np.clip(p0[:, 1], -19.99, -18.01)
    p0[:, 2] = np.clip(p0[:, 2], 8001, 11999)
    sampler = emcee.EnsembleSampler(NWALKERS, 3, logp)
    sampler.run_mcmc(p0, NSTEPS)
    post = sampler.get_chain()[BURN:].reshape(-1, 3)
    df = pd.DataFrame(post, columns=["s0", "MB", "H0rd"])
    df.to_csv(RESULTS_DIR / "lcos_z001_post.csv", index=False)
    ul95 = float(np.percentile(df["s0"], 95))
    med = float(np.percentile(df["s0"], 50))
    out["z001_posterior"] = {"s0_median": med, "s0_95UL": ul95,
                             "acceptance": float(np.mean(sampler.acceptance_fraction))}
    print(f"  posterior(z001): s0 median {med:.3f}, 95% UL {ul95:.3f} "
          f"(published UL on the full file: {PUBLISHED['s0_95UL']})")

    d = out["z001"]["delta_chi2"]
    if BARS["pass_dchi2"][0] <= d <= BARS["pass_dchi2"][1] and ul95 <= BARS["pass_UL"]:
        verdict = "PASS: Paper 1 stands cleanly under the standard cut; no action."
    elif d <= BARS["note_dchi2_hi"] and ul95 <= BARS["note_UL"]:
        verdict = "ROBUSTNESS NOTE: record the shifted numbers; no retraction."
    else:
        verdict = "REASSESS: outside the pre-stated bars."
    out["verdict"] = verdict
    with open(RESULTS_DIR / "lcos_z001_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
