#!/usr/bin/env python3
"""
Clock-asymmetry (epsilon) family fit for the Lambda-cos pipeline.

Registration: mode-identity-theory files/framework/files/working/files/
clock-asymmetry-fit.md, frozen at commits dc8f9f6 (spec) and ff875be
(pre-run wording guard). This script implements that registration and
nothing else.

The family is the continuous closure of the Appendix A clock-exponent
models: dt/dtau = S^n with

    n(eps) = -1/2 - eps

so that

    H^2(S) prop. (1 - S^2) S^(2n-2),   S = s0/(1+z)

becomes, normalized at z = 0 and closed with Omega_Lambda = 0.685,

    E^2 = Om/(1-s0^2) * [ (1+z)^(3+2eps) - s0^2 (1+z)^(1+2eps) ] + OL.

Slice dictionary (frozen): eps = -3/2, -1/2, 0, +1/2 are Models C, A,
D, B of fit_clock_exponents.py; eps = +1 is the registered completing
fit (no published anchor; it is part of Tier 1, not a gate).

Model functions, data handling, optimizer, and sampler conventions are
mirrored verbatim from scripts/fit_clock_exponents.py (paths made
robust to the working directory). The target exponent enters only
through n(eps); nothing else is new.

Modes (run from anywhere):
    python scripts/fit_clock_asymmetry.py gates    # slice reproduction only
    python scripts/fit_clock_asymmetry.py tier1    # profile Delta-chi2(eps) grid
    python scripts/fit_clock_asymmetry.py tier2    # joint 4-parameter MCMC
    python scripts/fit_clock_asymmetry.py manifest # write run manifest only

GATES (frozen; must pass before tier1/tier2 may run):
    For each anchored slice, the deterministic profile optimum must
    reproduce the published Delta-chi2 (tables/clock_exponent_appendix_A_fits.csv)
    within tol = max(0.5, 0.001 * |published|), absolute chi-square units.
    The deterministic optimizer may land slightly below a published
    chain-argmax value; that direction is expected and passes within tol.
    A failure outside tol stops everything: the failure is the result.

Sampled parameters in Tier 2: (s0, H0rd, M_B, eps) -- four live
parameters. M_B is sampled, not analytically marginalized, exactly as
in the baseline scripts. H0 appears only through H0rd (BAO) and M_B
(SN), as in the baseline. Omega_Lambda = 0.685 is FIXED; the later
free-Omega_Lambda scan is robustness, not authority, and is not
implemented here.

Both references are reported everywhere:
    delta_lcdm = chi2 - 1772.456        (LCDM baseline of the pipeline)
    delta_eps0 = chi2 - chi2(eps = 0)   (the symmetric-kernel point)

Outputs (results/):
    clock_asymmetry_gates.csv
    clock_asymmetry_tier1_curve.csv
    clock_asymmetry_tier1_summary.json
    clock_asymmetry_tier2_chain.npy
    clock_asymmetry_tier2_postburn.csv
    clock_asymmetry_tier2_summary.json
    clock_asymmetry_manifest.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# -----------------------------
# Configuration (frozen)
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
TABLES_DIR = BASE / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_LAMBDA = 0.685
OMEGA_M = 1.0 - OMEGA_LAMBDA
LCDM_CHI2_MIN = 1772.456

C_LIGHT = 299792.458  # km/s
H0_SN = 70.0          # absorbed into M_B for SN likelihood

NWALKERS = 32
NSTEPS = 5000
BURN = 1000
RNG_SEED = 12345

EPS_PRIOR = (-1.6, 1.1)          # flat prior, frozen in the registration
EPS_GRID_COARSE = 0.01           # step over the full prior range
EPS_GRID_FINE = 0.002            # step over the refinement window
EPS_FINE_WINDOW = (-0.12, 0.12)  # refinement window around the symmetric point

GATE_SLICES = {"C": -1.5, "A": -0.5, "D": 0.0, "B": 0.5}  # eps values
GATE_TOL_ABS = 0.5
GATE_TOL_REL = 0.001

REGISTRATION_REFS = {
    "registration_repo": "dmobius3/mode-identity-theory",
    "registration_file": "files/framework/files/working/files/clock-asymmetry-fit.md",
    "registration_commits": ["dc8f9f6", "ff875be"],
}


def n_of_eps(eps: float) -> float:
    return -0.5 - eps


# -----------------------------
# Data loading (mirrored from fit_clock_exponents.py)
# -----------------------------
pantheon = pd.read_csv(DATA_DIR / "pantheon_plus.csv")
z_sn = pantheon["zHD"].to_numpy(float)
m_b_corr = pantheon["m_b_corr"].to_numpy(float)

cov_sn = np.load(DATA_DIR / "pantheon_plus_cov.npy")
cho_sn = cho_factor(cov_sn, lower=True, check_finite=False)

bao = pd.read_csv(DATA_DIR / "desi_dr2_bao.csv")
z_bao = bao["z_eff"].to_numpy(float)
bao_values = bao["value"].to_numpy(float)
cov_bao = np.load(DATA_DIR / "desi_dr2_bao_cov.npy")
inv_cov_bao = np.linalg.inv(cov_bao)

zmax = max(float(np.max(z_sn)), float(np.max(z_bao))) * 1.002
z_grid = np.linspace(0.0, zmax, 5000)


# -----------------------------
# Model functions (mirrored; eps enters only via n_of_eps)
# -----------------------------
def e2_clock_model(z: np.ndarray, s0: float, n_exp: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if not (0.0 < s0 < 1.0):
        return np.full_like(z, np.nan)

    S = s0 / (1.0 + z)
    ratio = ((1.0 - S**2) * S**(2.0 * n_exp - 2.0)) / (
        (1.0 - s0**2) * s0**(2.0 * n_exp - 2.0)
    )
    return OMEGA_M * ratio + OMEGA_LAMBDA


def get_e_and_integral(s0: float, n_exp: float):
    e2 = e2_clock_model(z_grid, s0, n_exp)
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        return None, None
    e = np.sqrt(e2)
    integral = cumulative_trapezoid(1.0 / e, z_grid, initial=0.0)
    return e, integral


def sn_chi2(s0: float, n_exp: float, M_B: float) -> float:
    e, integral = get_e_and_integral(s0, n_exp)
    if e is None:
        return np.inf

    I = np.interp(z_sn, z_grid, integral)
    d_l_mpc = (1.0 + z_sn) * (C_LIGHT / H0_SN) * I
    if np.any(d_l_mpc <= 0) or np.any(~np.isfinite(d_l_mpc)):
        return np.inf

    mu_model = 5.0 * np.log10(d_l_mpc) + 25.0
    delta = m_b_corr - M_B - mu_model
    return float(delta @ cho_solve(cho_sn, delta, check_finite=False))


def bao_chi2(s0: float, H0rd: float, n_exp: float) -> float:
    e, integral = get_e_and_integral(s0, n_exp)
    if e is None:
        return np.inf

    I = np.interp(z_bao, z_grid, integral)
    E_bao = np.interp(z_bao, z_grid, e)

    D_M = C_LIGHT / H0rd * I
    D_H = C_LIGHT / H0rd / E_bao
    D_V = (z_bao * D_M**2 * D_H) ** (1.0 / 3.0)

    model = np.empty(len(bao), dtype=float)
    for i, obs in enumerate(bao["observable"]):
        if obs == "DV_rd":
            model[i] = D_V[i]
        elif obs == "DM_rd":
            model[i] = D_M[i]
        elif obs == "DH_rd":
            model[i] = D_H[i]
        else:
            raise ValueError(f"Unknown BAO observable: {obs}")

    if np.any(~np.isfinite(model)):
        return np.inf
    delta = model - bao_values
    return float(delta @ inv_cov_bao @ delta)


def total_chi2_at(theta3: np.ndarray, n_exp: float) -> float:
    s0, H0rd, M_B = theta3
    if not (0.001 <= s0 <= 0.99 and 8000.0 <= H0rd <= 12000.0 and -20.0 <= M_B <= -18.0):
        return np.inf
    c_sn = sn_chi2(s0, n_exp, M_B)
    if not np.isfinite(c_sn):
        return np.inf
    return c_sn + bao_chi2(s0, H0rd, n_exp)


# -----------------------------
# Deterministic profile optimum (mirrored)
# -----------------------------
def best_initial_point(n_exp: float):
    starts = [
        np.array([0.05, 10000.0, -19.35]),
        np.array([0.20, 10000.0, -19.35]),
        np.array([0.50, 9500.0, -19.30]),
        np.array([0.85, 9300.0, -19.25]),
        np.array([0.95, 9200.0, -19.10]),
    ]
    bounds = [(0.001, 0.99), (8000.0, 12000.0), (-20.0, -18.0)]

    best = None
    for start in starts:
        res = minimize(
            lambda x: total_chi2_at(x, n_exp),
            start,
            method="Nelder-Mead",
            options={"maxiter": 1200, "xatol": 1e-8, "fatol": 1e-6},
        )
        res2 = minimize(
            lambda x: total_chi2_at(x, n_exp),
            res.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1200, "ftol": 1e-9},
        )
        candidate = res2 if res2.fun <= res.fun else res
        if best is None or candidate.fun < best.fun:
            best = candidate

    x = np.array(best.x, dtype=float)
    x[0] = np.clip(x[0], 0.001, 0.99)
    x[1] = np.clip(x[1], 8000.0, 12000.0)
    x[2] = np.clip(x[2], -20.0, -18.0)
    return x, float(total_chi2_at(x, n_exp))


def profile_at_eps(eps: float):
    x, chi2 = best_initial_point(n_of_eps(eps))
    return {
        "eps": eps,
        "n": n_of_eps(eps),
        "s0": x[0],
        "H0rd": x[1],
        "M_B": x[2],
        "chi2_total": chi2,
        "delta_lcdm": chi2 - LCDM_CHI2_MIN,
    }


# -----------------------------
# Gates: slice reproduction against the published table
# -----------------------------
def run_gates() -> bool:
    anchors = pd.read_csv(TABLES_DIR / "clock_exponent_appendix_A_fits.csv")
    anchors = anchors.set_index("Model")

    rows = []
    all_pass = True
    d_row_chi2 = None

    for model, eps in GATE_SLICES.items():
        published = float(anchors.loc[model, "Delta_chi2_vs_LCDM_1772p456"])
        rep = profile_at_eps(eps)
        if model == "D":
            d_row_chi2 = rep["chi2_total"]
        tol = max(GATE_TOL_ABS, GATE_TOL_REL * abs(published))
        diff = rep["delta_lcdm"] - published
        ok = abs(diff) <= tol
        all_pass &= ok
        rows.append({
            "model": model,
            "eps": eps,
            "n": rep["n"],
            "published_delta_lcdm": published,
            "reproduced_delta_lcdm": rep["delta_lcdm"],
            "diff": diff,
            "tolerance": tol,
            "pass": bool(ok),
            "s0": rep["s0"],
            "H0rd": rep["H0rd"],
            "M_B": rep["M_B"],
        })
        print(f"  gate {model} (eps={eps:+.1f}): published {published:+.2f}, "
              f"reproduced {rep['delta_lcdm']:+.2f}, diff {diff:+.3f}, "
              f"tol {tol:.3f} -> {'PASS' if ok else 'FAIL'}")

    out = pd.DataFrame(rows)
    if d_row_chi2 is not None:
        out["delta_eps0"] = out["reproduced_delta_lcdm"] - (d_row_chi2 - LCDM_CHI2_MIN)
    out.to_csv(RESULTS_DIR / "clock_asymmetry_gates.csv", index=False)
    print(f"\nGATES {'PASS' if all_pass else 'FAIL'}. "
          f"Saved: {RESULTS_DIR / 'clock_asymmetry_gates.csv'}")
    return all_pass


def gates_passed() -> bool:
    path = RESULTS_DIR / "clock_asymmetry_gates.csv"
    if not path.exists():
        return False
    return bool(pd.read_csv(path)["pass"].all())


# -----------------------------
# Tier 1: profile curve in eps
# -----------------------------
def eps_grid() -> np.ndarray:
    lo, hi = EPS_PRIOR
    coarse = np.arange(lo, hi + 1e-12, EPS_GRID_COARSE)
    fine = np.arange(EPS_FINE_WINDOW[0], EPS_FINE_WINDOW[1] + 1e-12, EPS_GRID_FINE)
    grid = np.unique(np.round(np.concatenate([coarse, fine]), 6))
    return grid


def interval_crossings(eps_vals, delta_min_curve, threshold):
    """Linear-interpolated crossings of delta-above-minimum through a threshold."""
    lo_edge, hi_edge = None, None
    d = delta_min_curve
    for i in range(len(eps_vals) - 1):
        a, b = d[i], d[i + 1]
        if (a - threshold) * (b - threshold) < 0:
            x = eps_vals[i] + (threshold - a) * (eps_vals[i + 1] - eps_vals[i]) / (b - a)
            if eps_vals[i] < eps_vals[np.argmin(d)]:
                lo_edge = x if lo_edge is None else max(lo_edge, x)
            else:
                hi_edge = x if hi_edge is None else min(hi_edge, x)
    return lo_edge, hi_edge


def run_tier1():
    if not gates_passed():
        sys.exit("Tier 1 refused: gates have not passed. Run 'gates' first.")

    grid = eps_grid()
    print(f"Tier 1: {len(grid)} grid points on [{EPS_PRIOR[0]}, {EPS_PRIOR[1]}]")
    rows = []
    for k, eps in enumerate(grid):
        rep = profile_at_eps(float(eps))
        rows.append(rep)
        if k % 20 == 0:
            print(f"  [{k+1}/{len(grid)}] eps={eps:+.3f}  "
                  f"delta_lcdm={rep['delta_lcdm']:+.2f}")

    curve = pd.DataFrame(rows)
    chi2_eps0 = float(curve.loc[np.isclose(curve["eps"], 0.0), "chi2_total"].iloc[0])
    curve["delta_eps0"] = curve["chi2_total"] - chi2_eps0
    curve.to_csv(RESULTS_DIR / "clock_asymmetry_tier1_curve.csv", index=False)

    chi2_min = float(curve["chi2_total"].min())
    eps_hat = float(curve.loc[curve["chi2_total"].idxmin(), "eps"])
    dmin = (curve["chi2_total"] - chi2_min).to_numpy()
    e_vals = curve["eps"].to_numpy()

    lo68, hi68 = interval_crossings(e_vals, dmin, 1.0)
    lo95, hi95 = interval_crossings(e_vals, dmin, 3.84)

    summary = {
        "eps_hat": eps_hat,
        "chi2_min": chi2_min,
        "delta_lcdm_at_min": chi2_min - LCDM_CHI2_MIN,
        "chi2_eps0": chi2_eps0,
        "delta_eps0_at_min": chi2_min - chi2_eps0,
        "interval_68": [lo68, hi68],
        "interval_95": [lo95, hi95],
        "halfwidth_68_lower": None if lo68 is None else eps_hat - lo68,
        "halfwidth_68_upper": None if hi68 is None else hi68 - eps_hat,
        "meaningful_measurement_bar": 0.05,
        "grid": {"coarse": EPS_GRID_COARSE, "fine": EPS_GRID_FINE,
                 "fine_window": EPS_FINE_WINDOW, "prior": EPS_PRIOR},
    }
    with open(RESULTS_DIR / "clock_asymmetry_tier1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# -----------------------------
# Tier 2: joint 4-parameter MCMC (s0, H0rd, M_B, eps)
# -----------------------------
def log_prob4(theta: np.ndarray) -> float:
    s0, H0rd, M_B, eps = theta
    if not (EPS_PRIOR[0] <= eps <= EPS_PRIOR[1]):
        return -np.inf
    chi2 = total_chi2_at(np.array([s0, H0rd, M_B]), n_of_eps(eps))
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def run_tier2():
    if not gates_passed():
        sys.exit("Tier 2 refused: gates have not passed. Run 'gates' first.")

    import emcee  # lazy: gates and tier1 do not need it

    rng = np.random.default_rng(RNG_SEED)
    center3, chi2_eps0 = best_initial_point(n_of_eps(0.0))
    center = np.array([center3[0], center3[1], center3[2], 0.0])
    scales = np.array([0.01, 25.0, 0.01, 0.05])
    p0 = center + scales * rng.normal(size=(NWALKERS, 4))
    p0[:, 0] = np.clip(p0[:, 0], 0.001, 0.99)
    p0[:, 1] = np.clip(p0[:, 1], 8000.0, 12000.0)
    p0[:, 2] = np.clip(p0[:, 2], -20.0, -18.0)
    p0[:, 3] = np.clip(p0[:, 3], EPS_PRIOR[0] + 1e-3, EPS_PRIOR[1] - 1e-3)

    sampler = emcee.EnsembleSampler(NWALKERS, 4, log_prob4)
    sampler.run_mcmc(p0, NSTEPS, progress=True)

    chain = sampler.get_chain()
    np.save(RESULTS_DIR / "clock_asymmetry_tier2_chain.npy", chain)
    post = chain[BURN:].reshape(-1, 4)
    df = pd.DataFrame(post, columns=["s0", "H0rd", "M_B", "eps"])
    df.to_csv(RESULTS_DIR / "clock_asymmetry_tier2_postburn.csv", index=False)

    logp = np.array([log_prob4(t) for t in post])
    best = post[np.argmax(logp)]
    chi2_best = -2.0 * float(np.max(logp))

    eps_samples = df["eps"].to_numpy()
    q = np.percentile(eps_samples, [2.5, 16, 50, 84, 97.5])
    summary = {
        "sampled_parameters": ["s0", "H0rd", "M_B", "eps"],
        "eps_median": float(q[2]),
        "eps_68": [float(q[1]), float(q[3])],
        "eps_95": [float(q[0]), float(q[4])],
        "halfwidth_68_lower": float(q[2] - q[1]),
        "halfwidth_68_upper": float(q[3] - q[2]),
        "meaningful_measurement_bar": 0.05,
        "corr_s0_eps": float(np.corrcoef(df["s0"], df["eps"])[0, 1]),
        "best_point": {"s0": float(best[0]), "H0rd": float(best[1]),
                       "M_B": float(best[2]), "eps": float(best[3])},
        "chi2_best": chi2_best,
        "delta_lcdm_best": chi2_best - LCDM_CHI2_MIN,
        "delta_eps0_best": chi2_best - chi2_eps0,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "nwalkers": NWALKERS, "nsteps": NSTEPS, "burn": BURN, "seed": RNG_SEED,
    }
    with open(RESULTS_DIR / "clock_asymmetry_tier2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# -----------------------------
# Manifest
# -----------------------------
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    import scipy
    manifest = {
        "script": "scripts/fit_clock_asymmetry.py",
        "lambda_cos_commit": commit,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": RNG_SEED,
        "omega_lambda_fixed": OMEGA_LAMBDA,
        "lcdm_reference_chi2": LCDM_CHI2_MIN,
        "eps_prior": EPS_PRIOR,
        "gate_slices": GATE_SLICES,
        "gate_tolerance": {"abs": GATE_TOL_ABS, "rel": GATE_TOL_REL},
        "grid": {"coarse": EPS_GRID_COARSE, "fine": EPS_GRID_FINE,
                 "fine_window": EPS_FINE_WINDOW},
        "sampled_parameters_tier2": ["s0", "H0rd", "M_B", "eps"],
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
        "data_md5": {p.name: file_md5(p) for p in sorted(DATA_DIR.glob("*"))
                     if p.suffix in {".csv", ".npy"}},
        **REGISTRATION_REFS,
    }
    with open(RESULTS_DIR / "clock_asymmetry_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'clock_asymmetry_manifest.json'}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gates"
    if mode == "gates":
        write_manifest()
        ok = run_gates()
        sys.exit(0 if ok else 1)
    elif mode == "tier1":
        run_tier1()
    elif mode == "tier2":
        run_tier2()
    elif mode == "all":
        write_manifest()
        if not run_gates():
            sys.exit("Gates failed; tiers refused.")
        run_tier1()
        run_tier2()
    elif mode == "manifest":
        write_manifest()
    else:
        sys.exit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
