#!/usr/bin/env python3
"""
The gamma-CDM gate: the frozen six-model comparison that decides
whether the soft matter-exponent deformation becomes a paper or a note.

Canonical spec: ~/Desktop/MITv6/DESI Paper/Lcos Paper 2/gameplan.md
(sections 3-4, numerical thresholds and priors). This script implements
that spec and nothing else. Pure phenomenology; no external framework
language anywhere.

MODELS (flat unless stated; all carry nuisances M_B in [-20,-18],
H0rd in [8000,12000]; Omega_m flat [0,1]):
    LCDM      E^2 = Om (1+z)^3 + (1-Om)                       k=3
    gamma-CDM E^2 = Om (1+z)^gamma + (1-Om)                   k=4
    wCDM      E^2 = Om (1+z)^3 + (1-Om)(1+z)^(3(1+w0))        k=4
    CPL       f_DE = (1+z)^(3(1+w0+wa)) exp(-3 wa z/(1+z))    k=5
    BA        f_DE = (1+z)^(3(1+w0)) (1+z^2)^(3 wa/2)         k=5
    JBP       f_DE = (1+z)^(3(1+w0)) exp(1.5 wa z^2/(1+z)^2)  k=5
    oLCDM     E^2 = Om (1+z)^3 + Ok (1+z)^2 + (1-Om-Ok)       k=4  [diagnostic]
Priors frozen: gamma [2,4] primary, [0,5] robustness; w0 [-3,1];
wa [-3,2]; Ok [-0.5,0.5].

DATA VARIANTS: all (shipped Pantheon+ file) / z>0.01 / z>0.1, cuts by a
single index mask on the SN vector and covariance rows+columns; DESI
DR2 BAO (13 pts) in every variant. Counts N_all, N(z>0.01), N(z>0.1)
reported everywhere. Gate B is evaluated on the z>0.01 variant.

GATES (frozen numbers; A and B decide the paper):
  A: post-0.01: gamma_hat < 3 with 3 outside the 68% profile interval
     AND dchi2(gamma vs LCDM) <= -4; post-0.1: dchi2 <= -1, gamma_hat < 3.
  B: on z>0.01: dAIC(gamma-LCDM) < -2 AND AIC(gamma) <= min AIC(zoo)+2
     AND ln B (Savage-Dickey at gamma=3, primary prior) gives
     Delta lnZ(gamma vs LCDM) = -ln B01 > -1; -ln B01 < -3 auto-fail.
  C: on z>0.01: SN-only and BAO-only 68% profile intervals overlap, or
     both contain the joint gamma_hat.
  D: template image of the post-0.01 gamma best fit (noise-free BAO
     mock at the 13 DR2 points, CPL fit with DESI covariance weighting,
     Paper 1's template-bias pattern) has wa < 0; and the model-native
     effective rho_DE stays positive for z <= 1.5 at the best fit
     (secondary fiducial split Om=0.315 also reported).
  Curvature diagnostic: if oLCDM captures >= 70% of gamma-CDM's dchi2
     improvement on the same variant, flag "curvature costume".

REPRODUCTION GATES (must pass before anything else runs):
  LCDM on the all-variant reproduces 1772.456 within 0.5; gamma-CDM on
  the all-variant reproduces the prior finding (chi2 about 1763.8,
  dLCDM about -8.7) within 1.0 -- the identical model family found by
  the earlier free-OL diagnostic, reparameterized.

MODES: gates | fits | mcmc | verdict | all
OUTPUTS (results/): gamma_gate_fits.csv, gamma_gate_profile_<variant>.csv,
  gamma_gate_split.csv, gamma_gate_template.json, gamma_gate_mcmc_<prior>.csv,
  gamma_gate_evidence.json, gamma_gate_verdict.json, gamma_gate_manifest.json
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
from scipy.stats import gaussian_kde

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

C_LIGHT = 299792.458
H0_SN = 70.0
LCDM_REF_ALL = 1772.456
GAMMA_FINDING_CHI2 = 1763.8   # free-OL diagnostic corner, same family
REPRO_TOL_LCDM = 0.5
REPRO_TOL_GAMMA = 1.0

NWALKERS = 32
NSTEPS = 5000
BURN = 1000
RNG_SEED = 12345

PRIORS = {
    "Om": (0.0, 1.0), "MB": (-20.0, -18.0), "H0rd": (8000.0, 12000.0),
    "gamma_primary": (2.0, 4.0), "gamma_robust": (0.0, 5.0),
    "w0": (-3.0, 1.0), "wa": (-3.0, 2.0), "Ok": (-0.5, 0.5),
}
GAMMA_GRID = np.round(np.arange(2.0, 4.0 + 1e-9, 0.01), 4)

# ---------------- data ----------------
_pan = pd.read_csv(DATA_DIR / "pantheon_plus.csv")
Z_ALL = _pan["zHD"].to_numpy(float)
MB_ALL = _pan["m_b_corr"].to_numpy(float)
COV_ALL = np.load(DATA_DIR / "pantheon_plus_cov.npy")

bao = pd.read_csv(DATA_DIR / "desi_dr2_bao.csv")
z_bao = bao["z_eff"].to_numpy(float)
bao_values = bao["value"].to_numpy(float)
bao_obs = bao["observable"].to_numpy()
COV_BAO = np.load(DATA_DIR / "desi_dr2_bao_cov.npy")
INV_COV_BAO = np.linalg.inv(COV_BAO)

Z_GRID = np.linspace(0.0, max(float(Z_ALL.max()), float(z_bao.max())) * 1.002, 5000)


class SNVariant:
    def __init__(self, name, zmin):
        mask = Z_ALL > zmin if zmin > 0 else np.ones_like(Z_ALL, dtype=bool)
        self.name, self.zmin = name, zmin
        self.z = Z_ALL[mask]
        self.mb = MB_ALL[mask]
        cov = COV_ALL[np.ix_(mask, mask)]
        self.cho = cho_factor(cov, lower=True, check_finite=False)
        self.n = int(mask.sum())


VARIANTS = {v.name: v for v in [SNVariant("all", 0.0),
                                SNVariant("z001", 0.01),
                                SNVariant("z010", 0.10)]}

# ---------------- models ----------------
def e2_flat(z, Om, fde):
    return Om * (1 + z) ** 3 + (1 - Om) * fde


def model_e2(name, z, p):
    z = np.asarray(z, float)
    if name == "LCDM":
        Om, = p
        return Om * (1 + z) ** 3 + (1 - Om)
    if name == "gammaCDM":
        Om, g = p
        return Om * (1 + z) ** g + (1 - Om)
    if name == "wCDM":
        Om, w0 = p
        return e2_flat(z, Om, (1 + z) ** (3 * (1 + w0)))
    if name == "CPL":
        Om, w0, wa = p
        return e2_flat(z, Om, (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)))
    if name == "BA":
        Om, w0, wa = p
        return e2_flat(z, Om, (1 + z) ** (3 * (1 + w0)) * (1 + z ** 2) ** (1.5 * wa))
    if name == "JBP":
        Om, w0, wa = p
        return e2_flat(z, Om, (1 + z) ** (3 * (1 + w0)) * np.exp(1.5 * wa * z ** 2 / (1 + z) ** 2))
    if name == "oLCDM":
        Om, Ok = p
        return Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + (1 - Om - Ok)
    raise ValueError(name)


MODELS = {
    "LCDM":     {"shape": ["Om"],               "k": 3, "starts": [[0.30]]},
    "gammaCDM": {"shape": ["Om", "gamma"],      "k": 4, "starts": [[0.30, 3.0], [0.375, 2.84], [0.32, 2.6], [0.30, 3.3]]},
    "wCDM":     {"shape": ["Om", "w0"],         "k": 4, "starts": [[0.30, -1.0], [0.32, -0.9], [0.28, -1.1]]},
    "CPL":      {"shape": ["Om", "w0", "wa"],   "k": 5, "starts": [[0.30, -1.0, 0.0], [0.33, -0.7, -1.0], [0.30, -0.9, -0.4], [0.28, -1.1, 0.5]]},
    "BA":       {"shape": ["Om", "w0", "wa"],   "k": 5, "starts": [[0.30, -1.0, 0.0], [0.33, -0.7, -0.7], [0.30, -0.9, -0.3]]},
    "JBP":      {"shape": ["Om", "w0", "wa"],   "k": 5, "starts": [[0.30, -1.0, 0.0], [0.33, -0.7, -1.4], [0.30, -0.9, -0.5]]},
    "oLCDM":    {"shape": ["Om", "Ok"],         "k": 4, "starts": [[0.30, 0.0], [0.29, 0.05], [0.31, -0.05]]},
}
SHAPE_BOUNDS = {"Om": PRIORS["Om"], "gamma": PRIORS["gamma_primary"],
                "w0": PRIORS["w0"], "wa": PRIORS["wa"], "Ok": PRIORS["Ok"]}
ZOO = ["wCDM", "CPL", "BA", "JBP"]


def sinn(I, Ok):
    if abs(Ok) < 1e-8:
        return I
    if Ok > 0:
        s = np.sqrt(Ok)
        return np.sinh(s * I) / s
    s = np.sqrt(-Ok)
    return np.sin(s * I) / s


def chi2_parts(name, shape, MB, H0rd, sn: SNVariant, use_sn=True, use_bao=True):
    e2 = model_e2(name, Z_GRID, shape)
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        return np.inf, np.inf
    e = np.sqrt(e2)
    I = cumulative_trapezoid(1.0 / e, Z_GRID, initial=0.0)
    Ok = shape[1] if name == "oLCDM" else 0.0

    c_sn = 0.0
    if use_sn:
        Isn = np.interp(sn.z, Z_GRID, I)
        dl = (1 + sn.z) * (C_LIGHT / H0_SN) * sinn(Isn, Ok)
        if np.any(dl <= 0) or np.any(~np.isfinite(dl)):
            return np.inf, np.inf
        delta = sn.mb - MB - (5 * np.log10(dl) + 25)
        c_sn = float(delta @ cho_solve(sn.cho, delta, check_finite=False))

    c_bao = 0.0
    if use_bao:
        Ib = np.interp(z_bao, Z_GRID, I)
        Eb = np.interp(z_bao, Z_GRID, e)
        DM = C_LIGHT / H0rd * sinn(Ib, Ok)
        DH = C_LIGHT / H0rd / Eb
        DV = (z_bao * DM ** 2 * DH) ** (1 / 3)
        model = np.where(bao_obs == "DV_rd", DV, np.where(bao_obs == "DM_rd", DM, DH))
        delta = model - bao_values
        c_bao = float(delta @ INV_COV_BAO @ delta)

    return c_sn, c_bao


def total_chi2(name, theta, sn, use_sn=True, use_bao=True, gamma_bounds=None):
    spec = MODELS[name]
    ns = len(spec["shape"])
    shape, MB, H0rd = list(theta[:ns]), theta[ns], theta[ns + 1]
    for val, pname in zip(shape, spec["shape"]):
        lo, hi = (gamma_bounds if (pname == "gamma" and gamma_bounds) else SHAPE_BOUNDS[pname])
        if not (lo <= val <= hi):
            return np.inf
    if not (PRIORS["MB"][0] <= MB <= PRIORS["MB"][1] and PRIORS["H0rd"][0] <= H0rd <= PRIORS["H0rd"][1]):
        return np.inf
    c_sn, c_bao = chi2_parts(name, shape, MB, H0rd, sn, use_sn, use_bao)
    return c_sn + c_bao


def profile(name, sn, use_sn=True, use_bao=True, fix=None, gamma_bounds=None):
    """Deterministic multi-start optimum; fix = dict(paramname=value)."""
    spec = MODELS[name]
    names = spec["shape"] + ["MB", "H0rd"]
    fix = fix or {}

    def pack(free):
        full, j = [], 0
        for nm in names:
            if nm in fix:
                full.append(fix[nm])
            else:
                full.append(free[j]); j += 1
        return np.array(full)

    free_names = [nm for nm in names if nm not in fix]
    bounds = []
    for nm in free_names:
        if nm == "MB":
            bounds.append(PRIORS["MB"])
        elif nm == "H0rd":
            bounds.append(PRIORS["H0rd"])
        elif nm == "gamma" and gamma_bounds:
            bounds.append(gamma_bounds)
        else:
            bounds.append(SHAPE_BOUNDS[nm])

    fun = lambda x: total_chi2(name, pack(x), sn, use_sn, use_bao, gamma_bounds)
    best = None
    for s in spec["starts"]:
        start_full = dict(zip(spec["shape"], s)); start_full.update({"MB": -19.35, "H0rd": 10000.0})
        x0 = np.array([start_full[nm] for nm in free_names])
        r1 = minimize(fun, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-6})
        r2 = minimize(fun, r1.x, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-9})
        cand = r2 if r2.fun <= r1.fun else r1
        if best is None or cand.fun < best.fun:
            best = cand
    full = pack(np.array(best.x))
    return dict(zip(names, full)), float(best.fun)


def crossings(x, d, thr):
    ihat = int(np.argmin(d)); lo = hi = None
    for j in range(len(x) - 1):
        if (d[j] - thr) * (d[j + 1] - thr) < 0:
            xc = x[j] + (thr - d[j]) * (x[j + 1] - x[j]) / (d[j + 1] - d[j])
            if x[j] < x[ihat]:
                lo = xc
            elif hi is None:
                hi = xc
    return lo, hi


# ---------------- stages ----------------
def file_md5(p):
    h = hashlib.md5(); h.update(Path(p).read_bytes()); return h.hexdigest()


def write_manifest():
    manifest = {
        "script": "scripts/gamma_gate.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "-C", str(BASE), "rev-parse", "HEAD"], text=True).strip(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "~/Desktop/MITv6/DESI Paper/Lcos Paper 2/gameplan.md sections 3-4",
        "counts": {"N_all": VARIANTS["all"].n, "N_z>0.01": VARIANTS["z001"].n, "N_z>0.10": VARIANTS["z010"].n,
                   "N_BAO": len(bao)},
        "priors": PRIORS, "seed": RNG_SEED,
        "gateB_variant": "z001",
        "thresholds": {"A_dchi2_001": -4.0, "A_dchi2_010": -1.0, "B_dAIC": -2.0,
                       "B_zoo_margin": 2.0, "B_lnZ_floor": -1.0, "B_lnZ_autofail": -3.0,
                       "curvature_costume_frac": 0.70, "D_rhoDE_positive_to_z": 1.5},
        "data_md5": {p.name: file_md5(p) for p in sorted(DATA_DIR.glob("*")) if p.suffix in {".csv", ".npy"}},
    }
    with open(RESULTS_DIR / "gamma_gate_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["counts"], indent=2))


def run_repro_gates():
    ok = True
    _, chi2_l = profile("LCDM", VARIANTS["all"])
    d1 = chi2_l - LCDM_REF_ALL
    print(f"  repro LCDM(all): {chi2_l:.3f} vs {LCDM_REF_ALL} (diff {d1:+.3f}) "
          f"-> {'PASS' if abs(d1) <= REPRO_TOL_LCDM else 'FAIL'}")
    ok &= abs(d1) <= REPRO_TOL_LCDM
    pg, chi2_g = profile("gammaCDM", VARIANTS["all"])
    d2 = chi2_g - GAMMA_FINDING_CHI2
    print(f"  repro gammaCDM(all): {chi2_g:.3f} (gamma={pg['gamma']:.3f}, Om={pg['Om']:.3f}) "
          f"vs {GAMMA_FINDING_CHI2} (diff {d2:+.3f}) -> {'PASS' if abs(d2) <= REPRO_TOL_GAMMA else 'FAIL'}")
    ok &= abs(d2) <= REPRO_TOL_GAMMA
    with open(RESULTS_DIR / "gamma_gate_repro.json", "w") as f:
        json.dump({"lcdm_all": chi2_l, "gamma_all": chi2_g, "gamma_hat_all": pg["gamma"],
                   "Om_hat_all": pg["Om"], "pass": bool(ok)}, f, indent=2)
    print(f"REPRODUCTION GATES {'PASS' if ok else 'FAIL'}")
    return ok


def repro_passed():
    p = RESULTS_DIR / "gamma_gate_repro.json"
    return p.exists() and json.load(open(p)).get("pass", False)


def run_fits():
    if not repro_passed():
        sys.exit("fits refused: reproduction gates have not passed")
    rows = []
    for vname, sn in VARIANTS.items():
        N = sn.n + len(bao)
        base = None
        for mname in ["LCDM", "gammaCDM", "wCDM", "CPL", "BA", "JBP", "oLCDM"]:
            params, chi2 = profile(mname, sn)
            k = MODELS[mname]["k"]
            aic, bic = chi2 + 2 * k, chi2 + k * np.log(N)
            if mname == "LCDM":
                base = {"chi2": chi2, "aic": aic}
            rows.append({"variant": vname, "N_SN": sn.n, "model": mname, "k": k,
                         "chi2": chi2, "dchi2_vs_LCDM": chi2 - base["chi2"],
                         "AIC": aic, "dAIC_vs_LCDM": aic - base["aic"],
                         "BIC": bic, **{f"p_{a}": b for a, b in params.items()}})
            print(f"  [{vname}] {mname}: chi2={chi2:.3f} dchi2={chi2 - base['chi2']:+.3f} "
                  f"dAIC={aic - base['aic']:+.3f}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "gamma_gate_fits.csv", index=False)

    # gamma profile curves (68/95 intervals) on the cut variants
    for vname in ["z001", "z010"]:
        sn = VARIANTS[vname]
        rows = []
        for g in GAMMA_GRID:
            _, chi2 = profile("gammaCDM", sn, fix={"gamma": float(g)})
            rows.append({"gamma": float(g), "chi2": chi2})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"gamma_gate_profile_{vname}.csv", index=False)

    # SN-only / BAO-only gamma profiles on z001 (Gate C)
    sn = VARIANTS["z001"]
    rows = []
    for g in GAMMA_GRID:
        _, c_sn = profile("gammaCDM", sn, use_bao=False, fix={"gamma": float(g), "H0rd": 10000.0})
        _, c_bao = profile("gammaCDM", sn, use_sn=False, fix={"gamma": float(g), "MB": -19.35})
        rows.append({"gamma": float(g), "chi2_sn": c_sn, "chi2_bao": c_bao})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "gamma_gate_split.csv", index=False)

    # Gate D: template image + effective-fluid check, post-0.01 best fit
    pbest, _ = profile("gammaCDM", VARIANTS["z001"])
    Om_g, g_g = pbest["Om"], pbest["gamma"]
    e2 = model_e2("gammaCDM", Z_GRID, [Om_g, g_g])
    e = np.sqrt(e2); I = cumulative_trapezoid(1.0 / e, Z_GRID, initial=0.0)
    Ib = np.interp(z_bao, Z_GRID, I); Eb = np.interp(z_bao, Z_GRID, e)
    H0rd_m = pbest["H0rd"]
    DM = C_LIGHT / H0rd_m * Ib; DH = C_LIGHT / H0rd_m / Eb
    DV = (z_bao * DM ** 2 * DH) ** (1 / 3)
    mock = np.where(bao_obs == "DV_rd", DV, np.where(bao_obs == "DM_rd", DM, DH))

    def cpl_mock_chi2(x):
        Om, w0, wa, H0rd = x
        if not (0 < Om < 1 and -3 <= w0 <= 1 and -3 <= wa <= 2 and 8000 <= H0rd <= 12000):
            return np.inf
        e2c = model_e2("CPL", Z_GRID, [Om, w0, wa])
        if np.any(e2c <= 0):
            return np.inf
        ec = np.sqrt(e2c); Ic = cumulative_trapezoid(1.0 / ec, Z_GRID, initial=0.0)
        Icb = np.interp(z_bao, Z_GRID, Ic); Ecb = np.interp(z_bao, Z_GRID, ec)
        DMc = C_LIGHT / H0rd * Icb; DHc = C_LIGHT / H0rd / Ecb
        DVc = (z_bao * DMc ** 2 * DHc) ** (1 / 3)
        mc = np.where(bao_obs == "DV_rd", DVc, np.where(bao_obs == "DM_rd", DMc, DHc))
        d = mc - mock
        return float(d @ INV_COV_BAO @ d)

    best = None
    for s in [[Om_g, -1.0, 0.0, H0rd_m], [Om_g, -0.9, -0.5, H0rd_m], [0.315, -1.05, 0.3, H0rd_m]]:
        r = minimize(cpl_mock_chi2, np.array(s), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    w0_img, wa_img = float(best.x[1]), float(best.x[2])

    zchk = np.linspace(0, 1.5, 200)
    rho_native = model_e2("gammaCDM", zchk, [Om_g, g_g]) - Om_g * (1 + zchk) ** 3
    rho_fid = model_e2("gammaCDM", zchk, [Om_g, g_g]) - 0.315 * (1 + zchk) ** 3
    template = {"gamma_hat_z001": g_g, "Om_hat_z001": Om_g,
                "cpl_image": {"w0": w0_img, "wa": wa_img, "mock_fit_chi2": float(best.fun)},
                "rhoDE_native_min_z<=1.5": float(rho_native.min()),
                "rhoDE_fiducial_min_z<=1.5": float(rho_fid.min())}
    with open(RESULTS_DIR / "gamma_gate_template.json", "w") as f:
        json.dump(template, f, indent=2)
    print(json.dumps(template, indent=2))


def run_mcmc():
    if not repro_passed():
        sys.exit("mcmc refused: reproduction gates have not passed")
    import emcee
    sn = VARIANTS["z001"]
    out = {}
    for tag, gb in [("primary", PRIORS["gamma_primary"]), ("robust", PRIORS["gamma_robust"])]:
        def logp(theta):
            c = total_chi2("gammaCDM", theta, sn, gamma_bounds=gb)
            return -0.5 * c if np.isfinite(c) else -np.inf
        rng = np.random.default_rng(RNG_SEED)
        pbest, _ = profile("gammaCDM", sn, gamma_bounds=gb)
        center = np.array([pbest["Om"], pbest["gamma"], pbest["MB"], pbest["H0rd"]])
        p0 = center + np.array([0.01, 0.03, 0.01, 25.0]) * rng.normal(size=(NWALKERS, 4))
        p0[:, 0] = np.clip(p0[:, 0], 0.01, 0.99)
        p0[:, 1] = np.clip(p0[:, 1], gb[0] + 1e-3, gb[1] - 1e-3)
        p0[:, 2] = np.clip(p0[:, 2], -19.99, -18.01)
        p0[:, 3] = np.clip(p0[:, 3], 8001, 11999)
        sampler = emcee.EnsembleSampler(NWALKERS, 4, logp)
        sampler.run_mcmc(p0, NSTEPS, progress=True)
        post = sampler.get_chain()[BURN:].reshape(-1, 4)
        df = pd.DataFrame(post, columns=["Om", "gamma", "MB", "H0rd"])
        df.to_csv(RESULTS_DIR / f"gamma_gate_mcmc_{tag}.csv", index=False)

        gs = df["gamma"].to_numpy()
        prior_density = 1.0 / (gb[1] - gb[0])
        lnB = {}
        for bw in [0.01, 0.015, 0.02, 0.03]:
            kde = gaussian_kde(gs, bw_method=bw / gs.std())
            lnB[bw] = float(np.log(kde(3.0)[0] / prior_density))
        q = np.percentile(gs, [2.5, 16, 50, 84, 97.5])
        out[tag] = {"prior": list(gb), "gamma_median": float(q[2]),
                    "gamma_68": [float(q[1]), float(q[3])], "gamma_95": [float(q[0]), float(q[4])],
                    "lnB01_savage_dickey_at_gamma3_by_bw": lnB,
                    "lnB01_mean": float(np.mean(list(lnB.values()))),
                    "delta_lnZ_gamma_vs_LCDM": -float(np.mean(list(lnB.values()))),
                    "acceptance": float(np.mean(sampler.acceptance_fraction))}
        print(f"  mcmc[{tag}]: gamma = {q[2]:.3f} 68% [{q[1]:.3f},{q[3]:.3f}]  "
              f"lnB01 = {out[tag]['lnB01_mean']:+.2f}")
    with open(RESULTS_DIR / "gamma_gate_evidence.json", "w") as f:
        json.dump(out, f, indent=2)


def run_verdict():
    fits = pd.read_csv(RESULTS_DIR / "gamma_gate_fits.csv")
    ev = json.load(open(RESULTS_DIR / "gamma_gate_evidence.json"))
    tpl = json.load(open(RESULTS_DIR / "gamma_gate_template.json"))

    def row(v, m):
        return fits[(fits.variant == v) & (fits.model == m)].iloc[0]

    # Gate A
    a01 = row("z001", "gammaCDM"); a10 = row("z010", "gammaCDM")
    prof = pd.read_csv(RESULTS_DIR / "gamma_gate_profile_z001.csv")
    d = prof.chi2.to_numpy() - prof.chi2.min()
    lo68, hi68 = crossings(prof.gamma.to_numpy(), d, 1.0)
    ghat = float(prof.gamma[prof.chi2.idxmin()])
    A = (ghat < 3.0 and (hi68 is not None and hi68 < 3.0)
         and a01.dchi2_vs_LCDM <= -4.0
         and a10.dchi2_vs_LCDM <= -1.0 and row("z010", "gammaCDM").p_gamma < 3.0)

    # Gate B (z001)
    zoo_aic = min(row("z001", m).AIC for m in ZOO)
    dlnZ = ev["primary"]["delta_lnZ_gamma_vs_LCDM"]
    B = (a01.dAIC_vs_LCDM < -2.0 and row("z001", "gammaCDM").AIC <= zoo_aic + 2.0
         and dlnZ > -1.0)
    if dlnZ < -3.0:
        B = False

    # Gate C (z001 split)
    sp = pd.read_csv(RESULTS_DIR / "gamma_gate_split.csv")
    dsn = sp.chi2_sn.to_numpy() - sp.chi2_sn.min()
    dbao = sp.chi2_bao.to_numpy() - sp.chi2_bao.min()
    sn_lo, sn_hi = crossings(sp.gamma.to_numpy(), dsn, 1.0)
    bao_lo, bao_hi = crossings(sp.gamma.to_numpy(), dbao, 1.0)
    def iv(lo, hi):
        return (lo if lo is not None else GAMMA_GRID[0], hi if hi is not None else GAMMA_GRID[-1])
    s_iv, b_iv = iv(sn_lo, sn_hi), iv(bao_lo, bao_hi)
    overlap = not (s_iv[1] < b_iv[0] or b_iv[1] < s_iv[0])
    C = overlap or (s_iv[0] <= ghat <= s_iv[1] and b_iv[0] <= ghat <= b_iv[1])

    # Gate D
    D = (tpl["cpl_image"]["wa"] < 0.0) and (tpl["rhoDE_native_min_z<=1.5"] > 0.0)

    # curvature costume
    frac = row("z001", "oLCDM").dchi2_vs_LCDM / a01.dchi2_vs_LCDM if a01.dchi2_vs_LCDM < 0 else np.nan
    costume = bool(np.isfinite(frac) and frac >= 0.70)

    verdict = {
        "counts": {"N_all": VARIANTS["all"].n, "N_z>0.01": VARIANTS["z001"].n, "N_z>0.10": VARIANTS["z010"].n},
        "gateA": {"pass": bool(A), "gamma_hat_z001": ghat, "gamma_68_z001": [lo68, hi68],
                  "dchi2_z001": float(a01.dchi2_vs_LCDM), "dchi2_z010": float(a10.dchi2_vs_LCDM),
                  "dchi2_all_reference": float(row("all", "gammaCDM").dchi2_vs_LCDM)},
        "gateB": {"pass": bool(B), "dAIC_vs_LCDM_z001": float(a01.dAIC_vs_LCDM),
                  "AIC_gamma": float(a01.AIC), "min_AIC_zoo": float(zoo_aic),
                  "delta_lnZ_primary": dlnZ,
                  "delta_lnZ_robust": ev["robust"]["delta_lnZ_gamma_vs_LCDM"]},
        "gateC": {"pass": bool(C), "sn_68": list(s_iv), "bao_68": list(b_iv)},
        "gateD": {"pass": bool(D), **tpl["cpl_image"],
                  "rhoDE_native_min": tpl["rhoDE_native_min_z<=1.5"]},
        "curvature_costume_flag": costume, "oLCDM_capture_fraction": float(frac) if np.isfinite(frac) else None,
        "paper": bool(A and B),
        "tree": "A and B decide; C and D are strength modifiers; C-fail forces the tension-absorber discussion to lead",
    }
    with open(RESULTS_DIR / "gamma_gate_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    RESULTS_DIR.mkdir(exist_ok=True)
    if mode in ("gates", "all"):
        write_manifest()
        if not run_repro_gates():
            sys.exit("reproduction gates FAILED; run halted (technical, not physics)")
        if mode == "gates":
            return
    if mode in ("fits", "all"):
        run_fits()
    if mode in ("mcmc", "all"):
        run_mcmc()
    if mode in ("verdict", "all"):
        run_verdict()


if __name__ == "__main__":
    main()
