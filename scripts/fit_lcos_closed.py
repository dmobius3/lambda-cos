"""
Λcos with a closed transverse distance at a fixed curvature radius.

The rate is the published Λcos fit's, unchanged (fit_lcos.py, Ω_Λ = 0.685):
    E^2(z) = (1-Ω_Λ)/(1-s0^2) [(1+z)^3 - s0^2 (1+z)] + Ω_Λ,      I(z) = ∫_0^z dz'/E(z'),
and only the transverse distance changes, at u = 1/ζ^2 with ζ = R H0 / c:
    D_M = (c/H0) ζ sin(I/ζ),   D_H = c/(H0 E),   D_V = (z D_M^2 D_H)^(1/3),   D_L = (1+z) D_M.
u = 0 is fit_lcos.py's model. Today's curvature radius equals R.

The radii, the two H0 rows, the bars, the gates and their tolerances follow an external
registration. This script implements them and refuses to evaluate u > 0 on the measured
data until the post-freeze pin check (G6b) passes on the registered tag.

From scripts/:
    python fit_lcos_closed.py qualify --out DIR
        G1 to G5 and G6a, their mutation arms, and a probe of G6b on temporary local tags.
        The measured data meet only the flat limit (G1, G2). Nothing is written into the repo.
    python fit_lcos_closed.py run --out DIR --registration-repo PATH --registration-commit SHA \
            --registration-sha256 HEX
        The environment check and G6b: the tag closed-distance-v1.0 must resolve to this checkout, and the
        registration, fetched afresh into PATH, must sit on its origin/main at SHA, have SHA-256 HEX, and
        name this commit, this script's SHA-256 and the tag. Then G1 to G5 again, then the comparison on
        the measured data, once.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-threaded BLAS: deterministic sums, one process per core
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True              # keeps the checkout clean for G6a

import argparse, contextlib, hashlib, io, json, platform, shutil, subprocess, tempfile, time
from multiprocessing import Pool
import numpy as np, pandas as pd, scipy

def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

# ---------------- registered terms ----------------
OMEGA_L = 0.685
C_LIGHT = 299792.458                        # km/s
H0_SN = 70.0                                # absorbed into M_B, as in fit_lcos.py
N_GRID = 4000                               # fit_lcos.py's grid on [0, 2.5]
Z_MAX = 2.5
S0_GRID = np.concatenate([[0.001], np.round(np.arange(1, 99) * 0.01, 2), [0.99]])
S0_BOX, H0RD_BOX, MB_BOX = (0.001, 0.99), (8000.0, 12000.0), (-20.0, -18.0)   # strict box, closed
U_GRID = np.round(np.arange(1001) * 0.001, 3)                                # u = 1/zeta^2 on [0, 1]
RADII_GPC = {"A": 6.13, "C": 10.5, "B": 19.7}
H0_ROWS = {"primary": 67.36, "robustness": 73.04}
SYN_S0 = (0.001, 0.1)
BAR_DISFAVORED, BAR_EXCLUDED = 9.0, 25.0
REGION_LEVEL = 2.71
PREFERENCE_BAR = 9.0
INTERIOR_CUT = 1e-6                         # a minimum counts as interior only this far below the flat value
TAG = "closed-distance-v1.0"                # the registered tag; the run accepts no other
REG_PATH = "files/framework/files/working/files/nonflat-distance-comparison.md"
BASE = (N_GRID, OMEGA_L, 1.0 / 3.0, "sin", False)     # (grid, Ω_Λ, D_V exponent, transverse, curvature in E)

# Tolerances. Qualification checks that each sits at least ten times above its measured floor
# (FLOOR_MARGIN), and its record lists both.
FLOOR_MARGIN = 10.0
ENV_PINNED = {"python": "3.9.6", "numpy": "2.0.2", "scipy": "1.13.1", "pandas": "2.3.3"}
INVALID = 1e30                              # chi^2 of a model with no valid rate or distance (arms only)
TOL = {
    "G1_rel": 1e-9,
    "G2_chi2": 0.01, "G2_s0sq": 1e-5, "G2_h0rd": 1.0, "G2_mb": 1e-3,
    "G3_below": 1e-6, "G3_forecast_rel": 0.005,
    "G4_zeta_rel": 1e-4, "G4_s0sq": 1e-5, "G4_chi2": 1e-6,
    "G5_dchi2": 0.01, "G5_zeta_rel": 1e-3, "G5_s0sq": 1e-5,
    "R_bound_rel": 1e-3, "R_zeta_rel": 1e-4,
}

# SHA-256 pins, computed from the files when this script was built
PINS = {
    "data/pantheon_plus.csv": "46a9334d15f7115435f662038692061a06fa36ba87e61e85bc17f5ac86c439a8",
    "data/pantheon_plus_cov.npy": "5c1b855c02e6a96ca8aa651e5e645af2db3b5f15f0c3647ca93c10ca27f8197e",
    "data/desi_dr2_bao.csv": "c746badffdef7f5e6b475d075f55a4de12173e991c94cb3a00eb498b9b5c254a",
    "data/desi_dr2_bao_cov.npy": "4d6ce177d7e96fb63199af88c2688a926dbb166acd61dc9f75dc0da7b4825c03",
    "results/lcos_summary.json": "b234fd104fedce1f333ec9b058448694ed945ea8ff1f86cdb11ef488fc2ac2bd",
    "data/closed_distance_synthetic.json": "80c4f32c43accefa2992abfdea72dc5030ac8df53d6fa74ff16389583b12c68d",
    "data/closed_distance_forecast.json": "2e16bd4f9a92232e8b0e82f1151408d2a29a3aaa05ace4adcd43ff8235a9e5ef",
}
SPAN_SHA256 = "75d1089fb20328245a1ea1122b3844ac834af0e8e2d1f6cfa35b48870c788e5d"                    # fit_lcos.py, its lines before the MCMC marker
MCMC_MARKER = "# ---------------- MCMC"

# ---------------- data ----------------
_DATA = {}

def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

_COV = {}

def _sn_cov():
    """The supernova covariance's Cholesky factor, shared by every data set (they differ only in values)."""
    if not _COV:
        C = np.load(os.path.join(DATA, "pantheon_plus_cov.npy"))
        cfac = cho_factor(C)
        c1 = cho_solve(cfac, np.ones(C.shape[0]))
        _COV.update(cfac=cfac, c1=c1, s11=float(np.ones(C.shape[0]) @ c1))
    return _COV

def _measured():
    sn = pd.read_csv(os.path.join(DATA, "pantheon_plus.csv"))
    bao = pd.read_csv(os.path.join(DATA, "desi_dr2_bao.csv"))
    return (sn.zHD.values, sn.m_b_corr.values, bao.z_eff.values, bao.observable.values, bao.value.values,
            np.load(os.path.join(DATA, "desi_dr2_bao_cov.npy")))

def load(key):
    """key: 'measured', 'measured-noDV' (G2's arm), a synthetic set name, or 'flat+zeta3' (G3's arm)."""
    if key in _DATA:
        return _DATA[key]
    z_sn, m, bz, bobs, bval, Cb = _measured()
    if key == "measured-noDV":
        keep = np.arange(1, len(bz))
        assert bobs[0] == "DV_rd"
        bz, bobs, bval, Cb = bz[keep], bobs[keep], bval[keep], Cb[np.ix_(keep, keep)]
    elif key != "measured":
        syn = read_json(os.path.join(DATA, "closed_distance_synthetic.json"))
        base = "flat" if key == "flat+zeta3" else key
        bval, m = np.array(syn[base + "_bao"]), np.array(syn[base + "_sn"])
        if key == "flat+zeta3":         # a closed signal at zeta = 3 injected into a copy of the flat set
            bf = read_json(os.path.join(REPO, "results", "lcos_summary.json"))["best_fit"]
            tmp = {"z_sn": z_sn, "bz": bz, "bobs": bobs}
            f0, mu0 = predict(BASE, tmp, bf["s0"], 0.0)
            f3, mu3 = predict(BASE, tmp, bf["s0"], 1.0 / 9.0)
            bval = bval + (C_LIGHT / bf["H0rd"]) * (f3 - f0)
            m = m + (mu3 - mu0)
    d = {"z_sn": z_sn, "m": m, "bz": bz, "bobs": bobs, "y": bval, "Cbi": np.linalg.inv(Cb), **_sn_cov()}
    _DATA[key] = d
    return d

# ---------------- model ----------------
def E_of(cfg, z, s0, u=0.0):
    N, OL, dv, kind, curv = cfg
    with np.errstate(invalid="ignore"):
        return np.sqrt(_E2(cfg, z, s0, u))

def _E2(cfg, z, s0, u):
    N, OL, dv, kind, curv = cfg
    E2 = (1 - OL) / (1 - s0**2) * (1 + z)**3 - (1 - OL) * s0**2 / (1 - s0**2) * (1 + z) + OL
    if curv:                             # G4's arm only: a curvature term put into the rate
        E2 = E2 - u * (1 + z)**2
    return E2

def transverse(I, u, kind):
    if u == 0.0:
        return I
    r = np.sqrt(u)
    return (np.sin(I * r) if kind == "sin" else np.sinh(I * r)) / r

_EI = {}
_GRIDSET = set(float(v) for v in S0_GRID)

def _E_I(cfg, s0, u):
    N, OL, dv, kind, curv = cfg
    key = (N, OL, float(s0))
    if not curv and key in _EI:
        return _EI[key]
    z = np.linspace(0, Z_MAX, N)
    E = E_of(cfg, z, s0, u)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (z, E, cumulative_trapezoid(1 / E, z, initial=0))
    if not curv and float(s0) in _GRIDSET:
        _EI[key] = out
    return out

def predict(cfg, d, s0, u):
    """Dimensionless BAO shapes f (times c/(H0 r_d) gives r_d units) and the SN modulus mu,
    the modulus written as fit_lcos.py writes it: 5 log10[(1+z)(c/70) D] + 25."""
    N, OL, dv, kind, curv = cfg
    z, E, I = _E_I(cfg, s0, u)
    Iz, Ez = np.interp(d["bz"], z, I), np.interp(d["bz"], z, E)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm, dh = transverse(Iz, u, kind), 1 / Ez
    f = np.array([(zz * a**2 * b)**dv if o == "DV_rd" else (a if o == "DM_rd" else b)
                  for zz, a, b, o in zip(d["bz"], dm, dh, d["bobs"])])
    Is = np.interp(d["z_sn"], z, I)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = 5 * np.log10((1 + d["z_sn"]) * (C_LIGHT / H0_SN) * transverse(Is, u, kind)) + 25
    return f, mu

def chi2_profiled(d, F, MU):
    """F: BAO shapes (n_bao x K), MU: SN moduli (n_sn x K). M_B and c/(H0 r_d) profiled analytically,
    each clipped to the box. Returns (total, SN, BAO, H0rd, M_B), each of length K. A column whose rate or
    distances are not finite and positive (only an arm's model can produce one) gets chi^2 = INVALID."""
    ok = np.all(np.isfinite(F) & (F > 0), axis=0) & np.all(np.isfinite(MU), axis=0)
    out = [np.full(F.shape[1], INVALID), np.full(F.shape[1], INVALID), np.full(F.shape[1], INVALID),
           np.full(F.shape[1], np.nan), np.full(F.shape[1], np.nan)]
    if ok.any():
        for o, v in zip(out, _chi2_valid(d, F[:, ok], MU[:, ok])):
            o[ok] = v
    return tuple(out)

def _chi2_valid(d, F, MU):
    y, Cbi = d["y"], d["Cbi"]
    A = np.einsum("ik,ij,j->k", F, Cbi, y) / np.einsum("ik,ij,jk->k", F, Cbi, F)
    A = np.clip(A, C_LIGHT / H0RD_BOX[1], C_LIGHT / H0RD_BOX[0])
    R = A * F - y[:, None]
    cb = np.einsum("ik,ij,jk->k", R, Cbi, R)
    D = d["m"][:, None] - MU
    MB = np.clip(d["c1"] @ D / d["s11"], MB_BOX[0], MB_BOX[1])
    RS = D - MB
    cs = np.sum(RS * cho_solve(d["cfac"], RS), axis=0)
    return cs + cb, cs, cb, C_LIGHT / A, MB

def _chi2_at(cfg, d, s0, u):
    f, mu = predict(cfg, d, s0, u)
    return chi2_profiled(d, f[:, None], mu[:, None])

def profile(cfg, key, u, authorized=False):
    """Profile over (s0, H0 r_d, M_B) at u: s0 on its grid, then bounded refinement between the best
    grid point's neighbors to 1e-6."""
    if key.startswith("measured") and u > 0 and not authorized:
        raise RuntimeError("a finite zeta on the measured data before the registered tag")
    d = load(key)
    Fs, MUs = zip(*(predict(cfg, d, s, u) for s in S0_GRID))
    tot = chi2_profiled(d, np.array(Fs).T, np.array(MUs).T)[0]
    k = int(np.argmin(tot))
    lo, hi = S0_GRID[max(k - 1, 0)], S0_GRID[min(k + 1, len(S0_GRID) - 1)]
    r = minimize_scalar(lambda s: float(_chi2_at(cfg, d, s, u)[0][0]), bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-6})
    s_best = float(S0_GRID[k]) if tot[k] <= r.fun else float(r.x)
    t, cs, cb, h0rd, mb = (float(v[0]) for v in _chi2_at(cfg, d, s_best, u))
    return {"u": u, "chi2": t, "chi2_sn": cs, "chi2_bao": cb, "s0": s_best, "h0rd": h0rd, "mb": mb}

def _task(args):
    return profile(*args)

def curve(pool, cfg, key, us, authorized=False):
    return pool.map(_task, [(cfg, key, float(u), authorized) for u in us], chunksize=8)

def refine_u(cfg, key, pts, authorized=False):
    """Refine the free-zeta minimum between the best grid point's neighbors (bounded minimization in u)."""
    c2 = np.array([p["chi2"] for p in pts])
    k = int(np.argmin(c2))
    lo, hi = U_GRID[max(k - 1, 0)], U_GRID[min(k + 1, len(U_GRID) - 1)]
    r = minimize_scalar(lambda u: profile(cfg, key, float(u), authorized)["chi2"], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-10})
    best = pts[k] if c2[k] <= r.fun else profile(cfg, key, float(r.x), authorized)
    return best

def zeta_of(R_gpc, H0):
    return R_gpc * 1000.0 * H0 / C_LIGHT

def readings():
    return [(lab, row, zeta_of(R, H0), H0) for row, H0 in H0_ROWS.items() for lab, R in RADII_GPC.items()]

# ---------------- gates ----------------
def run_published_span():
    """fit_lcos.py from its first line to the line before its MCMC marker, with sys.argv pinned and its
    own directory as the working directory. Returns its namespace and the span's SHA-256."""
    with open(os.path.join(HERE, "fit_lcos.py"), encoding="utf-8") as fh:
        src = fh.read().split("\n")
    k = [i for i, l in enumerate(src) if l.startswith(MCMC_MARKER)]
    assert len(k) == 1, "the MCMC marker must occur once"
    span = "\n".join(src[:k[0]]) + "\n"
    sha = hashlib.sha256(span.encode("utf-8")).hexdigest()
    cwd, argv, path = os.getcwd(), sys.argv[:], list(sys.path)
    ns = {"__name__": "fit_lcos_span", "__file__": os.path.join(HERE, "fit_lcos.py")}
    try:
        os.chdir(HERE); sys.argv = ["fit_lcos.py"]; sys.path.insert(0, HERE)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(span, os.path.join(HERE, "fit_lcos.py"), "exec"), ns)
    finally:
        os.chdir(cwd); sys.argv = argv; sys.path[:] = path
    return ns, sha, k[0]

def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))

def gate_G1(cfg=BASE):
    ns, sha, n = run_published_span()
    d = load("measured")
    zg = ns["zgrid"]
    worst = {"E": 0.0, "sn_chi2": 0.0, "bao_chi2": 0.0, "bao_pred": 0.0}
    s0s, mbs, h0rds = (0.001, 0.05, 0.1, 0.2, 0.4), (-19.40, -19.354, -19.30), (9900.0, 10025.43, 10150.0)
    bao_pub, cinv_pub = ns["bao"], ns["Cbao_inv"]
    negative = 0
    try:
        for s0 in s0s:
            worst["E"] = max(worst["E"], rel(E_of(cfg, zg, s0), ns["E"](zg, s0)))
            f, mu = predict(cfg, d, s0, 0.0)
            for mb in mbs:
                r = d["m"] - mb - mu
                worst["sn_chi2"] = max(worst["sn_chi2"], rel(r @ cho_solve(d["cfac"], r), ns["sn_chi2"](s0, mb)))
            for h in h0rds:
                pred = (C_LIGHT / h) * f
                r = pred - d["y"]
                worst["bao_chi2"] = max(worst["bao_chi2"], rel(r @ d["Cbi"] @ r, ns["bao_chi2"](s0, h)))
                ns["bao"] = bao_pub.assign(value=0.0)          # data vector to zero
                for i in range(len(pred)):
                    P = np.zeros_like(cinv_pub); P[i, i] = 1.0  # unit projector: bao_chi2 returns pred_i^2
                    ns["Cbao_inv"] = P
                    sq = ns["bao_chi2"](s0, h)
                    negative += int(sq < 0)
                    worst["bao_pred"] = max(worst["bao_pred"], rel(pred[i], np.sqrt(max(sq, 0.0))))
                ns["bao"], ns["Cbao_inv"] = bao_pub, cinv_pub
    finally:
        ns["bao"], ns["Cbao_inv"] = bao_pub, cinv_pub
    floor = max(worst.values())
    ok = sha == SPAN_SHA256 and negative == 0 and floor <= TOL["G1_rel"]
    return {"gate": "G1", "pass": ok, "span_lines": n, "span_sha256": sha, "span_pinned": sha == SPAN_SHA256,
            "worst_rel": worst, "floor": floor, "tol": TOL["G1_rel"]}

def gate_G2(cfg=BASE, key="measured"):
    pub = read_json(os.path.join(REPO, "results", "lcos_summary.json"))
    p = profile(cfg, key, 0.0)
    dev = {"chi2": abs(p["chi2"] - pub["chi2"]["total"]), "chi2_sn": abs(p["chi2_sn"] - pub["chi2"]["SN"]),
           "chi2_bao": abs(p["chi2_bao"] - pub["chi2"]["BAO"]),
           "s0sq": abs(p["s0"]**2 - pub["best_fit"]["s0"]**2), "h0rd": abs(p["h0rd"] - pub["best_fit"]["H0rd"]),
           "mb": abs(p["mb"] - pub["best_fit"]["MB"])}
    ok = (max(dev["chi2"], dev["chi2_sn"], dev["chi2_bao"]) <= TOL["G2_chi2"] and dev["s0sq"] <= TOL["G2_s0sq"]
          and dev["h0rd"] <= TOL["G2_h0rd"] and dev["mb"] <= TOL["G2_mb"])
    return {"gate": "G2", "pass": ok, "profile": p, "deviation": dev,
            "tol": {k: TOL[k] for k in ("G2_chi2", "G2_s0sq", "G2_h0rd", "G2_mb")}}

def gate_G3(pool, cfg=BASE, key="flat"):
    fc = read_json(os.path.join(DATA, "closed_distance_forecast.json"))
    pts = curve(pool, cfg, key, U_GRID)
    c0 = pts[0]["chi2"]
    below = max(0.0, max(c0 - p["chi2"] for p in pts[1:]))
    dch = {}
    worst_rel = 0.0
    for lab, row, zeta, H0 in readings():
        dc = profile(cfg, key, 1.0 / zeta**2)["chi2"] - c0
        ref = fc["cells"][f"{lab}@{H0}"]["dchi2"]
        dch[f"{lab}@{H0}"] = dc
        worst_rel = max(worst_rel, abs(dc / ref - 1))
    ok = below <= TOL["G3_below"] and worst_rel <= TOL["G3_forecast_rel"]
    return {"gate": "G3", "pass": ok, "flat_chi2": c0, "worst_below_flat": below, "dchi2": dch,
            "worst_forecast_rel": worst_rel, "tol": {"G3_below": TOL["G3_below"], "G3_forecast_rel": TOL["G3_forecast_rel"]},
            "curve_chi2": [p["chi2"] for p in pts]}

def gate_G4(pool, cfg=BASE, stop_at_first_failure=False):
    sets, ok_all, worst = [], True, {"zeta_rel": 0.0, "s0sq": 0.0, "chi2": 0.0}
    for lab, row, zeta, H0 in readings():
        for s0t in SYN_S0:
            key = f"{lab}_{H0}_{s0t}"
            pts = curve(pool, cfg, key, U_GRID)
            b = refine_u(cfg, key, pts)
            zh = 1.0 / np.sqrt(b["u"]) if b["u"] > 0 else float("inf")
            e = {"zeta_rel": abs(zh / zeta - 1), "s0sq": abs(b["s0"]**2 - s0t**2), "chi2": b["chi2"]}
            ok = e["zeta_rel"] <= TOL["G4_zeta_rel"] and e["s0sq"] <= TOL["G4_s0sq"] and e["chi2"] <= TOL["G4_chi2"]
            for kk in worst:
                worst[kk] = max(worst[kk], e[kk])
            sets.append({"set": key, "zeta_true": zeta, "zeta_hat": zh, "s0_hat": b["s0"], "chi2_min": b["chi2"],
                         "error": e, "pass": ok, "curve_chi2": [p["chi2"] for p in pts]})
            ok_all &= ok
            if stop_at_first_failure and not ok:
                return {"gate": "G4", "pass": False, "sets": sets, "worst": worst, "stopped_early": True}
    return {"gate": "G4", "pass": ok_all, "sets": sets, "worst": worst,
            "tol": {k: TOL[k] for k in ("G4_zeta_rel", "G4_s0sq", "G4_chi2")}}

def gate_G5(pool, g3, g4, cfg_fine=None, stop_after_G3=False):
    """Every quantity G3 and G4 report, recomputed with the instrument on 20 001 points (or, for the
    arm, on the grid of cfg_fine compared against a 20 001-point reference)."""
    fine = cfg_fine or (20001,) + BASE[1:]
    g3f = gate_G3(pool, fine)
    move = {"G3_dchi2": max(abs(g3f["dchi2"][k] - g3["dchi2"][k]) for k in g3["dchi2"]),
            "G3_flat_chi2": abs(g3f["flat_chi2"] - g3["flat_chi2"])}
    if not stop_after_G3:
        g4f = gate_G4(pool, fine)
        move["G4_zeta_rel"] = max(abs(a["zeta_hat"] / b["zeta_hat"] - 1) for a, b in zip(g4f["sets"], g4["sets"]))
        move["G4_s0sq"] = max(abs(a["s0_hat"]**2 - b["s0_hat"]**2) for a, b in zip(g4f["sets"], g4["sets"]))
        move["G4_chi2"] = max(abs(a["chi2_min"] - b["chi2_min"]) for a, b in zip(g4f["sets"], g4["sets"]))
    ok = (move["G3_dchi2"] <= TOL["G5_dchi2"] and move["G3_flat_chi2"] <= TOL["G5_dchi2"]
          and move.get("G4_chi2", 0.0) <= TOL["G5_dchi2"] and move.get("G4_zeta_rel", 0.0) <= TOL["G5_zeta_rel"]
          and move.get("G4_s0sq", 0.0) <= TOL["G5_s0sq"])
    return {"gate": "G5", "pass": ok, "grid": fine[0], "move": move,
            "tol": {k: TOL[k] for k in ("G5_dchi2", "G5_zeta_rel", "G5_s0sq")}}

def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True).stdout.strip()

def gate_G6a(paths=None):
    paths = paths or {k: os.path.join(REPO, k) for k in PINS}
    got = {k: sha256_file(p) for k, p in paths.items()}
    _, span_sha, _ = run_published_span()
    status = git("status", "--porcelain")
    ok = all(got[k] == PINS[k] for k in PINS) and span_sha == SPAN_SHA256 and status == ""
    return {"gate": "G6a", "pass": ok, "files": {k: got[k] == PINS[k] for k in PINS},
            "span": span_sha == SPAN_SHA256, "clean": status == "", "commit": git("rev-parse", "HEAD"),
            "script_sha256": sha256_file(os.path.abspath(__file__))}

def gate_G6b(reg_repo, reg_commit, reg_sha256, tag=TAG, reg_path=REG_PATH, reg_branch="origin/main"):
    """After the freeze. The tag resolves to this checkout, and G6a's pins hold. The registration, fetched
    afresh, sits on reg_branch at reg_commit, its page there has SHA-256 reg_sha256, and that page names
    this commit, this script's SHA-256 and the tag. Nothing a caller supplies can stand in for these."""
    try:
        tagged = git("rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        tagged = None
    a = gate_G6a()
    head, me = a["commit"], a["script_sha256"]
    reg = {"repo": reg_repo, "commit": reg_commit, "path": reg_path, "branch": reg_branch}
    try:
        subprocess.run(["git", "-C", reg_repo, "fetch", "-q", "origin"], check=True, capture_output=True)
        reg["on_branch"] = subprocess.run(["git", "-C", reg_repo, "merge-base", "--is-ancestor", reg_commit, reg_branch],
                                          capture_output=True).returncode == 0
        page = subprocess.run(["git", "-C", reg_repo, "show", f"{reg_commit}:{reg_path}"], check=True, capture_output=True).stdout
        reg["sha256"] = hashlib.sha256(page).hexdigest()
        text = page.decode("utf-8")
        reg["names"] = {"commit": head in text, "script_sha256": me in text, "tag": tag in text}
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError) as e:
        reg.update(on_branch=False, sha256=None, names={}, error=str(e))
    ok = (tagged == head and a["pass"] and reg["on_branch"] and reg["sha256"] == reg_sha256
          and len(reg["names"]) == 3 and all(reg["names"].values()))
    return {"gate": "G6b", "pass": bool(ok), "tag": tag, "tag_commit": tagged, "head": head,
            "registration": reg, "registration_sha256_expected": reg_sha256, "pins": a}

def _probe_registration(tmp, text, branch="main"):
    """A throwaway registration repo with a local bare origin, for probing G6b before the freeze."""
    origin, work = os.path.join(tmp, "origin.git"), os.path.join(tmp, "reg")
    g = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()
    g("init", "-q", "--bare", origin)
    g("clone", "-q", origin, work)
    os.makedirs(os.path.dirname(os.path.join(work, REG_PATH)), exist_ok=True)
    with open(os.path.join(work, REG_PATH), "w", encoding="utf-8") as fh:
        fh.write(text)
    g("-C", work, "add", REG_PATH)
    g("-C", work, "-c", "user.name=qualification-probe", "-c", "user.email=probe@localhost", "commit", "-q", "-m", "probe")
    g("-C", work, "push", "-q", "origin", f"HEAD:{branch}")
    commit = g("-C", work, "rev-parse", "HEAD")
    return work, commit, hashlib.sha256(text.encode("utf-8")).hexdigest()

def probe_G6b():
    """G6b's parent and its four arms, on temporary local tags and throwaway registrations."""
    head = git("rev-parse", "HEAD")
    parent_commit = git("rev-parse", "HEAD~1")
    me = sha256_file(os.path.abspath(__file__))
    ptag = "closed-distance-probe"
    good = f"registration\ncommit {head}\nscript {me}\ntag {ptag}\n"
    out = {}
    tmp = tempfile.mkdtemp()
    try:
        def case(name, tag_target, text, branch="main", sha_override=None):
            d = tempfile.mkdtemp(dir=tmp)
            work, commit, sha = _probe_registration(d, text, branch)
            git("tag", ptag, tag_target)
            try:
                out[name] = gate_G6b(work, commit, sha_override or sha, tag=ptag)["pass"]
            finally:
                git("tag", "-d", ptag)
        case("parent", "HEAD", good)
        case("tag on another commit", "HEAD~1", good)
        case("registration not on main", "HEAD", good, branch="side")
        case("registration hash wrong", "HEAD", good, sha_override="0" * 64)
        case("registration names another commit", "HEAD", good.replace(head, parent_commit))
    finally:
        shutil.rmtree(tmp)
    return {"parent_passes": out["parent"],
            "arms_fail": {k: not v for k, v in out.items() if k != "parent"}}

# ---------------- qualification ----------------
def floor_rows(g):
    """Each measured floor beside its tolerance, and whether the tolerance sits FLOOR_MARGIN times above it."""
    rows = [("G1", "worst relative deviation from the published code", g["G1"]["floor"], "G1_rel"),
            ("G2", "chi2, worst of the total and its parts", max(g["G2"]["deviation"][k] for k in ("chi2", "chi2_sn", "chi2_bao")), "G2_chi2"),
            ("G2", "s0^2", g["G2"]["deviation"]["s0sq"], "G2_s0sq"),
            ("G2", "H0 r_d (km/s)", g["G2"]["deviation"]["h0rd"], "G2_h0rd"),
            ("G2", "M_B", g["G2"]["deviation"]["mb"], "G2_mb"),
            ("G3", "chi2 of the flat set at u = 0", g["G3"]["flat_chi2"], "G3_below"),
            ("G3", "worst relative difference from the forecast", g["G3"]["worst_forecast_rel"], "G3_forecast_rel"),
            ("G4", "worst relative zeta error", g["G4"]["worst"]["zeta_rel"], "G4_zeta_rel"),
            ("G4", "worst s0^2 error", g["G4"]["worst"]["s0sq"], "G4_s0sq"),
            ("G4", "largest chi2 at the refined minimum", g["G4"]["worst"]["chi2"], "G4_chi2"),
            ("G5", "largest move in chi2", max(g["G5"]["move"][k] for k in ("G3_dchi2", "G3_flat_chi2", "G4_chi2")), "G5_dchi2"),
            ("G5", "largest relative move in zeta", g["G5"]["move"]["G4_zeta_rel"], "G5_zeta_rel"),
            ("G5", "largest move in s0^2", g["G5"]["move"]["G4_s0sq"], "G5_s0sq"),
            ("R", "the flat set's bound against the forecast's, relative", g["R"]["bound_rel"], "R_bound_rel"),
            ("R", "the closed set's recovered zeta, relative", g["R"]["zeta_rel"], "R_zeta_rel")]
    return [{"gate": a, "quantity": q, "floor": float(f), "tolerance_key": k, "tolerance": TOL[k],
             "clears": bool(TOL[k] >= FLOOR_MARGIN * f)} for a, q, f, k in rows]

def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "platform": platform.platform()}

def check_environment():
    """The script's first act: the numerical stack must be the pinned one."""
    env = environment()
    bad = {k: [env[k], v] for k, v in ENV_PINNED.items() if env[k] != v}
    return {"pass": not bad, "mismatch": bad, "environment": env}

def qualify(out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rec = {"environment": check_environment(), "tolerances": dict(TOL), "gates": {}, "arms": {}}
    if not rec["environment"]["pass"]:
        json.dump(rec, open(os.path.join(out, "qualification.json"), "w"), indent=1, default=_jsonable)
        sys.exit("the environment is not the pinned one: qualification stops")
    with Pool(os.cpu_count()) as pool:
        rec["gates"]["G1"] = g1 = gate_G1()
        rec["gates"]["G2"] = g2 = gate_G2()
        rec["gates"]["G3"] = g3 = gate_G3(pool)
        rec["gates"]["G4"] = g4 = gate_G4(pool)
        rec["gates"]["G5"] = g5 = gate_G5(pool, g3, g4)
        rec["gates"]["R"] = gate_R = rehearsal(pool)
        rec["gates"]["G6a"] = g6 = gate_G6a()
        parents = {k: v["pass"] for k, v in rec["gates"].items()}
        arms = rec["arms"]
        if parents["G1"]:
            arms["G1 Ω_Λ = 0.686"] = not gate_G1((N_GRID, 0.686, 1 / 3, "sin", False))["pass"]
            arms["G1 D_V exponent 1/2"] = not gate_G1((N_GRID, OMEGA_L, 0.5, "sin", False))["pass"]
        if parents["G2"]:
            arms["G2 D_V point dropped"] = not gate_G2(key="measured-noDV")["pass"]
        if parents["G3"]:
            arms["G3 closed signal at zeta = 3"] = not gate_G3(pool, key="flat+zeta3")["pass"]
        if parents["G4"]:
            arms["G4 sinh for sin"] = not gate_G4(pool, (N_GRID, OMEGA_L, 1 / 3, "sinh", False), True)["pass"]
            arms["G4 curvature in E"] = not gate_G4(pool, (N_GRID, OMEGA_L, 1 / 3, "sin", True), True)["pass"]
        if parents["G5"]:
            ref = gate_G3(pool, (20001,) + BASE[1:])
            coarse = gate_G3(pool, (400,) + BASE[1:])
            mv = max(abs(coarse["dchi2"][k] - ref["dchi2"][k]) for k in ref["dchi2"])
            arms["G5 grid cut to 400 points"] = mv > TOL["G5_dchi2"]
            rec["arms_detail"] = {"G5_400_move": mv}
        if parents["R"]:
            arms["R region level 3.84"] = not rehearsal(pool, 3.84)["pass"]
        if parents["G6a"]:
            tmp = tempfile.mkdtemp()
            try:
                bad = dict((k, os.path.join(REPO, k)) for k in PINS)
                k0 = "data/desi_dr2_bao.csv"
                cp = os.path.join(tmp, "copy.csv"); shutil.copy(bad[k0], cp)
                with open(cp, "r+b") as f:
                    f.seek(10); c = f.read(1); f.seek(10); f.write(bytes([c[0] ^ 1]))
                bad[k0] = cp
                arms["G6a one byte changed"] = not gate_G6a(bad)["pass"]
            finally:
                shutil.rmtree(tmp)
        # G6b, probed on temporary local tags and throwaway registrations: its parent passes, its four arms fail
        rec["G6b_probe"] = probe_G6b()
        head = git("rev-parse", "HEAD")
        me = sha256_file(os.path.abspath(__file__))
    rec["floors"] = floor_rows(rec["gates"])
    rec["tolerances_clear_floors"] = all(r["clears"] for r in rec["floors"])
    rec["all_gates_pass"] = all(parents.values())
    rec["all_arms_fire"] = (all(arms.values()) and len(arms) == 9 and rec["G6b_probe"]["parent_passes"]
                            and len(rec["G6b_probe"]["arms_fail"]) == 4 and all(rec["G6b_probe"]["arms_fail"].values()))
    rec["seconds"] = round(time.time() - t0, 1)
    rec["commit"], rec["script_sha256"] = head, me
    json.dump(rec, open(os.path.join(out, "qualification.json"), "w"), indent=1, default=_jsonable)
    print(json.dumps({"all_gates_pass": rec["all_gates_pass"], "all_arms_fire": rec["all_arms_fire"], "tolerances_clear_floors": rec["tolerances_clear_floors"],
                      "gates": parents, "arms": arms, "G6b_probe": rec["G6b_probe"], "seconds": rec["seconds"]}, indent=1))
    return rec

# ---------------- the run, once, from the tag ----------------
def grade(dc):
    return "EXCLUDED" if dc >= BAR_EXCLUDED else ("DISFAVORED" if dc >= BAR_DISFAVORED else "NOT EXCLUDED")

def region(pts, best, key, authorized, level_above=REGION_LEVEL):
    """The region below level_above over the refined minimum, as intervals in u with edges refined by
    bisection to 1e-4 in zeta, then in zeta and in Omega_0 R (Gpc, both rows); zeta95 is its smallest zeta."""
    level = best["chi2"] + level_above
    inside = [bool(p["chi2"] < level) for p in pts]
    f = lambda u: profile(BASE, key, float(u), authorized)["chi2"] - level
    def edge(a, b):                                  # a inside, b outside
        fa = f(a)
        while abs(1 / np.sqrt(max(a, 1e-300)) - 1 / np.sqrt(max(b, 1e-300))) > 1e-4 and abs(b - a) > 1e-12:
            m = 0.5 * (a + b); fm = f(m)
            if (fm < 0) == (fa < 0): a, fa = m, fm
            else: b = m
        return 0.5 * (a + b)
    ivals, k = [], 0
    while k < len(pts):
        if inside[k]:
            j = k
            while j + 1 < len(pts) and inside[j + 1]:
                j += 1
            lo = 0.0 if k == 0 else edge(U_GRID[k], U_GRID[k - 1])
            hi = 1.0 if j == len(pts) - 1 else edge(U_GRID[j], U_GRID[j + 1])
            ivals.append((lo, hi))
            k = j + 1
        else:
            k += 1
    zeta_iv = [(1 / np.sqrt(hi), (1 / np.sqrt(lo)) if lo > 0 else float("inf")) for lo, hi in ivals]
    out = {"level": level, "intervals_u": ivals, "intervals_zeta": zeta_iv,
           "intervals_R_gpc": {row: [(a * C_LIGHT / H0 / 1000.0, b * C_LIGHT / H0 / 1000.0) for a, b in zeta_iv]
                               for row, H0 in H0_ROWS.items()},
           "reaches_u1": inside[-1], "flat_inside": inside[0], "best_u": best["u"],
           "zeta95": None if (inside[-1] or not ivals) else 1 / np.sqrt(max(h for _, h in ivals))}
    if out["zeta95"] is not None:
        out["R95_gpc"] = {row: out["zeta95"] * C_LIGHT / H0 / 1000.0 for row, H0 in H0_ROWS.items()}
    return out

def analyze(pool, key, authorized, level_above=REGION_LEVEL):
    """The registered computation on one data set: each reading's profiled dchi2 and grade in both rows,
    the free-zeta profile with its refined minimum, the region and its bound, and the preference."""
    flat = profile(BASE, key, 0.0, authorized)
    cells = {}
    for lab, row, zeta, H0 in readings():
        p = profile(BASE, key, 1.0 / zeta**2, authorized)
        cells[f"{lab}@{H0}"] = {"zeta": zeta, "dchi2": p["chi2"] - flat["chi2"], "grade": grade(p["chi2"] - flat["chi2"]),
                                "closed": p, "flat": flat}
    grades = {}
    for lab in RADII_GPC:
        a, b = (cells[f"{lab}@{H0}"]["grade"] for H0 in H0_ROWS.values())
        grades[lab] = a if a == b else "SCALE-DEPENDENT"
    pts = curve(pool, BASE, key, U_GRID, authorized)
    best = refine_u(BASE, key, pts, authorized)
    reg = region(pts, best, key, authorized, level_above)
    pref = flat["chi2"] - best["chi2"]
    return {"cells": cells, "grades": grades, "free_zeta": {"best": best, "curve_chi2": [p["chi2"] for p in pts]},
            "region": reg, "preference": pref, "interior_minimum": bool(pref > INTERIOR_CUT),
            "closure_preferred": bool(pref >= PREFERENCE_BAR)}

def rehearsal(pool, level_above=REGION_LEVEL):
    """The run's analysis, rehearsed on the fixed synthetic sets: on the flat set it must give the forecast's
    grades and bound and no preference; on the C@67.36 closed set it must recover zeta inside its region."""
    fc = read_json(os.path.join(DATA, "closed_distance_forecast.json"))
    fl = analyze(pool, "flat", False, level_above)
    expect = {}
    for lab in RADII_GPC:
        a, b = (grade(fc["cells"][f"{lab}@{H0}"]["dchi2"]) for H0 in H0_ROWS.values())
        expect[lab] = a if a == b else "SCALE-DEPENDENT"
    zt = zeta_of(RADII_GPC["C"], H0_ROWS["primary"])
    cl = analyze(pool, f"C_{H0_ROWS['primary']}_0.001", False, level_above)
    zb = cl["free_zeta"]["best"]["u"]
    zh = 1 / np.sqrt(zb) if zb > 0 else float("inf")
    bound_rel = abs(fl["region"]["zeta95"] / fc["bound"]["zeta95"] - 1) if fl["region"]["zeta95"] else float("inf")
    zeta_rel = abs(zh / zt - 1)
    in_region = any(a <= zt <= b for a, b in cl["region"]["intervals_zeta"])
    ok = (fl["grades"] == expect and fl["region"]["flat_inside"] and fl["preference"] <= TOL["G3_below"]
          and bound_rel <= TOL["R_bound_rel"] and zeta_rel <= TOL["R_zeta_rel"] and in_region
          and not cl["region"]["flat_inside"] and cl["closure_preferred"]
          and not fl["interior_minimum"] and cl["interior_minimum"])
    return {"gate": "R", "pass": bool(ok), "flat_grades": fl["grades"], "expected_grades": expect,
            "flat_zeta95": fl["region"]["zeta95"], "forecast_zeta95": fc["bound"]["zeta95"], "bound_rel": bound_rel,
            "flat_preference": fl["preference"], "closed_zeta_hat": zh, "closed_zeta_true": zt, "zeta_rel": zeta_rel,
            "closed_in_region": in_region, "closed_preference": cl["preference"], "flat": fl, "closed": cl,
            "tol": {k: TOL[k] for k in ("R_bound_rel", "R_zeta_rel")}}

def run(out, reg_repo, reg_commit, reg_sha256):
    os.makedirs(out, exist_ok=True)
    rec = {"environment": check_environment(), "tolerances": dict(TOL),
           "registration": {"commit": reg_commit, "sha256": reg_sha256, "path": REG_PATH}}
    if not rec["environment"]["pass"]:
        json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
        sys.exit("the environment is not the pinned one: nothing runs (Uninformative)")
    rec["G6b"] = g6b = gate_G6b(reg_repo, reg_commit, reg_sha256)
    if not g6b["pass"]:
        json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
        sys.exit("G6b failed: nothing runs on the measured data (Uninformative)")
    with Pool(os.cpu_count()) as pool:
        g = rec["gates"] = {"G1": gate_G1(), "G2": gate_G2()}
        g["G3"] = gate_G3(pool); g["G4"] = gate_G4(pool); g["G5"] = gate_G5(pool, g["G3"], g["G4"])
        if not all(v["pass"] for v in g.values()):
            json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
            sys.exit("a gate failed after the freeze: the run ends as Uninformative")
        rec.update(analyze(pool, "measured", True))
    json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
    print(json.dumps({"grades": rec["grades"], "preference": rec["preference"], "zeta95": rec["region"]["zeta95"]},
                     indent=1, default=_jsonable))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Λcos with a closed transverse distance at a fixed curvature radius.")
    sub = ap.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("qualify"); q.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--registration-repo", required=True)
    r.add_argument("--registration-commit", required=True); r.add_argument("--registration-sha256", required=True)
    a = ap.parse_args()
    if os.path.realpath(a.out).startswith(os.path.realpath(REPO) + os.sep) and a.mode == "qualify":
        sys.exit("qualification writes outside the repo, so the checkout stays clean")
    if a.mode == "qualify":
        qualify(a.out)
    else:
        run(a.out, a.registration_repo, a.registration_commit, a.registration_sha256)
