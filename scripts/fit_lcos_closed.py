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
    python fit_lcos_closed.py run --out DIR --tag TAG --expect-commit SHA \
            --expect-script-sha256 HEX --registration-sha256 HEX
        G6b, then G1 to G5 again, then the comparison on the measured data, once.
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
BASE = (N_GRID, OMEGA_L, 1.0 / 3.0, "sin", False)     # (grid, Ω_Λ, D_V exponent, transverse, curvature in E)

# Tolerances: provisional until qualification sets them from the measured floors.
TOL = {
    "G1_rel": 1e-9,
    "G2_chi2": 0.01, "G2_s0sq": 1e-5, "G2_h0rd": 1.0, "G2_mb": 1e-3,
    "G3_below": 1e-6, "G3_forecast_rel": 0.005,
    "G4_zeta_rel": 1e-4, "G4_s0sq": 1e-5, "G4_chi2": 1e-6,
    "G5_dchi2": 0.01, "G5_zeta_rel": 1e-3, "G5_s0sq": 1e-5,
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
        syn = json.load(open(os.path.join(DATA, "closed_distance_synthetic.json")))
        base = "flat" if key == "flat+zeta3" else key
        bval, m = np.array(syn[base + "_bao"]), np.array(syn[base + "_sn"])
        if key == "flat+zeta3":         # a closed signal at zeta = 3 injected into a copy of the flat set
            bf = json.load(open(os.path.join(REPO, "results", "lcos_summary.json")))["best_fit"]
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
    E2 = (1 - OL) / (1 - s0**2) * (1 + z)**3 - (1 - OL) * s0**2 / (1 - s0**2) * (1 + z) + OL
    if curv:                             # G4's arm only: a curvature term put into the rate
        E2 = E2 - u * (1 + z)**2
    return np.sqrt(E2)

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
    dm, dh = transverse(Iz, u, kind), 1 / Ez
    f = np.array([(zz * a**2 * b)**dv if o == "DV_rd" else (a if o == "DM_rd" else b)
                  for zz, a, b, o in zip(d["bz"], dm, dh, d["bobs"])])
    Is = np.interp(d["z_sn"], z, I)
    mu = 5 * np.log10((1 + d["z_sn"]) * (C_LIGHT / H0_SN) * transverse(Is, u, kind)) + 25
    return f, mu

def chi2_profiled(d, F, MU):
    """F: BAO shapes (n_bao x K), MU: SN moduli (n_sn x K). M_B and c/(H0 r_d) profiled analytically,
    each clipped to the box. Returns (total, SN, BAO, H0rd, M_B), each of length K."""
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
    src = open(os.path.join(HERE, "fit_lcos.py"), encoding="utf-8").read().split("\n")
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
    pub = json.load(open(os.path.join(REPO, "results", "lcos_summary.json")))
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
    fc = json.load(open(os.path.join(DATA, "closed_distance_forecast.json")))
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

def gate_G6b(tag, expect_commit, expect_script):
    try:
        tagged = git("rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        tagged = None
    a = gate_G6a()
    ok = (tagged == expect_commit and a["commit"] == expect_commit and a["pass"]
          and a["script_sha256"] == expect_script)
    return {"gate": "G6b", "pass": ok, "tag": tag, "tag_commit": tagged, "head": a["commit"],
            "expect_commit": expect_commit, "script_ok": a["script_sha256"] == expect_script, "pins": a}

# ---------------- qualification ----------------
def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "platform": platform.platform()}

def qualify(out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rec = {"environment": environment(), "tolerances": dict(TOL), "gates": {}, "arms": {}}
    with Pool(os.cpu_count()) as pool:
        rec["gates"]["G1"] = g1 = gate_G1()
        rec["gates"]["G2"] = g2 = gate_G2()
        rec["gates"]["G3"] = g3 = gate_G3(pool)
        rec["gates"]["G4"] = g4 = gate_G4(pool)
        rec["gates"]["G5"] = g5 = gate_G5(pool, g3, g4)
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
        # G6b, probed on temporary local tags: its parent passes on HEAD, its arm fails on another commit
        head = git("rev-parse", "HEAD")
        me = sha256_file(os.path.abspath(__file__))
        probes = {}
        for name, target in (("closed-distance-probe-pass", "HEAD"), ("closed-distance-probe-fail", "HEAD~1")):
            git("tag", name, target)
            try:
                probes[name] = gate_G6b(name, head, me)["pass"]
            finally:
                git("tag", "-d", name)
        rec["G6b_probe"] = {"parent_passes": probes["closed-distance-probe-pass"],
                            "arm_fails": not probes["closed-distance-probe-fail"]}
    rec["all_gates_pass"] = all(parents.values())
    rec["all_arms_fire"] = all(arms.values()) and len(arms) == 8 and rec["G6b_probe"]["parent_passes"] and rec["G6b_probe"]["arm_fails"]
    rec["seconds"] = round(time.time() - t0, 1)
    rec["commit"], rec["script_sha256"] = head, me
    json.dump(rec, open(os.path.join(out, "qualification.json"), "w"), indent=1, default=_jsonable)
    print(json.dumps({"all_gates_pass": rec["all_gates_pass"], "all_arms_fire": rec["all_arms_fire"],
                      "gates": parents, "arms": arms, "G6b_probe": rec["G6b_probe"], "seconds": rec["seconds"]}, indent=1))
    return rec

# ---------------- the run, once, from the tag ----------------
def grade(dc):
    return "EXCLUDED" if dc >= BAR_EXCLUDED else ("DISFAVORED" if dc >= BAR_DISFAVORED else "NOT EXCLUDED")

def region(pts, best):
    """The region below REGION_LEVEL above the refined minimum, as intervals in u with edges refined by
    bisection to 1e-4 in zeta; zeta95 is its smallest zeta."""
    level = best["chi2"] + REGION_LEVEL
    inside = [p["chi2"] < level for p in pts]
    f = lambda u: profile(BASE, "measured", float(u), True)["chi2"] - level
    def edge(a, b):                                  # f(a) < 0 <= f(b) or the reverse
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
    u_max = max(h for _, h in ivals)
    return {"level": level, "intervals_u": ivals, "reaches_u1": inside[-1],
            "zeta95": None if inside[-1] else 1 / np.sqrt(u_max), "flat_inside": inside[0]}

def run(out, tag, expect_commit, expect_script, registration_sha):
    os.makedirs(out, exist_ok=True)
    rec = {"environment": environment(), "tolerances": dict(TOL), "registration_sha256": registration_sha}
    rec["G6b"] = g6b = gate_G6b(tag, expect_commit, expect_script)
    if not g6b["pass"]:
        json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
        sys.exit("G6b failed: nothing runs on the measured data (Uninformative)")
    with Pool(os.cpu_count()) as pool:
        g = rec["gates"] = {"G1": gate_G1(), "G2": gate_G2()}
        g["G3"] = gate_G3(pool); g["G4"] = gate_G4(pool); g["G5"] = gate_G5(pool, g["G3"], g["G4"])
        if not all(v["pass"] for v in g.values()):
            json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
            sys.exit("a gate failed after the freeze: the run ends as Uninformative")
        flat = profile(BASE, "measured", 0.0, True)
        cells = {}
        for lab, row, zeta, H0 in readings():
            p = profile(BASE, "measured", 1.0 / zeta**2, True)
            cells[f"{lab}@{H0}"] = {"zeta": zeta, "dchi2": p["chi2"] - flat["chi2"], "grade": grade(p["chi2"] - flat["chi2"]),
                                    "closed": p, "flat": flat}
        grades = {}
        for lab in RADII_GPC:
            a, b = (cells[f"{lab}@{H0}"]["grade"] for H0 in H0_ROWS.values())
            grades[lab] = a if a == b else "SCALE-DEPENDENT"
        pts = curve(pool, BASE, "measured", U_GRID, True)
        best = refine_u(BASE, "measured", pts, True)
        reg = region(pts, best)
        if reg["zeta95"] is not None:
            reg["R95_gpc"] = {row: reg["zeta95"] * C_LIGHT / H0 / 1000.0 for row, H0 in H0_ROWS.items()}
        rec.update({"cells": cells, "grades": grades, "free_zeta": {"best": best, "curve_chi2": [p["chi2"] for p in pts]},
                    "region": reg, "preference": flat["chi2"] - best["chi2"],
                    "closure_preferred": flat["chi2"] - best["chi2"] >= PREFERENCE_BAR})
    json.dump(rec, open(os.path.join(out, "run.json"), "w"), indent=1, default=_jsonable)
    print(json.dumps({"grades": grades, "preference": rec["preference"], "zeta95": reg["zeta95"]}, indent=1, default=_jsonable))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Λcos with a closed transverse distance at a fixed curvature radius.")
    sub = ap.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("qualify"); q.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--tag", required=True)
    r.add_argument("--expect-commit", required=True); r.add_argument("--expect-script-sha256", required=True)
    r.add_argument("--registration-sha256", required=True)
    a = ap.parse_args()
    if os.path.realpath(a.out).startswith(os.path.realpath(REPO) + os.sep) and a.mode == "qualify":
        sys.exit("qualification writes outside the repo, so the checkout stays clean")
    if a.mode == "qualify":
        qualify(a.out)
    else:
        run(a.out, a.tag, a.expect_commit, a.expect_script_sha256, a.registration_sha256)
