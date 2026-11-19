"""
SN against BAO on s0: does one s0 describe both datasets in Model D+Lambda?

Model D+Lambda is fit_lcos.py's, at Omega_L = 0.685. With u = s0^2,
    E^2(z) = (1 - Omega_L)/(1 - u) (1+z)^3 - (1 - Omega_L) u/(1 - u) (1+z) + Omega_L,   I(z) = int_0^z dz'/E(z').
The supernova chi^2 and the BAO chi^2 are fit_lcos.py's, on its z grid (4000 points on [0, 2.5]), with M_B and
c/(H0 r_d) profiled analytically and each clipped to the box, as fit_lcos_closed.py profiles them. s0 runs over
[0.001, 0.99], so u over [0.001^2, 0.99^2].

For each row of supernovae (all 1701 light curves; zHD > 0.01; zHD > 0.1) the statistic is the parameter
goodness-of-fit
    T = min_u [chi2_SN(u) + chi2_BAO(u)] - min_u chi2_SN(u) - min_u chi2_BAO(u).
Every minimum over u is taken the same way, for the data and for every mock: on a uniform grid of K_U points in u,
then a three-point parabola about the best grid point, kept between that point's neighbours and inside the box.

The calibration (the recon's §3.4). J = {(s0, H0rd) in the box : chi2_BAO(s0, H0rd) <= q}, with chi2_BAO not
profiled and q the (1 - BETA) quantile of chi^2 with one degree of freedom per BAO point; J comes from the BAO data
alone. The truths are TRUTHS through the first value past J's s0 range. At each value inside J, H0rd takes the BAO
data's own profiled value and the two ends of J's interval there; past J it takes the profiled value. The midpoints
between neighbouring truths are calibrated by the same rule. Mocks are drawn from the model at each truth through
the row's own covariances, supernovae and BAO independent, with M_B at the published best fit; the stream is fixed
by (SEED + k, row, truth), k = 0, 1, 2 for the profiled value, the low end and the high end. The p-value at a truth
is (1 + #{T_mock >= T})/(1 + N), with SE(p) = sqrt(p (1 - p)/N). p1 is the largest over the truths; the midpoints
beside its s0 are added and the largest taken again, p2. If p2 - p1 exceeds 3 SE(p1), the calibration is not
bounded and the row Uninformative; otherwise the graded p is min(1, p2 + 2 SE(p2) + BETA). Each bar b is equivalently a
threshold on T: at one truth, the (c + 1)-th largest mock T, c the largest count with p_c + 2 SE(p_c) + BETA < b and
p_c = (1 + c)/(1 + N); the row's threshold is the largest over the truths. Each mock set keeps its TAIL_KEEP largest
T and its count at or above the observed T, so that the thresholds and the graded p can be recomputed elsewhere.

The guard. Until the run is authorized, the cut and stress rows' supernovae are evaluated only in flat LCDM and in
Model D+Lambda at the floor, s0 = 0.001 with Omega_L = 0.685. Any other evaluation of those rows raises GuardError,
whether alone or summed with BAO. The script reads repository files only through an allowlist.

From scripts/:
    python sn_bao_consistency.py probe --out DIR
        The construction checks: the environment, the pinned inputs, parity with fit_lcos.py's own functions, the
        committed anchors that may be evaluated before the freeze (with fit_lcos.py's numerics and with the numerics
        they were computed with), the guard, a smoke test of the mocks, and the calibration's design (J and its
        truths, from the BAO data). Writes DIR/probe.json and nothing in the repo.
    python sn_bao_consistency.py qualify --out DIR
        The registration's gates (G1 to G7) and the rehearsal R, each arm after its parent passes, the floors beside
        the tolerances, and the frozen thresholds with their Monte Carlo errors. Writes DIR/qualification.json and
        nothing in the repo; the measured data meet only what the guard allows.
    python sn_bao_consistency.py check-pins --out DIR --registration PATH
        The run's first guard alone: the environment and the registration's pins. It evaluates nothing.
    python sn_bao_consistency.py run --out DIR --registration PATH
        The environment, the pins in the registration's Pins table, the gates at the frozen tolerances, then the
        analysis, once, with G6 on its own calibration and G5 on the graded rows (the stress row's move is reported
        beside it). Writes DIR/run.json and DIR/tails.npz.
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-threaded BLAS: deterministic sums, one process per core
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True              # keeps the checkout clean

import argparse, contextlib, copy, hashlib, io, json, math, platform, re, shutil, subprocess, tempfile, time
from multiprocessing import Pool
import numpy as np, pandas as pd, scipy
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2_law

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------- registered terms ----------------
OMEGA_L = 0.685
OMEGA_L_ANCHORS = (0.68, 0.69, 0.70, 0.715)                 # reported, not graded
C_LIGHT = 299792.458                                        # km/s
H0_SN = 70.0                                                # absorbed into M_B, as in fit_lcos.py
GRID_PUBLISHED = (4000, 2.5)                                # fit_lcos.py's z grid: N points on [0, Z_MAX]
NUMERICS_PUBLISHED = (GRID_PUBLISHED, "upper")              # its grid, and the covariance triangle it reads
S0_BOX, H0RD_BOX, MB_BOX = (0.001, 0.99), (8000.0, 12000.0), (-20.0, -18.0)
U_BOX = (S0_BOX[0] ** 2, S0_BOX[1] ** 2)
K_U = 1961                                                  # the profile grid in u, a step of about 5e-4
OM_BOX, K_OM = (0.05, 0.70), 2601                           # flat LCDM's Omega_m grid, a step of 2.5e-4 (reported only)
ROWS = {"all": None, "cut": 0.01, "stress": 0.1}            # zHD above this; None keeps every light curve
GRADED = ("all", "cut")
TRUTHS = (0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99)
N_MOCKS = 100_000
SEED = 20261117
BARS = (("CONSISTENT", 0.05), ("MILD TENSION", 0.0027))      # p >= 0.05; 0.0027 <= p < 0.05; below, TENSION
BOUND_SE, UNBOUNDED_SE = 2.0, 3.0
BETA = 1e-5                                                 # the probability that J misses the truth
KINDS = ("profiled", "low", "high")                         # H0rd at a truth: the BAO data's own value, then J's two ends
J_BRACKET, J_XTOL = (0.05, 0.9), 1e-12                      # where J's s0 range ends is found by brentq
TAIL_KEEP = 6000                                            # the largest mock T kept from each set
DV_EXPONENT = 1.0 / 3.0                                     # D_V = (z D_M^2 D_H)^(1/3); one arm sets 1/2
SYNTHETIC, FORECAST, COVERAGE = "data/sn_bao_synthetic.json", "data/sn_bao_forecast.json", "data/sn_bao_coverage.json"
GENERATOR = "scripts/sn_bao_generator.py"                   # written apart; G6 recomputes with its code
T_QUALIFY = 5.8867                                          # the all row's disclosed T: G6's counts at qualification
M_SHIFT = 0.1                                               # G6: the truth's offset moved this far, either way
FINE_K = 10 * (K_U - 1) + 1                                 # G5: a u grid ten times finer
COARSE_K = (K_U - 1) // 10 + 1                              # G5's arm: one ten times coarser
PLANTED_REHEARSAL = "b_sn0.389_bao0.001"                    # R: the synthetic set the run is rehearsed on
NULL_SETS = ("a_null", "b_sn0.3_bao0.3")                    # G3: the synthetic sets where T must be 0
ENV_PINNED = {"python": "3.9.6", "numpy": "2.0.2", "scipy": "1.13.1", "pandas": "2.3.3"}
BATCH = 2000

# Tolerances. A deterministic gate's tolerance sits at least FLOOR_MARGIN times above its measured floor, and the
# qualification record lists both. A statistical comparison is held to a number of its standard errors instead.
FLOOR_MARGIN = 10.0
Z_LIMIT, NOISE_SE_LIMIT, COVERAGE_SE_LIMIT = 4.0, 5.0, 3.0
TOL = {
    "G1_rel": 1e-9,
    "G2_chi2": 1e-5, "G2_s0": 1e-5,
    "G3_T": 1e-7,
    "G4_T": 1e-2,
    "G5_T": 1e-6,
    "G6_top": 1e-5, "G6_h0rd": 0.01, "G6_tail": 1e-12, "G6_M": 1e-9,
}

# SHA-256 pins, written from the files when this script was built (the bench's Step 1 record)
PINS = {
    "data/pantheon_plus.csv": "46a9334d15f7115435f662038692061a06fa36ba87e61e85bc17f5ac86c439a8",
    "data/pantheon_plus_cov.npy": "5c1b855c02e6a96ca8aa651e5e645af2db3b5f15f0c3647ca93c10ca27f8197e",
    "data/desi_dr2_bao.csv": "c746badffdef7f5e6b475d075f55a4de12173e991c94cb3a00eb498b9b5c254a",
    "data/desi_dr2_bao_cov.npy": "4d6ce177d7e96fb63199af88c2688a926dbb166acd61dc9f75dc0da7b4825c03",
    "results/lcos_summary.json": "b234fd104fedce1f333ec9b058448694ed945ea8ff1f86cdb11ef488fc2ac2bd",
    # the generator's outputs, the bench's coverage record, and the generator itself (step3b/, byte for byte)
    "data/sn_bao_synthetic.json": "95b65237bf0f4237b41da1e19c48eaa53424c14d0ca77c7f21b1c8616a3a2717",
    "data/sn_bao_forecast.json": "3638aa06b790086395d324a61520315deb9859610be3f1f5708da461b63ffe6b",
    "data/sn_bao_coverage.json": "c3eeb225d3a9ce5a8f34dc523f1638570901e1ee7bd8b698f3230791e4905f28",
    "scripts/sn_bao_generator.py": "bf093cb30e6f0e16878611fb79c3e444610426ca88b8e9bba3ef6cf9e6d68e34",
}
SPAN_SHA256 = "75d1089fb20328245a1ea1122b3844ac834af0e8e2d1f6cfa35b48870c788e5d"                               # fit_lcos.py, its lines before the MCMC marker
MCMC_MARKER = "# ---------------- MCMC"

# The files this script may read. The cut row's joint s0 profile is not among them.
ALLOWED_READS = frozenset(list(PINS) + [
    "scripts/fit_lcos.py",
    "results/clock_asymmetry_diag_snonly_curve.csv", "results/clock_asymmetry_diag_baoonly_curve.csv",
    "results/clock_asymmetry_tier1_curve.csv", "results/lcos_all_s0_profile.csv", "results/lcos_z001_robustness.json",
    "results/gamma_gate_fits.csv", "results/gamma_gate_split.csv",
])


class GuardError(RuntimeError):
    pass


AUTH = {"ok": False, "by": None}


def repo_path(rel):
    if rel not in ALLOWED_READS:
        raise GuardError(f"{rel} is not on the list of files this script may read")
    return os.path.join(REPO, rel)


def read_json(rel):
    with open(repo_path(rel), encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


# ---------------- data ----------------
_RAW = {}


def raw():
    if not _RAW:
        sn = pd.read_csv(repo_path("data/pantheon_plus.csv"))
        bao = pd.read_csv(repo_path("data/desi_dr2_bao.csv"))
        _RAW.update(z=sn.zHD.values, m=sn.m_b_corr.values, C=np.load(repo_path("data/pantheon_plus_cov.npy")),
                    bz=bao.z_eff.values, bobs=bao.observable.values, by=bao.value.values,
                    Cb=np.load(repo_path("data/desi_dr2_bao_cov.npy")))
    return _RAW


def numerics_of_anchors():
    """How every committed anchor was computed: 5000 points on [0, 1.002 max(z)], and the covariance's lower triangle."""
    r = raw()
    return ((5000, max(float(r["z"].max()), float(r["bz"].max())) * 1.002), "lower")


def best_fit():
    return read_json("results/lcos_summary.json")["best_fit"]


# ---------------- model ----------------
def e2(model, OL, z, p):
    if model == "Lcos":                       # p = u = s0^2, written in fit_lcos.py's order of operations
        Om = 1 - OL
        return Om / (1 - p) * (1 + z) ** 3 - Om * p / (1 - p) * (1 + z) + OL
    if model == "LCDM":                       # p = Omega_m
        return p * (1 + z) ** 3 + (1 - p)
    raise ValueError(model)


def shapes(grid, OL, model, p, z_sn, bz, bobs, dv_exp=DV_EXPONENT):
    """The BAO shapes f (times c/(H0 r_d) gives the r_d units) and the supernova modulus, as fit_lcos.py writes it."""
    N, zmax = grid
    zg = np.linspace(0, zmax, N)
    E = np.sqrt(e2(model, OL, zg, p))
    I = cumulative_trapezoid(1 / E, zg, initial=0)
    Ib, Eb = np.interp(bz, zg, I), np.interp(bz, zg, E)
    f = np.where(bobs == "DV_rd", (bz * Ib ** 2 / Eb) ** dv_exp, np.where(bobs == "DM_rd", Ib, 1 / Eb))
    mu = 5 * np.log10((1 + z_sn) * (C_LIGHT / H0_SN) * np.interp(z_sn, zg, I)) + 25
    return f, mu


class Row:
    """One row of supernovae with the BAO data, whitened. Everything is measured against a reference model, Model
    D+Lambda at the floor with the published M_B, which only keeps the arithmetic small and cancels exactly."""

    def __init__(self, name, numerics=None, dv_exp=DV_EXPONENT, drop_bao=None, cov_block="masked"):
        """drop_bao and cov_block exist for the arms only: a BAO point dropped, and the cut applied to the data with
        the covariance's leading block kept in place of the cut's own block."""
        grid, triangle = numerics or NUMERICS_PUBLISHED
        r = raw()
        zmin = ROWS[name]
        mask = np.ones(len(r["z"]), bool) if zmin is None else r["z"] > zmin
        self.name, self.grid, self.triangle, self.n, self.dv_exp = name, grid, triangle, int(mask.sum()), dv_exp
        self.mask, self.synthetic, self.data_label = mask, False, "measured"
        self.z, self._m = r["z"][mask], r["m"][mask]
        # The covariance file is not exactly symmetric (778 entries differ from their transposes, by 3e-8 at most).
        # fit_lcos.py's cho_factor reads its upper triangle, and so does this script; "lower" reproduces the
        # committed anchors, whose scripts read the lower triangle.
        C = r["C"][np.ix_(mask, mask)] if cov_block == "masked" else r["C"][: self.n, : self.n]
        self.U = cholesky(C, lower=(triangle == "lower"), check_finite=False)
        keep = np.ones(len(r["bz"]), bool)
        if drop_bao is not None:
            keep[drop_bao] = False
        self.bkeep = keep
        self.Lb = cholesky(r["Cb"][np.ix_(keep, keep)], lower=True, check_finite=False)
        self.bz, self.bobs, self.by = r["bz"][keep], r["bobs"][keep], r["by"][keep]
        self.e = self.w(np.ones(self.n))
        self.ee = float(self.e @ self.e)
        self.M_ref = float(best_fit()["MB"])
        self.mu_ref = shapes(grid, OMEGA_L, "Lcos", U_BOX[0], self.z, self.bz, self.bobs, dv_exp)[1]
        self.yb = self.wb(self.by)                                   # the BAO data, whitened; free to use
        self._cols = {}

    def with_synthetic(self, set_name):
        """The same row on one of the generator's pinned synthetic sets, sharing this row's model columns. The guard
        lets a synthetic row through: it can only carry a set from the pinned synthetic file."""
        st = synthetic_sets()[set_name]
        r = copy.copy(self)
        r.synthetic, r.data_label = True, set_name
        r._m = np.asarray(st["sn"], float)[self.mask]
        r.by = np.asarray(st["bao"], float)[self.bkeep]
        r.yb = r.wb(r.by)
        return r

    def w(self, v):
        if self.triangle == "lower":                                 # C = L L^T, from the lower triangle
            return solve_triangular(self.U, v, lower=True, check_finite=False)
        return solve_triangular(self.U, v, trans="T", lower=False, check_finite=False)   # C = U^T U, upper

    def wb(self, v):
        return solve_triangular(self.Lb, v, lower=True, check_finite=False)

    def columns(self, model, OL, ps):
        """Whitened model columns: supernova offsets from the reference (n x K) and BAO shapes (13 x K). Model only."""
        key = (model, OL, tuple(np.asarray(ps, float).tolist()))
        if key not in self._cols:
            F, MU = zip(*(shapes(self.grid, OL, model, p, self.z, self.bz, self.bobs, self.dv_exp) for p in ps))
            MU = np.array(MU).T - self.mu_ref[:, None]
            self._cols[key] = (self.w(MU), self.wb(np.array(F).T))
        return self._cols[key]

    def measured_sn(self, model, OL, ps):
        """The measured supernovae, whitened, for an evaluation at the parameters ps: the guard decides."""
        guard(self, model, OL, ps)
        return self.w(self._m - self.mu_ref - self.M_ref)


def guard(row, model, OL, ps):
    if AUTH["ok"] or row.synthetic or row.name == "all" or model == "LCDM":
        return
    if model == "Lcos" and OL == OMEGA_L and np.all(np.asarray(ps, float) == U_BOX[0]):
        return
    raise GuardError(f"Model D+Lambda on the {row.name} row's supernovae at Omega_L = {OL}, away from the floor or "
                     f"off the published Omega_L, before the run is authorized")


# ---------------- chi^2 curves ----------------
def sn_curves(row, Mt, Y, profile_M=True, want_offsets=False):
    """chi^2_SN with M_B profiled and clipped, for each model column of Mt (K) and each data vector of Y (N): K x N.
    want_offsets also returns the offset profiled with M unbounded, relative to the reference, K x N. profile_M=False
    holds the offset at the reference (an arm only)."""
    G = Mt.T @ Y
    ge = row.e @ Y
    yy = np.einsum("ij,ij->j", Y, Y)
    mm = np.einsum("ij,ij->j", Mt, Mt)
    em = row.e @ Mt
    a = ge[None, :] - em[:, None]
    if profile_M:
        d = np.clip(a / row.ee, MB_BOX[0] - row.M_ref, MB_BOX[1] - row.M_ref)
    else:
        d = np.zeros_like(a)
    S = yy[None, :] - 2 * G + mm[:, None] - 2 * d * a + d * d * row.ee
    return (S, a / row.ee) if want_offsets else S


def bao_curves(Ft, Bv):
    """chi^2_BAO with c/(H0 r_d) profiled and clipped: K x N, with the amplitudes."""
    G = Ft.T @ Bv
    ff = np.einsum("ij,ij->j", Ft, Ft)
    bb = np.einsum("ij,ij->j", Bv, Bv)
    A = np.clip(G / ff[:, None], C_LIGHT / H0RD_BOX[1], C_LIGHT / H0RD_BOX[0])
    return bb[None, :] - 2 * A * G + A * A * ff[:, None], A


def curve_min(V, x):
    """Each column's minimum over the uniform grid x: the best grid point, then the three-point parabola about it,
    kept between that point's neighbours. Returns the values and their locations."""
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[:, None]
    K, N = V.shape
    k = np.argmin(V, axis=0)
    cols = np.arange(N)
    if K < 3:
        return V[k, cols], np.asarray(x)[k]
    h = float(x[1] - x[0])
    kc = np.clip(k, 1, K - 2)
    v0, v1, v2 = V[kc - 1, cols], V[kc, cols], V[kc + 1, cols]
    a = (v2 - 2 * v1 + v0) / (2 * h * h)
    b = (v2 - v0) / (2 * h)
    lo = (np.maximum(k - 1, 0) - kc) * h
    hi = (np.minimum(k + 1, K - 1) - kc) * h
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a > 0, -b / (2 * a), np.where(v0 <= v2, lo, hi))
    t = np.clip(t, lo, hi)
    vp = v1 + b * t + a * t * t
    best = V[k, cols]
    take = vp < best
    return np.where(take, vp, best), np.where(take, x[kc] + t, x[k])


def statistic(S, B, x):
    s, xs = curve_min(S, x)
    b, xb = curve_min(B, x)
    j, xj = curve_min(S + B, x)
    return j - s - b, {"sn": (s, xs), "bao": (b, xb), "joint": (j, xj)}


def crossings(x, d, level):
    """Where the curve d (its minimum at zero) crosses level, on either side of its minimum; None at an edge."""
    i = int(np.argmin(d))
    lo = hi = None
    for j in range(len(x) - 1):
        if (d[j] - level) * (d[j + 1] - level) < 0:
            xc = x[j] + (level - d[j]) * (x[j + 1] - x[j]) / (d[j + 1] - d[j])
            if x[j] < x[i]:
                lo = xc
            elif hi is None:
                hi = xc
    return lo, hi


U_GRID = np.linspace(U_BOX[0], U_BOX[1], K_U)
OM_GRID = np.linspace(OM_BOX[0], OM_BOX[1], K_OM)


def measured_curves(row, model="Lcos", OL=OMEGA_L, x=None):
    """The measured chi^2 curves of one row over x (guarded), with the profiled nuisances."""
    x = (U_GRID if model == "Lcos" else OM_GRID) if x is None else np.asarray(x, float)
    y = row.measured_sn(model, OL, x)
    Mt, Ft = row.columns(model, OL, x)
    S = sn_curves(row, Mt, y[:, None])[:, 0]
    B, A = bao_curves(Ft, row.yb[:, None])
    return x, S, B[:, 0], C_LIGHT / A[:, 0]


def measured_row(row, model="Lcos", OL=OMEGA_L):
    x, S, B, h0rd = measured_curves(row, model, OL)
    T, parts = statistic(S, B, x)
    label = "s0" if model == "Lcos" else "Omega_m"
    to_label = (lambda v: math.sqrt(max(v, 0.0))) if model == "Lcos" else float
    out = {"row": row.name, "n_sn": row.n, "model": model, "Omega_L": OL if model == "Lcos" else None, "T": float(T[0])}
    for part, curve in (("sn", S), ("bao", B), ("joint", S + B)):
        v, at = parts[part]
        lo, hi = crossings(x, curve - curve.min(), 1.0)
        out[part] = {"chi2_min": float(v[0]), label: to_label(float(at[0])),
                     f"{label}_within_1": [None if lo is None else to_label(lo), None if hi is None else to_label(hi)]}
    if model == "Lcos":
        out["floor"] = {"chi2_sn": float(S[0]), "chi2_bao": float(B[0]), "H0rd_bao": float(h0rd[0])}
    return out


def chi2_at_floor(row):
    """Model D+Lambda at the floor on any row: allowed before the run."""
    x, S, B, h0rd = measured_curves(row, "Lcos", OMEGA_L, x=[U_BOX[0]])
    return {"chi2_sn": float(S[0]), "chi2_bao": float(B[0]), "joint": float(S[0] + B[0]), "H0rd_bao": float(h0rd[0])}


# ---------------- mocks and the calibration ----------------
def truth(row, s0_t, OL=OMEGA_L, h0rd_scale=1.0):
    """The null truth's whitened supernova offset and BAO prediction, c/(H0 r_d) at the BAO data's own value there."""
    Mt, Ft = row.columns("Lcos", OL, [s0_t ** 2])
    _, A = bao_curves(Ft, row.yb[:, None])
    A_t = float(A[0, 0]) / h0rd_scale
    return Mt[:, 0], A_t * Ft[:, 0], C_LIGHT / A_t


def truth_code(s0_t):
    return int(round(s0_t * 1_000_000))


def mock_T(row, s0_t, n, seed=SEED, OL=OMEGA_L, h0rd_scale=1.0, noise=True, diagnostics=False, m_shift=0.0, draw=None,
           profile_M=True):
    """T for n mocks at the null truth s0_t; the stream is fixed by (seed, row, truth).
    diagnostics also returns, for G6: the summed squared norms of the whitened supernova and BAO noise, and the range
    of the offset profiled with M unbounded over every grid point of every mock.
    m_shift moves the truth's offset (G6's invariance check). draw (an arm only) draws the supernova noise in data
    space through another factor and whitens it with the row's own. profile_M=False (an arm only) holds the offset."""
    Mt, Ft = row.columns("Lcos", OL, U_GRID)
    mt, bt, _ = truth(row, s0_t, OL, h0rd_scale)
    if m_shift:
        mt = mt + m_shift * row.e
    rng = np.random.default_rng(np.random.SeedSequence([seed, list(ROWS).index(row.name), truth_code(s0_t)]))
    out = np.empty(n)
    dg = {"N": n, "sn_noise_sq": 0.0, "bao_noise_sq": 0.0, "n_sn": row.n, "n_bao": len(bt),
          "M_grid": [math.inf, -math.inf]}
    for i in range(0, n, BATCH):
        nb = min(BATCH, n - i)
        eps = rng.standard_normal((row.n, nb)) if noise else np.zeros((row.n, nb))
        epb = rng.standard_normal((len(bt), nb)) if noise else np.zeros((len(bt), nb))
        if draw is not None:
            eps = row.w(draw @ eps)
        if diagnostics:
            S, off = sn_curves(row, Mt, mt[:, None] + eps, profile_M, True)
        else:
            S = sn_curves(row, Mt, mt[:, None] + eps, profile_M)
        B, _ = bao_curves(Ft, bt[:, None] + epb)
        out[i:i + nb] = statistic(S, B, U_GRID)[0]
        if diagnostics:
            M = row.M_ref + off                                  # K x nb: the unbounded offset at every grid point
            dg["M_grid"] = [min(dg["M_grid"][0], float(M.min())), max(dg["M_grid"][1], float(M.max()))]
            dg["sn_noise_sq"] += float(np.einsum("ij,ij->", eps, eps))
            dg["bao_noise_sq"] += float(np.einsum("ij,ij->", epb, epb))
    return (out, dg) if diagnostics else out


def p_value(Tm, T_obs):
    N = len(Tm)
    p = (1 + int(np.count_nonzero(Tm >= T_obs))) / (1 + N)
    return p, math.sqrt(p * (1 - p) / N)


def bound_p(p, n):
    """p plus twice its Monte Carlo standard error, SE(p) = sqrt(p (1 - p)/n)."""
    return p + BOUND_SE * math.sqrt(p * (1 - p) / n)


def bar_count(n, bar, beta=BETA):
    """The largest count c whose p, (1 + c)/(1 + n), with twice its standard error and beta, is below the bar."""
    c = -1
    while bound_p((1 + (c + 1)) / (1 + n), n) + beta < bar:
        c += 1
    return c


def count_threshold(tail, n, bar, beta=BETA):
    """One truth's threshold at a bar: the (c + 1)-th largest mock T, c = bar_count(n, bar, beta). tail: the largest
    mock T, ascending."""
    c = bar_count(n, bar, beta)
    if not 0 <= c < len(tail) - 1:
        raise ValueError("the kept tail is too short for this bar")
    return float(tail[-(c + 1)])


def grade(p):
    if p is None:
        return "UNINFORMATIVE"
    return "CONSISTENT" if p >= BARS[0][1] else ("MILD TENSION" if p >= BARS[1][1] else "TENSION")


def bao_profile(row, s0):
    """At s0: the BAO chi^2 profiled over H0rd in its box, that H0rd, the whitened shape's squared length ff, and the
    unclipped amplitude a0. The BAO data only, so free to use before the run."""
    Mt, Ft = row.columns("Lcos", OMEGA_L, [s0 ** 2])
    B, A = bao_curves(Ft, row.yb[:, None])
    f = Ft[:, 0]
    ff = float(f @ f)
    return float(B[0, 0]), C_LIGHT / float(A[0, 0]), ff, float(f @ row.yb) / ff


def calibration_design(row, dof=None):
    """The recon's §3.4, from the BAO data alone: q, where J's s0 range ends, the lattice, the truths and the
    midpoints, each truth as (s0, kind, H0rd). dof defaults to the number of BAO points (another value: an arm only)."""
    q = float(chi2_law.isf(BETA, len(row.yb) if dof is None else dof))
    prof = lambda s: bao_profile(row, s)
    top = brentq(lambda s: prof(s)[0] - q, J_BRACKET[0], J_BRACKET[1], xtol=J_XTOL)

    def at(s):
        B, H, ff, a0 = prof(s)
        out = [(s, "profiled", H)]
        if B < q:
            half = math.sqrt((q - B) / ff)
            out += [(s, "low", C_LIGHT / (a0 + half)), (s, "high", C_LIGHT / (a0 - half))]
        return out

    lattice = [b for b in TRUTHS if b <= top] + [min(b for b in TRUTHS if b > top)]
    truths = [t for s in lattice for t in at(s)]
    midpoints = [t for a, b in zip(lattice[:-1], lattice[1:]) for t in at(round((a + b) / 2, 6))]
    return {"beta": BETA, "q": q, "top": top, "lattice": lattice, "truths": truths, "midpoints": midpoints}


_POOL_ROWS = {}


def _mock_task(args):
    name, s0_t, kind, H, n, seed, diagnostics = args
    if name not in _POOL_ROWS:
        _POOL_ROWS[name] = Row(name)
    row = _POOL_ROWS[name]
    H_prof = truth(row, s0_t)[2]
    scale = 1.0 if kind == "profiled" else H / H_prof
    got = mock_T(row, s0_t, n, seed + KINDS.index(kind), h0rd_scale=scale, diagnostics=diagnostics)
    Tm, dg = got if diagnostics else (got, None)
    return name, s0_t, kind, Tm, dg


def calibrate(rows, truths, n=N_MOCKS, seed=SEED, processes=4, diagnostics=False):
    """Mock T at each truth (s0, kind, H0rd), for each row: {row: {(s0, kind): T}}; with diagnostics, also
    {row: {(s0, kind): G6's diagnostics}}."""
    tasks = [(r, s, k, H, n, seed, diagnostics) for r in rows for s, k, H in truths]
    out, dgs = {r: {} for r in rows}, {r: {} for r in rows}
    with Pool(processes) as pool:
        for name, s0_t, kind, Tm, dg in pool.imap_unordered(_mock_task, tasks):
            out[name][(s0_t, kind)] = Tm
            dgs[name][(s0_t, kind)] = dg
    return (out, dgs) if diagnostics else out


def set_summary(Tm, T_obs):
    """One mock set at the observed T: N, P(T = 0), three quantiles, the count at or above T_obs, p and its SE, and
    the kept tail (the TAIL_KEEP largest T, ascending)."""
    p, se = p_value(Tm, T_obs)
    q = np.quantile(Tm, [0.5, 0.95, 0.9973])
    return {"N": int(len(Tm)), "P_T0": float(np.mean(Tm < 1e-9)), "q50": float(q[0]), "q95": float(q[1]),
            "q9973": float(q[2]), "count": int(np.count_nonzero(Tm >= T_obs)), "p": p, "se": se,
            "tail": np.sort(Tm)[-TAIL_KEEP:]}


def row_thresholds(sums, design, beta=BETA):
    """Each bar as a threshold on T: the largest, over the truths, of each truth's threshold by counts, each at its own
    number of mocks. beta other than BETA: an arm only."""
    keys = [(s, k) for s, k, _ in design["truths"]]
    return {str(b): max(count_threshold(sums[t]["tail"], sums[t]["N"], b, beta) for t in keys) for _, b in BARS}


def graded_p(sums, design, beta=BETA, cap=1.0):
    """The recon's §3.4: p1, the largest over the truths; the midpoints beside its s0 added, giving p2; not bounded
    if p2 - p1 exceeds 3 SE(p1); otherwise min(1, p2 + 2 SE(p2) + beta) (BETA, and a cap of 1; another beta or
    another cap: an arm only)."""
    keys = [(s, k) for s, k, _ in design["truths"]]
    k1 = max(keys, key=lambda t: sums[t]["p"])
    L = design["lattice"]
    i = L.index(k1[0])
    mids = [round((L[j] + L[j + 1]) / 2, 6) for j in (i - 1, i) if 0 <= j < len(L) - 1]
    k2 = max([k1] + [(s, k) for s, k, _ in design["midpoints"] if s in mids], key=lambda t: sums[t]["p"])
    p1, se1, p2, se2 = sums[k1]["p"], sums[k1]["se"], sums[k2]["p"], sums[k2]["se"]
    bounded = p2 - p1 <= UNBOUNDED_SE * se1
    moved = p2 - p1                                              # a p of 1 has no standard error: nothing can move
    return {"p1": p1, "at1": list(k1), "p2": p2, "at2": list(k2), "bounded": bounded,
            "moved_in_se": moved / se1 if se1 > 0 else (0.0 if moved == 0 else math.inf),
            "p": min(cap, p2 + BOUND_SE * se2 + beta) if bounded else None}


def grid_x10(row, model="Lcos", OL=OMEGA_L):
    """G5 on a measured row: T on the registered u grid and on one ten times finer. Guarded like any measured
    evaluation."""
    fine = np.linspace(U_BOX[0], U_BOX[1], 10 * (K_U - 1) + 1)
    x, S, B, _ = measured_curves(row, model, OL, x=fine)
    row._cols.pop((model, OL, tuple(fine.tolist())), None)        # large: kept out of the cache
    return {"T": measured_row(row, model, OL)["T"], "T_x10": float(statistic(S, B, fine)[0][0])}


# ---------------- the construction checks (probe) ----------------
def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "platform": platform.platform()}


def check_environment():
    env = environment()
    bad = {k: [env[k], v] for k, v in ENV_PINNED.items() if env[k] != v}
    return {"pass": not bad, "mismatch": bad, "environment": env}


def check_pins():
    got = {rel: sha256_file(repo_path(rel)) for rel in PINS}
    return {"pass": got == PINS, "files": {rel: {"sha256": got[rel], "pinned": got[rel] == PINS[rel]} for rel in PINS}}


def run_published_span():
    """fit_lcos.py up to its MCMC marker, with sys.argv pinned and its own directory as the working directory."""
    with open(repo_path("scripts/fit_lcos.py"), encoding="utf-8") as fh:
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


def parity(OL=OMEGA_L, dv_exp=DV_EXPONENT):
    """The instrument's E, chi^2_SN, chi^2_BAO and BAO predictions against fit_lcos.py's own, on the published sample.
    OL and dv_exp other than the registered ones: arms only."""
    ns, sha, n = run_published_span()
    row = Row("all", NUMERICS_PUBLISHED)
    zg = ns["zgrid"]
    worst = {"E": 0.0, "sn_chi2": 0.0, "bao_chi2": 0.0, "bao_pred": 0.0}
    bao_pub, cinv_pub = ns["bao"], ns["Cbao_inv"]
    try:
        for s0 in (0.001, 0.05, 0.1, 0.2, 0.4):
            worst["E"] = max(worst["E"], rel(np.sqrt(e2("Lcos", OL, zg, s0 ** 2)), ns["E"](zg, s0)))
            f, mu = shapes(GRID_PUBLISHED, OL, "Lcos", s0 ** 2, row.z, row.bz, row.bobs, dv_exp)
            guard(row, "Lcos", OMEGA_L, [s0 ** 2])                 # the published sample: always allowed
            for mb in (-19.40, -19.354, -19.30):
                r = row.w(row._m - mb - mu)
                worst["sn_chi2"] = max(worst["sn_chi2"], rel(r @ r, ns["sn_chi2"](s0, mb)))
            for h in (9900.0, 10025.43, 10150.0):
                pred = (C_LIGHT / h) * f
                r = row.wb(pred - row.by)
                worst["bao_chi2"] = max(worst["bao_chi2"], rel(r @ r, ns["bao_chi2"](s0, h)))
                ns["bao"] = bao_pub.assign(value=0.0)
                for i in range(len(pred)):
                    P = np.zeros_like(cinv_pub); P[i, i] = 1.0
                    ns["Cbao_inv"] = P
                    worst["bao_pred"] = max(worst["bao_pred"], rel(pred[i], math.sqrt(max(ns["bao_chi2"](s0, h), 0.0))))
                ns["bao"], ns["Cbao_inv"] = bao_pub, cinv_pub
    finally:
        ns["bao"], ns["Cbao_inv"] = bao_pub, cinv_pub
    return {"span_lines": n, "span_sha256": sha, "span_pinned": sha == SPAN_SHA256, "worst_rel": worst,
            "floor": max(worst.values())}


def committed_anchors():
    """The committed values that may be evaluated before the freeze, read from their files (never typed)."""
    def eps0(rel_path, col):
        df = pd.read_csv(repo_path(rel_path))
        r = df.loc[np.isclose(df["eps"], 0.0)].iloc[0]
        return float(r[col]), float(r["s0"])
    rob = read_json("results/lcos_z001_robustness.json")
    fits = pd.read_csv(repo_path("results/gamma_gate_fits.csv"))
    lcdm_cut = fits[(fits.variant == "z001") & (fits.model == "LCDM")].iloc[0]
    split = pd.read_csv(repo_path("results/gamma_gate_split.csv"))
    g3 = split.loc[np.isclose(split["gamma"], 3.0)].iloc[0]
    prof = pd.read_csv(repo_path("results/lcos_all_s0_profile.csv"))
    at39 = prof.loc[np.isclose(prof["s0"], 0.39)].iloc[0]
    sn_c, sn_s = eps0("results/clock_asymmetry_diag_snonly_curve.csv", "chi2_sn")
    bao_c, bao_s = eps0("results/clock_asymmetry_diag_baoonly_curve.csv", "chi2_bao")
    j_c, j_s = eps0("results/clock_asymmetry_tier1_curve.csv", "chi2_total")
    return {"all": {"sn": (sn_c, sn_s), "bao": (bao_c, bao_s), "joint": (j_c, j_s), "T": j_c - sn_c - bao_c,
                    "joint_at_0.39_above_min": float(at39["dchi2"]), "joint_robustness_file": rob["all"]["chi2_Lcos"]},
            "cut": {"lcdm_joint": float(lcdm_cut["chi2"]), "lcdm_sn": float(g3["chi2_sn"]),
                    "lcdm_bao": float(g3["chi2_bao"]), "lcos_joint_at_floor": rob["z001"]["chi2_Lcos"],
                    "lcos_s0_at_floor": rob["z001"]["s0_best"], "n": rob["counts"]["z001"]},
            "limits_95": {"all": rob["profile_UL"]["s0_95UL_full"], "cut": rob["profile_UL"]["s0_95UL_z001"]}}


def instrument_anchors(numerics, drop_bao=None, cov_block="masked"):
    """drop_bao and cov_block other than the registered ones: arms only."""
    allr = Row("all", numerics, drop_bao=drop_bao)
    cut = Row("cut", numerics, drop_bao=drop_bao, cov_block=cov_block)
    a = measured_row(allr)
    x, S, B, _ = measured_curves(allr)
    J = S + B
    j39 = measured_curves(allr, x=[0.39 ** 2])
    lc = measured_row(cut, "LCDM")
    fl = chi2_at_floor(cut)
    return {"all": {"sn": (a["sn"]["chi2_min"], a["sn"]["s0"]), "bao": (a["bao"]["chi2_min"], a["bao"]["s0"]),
                    "joint": (a["joint"]["chi2_min"], a["joint"]["s0"]), "T": a["T"],
                    "joint_at_0.39_above_min": float(j39[1][0] + j39[2][0]) - a["joint"]["chi2_min"],
                    "sn_within_1": a["sn"]["s0_within_1"], "bao_within_1": a["bao"]["s0_within_1"]},
            "cut": {"n": cut.n, "lcdm_joint": lc["joint"]["chi2_min"], "lcdm_sn": lc["sn"]["chi2_min"],
                    "lcdm_bao": lc["bao"]["chi2_min"], "lcdm_T": lc["T"], "lcos_joint_at_floor": fl["joint"],
                    "lcos_sn_at_floor": fl["chi2_sn"], "lcos_bao_at_floor": fl["chi2_bao"]}}


def compare_anchors(inst, com):
    rows = []
    for part in ("sn", "bao", "joint"):
        rows.append((f"all {part}: chi2 min", inst["all"][part][0], com["all"][part][0]))
        rows.append((f"all {part}: s0 at the min", inst["all"][part][1], com["all"][part][1]))
    rows.append(("all T", inst["all"]["T"], com["all"]["T"]))
    rows.append(("all joint at s0 = 0.39, above its min", inst["all"]["joint_at_0.39_above_min"], com["all"]["joint_at_0.39_above_min"]))
    for k in ("lcdm_joint", "lcdm_sn", "lcdm_bao", "lcos_joint_at_floor"):
        rows.append((f"cut {k}", inst["cut"][k], com["cut"][k]))
    rows.append(("cut T in flat LCDM", inst["cut"]["lcdm_T"], com["cut"]["lcdm_joint"] - com["cut"]["lcdm_sn"] - com["cut"]["lcdm_bao"]))
    return [{"quantity": q, "instrument": float(i), "committed": float(c), "diff": float(i - c)} for q, i, c in rows]


def guard_probe():
    cut, stress = Row("cut"), Row("stress")
    refused, allowed = [], []
    for label, fn in (("cut row, D+Lambda supernova curve over the u grid", lambda: measured_curves(cut)),
                      ("cut row, D+Lambda at s0 = 0.1", lambda: measured_curves(cut, x=[0.01])),
                      ("cut row, D+Lambda at the floor with Omega_L = 0.69", lambda: measured_curves(cut, OL=0.69, x=[U_BOX[0]])),
                      ("cut row, the whole D+Lambda row", lambda: measured_row(cut)),
                      ("stress row, D+Lambda at s0 = 0.2", lambda: measured_curves(stress, x=[0.04])),
                      ("the cut row's joint s0 profile, read from results/", lambda: repo_path("results/lcos_z001_s0_profile.csv"))):
        try:
            fn()
            refused.append({"what": label, "refused": False})
        except GuardError as err:
            refused.append({"what": label, "refused": True, "message": str(err)})
    for label, fn in (("cut row, flat LCDM", lambda: measured_row(cut, "LCDM")),
                      ("cut row, D+Lambda at the floor", lambda: chi2_at_floor(cut)),
                      ("stress row, flat LCDM", lambda: measured_row(stress, "LCDM")),
                      ("stress row, D+Lambda at the floor", lambda: chi2_at_floor(stress))):
        try:
            fn()
            allowed.append({"what": label, "ran": True})
        except GuardError as err:
            allowed.append({"what": label, "ran": False, "message": str(err)})
    return {"pass": all(r["refused"] for r in refused) and all(a["ran"] for a in allowed) and not AUTH["ok"],
            "refused": refused, "allowed": allowed}


def mock_smoke(n=10_000):
    out = {}
    for name in GRADED:
        row = Row(name)
        nf = {str(s): float(mock_T(row, s, 4, noise=False)[0]) for s in (0.001, 0.2, 0.4, 0.99)}
        t0 = time.time()
        Tm = mock_T(row, 0.001, n)
        dt = time.time() - t0
        q = np.quantile(Tm, [0.5, 0.95, 0.9973])
        out[name] = {"noise_free_T": nf, "n": n, "seconds": dt, "seconds_per_100k": dt * 100_000 / n,
                     "P(T=0)": float(np.mean(Tm < 1e-9)), "min_T": float(Tm.min()),
                     "quantiles_0.5_0.95_0.9973": [float(v) for v in q]}
    return out


def probe(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rec = {"environment": check_environment()}
    if not rec["environment"]["pass"]:
        json.dump(rec, open(os.path.join(out_dir, "probe.json"), "w"), indent=1, default=_jsonable)
        sys.exit("the environment is not the pinned one: the probe stops")
    t0 = time.time()
    rec["pins"] = check_pins()
    rec["parity"] = parity()
    com = committed_anchors()
    rec["anchors"] = {"committed": com,
                      "published_numerics": {"numerics": NUMERICS_PUBLISHED, "values": instrument_anchors(NUMERICS_PUBLISHED)},
                      "anchors_numerics": {"numerics": numerics_of_anchors(), "values": instrument_anchors(numerics_of_anchors())}}
    for g in ("published_numerics", "anchors_numerics"):
        rec["anchors"][g]["against_committed"] = compare_anchors(rec["anchors"][g]["values"], com)
    rec["guard"] = guard_probe()
    rec["mocks"] = mock_smoke()
    rec["design"] = calibration_design(Row("all"))
    rec["seconds"] = time.time() - t0
    rec["authorized"] = AUTH["ok"]
    with open(os.path.join(out_dir, "probe.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=_jsonable)
    return rec


# ---------------- the run ----------------
def registration_pins(path):
    """The Pins table of the registration page: '| pin | `value` |' rows under a heading '### Pins'."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"^### Pins\n(.*?)(?=^#|\Z)", text, re.S | re.M)
    if not m:
        raise SystemExit("the registration has no Pins table")
    return dict(re.findall(r"^\| ([^|`]+?) \| `([0-9a-f]{40}|[0-9a-f]{64})` \|$", m.group(1), re.M))


def authorize(pins):
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    clean = not subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout
    checks = {"design commit": head == pins.get("design commit"),
              "clean checkout": clean,
              "script": sha256_file(os.path.abspath(__file__)) == pins.get("script")}
    for rel in PINS:
        checks[rel] = sha256_file(repo_path(rel)) == pins.get(rel) == PINS[rel]
    ok = all(checks.values())
    if ok:
        AUTH.update(ok=True, by=pins.get("design commit"))
    return {"pass": ok, "checks": checks}


def check_pins_only(registration):
    """The run's first guard and nothing else: the environment and the registration's pins. It evaluates nothing."""
    rec = {"environment": check_environment(), "pins": authorize(registration_pins(registration))}
    AUTH.update(ok=False, by=None)
    return rec


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True).stdout.strip()


# ---------------- synthetic sets (the generator's, pinned) and shared rows ----------------
_SYN, _BASE = {}, {}


def synthetic_sets():
    if not _SYN:
        _SYN.update(read_json(SYNTHETIC)["sets"])
    return _SYN


def base_row(name, dv_exp=DV_EXPONENT, drop_bao=None, cov_block="masked"):
    """One row per configuration, kept, so that its synthetic copies share its model columns."""
    key = (name, dv_exp, drop_bao, cov_block)
    if key not in _BASE:
        _BASE[key] = Row(name, dv_exp=dv_exp, drop_bao=drop_bao, cov_block=cov_block)
    return _BASE[key]


def T_on(row, x):
    """T on a row's own data (measured, so guarded; or synthetic) over the grid x."""
    x, S, B, _ = measured_curves(row, x=x)
    return float(statistic(S, B, x)[0][0])


def generator():
    """The generator's module, from scripts/. It does not import this script."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import sn_bao_generator
    return sn_bao_generator


def limit_truths(row):
    """The truths at the rows' published 95% profile limits, H0rd profiled: reported p-values only."""
    return [(s, "profiled", bao_profile(row, s)[1]) for s in sorted(set(committed_anchors()["limits_95"].values()))]


def chi2_tail_closed(x, k):
    """P(chi^2 with k degrees of freedom > x), in closed form: this script's own, written apart from scipy's."""
    h = x / 2.0
    if k % 2 == 0:
        term = tot = 1.0
        for j in range(1, k // 2):
            term *= h / j
            tot += term
        return math.exp(-h) * tot
    tot, term = math.erfc(math.sqrt(h)), 2.0 * math.sqrt(h / math.pi)
    for j in range(1, (k + 1) // 2):
        tot += math.exp(-h) * term
        term *= h / (j + 0.5)
    return tot


def _worst(d, key=abs):
    k = max(d, key=lambda kk: key(d[kk]))
    return k, d[k]


# ---------------- the gates ----------------
def gate_G1(OL=OMEGA_L, dv_exp=DV_EXPONENT):
    """Parity: the instrument's E, chi^2_SN, chi^2_BAO and BAO predictions equal fit_lcos.py's at fixed points."""
    p = parity(OL, dv_exp)
    ok = p["span_pinned"] and p["floor"] <= TOL["G1_rel"]
    part, v = _worst(p["worst_rel"])
    return {"gate": "G1", "pass": bool(ok), "parity": p, "floor": p["floor"], "tol": TOL["G1_rel"],
            "why": None if ok else f"{part} departs from fit_lcos.py's by {v:.2e}, relative"}


def gate_G2(drop_bao=None, cov_block="masked", report_published=False):
    """Anchors: the committed values, on the numerics they were computed with (§2.2): every one on the all row, and on
    the cut row the flat LCDM reference and the joint value at the floor."""
    com = committed_anchors()
    cmp_ = compare_anchors(instrument_anchors(numerics_of_anchors(), drop_bao, cov_block), com)
    is_s0 = lambda r: "s0 at" in r["quantity"]
    chi2_dev = max(abs(r["diff"]) for r in cmp_ if not is_s0(r))
    s0_dev = max(abs(r["diff"]) for r in cmp_ if is_s0(r))
    ok = chi2_dev <= TOL["G2_chi2"] and s0_dev <= TOL["G2_s0"]
    w = max(cmp_, key=lambda r: abs(r["diff"]) / (TOL["G2_s0"] if is_s0(r) else TOL["G2_chi2"]))
    out = {"gate": "G2", "pass": bool(ok), "against_committed": cmp_, "chi2_dev": chi2_dev, "s0_dev": s0_dev,
           "tol": {"G2_chi2": TOL["G2_chi2"], "G2_s0": TOL["G2_s0"]},
           "why": None if ok else f"{w['quantity']} departs from its committed value by {w['diff']:+.3e}"}
    if report_published:
        out["published_numerics_reported"] = compare_anchors(instrument_anchors(NUMERICS_PUBLISHED), com)
    return out


def gate_G3(null_sets=NULL_SETS):
    """Null: T = 0 on the synthetic sets where both datasets share one s0."""
    T = {f"{s}|{name}": T_on(base_row(name).with_synthetic(s), U_GRID) for s in null_sets for name in GRADED}
    k, v = _worst(T)
    ok = abs(v) <= TOL["G3_T"]
    return {"gate": "G3", "pass": bool(ok), "T": T, "floor": abs(v), "tol": TOL["G3_T"],
            "why": None if ok else f"T = {v:.4g} on {k}, where it must be 0"}


def gate_G4(dv_exp=DV_EXPONENT):
    """Planted tension: T on the planted synthetic sets equals the generator's."""
    dev = {}
    for s, st in synthetic_sets().items():
        if not s.startswith("b_"):
            continue
        for name in GRADED:
            Ti = T_on(base_row(name, dv_exp=dv_exp).with_synthetic(s), U_GRID)
            dev[f"{s}|{name}"] = {"instrument": Ti, "generator": st["T"][name]["T"], "diff": Ti - st["T"][name]["T"]}
    k, v = _worst(dev, key=lambda d: abs(d["diff"]))
    ok = abs(v["diff"]) <= TOL["G4_T"]
    return {"gate": "G4", "pass": bool(ok), "sets": dev, "floor": abs(v["diff"]), "tol": TOL["G4_T"],
            "why": None if ok else f"T on {k} is {v['instrument']:.5f} against the generator's {v['generator']:.5f}"}


def gate_G5(k=None):
    """Grids: T on every synthetic set moves little on a u grid ten times finer. k other than the registered grid's
    points: an arm only."""
    grid = U_GRID if k is None else np.linspace(U_BOX[0], U_BOX[1], k)
    fine = np.linspace(U_BOX[0], U_BOX[1], FINE_K)
    move = {}
    for s in synthetic_sets():
        for name in GRADED:
            r = base_row(name).with_synthetic(s)
            move[f"{s}|{name}"] = T_on(r, fine) - T_on(r, grid)
    kk, v = _worst(move)
    ok = abs(v) <= TOL["G5_T"]
    return {"gate": "G5", "pass": bool(ok), "moves": move, "points": len(grid), "fine_points": FINE_K, "floor": abs(v),
            "tol": TOL["G5_T"], "why": None if ok else f"T on {kk} moves by {v:+.4g} on the finer grid"}


def gate_G5_measured(res):
    """G5 at the run: each graded row's T on the finer grid. A row that is reported and not graded cannot fail the
    gate: its move is recorded beside it."""
    move = {name: res["grid_x10"][name]["T_x10"] - res["grid_x10"][name]["T"] for name in GRADED}
    kk, v = _worst(move)
    ok = abs(v) <= TOL["G5_T"]
    reported = {name: {"move": r["T_x10"] - r["T"], "within_tolerance": bool(abs(r["T_x10"] - r["T"]) <= TOL["G5_T"])}
                for name, r in res["grid_x10"].items() if name not in GRADED}
    return {"gate": "G5 (graded rows)", "pass": bool(ok), "moves": move, "reported": reported, "tol": TOL["G5_T"],
            "why": None if ok else f"the {kk} row's T moves by {v:+.4g} on the finer grid"}


# ---- G6, part by part ----
def g6_design(design, gen_design):
    """J's top, its lattice, its truths and midpoints, and their H0rd values, against the generator's."""
    mine = [tuple(t) for t in design["truths"] + design["midpoints"]]
    theirs = [tuple(t) for t in gen_design["truths"] + gen_design["midpoints"]]
    keys_ok = [t[:2] for t in mine] == [t[:2] for t in theirs] and list(design["lattice"]) == list(gen_design["lattice"])
    dtop = abs(design["top"] - gen_design["top"])
    dh = max(abs(a[2] - b[2]) for a, b in zip(mine, theirs)) if keys_ok else math.inf
    ok = keys_ok and dtop <= TOL["G6_top"] and dh <= TOL["G6_h0rd"]
    why = None if ok else ("the lattice or the truths differ from the generator's" if not keys_ok else
                           f"J's top differs by {dtop:.2e} or an H0rd by {dh:.2e} km/s")
    return {"pass": bool(ok), "keys_equal": keys_ok, "top": design["top"], "top_generator": gen_design["top"],
            "top_diff": dtop, "h0rd_diff": dh, "lattice": list(design["lattice"]), "why": why}


def g6_q(design, k_bao, gen_design):
    """q's upper tail is beta under chi^2 with as many degrees of freedom as BAO points, by this script's own closed
    form and by the generator's."""
    tail = chi2_tail_closed(design["q"], k_bao)
    gtail = gen_design["tail_at_q_closed_form"]
    d = max(abs(tail - BETA), abs(gtail - BETA))
    ok = d <= TOL["G6_tail"] and gen_design["k"] == k_bao and design["beta"] == BETA
    return {"pass": bool(ok), "q": design["q"], "k": k_bao, "tail_closed_form": tail, "tail_generator": gtail,
            "floor": d, "why": None if ok else f"q's upper tail with {k_bao} degrees of freedom is {tail:.6e}, not {BETA}"}


def g6_coverage(path=None):
    """The bench's coverage record, read back: J missed the truth at the rate beta in 20 million BAO mocks."""
    rec = read_json(COVERAGE) if path is None else json.load(open(path, encoding="utf-8"))
    rs = rec["records"]
    N = sum(r["N"] for r in rs)
    miss = sum(int(round(r["outside_J_fraction"][str(BETA)] * r["N"])) for r in rs)
    exp, sd = BETA * N, math.sqrt(BETA * (1 - BETA) * N)
    z = (miss - exp) / sd
    ok = len(rs) == 5 and N == 20_000_000 and abs(z) <= COVERAGE_SE_LIMIT
    return {"pass": bool(ok), "N": N, "misses": miss, "expected": exp, "sd": sd, "z": z,
            "why": None if ok else f"{miss} misses in {N} BAO mocks against {exp:.0f} +- {sd:.0f}"}


def _set_keys(design):
    return [(s, k) for s, k, _ in design["truths"] + design["midpoints"]]


def g6_halves(cal, design, neighbour=False):
    """At every calibrated set and bar, the two halves of the mocks put the same share above the set's threshold: the
    two counts agree within Z_LIMIT standard errors. neighbour (an arm only) takes the second half from the next set."""
    keys, z = _set_keys(design), {}
    for name in cal:
        for i, key in enumerate(keys):
            T = cal[name][key]
            h = len(T) // 2
            A, B = T[:h], (cal[name][keys[(i + 1) % len(keys)]] if neighbour else T)[h:]
            tail = np.sort(T)[-TAIL_KEEP:]
            for _, b in BARS:
                th = count_threshold(tail, len(T), b)
                na, nb = int(np.count_nonzero(A >= th)), int(np.count_nonzero(B >= th))
                z[f"{name}|{key[0]}|{key[1]}|{b}"] = (na - nb) / math.sqrt(max(na + nb, 1))
    k, v = _worst(z)
    ok = abs(v) <= Z_LIMIT
    return {"pass": bool(ok), "worst": k, "z": v, "rms": math.sqrt(sum(x * x for x in z.values()) / len(z)), "n": len(z),
            "why": None if ok else f"the halves at {k} differ by {v:+.2f} standard errors"}


def g6_generator(cal, gen_fc, mislabel=False):
    """At every set and bar, the instrument's count of mocks at or above the generator's threshold agrees with the
    generator's own, c + 1, within Z_LIMIT standard errors. mislabel (an arm only) compares each of the generator's
    truths with the instrument's at the next lattice value."""
    GEN = generator()
    lattice = gen_fc["design"]["lattice"]
    z = {}
    for name in cal:
        for r in gen_fc["rows"][name]["by_truth"] + gen_fc["rows"][name]["midpoints"]:
            key = (r["s0"], r["kind"])
            if mislabel:
                if r["s0"] not in lattice or r["s0"] == lattice[-1]:
                    continue
                nxt = lattice[lattice.index(r["s0"]) + 1]
                key = (nxt, r["kind"]) if (nxt, r["kind"]) in cal[name] else (nxt, "profiled")
            T = cal[name][key]
            for _, b in BARS:
                ni, ng = int(np.count_nonzero(T >= r["thresholds"][str(b)])), GEN.count_for_bar(r["N"], b) + 1
                Ni, Ng = len(T), r["N"]
                pool = (ni + ng) / (Ni + Ng)
                z[f"{name}|{r['s0']}|{r['kind']}|{b}"] = (ni / Ni - ng / Ng) / math.sqrt(pool * (1 - pool) * (1 / Ni + 1 / Ng))
    k, v = _worst(z)
    ok = abs(v) <= Z_LIMIT
    return {"pass": bool(ok), "worst": k, "z": v, "rms": math.sqrt(sum(x * x for x in z.values()) / len(z)), "n": len(z),
            "why": None if ok else f"at {k} the codes' counts differ by {v:+.2f} standard errors"}


def g6_recompute(sums, design, out_dir, beta=BETA, nudge=0.0, cap=1.0):
    """Each row's thresholds and graded p, recomputed by the generator's code from the tails and counts this script
    saves, equal this script's exactly; and where every mock is at or above the observed T, both codes give a graded
    p of exactly 1. beta other than BETA, nudge other than 0 and cap other than 1: arms only."""
    GEN = generator()
    tp, cp = os.path.join(out_dir, "g6_tails.npz"), os.path.join(out_dir, "g6_counts.json")
    np.savez_compressed(tp, **{f"{n}|{s}|{k}": d["tail"] for n in sums for (s, k), d in sums[n].items()})
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump({n: {f"{s}|{k}": {"N": d["N"], "count": d["count"]} for (s, k), d in sums[n].items()} for n in sums}, fh)
    gen_design = GEN.design(GEN.BAOData(GEN.load(REPO)))
    theirs = GEN.recompute(tp, cp, gen_design)
    mine, diffs = {}, []
    for name in sums:
        thr = row_thresholds(sums[name], design, beta=beta)
        if nudge:
            thr[str(BARS[0][1])] += nudge
        gp = graded_p(sums[name], design, beta=beta, cap=cap)["p"]
        mine[name] = {"thresholds": thr, "graded_p": gp}
        if thr != theirs[name]["thresholds"]:
            diffs.append(f"the {name} row's thresholds {thr} against the generator's {theirs[name]['thresholds']}")
        if gp != theirs[name]["graded"]["p"]:
            diffs.append(f"the {name} row's graded p {gp!r} against the generator's {theirs[name]['graded']['p']!r}")
    # where every mock is at or above the observed T, each p is 1 and the graded p must be exactly 1, in both codes
    n0 = next(iter(next(iter(sums.values())).values()))["N"]
    ones = {(s, k): {"p": 1.0, "se": 0.0} for s, k, _ in design["truths"] + design["midpoints"]}
    theirs_ones = {(s, k): n0 for s, k, _ in gen_design["truths"] + gen_design["midpoints"]}
    zero = {"instrument": graded_p(ones, design, beta=beta, cap=cap)["p"],
            "generator": GEN.graded_from_counts(theirs_ones, n0, gen_design)["p"]}
    if not (zero["instrument"] == zero["generator"] == 1.0):
        diffs.append(f"where every mock is at or above the observed T, the graded p is {zero['instrument']!r} here and "
                     f"{zero['generator']!r} in the generator, where both must be 1")
    return {"pass": not diffs, "instrument": mine, "generator": {n: {"thresholds": theirs[n]["thresholds"],
            "graded_p": theirs[n]["graded"]["p"]} for n in theirs}, "at_T_zero": zero, "why": "; ".join(diffs) or None}


def g6_noise(dg):
    """The mocks are drawn through the row's own covariances: at every set the whitened noise's mean squared norm is
    the number of points, within NOISE_SE_LIMIT standard errors, for the supernovae and for BAO."""
    z = {}
    for name in dg:
        for (s, k), d in dg[name].items():
            for part, npts, sq in (("supernovae", d["n_sn"], d["sn_noise_sq"]), ("BAO", d["n_bao"], d["bao_noise_sq"])):
                z[f"{name}|{s}|{k}|{part}"] = (sq / d["N"] - npts) / math.sqrt(2 * npts / d["N"])
    kk, v = _worst(z)
    ok = abs(v) <= NOISE_SE_LIMIT
    return {"pass": bool(ok), "worst": kk, "z": v, "n": len(z),
            "why": None if ok else f"at {kk} the whitened noise's mean squared norm is {v:+.1f} standard errors off"}


def g6_offsets(dg, box=MB_BOX):
    """In every mock, the offset profiled with M unbounded lies inside its box at every grid point. The boxed and
    unbounded profiles then agree on the whole grid, and so at the supernova and joint minima and their neighbours."""
    margin = {}
    for name in dg:
        for (s, k), d in dg[name].items():
            lo, hi = d["M_grid"]
            margin[f"{name}|{s}|{k}"] = min(lo - box[0], box[1] - hi)
    kk, v = min(margin.items(), key=lambda it: it[1])
    ok = v > 0
    return {"pass": bool(ok), "box": list(box), "scope": "every grid point", "closest": kk, "margin": v,
            "why": None if ok else f"at {kk} an offset lies {-v:.4f} outside the box"}


def g6_M(profile_M=True, n=BATCH):
    """T's distribution does not move with the offset: mocks with the truth's offset moved by M_SHIFT either way, on
    the same stream, give the same T, at the floor and at s0 = 0.30 on both rows. profile_M=False: an arm only."""
    d = {}
    for name in GRADED:
        row = base_row(name)
        for s in (TRUTHS[0], 0.30):
            T0 = mock_T(row, s, n, profile_M=profile_M)
            for sh in (M_SHIFT, -M_SHIFT):
                d[f"{name}|{s}|{sh:+}"] = float(np.max(np.abs(mock_T(row, s, n, m_shift=sh, profile_M=profile_M) - T0)))
    kk, v = _worst(d)
    ok = v <= TOL["G6_M"]
    return {"pass": bool(ok), "moves": d, "floor": v, "tol": TOL["G6_M"],
            "why": None if ok else f"T moves by up to {v:.3g} when the offset moves ({kk})"}


def g6_refinement(sums, design, require_bounded=True, drop_midpoints=False):
    """The bound's one refinement runs: every midpoint beside the largest p is calibrated. At qualification the all
    row's graded p at T_QUALIFY must also be bounded. drop_midpoints: an arm only."""
    out = {}
    try:
        for name in sums:
            s_ = sums[name]
            if drop_midpoints:
                lat = {(s, k) for s, k, _ in design["truths"]}
                s_ = {key: v for key, v in s_.items() if key in lat}
            g = graded_p(s_, design)
            out[name] = {k: g[k] for k in ("at1", "at2", "moved_in_se", "bounded", "p")}
    except KeyError as err:
        return {"pass": False, "rows": out, "why": f"a midpoint beside the largest p is not calibrated: {err}"}
    ok = (not require_bounded) or out["all"]["bounded"]
    return {"pass": bool(ok), "rows": out, "why": None if ok else "the all row's graded p is not bounded"}


def gate_G6(design, cal, dg, sums, out_dir, require_bounded=True):
    """Calibration (the recon's §3.4), part by part."""
    fc = read_json(FORECAST)
    parts = {"design": g6_design(design, fc["design"]), "q": g6_q(design, len(base_row("all").yb), fc["design"]),
             "coverage": g6_coverage(), "halves": g6_halves(cal, design), "generator": g6_generator(cal, fc),
             "recompute": g6_recompute(sums, design, out_dir), "noise": g6_noise(dg), "offsets": g6_offsets(dg),
             "M": g6_M(), "refinement": g6_refinement(sums, design, require_bounded)}
    red = [f"{k}: {p['why']}" for k, p in parts.items() if not p["pass"]]
    return {"gate": "G6", "pass": not red, "parts": parts, "why": "; ".join(red) or None}


EXPECTED_NEW = ("data/sn_bao_coverage.json", "data/sn_bao_forecast.json", "data/sn_bao_synthetic.json",
                "scripts/sn_bao_consistency.py", "scripts/sn_bao_generator.py")


def gate_G7(paths=None, at_run=False):
    """Pins: every pinned file's SHA-256, fit_lcos.py's span, and the checkout's state (at qualification, exactly the
    design commit's new files untracked; at the run, clean). The commit and this script's SHA-256 are recorded."""
    paths = paths or {rel: repo_path(rel) for rel in PINS}
    got = {rel: sha256_file(p) for rel, p in paths.items()}
    _, span_sha, _ = run_published_span()
    status = sorted(l for l in git("status", "--porcelain").split("\n") if l)
    expect = [] if at_run else sorted(f"?? {rel}" for rel in EXPECTED_NEW)
    bad = [rel for rel in PINS if got[rel] != PINS[rel]]
    ok = not bad and span_sha == SPAN_SHA256 and status == expect
    why = None if ok else (f"{bad} differ from their pins" if bad else
                           ("fit_lcos.py's span differs" if span_sha != SPAN_SHA256 else f"the checkout holds {status}"))
    return {"gate": "G7", "pass": bool(ok), "files": {rel: got[rel] == PINS[rel] for rel in PINS},
            "span": span_sha == SPAN_SHA256, "status": status, "expected_status": expect, "commit": git("rev-parse", "HEAD"),
            "script_sha256": sha256_file(os.path.abspath(__file__)), "why": why}


def probe_pins():
    """The run's first guard, rehearsed in a scratch clone: the design files committed there on a throwaway commit, and
    throwaway registrations naming it. The parent passes; three arms fail."""
    tmp = tempfile.mkdtemp()
    out = {}
    try:
        clone = os.path.join(tmp, "clone")
        g = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()
        base = git("rev-parse", "HEAD")
        g("clone", "-q", "--no-checkout", REPO, clone)
        g("-C", clone, "checkout", "-q", base)
        for rel in EXPECTED_NEW:
            shutil.copy(os.path.join(REPO, rel), os.path.join(clone, rel))
        g("-C", clone, "add", *EXPECTED_NEW)
        g("-C", clone, "-c", "user.name=qualification-probe", "-c", "user.email=probe@localhost", "commit", "-q", "-m", "probe")
        commit = g("-C", clone, "rev-parse", "HEAD")
        script = os.path.join(clone, "scripts", "sn_bao_consistency.py")
        pins = {"design commit": commit, "script": sha256_file(script), **{rel: PINS[rel] for rel in PINS}}

        def case(name, pins_, stray=False):
            reg = os.path.join(tmp, f"{len(out)}.md")
            with open(reg, "w", encoding="utf-8") as fh:
                fh.write("### Pins\n\n| pin | value |\n|---|---|\n" + "".join(f"| {k} | `{v}` |\n" for k, v in pins_.items()))
            stray_path = os.path.join(clone, "scripts", "stray.txt")
            if stray:
                with open(stray_path, "w") as fh:
                    fh.write("stray\n")
            try:
                r = subprocess.run([sys.executable, "-B", script, "check-pins", "--registration", reg, "--out", os.path.join(tmp, "o")],
                                   capture_output=True, text=True, cwd=os.path.dirname(script))
                out[name] = r.returncode == 0
            finally:
                if stray:
                    os.remove(stray_path)

        case("parent", pins)
        case("the registration names the base commit", dict(pins, **{"design commit": base}))
        case("the script's hash wrong", dict(pins, script="0" * 64))
        case("a stray file in the checkout", pins, stray=True)
    finally:
        shutil.rmtree(tmp)
    return {"parent_passes": out.get("parent", False), "arms_fail": {k: not v for k, v in out.items() if k != "parent"}}


# ---------------- the analysis, for the run and its rehearsal ----------------
def analysis_core(rows, design, cal, out_dir=None):
    """The registered computation on given rows (measured at the run; a synthetic set in the rehearsal), graded with a
    given calibration."""
    res = {"rows": {}, "lcdm": {}, "omega_l_anchors": {}, "grid_x10": {}, "design": design,
           "data": {name: row.data_label for name, row in rows.items()}}
    for name, row in rows.items():
        res["rows"][name] = measured_row(row)
        res["lcdm"][name] = measured_row(row, "LCDM")
        res["omega_l_anchors"][name] = {str(OL): measured_row(row, "Lcos", OL)["T"] for OL in OMEGA_L_ANCHORS}
        res["grid_x10"][name] = grid_x10(row)
    limits = committed_anchors()["limits_95"]
    tails, sums = {}, {}
    for name in GRADED:
        T_obs = res["rows"][name]["T"]
        sums[name] = {key: set_summary(Tm, T_obs) for key, Tm in cal[name].items()}
        g = graded_p(sums[name], design)
        g["grade"] = grade(g["p"])
        g["chi2_1dof_p"] = math.erfc(math.sqrt(max(T_obs, 0.0) / 2))
        g["p_at_floor"] = sums[name][(TRUTHS[0], "profiled")]["p"]
        g["p_at_95_limit"] = sums[name][(limits[name], "profiled")]["p"]
        g["thresholds"] = row_thresholds(sums[name], design)
        g["sets"] = {f"{s}|{k}": {f: v for f, v in d.items() if f != "tail"} for (s, k), d in sums[name].items()}
        res["rows"][name]["graded"] = g
        tails.update({f"{name}|{s}|{k}": d["tail"] for (s, k), d in sums[name].items()})
    if out_dir:
        np.savez_compressed(os.path.join(out_dir, "tails.npz"), **tails)
    res["verdict"] = verdict_of([res["rows"][n]["graded"]["grade"] for n in GRADED])
    return res, sums


def verdict_of(grades):
    if "UNINFORMATIVE" in grades:
        return "Uninformative"
    if len(set(grades)) > 1:
        return "Mixed"
    return {"CONSISTENT": "Positive", "TENSION": "Negative", "MILD TENSION": "Inconclusive"}[grades[0]]


def rehearsal(design, cal, set_name=PLANTED_REHEARSAL, expect_set=PLANTED_REHEARSAL, move=None):
    """R: the run's analysis on a pinned synthetic set in place of the measured data. Its grades and Verdict must be
    the ones the generator's T and thresholds give, and its T the generator's within G4's tolerance. The run's closing
    steps run on it too: G5 on its graded rows, which a moved stress row cannot fail, and the run's outcome logic on
    three cases. set_name other than expect_set, and move: arms only."""
    rows = {name: base_row(name).with_synthetic(set_name) for name in ROWS}
    tmp = tempfile.mkdtemp()
    try:
        res, _ = analysis_core(rows, design, cal, tmp)
        tails_saved = os.path.exists(os.path.join(tmp, "tails.npz"))
    finally:
        shutil.rmtree(tmp)
    syn, fc = synthetic_sets()[expect_set], read_json(FORECAST)
    expect = {}
    for name in GRADED:
        T_g, th = syn["T"][name]["T"], fc["rows"][name]["thresholds"]
        expect[name] = "TENSION" if T_g >= th[str(BARS[1][1])] else ("MILD TENSION" if T_g >= th[str(BARS[0][1])] else "CONSISTENT")
    got = {name: res["rows"][name]["graded"]["grade"] for name in GRADED}
    dT = {name: res["rows"][name]["T"] - syn["T"][name]["T"] for name in GRADED}
    complete = tails_saved and set(res["grid_x10"]) == set(ROWS) and set(res["omega_l_anchors"]) == set(ROWS)
    if move:                                                     # an arm only: one row's finer-grid T moved
        res["grid_x10"][move[0]]["T_x10"] += move[1]
    g5 = gate_G5_measured(res)
    nudged = copy.deepcopy(res)
    for name in ROWS:
        if name not in GRADED:
            nudged["grid_x10"][name]["T_x10"] += 1e-3
    g5n = gate_G5_measured(nudged)
    stress_ok = g5n["pass"] == g5["pass"] and all(not r["within_tolerance"] for r in g5n["reported"].values())
    cases = (run_outcome(res, True) == {"uninformative": res["verdict"] == "Uninformative", "verdict": res["verdict"]}
             and run_outcome(res, False) == {"uninformative": True, "verdict": "Uninformative"}
             and run_outcome(dict(res, verdict="Uninformative"), True) == {"uninformative": True, "verdict": "Uninformative"})
    closing = {"G5_graded": bool(g5["pass"]), "stress_cannot_fail_G5": bool(stress_ok), "outcome_cases": bool(cases)}
    worst_dT = max(abs(v) for v in dT.values())
    ok = (got == expect and res["verdict"] == verdict_of([expect[n] for n in GRADED]) and worst_dT <= TOL["G4_T"] and complete
          and all(closing.values()))
    why = None if ok else (f"grades {got} against {expect}" if got != expect else
                           f"T off the generator's by {dT}" if worst_dT > TOL["G4_T"] else
                           f"G5 on the graded rows: {g5['why']}" if not g5["pass"] else
                           "a moved stress row fails G5" if not stress_ok else
                           "the run's outcome logic" if not cases else "a record is missing")
    return {"gate": "R", "pass": bool(ok), "data": set_name, "grades": got, "expected": expect, "verdict": res["verdict"],
            "T": {n: res["rows"][n]["T"] for n in ROWS}, "T_minus_generator": dT, "floor": max(abs(v) for v in dT.values()),
            "graded_p": {n: res["rows"][n]["graded"]["p"] for n in GRADED}, "tails_saved": tails_saved,
            "closing": closing, "G5_moves": g5["moves"], "why": why}


# ---------------- qualification ----------------
def floor_rows(G):
    rows = [("G1", "worst relative departure from fit_lcos.py", G["G1"]["floor"], "G1_rel"),
            ("G2", "chi2 and T against the committed anchors", G["G2"]["chi2_dev"], "G2_chi2"),
            ("G2", "s0 at the committed minima", G["G2"]["s0_dev"], "G2_s0"),
            ("G3", "T on the null sets", G["G3"]["floor"], "G3_T"),
            ("G4", "T on the planted sets against the generator's", G["G4"]["floor"], "G4_T"),
            ("G5", "T's move on the finer grid", G["G5"]["floor"], "G5_T"),
            ("G6", "J's top against the generator's", G["G6"]["parts"]["design"]["top_diff"], "G6_top"),
            ("G6", "H0rd at the truths against the generator's (km/s)", G["G6"]["parts"]["design"]["h0rd_diff"], "G6_h0rd"),
            ("G6", "q's upper tail against beta", G["G6"]["parts"]["q"]["floor"], "G6_tail"),
            ("G6", "T's move when the offset moves", G["G6"]["parts"]["M"]["floor"], "G6_M"),
            ("R", "the rehearsal's T against the generator's", G["R"]["floor"], "G4_T")]
    return [{"gate": a, "quantity": q, "floor": float(f), "tolerance_key": k, "tolerance": TOL[k],
             "clears": bool(TOL[k] >= FLOOR_MARGIN * f)} for a, q, f, k in rows]


def frozen_thresholds(cal, design):
    """Each graded row's thresholds, the truth that sets each, its count, and its Monte Carlo error: half the spread
    of the order statistics one binomial standard error either side of the threshold's."""
    out = {}
    for name in GRADED:
        out[name] = {}
        for _, b in BARS:
            best = None
            for s, k, _ in design["truths"]:
                srt = np.sort(cal[name][(s, k)])
                c = bar_count(len(srt), b)
                th = float(srt[-(c + 1)])
                if best is None or th > best["threshold"]:
                    w = int(round(math.sqrt(c + 1)))
                    best = {"threshold": th, "set_by": f"{s}|{k}", "count": c,
                            "mc_se": float((srt[-(c + 1 - w)] - srt[-(c + 1 + w)]) / 2)}
            out[name][str(b)] = best
    return out


def qualify(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rec = {"date": time.strftime("%Y-%m-%d"), "environment": check_environment(), "tolerances": dict(TOL), "floor_margin": FLOOR_MARGIN,
           "statistical_limits": {"Z_LIMIT": Z_LIMIT, "NOISE_SE_LIMIT": NOISE_SE_LIMIT, "COVERAGE_SE_LIMIT": COVERAGE_SE_LIMIT},
           "gates": {}, "arms": {}}
    if not rec["environment"]["pass"]:
        json.dump(rec, open(os.path.join(out_dir, "qualification.json"), "w"), indent=1, default=_jsonable)
        sys.exit("the environment is not the pinned one: qualification stops")
    G = rec["gates"]
    say = lambda s: print(f"[{time.time() - t0:7.0f} s] {s}", flush=True)
    G["G1"] = gate_G1(); say(f"G1 {G['G1']['pass']}")
    G["G2"] = gate_G2(report_published=True); say(f"G2 {G['G2']['pass']}")
    G["G3"] = gate_G3(); say(f"G3 {G['G3']['pass']}")
    G["G4"] = gate_G4(); say(f"G4 {G['G4']['pass']}")
    G["G5"] = gate_G5(); say(f"G5 {G['G5']['pass']}")
    G["G7"] = gate_G7(); say(f"G7 {G['G7']['pass']}")
    rec["G7_probe"] = probe_pins(); say(f"G7's probe {rec['G7_probe']}")
    rowA = base_row("all")
    design = calibration_design(rowA)
    cal, dg = calibrate(GRADED, design["truths"] + design["midpoints"] + limit_truths(rowA), diagnostics=True)
    say(f"calibrated {sum(len(v) for v in cal.values())} sets")
    sums = {n: {key: set_summary(T, T_QUALIFY) for key, T in cal[n].items()} for n in GRADED}
    G["G6"] = gate_G6(design, cal, dg, sums, out_dir, require_bounded=True); say(f"G6 {G['G6']['pass']} {G['G6']['why']}")
    G["R"] = rehearsal(design, cal); say(f"R {G['R']['pass']} {G['R']['verdict']}")
    parts = G["G6"]["parts"]
    fc = read_json(FORECAST)
    arms = rec["arms"]

    def arm(name, parent_ok, fn):
        if not parent_ok:
            arms[name] = {"parent_passed": False, "red": None, "why": "not run: its parent did not pass"}
        else:
            r = fn()
            arms[name] = {"parent_passed": True, "red": not r["pass"], "why": r.get("why")}
        say(f"arm '{name}': red {arms[name]['red']}: {arms[name]['why']}")

    tmp = tempfile.mkdtemp()
    try:
        arm("G1: Omega_L = 0.686", G["G1"]["pass"], lambda: gate_G1(OL=0.686))
        arm("G1: the D_V exponent set to 1/2", G["G1"]["pass"], lambda: gate_G1(dv_exp=0.5))
        arm("G2: a BAO point dropped", G["G2"]["pass"], lambda: gate_G2(drop_bao=0))
        arm("G2: the cut applied to the data, the covariance's leading block kept", G["G2"]["pass"], lambda: gate_G2(cov_block="leading"))
        arm("G3: a planted set given as a null set", G["G3"]["pass"], lambda: gate_G3(null_sets=(PLANTED_REHEARSAL,)))
        arm("G4: the D_V exponent set to 1/2", G["G4"]["pass"], lambda: gate_G4(dv_exp=0.5))
        arm("G5: the registered grid ten times coarser", G["G5"]["pass"], lambda: gate_G5(k=COARSE_K))
        floor_only = dict(design, lattice=[TRUTHS[0]], truths=[t for t in design["truths"] if t[0] == TRUTHS[0]], midpoints=[])
        arm("G6 design: the lattice cut down to the floor alone", parts["design"]["pass"], lambda: g6_design(floor_only, fc["design"]))
        arm("G6 q: q taken with 12 degrees of freedom", parts["q"]["pass"],
            lambda: g6_q(calibration_design(rowA, dof=12), len(rowA.yb), fc["design"]))
        cov = json.loads(open(repo_path(COVERAGE), encoding="utf-8").read())
        for r in cov["records"]:
            r["outside_J_fraction"][str(BETA)] *= 2
        cov_path = os.path.join(tmp, "coverage.json")
        json.dump(cov, open(cov_path, "w"))
        arm("G6 coverage: the record's misses doubled", parts["coverage"]["pass"], lambda: g6_coverage(cov_path))
        arm("G6 halves: each second half taken from the next set", parts["halves"]["pass"], lambda: g6_halves(cal, design, neighbour=True))
        arm("G6 generator: the truths mislabeled by one lattice step", parts["generator"]["pass"], lambda: g6_generator(cal, fc, mislabel=True))
        arm("G6 recompute: beta's term dropped", parts["recompute"]["pass"], lambda: g6_recompute(sums, design, tmp, beta=0.0))
        arm("G6 recompute: a threshold moved", parts["recompute"]["pass"], lambda: g6_recompute(sums, design, tmp, nudge=0.01))
        arm("G6 recompute: the cap removed", parts["recompute"]["pass"], lambda: g6_recompute(sums, design, tmp, cap=math.inf))
        cut = base_row("cut")
        L_lead = cholesky(raw()["C"][: cut.n, : cut.n], lower=True, check_finite=False)
        arm("G6 noise: a mock drawn with the wrong block", parts["noise"]["pass"],
            lambda: g6_noise({"cut": {(TRUTHS[0], "profiled"): mock_T(cut, TRUTHS[0], BATCH, diagnostics=True, draw=L_lead)[1]}}))
        Mr = rowA.M_ref
        arm("G6 offsets: the box narrowed to 0.005 about the reference offset", parts["offsets"]["pass"],
            lambda: g6_offsets(dg, box=(Mr - 0.005, Mr + 0.005)))
        arm("G6 M: the offset held at the reference, not profiled", parts["M"]["pass"], lambda: g6_M(profile_M=False))
        arm("G6 refinement: the midpoints left out", parts["refinement"]["pass"], lambda: g6_refinement(sums, design, True, drop_midpoints=True))
        bad = {rel: repo_path(rel) for rel in PINS}
        cp = os.path.join(tmp, "copy.csv")
        shutil.copy(bad["data/desi_dr2_bao.csv"], cp)
        with open(cp, "r+b") as f:
            f.seek(10); c = f.read(1); f.seek(10); f.write(bytes([c[0] ^ 1]))
        bad["data/desi_dr2_bao.csv"] = cp
        arm("G7: one byte changed in a pinned file", G["G7"]["pass"], lambda: gate_G7(bad))
        arm("R: the null set given as the rehearsal's data", G["R"]["pass"], lambda: rehearsal(design, cal, set_name="a_null"))
        arm("R: a graded row's finer-grid T moved", G["R"]["pass"], lambda: rehearsal(design, cal, move=("cut", 1e-3)))
    finally:
        shutil.rmtree(tmp)
    rec["floors"] = floor_rows(G)
    rec["tolerances_clear_floors"] = all(r["clears"] for r in rec["floors"])
    rec["frozen_thresholds"] = frozen_thresholds(cal, design)
    rec["design"] = design
    rec["all_gates_pass"] = all(v["pass"] for v in G.values())
    probe = rec["G7_probe"]
    # twenty-two arms: G1 two, G2 two, G3, G4 and G5 one each, G6 twelve, G7 one, R two; and the probe's three
    rec["all_arms_red"] = (all(a["red"] for a in arms.values()) and len(arms) == 22 and probe["parent_passes"]
                           and len(probe["arms_fail"]) == 3 and all(probe["arms_fail"].values()))
    rec["seconds"] = round(time.time() - t0, 1)
    rec["commit"], rec["script_sha256"] = git("rev-parse", "HEAD"), sha256_file(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "qualification.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=_jsonable)
    say(json.dumps({"all_gates_pass": rec["all_gates_pass"], "all_arms_red": rec["all_arms_red"],
                    "tolerances_clear_floors": rec["tolerances_clear_floors"]}))
    return rec


# ---------------- the run, once ----------------
def run_outcome(res, gates_pass):
    """The run's last step. The Verdict is Uninformative if G6 or G5 on the graded rows fails, or if the analysis's own
    Verdict is (a row's calibration not bounded). The flag is true in each case."""
    uninformative = (not gates_pass) or res["verdict"] == "Uninformative"
    return {"uninformative": bool(uninformative), "verdict": "Uninformative" if uninformative else res["verdict"]}


def run(out_dir, registration):
    os.makedirs(out_dir, exist_ok=True)
    rec = {"environment": check_environment(), "tolerances": dict(TOL)}
    dump = lambda: json.dump(rec, open(os.path.join(out_dir, "run.json"), "w"), indent=1, default=_jsonable)
    if not rec["environment"]["pass"]:
        dump()
        sys.exit("the environment is not the pinned one: the run stops")
    rec["pins"] = authorize(registration_pins(registration))
    if not rec["pins"]["pass"]:
        dump()
        sys.exit("a pin does not hold: the run stops, Uninformative")
    g = rec["gates"] = {"G1": gate_G1(), "G2": gate_G2(), "G3": gate_G3(), "G4": gate_G4(), "G5": gate_G5(),
                        "G7": gate_G7(at_run=True)}
    if not all(v["pass"] for v in g.values()):
        dump()
        sys.exit("a gate failed after the freeze: the run ends as Uninformative")
    rows = {name: Row(name) for name in ROWS}
    design = calibration_design(rows["all"])
    cal, dg = calibrate(GRADED, design["truths"] + design["midpoints"] + limit_truths(rows["all"]), diagnostics=True)
    res, sums = analysis_core(rows, design, cal, out_dir)
    g["G6"] = gate_G6(design, cal, dg, sums, out_dir, require_bounded=False)
    g["G5 (graded rows)"] = gate_G5_measured(res)
    rec["analysis"] = res
    rec.update(run_outcome(res, g["G6"]["pass"] and g["G5 (graded rows)"]["pass"]))
    dump()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mode", choices=("probe", "qualify", "run", "check-pins"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--registration")
    a = ap.parse_args()
    if a.mode in ("probe", "qualify", "check-pins") and os.path.realpath(a.out).startswith(os.path.realpath(REPO) + os.sep):
        sys.exit("this mode writes outside the repo, so the checkout stays clean")
    if a.mode == "probe":
        rec = probe(a.out)
        print(json.dumps({k: rec[k] for k in ("pins", "guard")}, indent=1, default=_jsonable)[:2000])
    elif a.mode == "qualify":
        qualify(a.out)
    elif a.mode == "check-pins":
        if not a.registration:
            sys.exit("check-pins needs --registration")
        rec = check_pins_only(a.registration)
        ok = rec["environment"]["pass"] and rec["pins"]["pass"]
        print(json.dumps({"environment": rec["environment"]["pass"], "pins": rec["pins"]}, default=_jsonable))
        sys.exit(0 if ok else 1)
    else:
        if not a.registration:
            sys.exit("the run needs --registration")
        run(a.out, a.registration)


if __name__ == "__main__":
    main()
