"""
The generator for the SN-against-BAO test on s0: synthetic sets, and the null distribution of T at each truth of the
recon's §3.4 (RECON_v4.md; Step 3b).

Written independently of the instrument (scripts/sn_bao_consistency.py in the Λcos repo), which it does not import.

Run, in Λcos's environment:
    python generate_registered.py LCOS_REPO OUT_DIR [--processes 4]
    python generate_registered.py LCOS_REPO OUT_DIR --recompute TAILS_NPZ SETS_JSON

It reads the Pantheon+ redshifts (the zHD column only; never the magnitudes), the supernova covariance's upper
triangle (the one fit_lcos.py's cho_factor reads), the DESI DR2 BAO table and covariance, and the published best fit
(results/lcos_summary.json). Its numerics differ from the instrument's on purpose:
  - a z grid of 20 001 points on [0, 2.5], with its own trapezoid;
  - s0 itself as the profile variable, on a grid of step 0.0005 from 0.001 to 0.99, with a parabola in s0 for the
    mocks and a bounded Brent minimization for the synthetic sets;
  - the chi^2 in data space, through the explicit inverse of each covariance;
  - the mocks drawn in data space, through numpy's Cholesky factor, from its own seed;
  - J's s0 range found by its own bisection on its own data-space BAO profile.
What it shares with the instrument is the registration's definitions: the model at Omega_L = 0.685, the box, the
rows, T, the confidence set J and its beta, the truths and midpoints, the nuisances at a truth (M_B at the published
best fit; c/(H0 r_d) at the BAO data's own profiled value there, or at J's two ends), and the rule by counts.

The mocks at a truth are drawn from the stream (GEN_SEED, 7, row, truth, k), k = 0, 1, 2 for the profiled value, the
low end and the high end: the streams the bench's rule check drew the generator's mocks from, so that its record can
be read against this generator's output.

Writes, into OUT_DIR:
  sn_bao_synthetic.json  (a) a noise-free set at the published joint best fit, where T must be 0; (b) planted sets,
                         the supernovae at one s0 and BAO at another, with the T the generator computes on each row
  sn_bao_forecast.json   (c) J and its truths, and T's null distribution at each truth and midpoint, on the all and cut
                         rows: N, P(T = 0), its quantiles, and each bar's threshold by counts; on the all row also the
                         count at or above T = 5.8867 and its p. Each row's thresholds, and the all row's graded p.
With --recompute it writes recompute.json instead: each row's thresholds and graded p, recomputed by this generator's
own code for the rule by counts from another code's saved tails (TAILS_NPZ, keys 'row|s0|kind', each the largest mock
T, ascending) and counts (SETS_JSON, {row: {'s0|kind': {'N', 'count'}}}, or the same under a top-level 'sets').
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse, json, math
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

# ---- the registration's definitions, restated ----
OMEGA_L = 0.685
C_KMS = 299792.458
H0_SN = 70.0
S0_LO, S0_HI = 0.001, 0.99
H0RD_LO, H0RD_HI = 8000.0, 12000.0
MB_LO, MB_HI = -20.0, -18.0
ROW_CUTS = {"all": None, "cut": 0.01}
TRUTHS = (0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99)
N_PER_TRUTH = 100_000
T_DISCLOSED = 5.8867                       # the all row's T as the recon discloses it
BETA = 1e-5                                # the probability that J misses the truth
KINDS = ("profiled", "low", "high")        # H0rd at a truth: the BAO data's own value, then J's two ends
BARS = (0.05, 0.0027)

# ---- this generator's own numerics ----
Z = np.linspace(0.0, 2.5, 20001)
S_GRID = np.round(S0_LO + 0.0005 * np.arange(1979), 6)          # 0.001 to 0.99, step 0.0005
GEN_SEED = 2026111802
BATCH = 2000
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.9973)
PLANTED = ((0.389, 0.001), (0.25, 0.10), (0.05, 0.30), (0.30, 0.30))   # (supernovae's s0, BAO's s0)
STREAM_TAG = 7                             # the tag in the rule check's generator streams
BISECT_TOL = 1e-12


def load(repo):
    sn = pd.read_csv(os.path.join(repo, "data", "pantheon_plus.csv"), usecols=["zHD"])
    bao = pd.read_csv(os.path.join(repo, "data", "desi_dr2_bao.csv"))
    C = np.load(os.path.join(repo, "data", "pantheon_plus_cov.npy"))
    Cb = np.load(os.path.join(repo, "data", "desi_dr2_bao_cov.npy"))
    with open(os.path.join(repo, "results", "lcos_summary.json"), encoding="utf-8") as fh:
        best = json.load(fh)["best_fit"]
    return {"z": sn["zHD"].to_numpy(float), "C": C, "bz": bao["z_eff"].to_numpy(float),
            "obs": bao["observable"].to_numpy(), "b": bao["value"].to_numpy(float), "Cb": Cb, "best": best}


def upper_symmetric(A):
    return np.triu(A) + np.triu(A, 1).T


def model(s0, z_sn, bz, obs):
    """The supernova moduli and the BAO shapes (times c/(H0 r_d)) at s0, on this generator's grid."""
    s2 = s0 * s0
    Om = 1.0 - OMEGA_L
    E = np.sqrt(Om * (1.0 + Z) ** 3 / (1.0 - s2) - Om * s2 * (1.0 + Z) / (1.0 - s2) + OMEGA_L)
    g = 1.0 / E
    I = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(Z))])
    mu = 5.0 * np.log10((1.0 + z_sn) * (C_KMS / H0_SN) * np.interp(z_sn, Z, I)) + 25.0
    Ib, Eb = np.interp(bz, Z, I), np.interp(bz, Z, E)
    f = np.empty(len(bz))
    for i, o in enumerate(obs):
        f[i] = np.cbrt(bz[i] * Ib[i] ** 2 / Eb[i]) if o == "DV_rd" else (Ib[i] if o == "DM_rd" else 1.0 / Eb[i])
    return mu, f


class Row:
    def __init__(self, d, name):
        cut = ROW_CUTS[name]
        keep = np.ones(len(d["z"]), bool) if cut is None else d["z"] > cut
        self.name, self.keep, self.z = name, keep, d["z"][keep]
        Cu = upper_symmetric(d["C"][np.ix_(keep, keep)])
        self.Cinv = np.linalg.inv(Cu)
        self.chol = np.linalg.cholesky(Cu)
        self.w1 = self.Cinv @ np.ones(len(self.z))
        self.s11 = float(np.sum(self.w1))
        self.Cbinv = np.linalg.inv(d["Cb"])
        self.cholb = np.linalg.cholesky(d["Cb"])
        self.bz, self.obs, self.b = d["bz"], d["obs"], d["b"]
        cols = [model(s, self.z, self.bz, self.obs) for s in S_GRID]
        self.MU = np.array([c[0] for c in cols]).T                    # n x K
        self.F = np.array([c[1] for c in cols]).T                     # 13 x K
        self.WMU = self.Cinv @ self.MU
        self.muWmu = np.sum(self.MU * self.WMU, axis=0)
        self.oneWmu = self.w1 @ self.MU
        self.WF = self.Cbinv @ self.F
        self.FWF = np.sum(self.F * self.WF, axis=0)

    # chi^2 curves over S_GRID, for data in columns; constants that cancel in T are kept, so the values are true chi^2
    def sn_curves(self, D):
        dWd = np.sum(D * (self.Cinv @ D), axis=0)
        muWd = self.WMU.T @ D
        a = (self.w1 @ D)[None, :] - self.oneWmu[:, None]
        M = np.clip(a / self.s11, MB_LO, MB_HI)
        return dWd[None, :] - 2.0 * muWd + self.muWmu[:, None] - 2.0 * M * a + M * M * self.s11

    def bao_curves(self, Bv):
        bWb = np.sum(Bv * (self.Cbinv @ Bv), axis=0)
        FWb = self.WF.T @ Bv
        A = np.clip(FWb / self.FWF[:, None], C_KMS / H0RD_HI, C_KMS / H0RD_LO)
        return bWb[None, :] - 2.0 * A * FWb + A * A * self.FWF[:, None], A


def sn_chi2(row, d, s0):
    """One supernova chi^2 at any s0, M_B profiled and clipped: for the Brent refinement."""
    mu, _ = model(s0, row.z, row.bz, row.obs)
    r = d - mu
    M = min(max(float(row.w1 @ r) / row.s11, MB_LO), MB_HI)
    r = r - M
    return float(r @ row.Cinv @ r)


def bao_chi2(row, b, s0):
    _, f = model(s0, row.z, row.bz, row.obs)
    A = min(max(float(f @ row.Cbinv @ b) / float(f @ row.Cbinv @ f), C_KMS / H0RD_HI), C_KMS / H0RD_LO)
    r = b - A * f
    return float(r @ row.Cbinv @ r)


def exact_min(fun, curve):
    """The minimum of fun over [S0_LO, S0_HI]: the best grid point of curve, then bounded Brent between its neighbours."""
    k = int(np.argmin(curve))
    lo, hi = S_GRID[max(k - 1, 0)], S_GRID[min(k + 1, len(S_GRID) - 1)]
    r = minimize_scalar(fun, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return (float(r.fun), float(r.x)) if r.fun < curve[k] else (float(curve[k]), float(S_GRID[k]))


def parabola_min(V):
    """Each column's minimum over S_GRID: the best point, then the parabola through it and its neighbours, written
    about the middle point (any spacing), its vertex kept inside the best point's neighbours."""
    K, N = V.shape
    k = np.argmin(V, axis=0)
    j = np.arange(N)
    c = np.clip(k, 1, K - 2)
    t0, t2 = S_GRID[c - 1] - S_GRID[c], S_GRID[c + 1] - S_GRID[c]
    y1 = V[c, j]
    d0, d2 = V[c - 1, j] - y1, V[c + 1, j] - y1
    den = t0 * t2 * (t2 - t0)
    a = (d2 * t0 - d0 * t2) / den                 # y = y1 + b t + a t^2 through the three points
    b = (d0 * t2 * t2 - d2 * t0 * t0) / den
    lo = S_GRID[np.maximum(k - 1, 0)] - S_GRID[c]
    hi = S_GRID[np.minimum(k + 1, K - 1)] - S_GRID[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = np.where(a > 0, -b / (2.0 * a), S_GRID[k] - S_GRID[c])
    tv = np.clip(tv, lo, hi)
    yv = y1 + b * tv + a * tv * tv
    best = V[k, j]
    return np.minimum(best, np.where(np.isfinite(yv), yv, best))


def truth_of(row, s0_t, best):
    """The truth's supernova moduli plus the published M_B, and its BAO, with c/(H0 r_d) the BAO data's own there."""
    mu, f = model(s0_t, row.z, row.bz, row.obs)
    A = min(max(float(f @ row.Cbinv @ row.b) / float(f @ row.Cbinv @ f), C_KMS / H0RD_HI), C_KMS / H0RD_LO)
    return mu + float(best["MB"]), A * f, C_KMS / A


# ---- J, its truths and midpoints (the recon's §3.4), from the BAO data alone ----
class BAOData:
    """Only what the BAO profile needs; model() wants supernova redshifts too, and one is enough for it."""

    def __init__(self, d):
        self.z, self.bz, self.obs, self.b = d["z"][:1], d["bz"], d["obs"], d["b"]
        self.Cbinv = np.linalg.inv(d["Cb"])


def bao_profile(bd, s0):
    """At s0, in data space: the BAO chi^2 profiled over H0rd in its box, that H0rd, the shape's weight ff, and the
    unclipped amplitude a0."""
    _, f = model(s0, bd.z, bd.bz, bd.obs)
    ff = float(f @ bd.Cbinv @ f)
    a0 = float(f @ bd.Cbinv @ bd.b) / ff
    A = min(max(a0, C_KMS / H0RD_HI), C_KMS / H0RD_LO)
    r = bd.b - A * f
    return float(r @ bd.Cbinv @ r), C_KMS / A, ff, a0


def chi2_upper_tail(x, k):
    """P(chi^2 with k degrees of freedom > x), in closed form, written apart from scipy's."""
    h = x / 2.0
    if k % 2 == 0:
        return math.exp(-h) * sum(h ** j / math.factorial(j) for j in range(k // 2))
    return math.erfc(math.sqrt(h)) + math.exp(-h) * sum(h ** (j - 0.5) / math.gamma(j + 0.5) for j in range(1, (k + 1) // 2))


def bisect_root(fun, lo, hi, tol=BISECT_TOL):
    """A root of fun on [lo, hi] by bisection; fun must change sign there."""
    flo = fun(lo) > 0
    if flo == (fun(hi) > 0):
        raise ValueError("no change of sign on the bracket")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if (fun(mid) > 0) == flo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def design(bd):
    """q, where J's s0 range ends, the lattice, the truths and the midpoints, each truth as (s0, kind, H0rd)."""
    k = len(bd.b)
    q = float(chi2.isf(BETA, k))
    top = bisect_root(lambda s: bao_profile(bd, s)[0] - q, 0.05, 0.9)

    def at(s):
        B, H, ff, a0 = bao_profile(bd, s)
        out = [(s, "profiled", H)]
        if B < q:
            half = math.sqrt((q - B) / ff)
            out += [(s, "low", C_KMS / (a0 + half)), (s, "high", C_KMS / (a0 - half))]
        return out

    lattice = [t for t in TRUTHS if t <= top] + [min(t for t in TRUTHS if t > top)]
    truths = [x for s in lattice for x in at(s)]
    midpoints = [x for a, b in zip(lattice[:-1], lattice[1:]) for x in at(round((a + b) / 2, 6))]
    return {"beta": BETA, "k": k, "q": q, "tail_at_q_closed_form": chi2_upper_tail(q, k), "top": top, "lattice": lattice,
            "truths": truths, "midpoints": midpoints}


# ---- this generator's own code for the rule by counts ----
def count_for_bar(n, bar, beta=BETA):
    """The largest count c whose p, (1 + c)/(1 + n), with twice its Monte Carlo standard error and beta, is below the
    bar."""
    c = -1
    while True:
        p = (2 + c) / (1 + n)
        if p + 2.0 * math.sqrt(p * (1 - p) / n) + beta < bar:
            c += 1
        else:
            return c


def threshold_from_tail(tail, n, bar, beta=BETA):
    """One truth's threshold at a bar: the (c + 1)-th largest mock T. tail: the largest mock T, ascending."""
    c = count_for_bar(n, bar, beta)
    if not 0 <= c < len(tail) - 1:
        raise ValueError("the kept tail is too short for this bar")
    return float(tail[-(c + 1)])


def graded_from_counts(counts, n, dz, beta=BETA, cap=1.0):
    """counts maps (s0, kind) to the count at or above the observed T. p1, the largest over the truths; the midpoints
    beside its s0 added, giving p2; not bounded if p2 - p1 exceeds 3 SE(p1); otherwise min(1, p2 + 2 SE(p2) + beta).
    Another cap: an arm only."""
    p = {t: (1 + c) / (1 + n) for t, c in counts.items()}
    se = {t: math.sqrt(p[t] * (1 - p[t]) / n) for t in p}
    keys = [(s, k) for s, k, _ in dz["truths"]]
    k1 = max(keys, key=lambda t: p[t])
    L = dz["lattice"]
    i = L.index(k1[0])
    beside = {round((L[j] + L[j + 1]) / 2, 6) for j in (i - 1, i) if 0 <= j < len(L) - 1}
    k2 = max([k1] + [(s, k) for s, k, _ in dz["midpoints"] if s in beside], key=lambda t: p[t])
    bounded = p[k2] - p[k1] <= 3.0 * se[k1]
    moved = p[k2] - p[k1]                     # a p of 1 has no standard error: then nothing can move
    return {"p1": p[k1], "at1": list(k1), "p2": p[k2], "at2": list(k2),
            "moved_in_se": moved / se[k1] if se[k1] > 0 else (0.0 if moved == 0 else math.inf),
            "bounded": bounded, "p": min(cap, p[k2] + 2.0 * se[k2] + beta) if bounded else None}


def recompute(tails_npz, sets_json, dz):
    """Each row's thresholds and graded p from another code's saved tails and counts."""
    sets = json.load(open(sets_json, encoding="utf-8"))
    sets = sets.get("sets", sets)
    tails = np.load(tails_npz)
    out = {}
    for row, by in sets.items():
        keyed = {(float(key.split("|")[0]), key.split("|")[1]): v for key, v in by.items()}
        n = {v["N"] for v in keyed.values()}
        if len(n) != 1:
            raise ValueError("the sets do not share one N")
        n = n.pop()
        lattice_keys = [(s, k) for s, k, _ in dz["truths"]]
        thr = {str(b): max(threshold_from_tail(tails[f"{row}|{s}|{k}"], n, b) for s, k in lattice_keys) for b in BARS}
        out[row] = {"N": n, "thresholds": thr, "graded": graded_from_counts({t: v["count"] for t, v in keyed.items()}, n, dz)}
    return out


# ---- the null distribution at one truth ----
_CACHE = {}


def null_task(args):
    repo, name, s0_t, kind, H, n = args
    if "d" not in _CACHE:
        _CACHE["d"] = load(repo)
    if name not in _CACHE:
        _CACHE[name] = Row(_CACHE["d"], name)
    d, row = _CACHE["d"], _CACHE[name]
    m_t, b_t, H_prof = truth_of(row, s0_t, d["best"])
    if kind != "profiled":
        b_t = b_t * (H_prof / H)
    seq = np.random.SeedSequence([GEN_SEED, STREAM_TAG, list(ROW_CUTS).index(name), int(round(s0_t * 1_000_000)), KINDS.index(kind)])
    rng = np.random.Generator(np.random.PCG64(seq))
    T = np.empty(n)
    for i in range(0, n, BATCH):
        nb = min(BATCH, n - i)
        D = m_t[:, None] + row.chol @ rng.standard_normal((len(m_t), nb))
        Bv = b_t[:, None] + row.cholb @ rng.standard_normal((len(b_t), nb))
        S = row.sn_curves(D)
        B, _ = row.bao_curves(Bv)
        T[i:i + nb] = parabola_min(S + B) - parabola_min(S) - parabola_min(B)
    q = np.quantile(T, QUANTILES)
    tail = np.sort(T)
    out = {"row": name, "s0": s0_t, "kind": kind, "H0rd": H_prof if kind == "profiled" else H, "N": n,
           "P_T0": float(np.mean(T < 1e-9)), "mean": float(T.mean()), "quantiles": {str(p): float(v) for p, v in zip(QUANTILES, q)},
           "thresholds": {str(b): threshold_from_tail(tail, n, b) for b in BARS}}
    if name == "all":
        c = int(np.count_nonzero(T >= T_DISCLOSED))
        out["count_at_or_above_T_disclosed"] = c
        out["p_of_T_disclosed"] = (1 + c) / (1 + n)
    return out


def synthetic(d):
    best = d["best"]
    allrow = Row(d, "all")
    mu_all, _ = model(best["s0"], allrow.z, allrow.bz, allrow.obs)
    sets = {}

    def make(s_sn, s_bao, label):
        mu, _ = model(s_sn, allrow.z, allrow.bz, allrow.obs)
        _, f = model(s_bao, allrow.z, allrow.bz, allrow.obs)
        return {"label": label, "s0_sn": s_sn, "s0_bao": s_bao, "M_B": float(best["MB"]), "H0rd": float(best["H0rd"]),
                "sn": (mu + float(best["MB"])).tolist(), "bao": ((C_KMS / float(best["H0rd"])) * f).tolist()}

    sets["a_null"] = make(float(best["s0"]), float(best["s0"]), "the published joint best fit; T must be 0")
    for s_sn, s_bao in PLANTED:
        sets[f"b_sn{s_sn}_bao{s_bao}"] = make(s_sn, s_bao, "planted: supernovae at one s0, BAO at another")
    for key, st in sets.items():
        st["T"] = {}
        for name in ROW_CUTS:
            row = allrow if name == "all" else Row(d, name)
            dsn = np.array(st["sn"])[row.keep]
            b = np.array(st["bao"])
            S = row.sn_curves(dsn[:, None])[:, 0]
            B = row.bao_curves(b[:, None])[0][:, 0]
            s_min, s_at = exact_min(lambda s: sn_chi2(row, dsn, s), S)
            b_min, b_at = exact_min(lambda s: bao_chi2(row, b, s), B)
            j_min, j_at = exact_min(lambda s: sn_chi2(row, dsn, s) + bao_chi2(row, b, s), S + B)
            st["T"][name] = {"T": j_min - s_min - b_min, "sn": [s_min, s_at], "bao": [b_min, b_at], "joint": [j_min, j_at]}
    return {"generator": "generate_registered.py", "z_grid": [0.0, 2.5, len(Z)], "s0_grid_step": 0.0005,
            "covariance": "upper triangle", "redshifts_n": len(d["z"]), "sets": sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("out"); ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--recompute", nargs=2, metavar=("TAILS_NPZ", "SETS_JSON"))
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    os.makedirs(a.out, exist_ok=True)
    d = load(repo)
    dz = design(BAOData(d))
    if a.recompute:
        out = {"generator": "generate_registered.py", "beta": BETA, "rows": recompute(a.recompute[0], a.recompute[1], dz)}
        with open(os.path.join(a.out, "recompute.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", os.path.join(a.out, "recompute.json"))
        return
    syn = synthetic(d)
    with open(os.path.join(a.out, "sn_bao_synthetic.json"), "w", encoding="utf-8") as fh:
        json.dump(syn, fh, indent=1, sort_keys=True)
    tasks = [(repo, name, s, k, H, N_PER_TRUTH) for name in ROW_CUTS for s, k, H in dz["truths"] + dz["midpoints"]]
    with Pool(a.processes) as pool:
        res = pool.map(null_task, tasks, chunksize=1)
    out = {"generator": "generate_registered.py", "seed": GEN_SEED, "stream_tag": STREAM_TAG, "N_per_truth": N_PER_TRUTH,
           "T_disclosed": T_DISCLOSED, "design": dz, "rows": {}}
    for name in ROW_CUTS:
        by = {(r["s0"], r["kind"]): r for r in res if r["row"] == name}
        lattice_keys = [(s, k) for s, k, _ in dz["truths"]]
        entry = {"by_truth": [by[t] for t in lattice_keys], "midpoints": [by[(s, k)] for s, k, _ in dz["midpoints"]],
                 "thresholds": {str(b): max(by[t]["thresholds"][str(b)] for t in lattice_keys) for b in BARS}}
        if name == "all":
            entry["graded"] = graded_from_counts({t: r["count_at_or_above_T_disclosed"] for t, r in by.items()}, N_PER_TRUTH, dz)
        out["rows"][name] = entry
    with open(os.path.join(a.out, "sn_bao_forecast.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", os.path.join(a.out, "sn_bao_synthetic.json"), "and", os.path.join(a.out, "sn_bao_forecast.json"))


if __name__ == "__main__":
    main()
