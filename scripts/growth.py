"""
Linear growth rate f(z) and f σ_8(z) for arbitrary cosmological models.

Solves the standard sub-horizon matter perturbation equation,

    δ'' + (2 + d ln E / d ln a) δ' - (3/2) Ω_m(z) δ = 0,

where primes are d/d ln a and Ω_m(z) = Ω_m,0 (1+z)^3 / E²(z). Returns the
growth factor δ(z), the growth rate f(z) = d ln δ / d ln a, and the
combination f σ_8(z) anchored to σ_{8,0} at z = 0.

For Λcos, the (1+z)¹ correction enters only through E(z); the matter
sector still clusters under the standard equation. No additional
perturbation parameters are needed (cf. paper §III, §VII.B).

Run:
    python growth.py --validate
    python growth.py --model LCDM
    python growth.py --model Lcos --s0 0.076

Outputs:
    results/growth_<model>.csv  (columns: z, delta, f, fsigma8)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ---------------- COSMOLOGY FACTORIES ----------------

def make_E_lcdm(Om):
    """Flat ΛCDM E(z) = √(Ω_m (1+z)³ + Ω_Λ) with Ω_Λ = 1 − Ω_m."""
    OL = 1.0 - Om
    def E(z):
        return np.sqrt(Om*(1+z)**3 + OL)
    return E

def make_E_lcos(s0, OL=0.685):
    """Λcos E(z) per Eq. (Hubble rate) of the paper. Ω_m = 1 − Ω_Λ at the reference value."""
    Om = 1.0 - OL
    α = Om / (1.0 - s0**2)
    β = Om * s0**2 / (1.0 - s0**2)
    def E(z):
        return np.sqrt(α*(1+z)**3 - β*(1+z) + OL)
    return E

# ---------------- GROWTH ODE ----------------

def dlnE_dlna_num(E_func, z, eps=1e-5):
    """Numerical derivative d ln E / d ln a at given z, via finite difference in x = ln a."""
    a = 1.0 / (1.0 + z)
    a_plus = a * np.exp(eps)
    a_minus = a * np.exp(-eps)
    z_plus = max(1.0/a_plus - 1.0, 0.0)
    z_minus = max(1.0/a_minus - 1.0, 0.0)
    return (np.log(E_func(z_plus)) - np.log(E_func(z_minus))) / (2.0 * eps)

def growth_rhs(x, y, Om0, E_func):
    """
    RHS of the growth ODE in x = ln a.
    y = [δ, dδ/dx], returns [dδ/dx, d²δ/dx²].
    """
    a = np.exp(x)
    z = max(1.0/a - 1.0, 0.0)
    E_now = E_func(z)
    Om_z = Om0 * (1.0+z)**3 / E_now**2
    dlnE = dlnE_dlna_num(E_func, z)
    δ, dδ = y
    d2δ = -(2.0 + dlnE) * dδ + 1.5 * Om_z * δ
    return [dδ, d2δ]

def solve_growth(Om0, E_func, z_init=1000.0, n_points=400):
    """
    Solve the growth ODE from matter-dominated initial conditions at z_init down to z = 0.
    Returns (z_grid, δ(z), f(z)) with z_grid in descending z order... actually let's go ascending z.
    """
    x_init = np.log(1.0 / (1.0 + z_init))
    x_end = 0.0
    x_eval = np.linspace(x_init, x_end, n_points)

    # Matter-dominated initial conditions: δ ∝ a, so dδ/dx = δ
    δ_init = np.exp(x_init)
    dδ_init = δ_init

    sol = solve_ivp(
        growth_rhs,
        [x_init, x_end],
        [δ_init, dδ_init],
        t_eval=x_eval,
        args=(Om0, E_func),
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Growth ODE failed: {sol.message}")

    δ = sol.y[0]
    dδ = sol.y[1]
    f = dδ / δ
    z = np.exp(-x_eval) - 1.0

    # Reverse so z is ascending
    return z[::-1], δ[::-1], f[::-1]

def fsigma8(z_grid, δ_grid, f_grid, sigma8_0=0.81):
    """f σ_8(z) anchored to σ_8(z=0) = sigma8_0. Uses σ_8(z) = σ_8(0) δ(z)/δ(0)."""
    δ0 = δ_grid[0]   # z = 0 entry (smallest z first after reverse)
    sigma8_z = sigma8_0 * δ_grid / δ0
    return f_grid * sigma8_z

# ---------------- VALIDATION ----------------

def validate_lcdm():
    """
    Sanity check: ΛCDM at Ω_m = 0.315 should satisfy f(z) ≈ Ω_m(z)^0.55 to ~1%.
    """
    Om = 0.315
    E = make_E_lcdm(Om)
    z, δ, f = solve_growth(Om, E, z_init=1000.0, n_points=600)

    Om_z = Om * (1+z)**3 / E(z)**2
    f_approx = Om_z**0.55

    z_check = [0.0, 0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330]
    print(f"{'z':>6}  {'f (ODE)':>10}  {'Ω_m(z)^0.55':>12}  {'Δ%':>6}  {'σ_8/σ_80':>10}  {'fσ_8':>10}")
    print("-" * 70)
    fσ8 = fsigma8(z, δ, f, sigma8_0=0.81)
    for zc in z_check:
        idx = np.argmin(np.abs(z - zc))
        f_ode = f[idx]
        f_apx = f_approx[idx]
        pct = 100*(f_ode - f_apx)/f_apx
        s8_ratio = δ[idx]/δ[0]
        print(f"{z[idx]:6.3f}  {f_ode:10.4f}  {f_apx:12.4f}  {pct:6.2f}  {s8_ratio:10.4f}  {fσ8[idx]:10.4f}")

    # Pass criterion: max deviation < 2% over all DESI redshifts
    max_dev = np.max(np.abs((f - f_approx)/f_approx)[z < 3])
    print()
    print(f"Max |Δf/f| for z < 3: {100*max_dev:.2f}%")
    if max_dev < 0.02:
        print("PASS: growth ODE agrees with Ω_m(z)^0.55 textbook approximation to <2%.")
    else:
        print("FAIL: growth ODE deviates by more than 2% from textbook approximation. Investigate.")
    return max_dev

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true",
                    help="Run ΛCDM validation against Ω_m(z)^0.55 textbook approximation.")
    ap.add_argument("--model", choices=["LCDM", "Lcos"], default=None,
                    help="Which model to solve.")
    ap.add_argument("--Om", type=float, default=0.315,
                    help="ΛCDM Ω_m (only used if --model LCDM).")
    ap.add_argument("--s0", type=float, default=0.076,
                    help="Λcos s_0 (only used if --model Lcos).")
    ap.add_argument("--OL", type=float, default=0.685,
                    help="Λcos Ω_Λ (only used if --model Lcos).")
    ap.add_argument("--sigma8_0", type=float, default=0.81,
                    help="σ_8 normalization at z=0.")
    ap.add_argument("--out_dir", default="../results/")
    args = ap.parse_args()

    if args.validate:
        validate_lcdm()
        return

    if args.model is None:
        ap.error("Provide --model LCDM or --model Lcos, or use --validate.")

    if args.model == "LCDM":
        Om0 = args.Om
        E = make_E_lcdm(Om0)
    else:
        Om0 = 1.0 - args.OL
        E = make_E_lcos(args.s0, args.OL)

    z, δ, f = solve_growth(Om0, E, z_init=1000.0, n_points=400)
    fσ8 = fsigma8(z, δ, f, sigma8_0=args.sigma8_0)
    df = pd.DataFrame({"z": z, "delta": δ, "f": f, "fsigma8": fσ8})
    if args.model == "LCDM":
        suffix = f"_Om{args.Om:.3f}".replace(".", "p")
    else:
        suffix = f"_s0{args.s0:.3f}".replace(".", "p")
    out_path = args.out_dir + f"growth_{args.model}{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows; z range {z[0]:.3f}-{z[-1]:.1f})")

if __name__ == "__main__":
    main()
