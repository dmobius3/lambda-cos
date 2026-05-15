# Λcos: Apparent Phantom Crossing as Template Bias

Code and data for the paper:

**B. Shatto, "Apparent Phantom Crossing as Template Bias: A Bounded-Clock Deformation of ΛCDM" (2026).**

This repository contains the analysis pipeline, figure-generation scripts, and the LaTeX source for the paper itself (under [`paper/`](paper/)).

---

## Overview

The Λcos model is a one-parameter deformation of the fiducial flat ΛCDM expansion history using a bounded auxiliary variable. It yields

$$\frac{H^2(z)}{H_0^2} = \alpha\,(1+z)^3 \;-\; \beta\,(1+z) \;+\; \Omega_\Lambda$$

with $\alpha$, $\beta$ determined by $s_0$ and a fixed reference $\Omega_\Lambda$. Under the fiducial-matter diagnostic split, the effective residual satisfies $w_\mathrm{eff}(z) > -1$.

Reproducible results in this repository:

- **Joint Pantheon+ + DESI DR2 BAO fit** for ΛCDM, Λcos, and wCDM (§5.2, §5.7)
- **Template-bias mock fits** across CPL, BA, JBP, and a three-parameter polynomial (§4.2)
- **CPL threshold scan** across $s_0 \in [0.01, 0.40]$ (§4.3)
- **Prior sensitivity** for Λcos under flat, $s_0^2$, and $\log_{10}(s_0)$ priors (§5.3)
- **Ω_Λ sensitivity scan** across 0.680–0.715 (§5.4)
- **CMB distance priors** for flat ΛCDM, non-flat ΛCDM, Λcos at fixed Ω_Λ, and Λcos with Ω_Λ free (§5.5)
- **Savage-Dickey Bayes factor** at the $s_0$ prior boundary (§5.5)
- **Clock exponent comparison** for Models A, B, C, D (Appendix A)
- **Linear growth comparison** against DESI DR1 ShapeFit+BAO $f\sigma_{s8}$ at six tracer effective redshifts (§6.C)
- **Figure generation** for Figs. 1–6

---

## Citation

If you use this code or data, please cite the paper and the Zenodo archive of this repository.

```bibtex
@article{Shatto2026Lambdacos,
  title  = {Apparent Phantom Crossing as Template Bias: A Bounded-Clock
            Deformation of ΛCDM},
  author = {Shatto, B.},
  year   = {2026}
}

@misc{ShattoLambdacosCode2026,
  author = {Shatto, B.},
  title  = {Λcos: Code and Data for "Apparent Phantom Crossing as Template Bias"},
  year   = {2026},
  doi    = {10.5281/zenodo.19798852}
}
```

---

## Repository contents

```
.
├── README.md                                  This file
├── LICENSE                                    MIT
├── requirements.txt                           Python dependencies
├── data/
│   ├── pantheon_plus.csv                      Pantheon+ SNe Ia magnitudes
│   ├── pantheon_plus_cov.npy                  1701 × 1701 statistical + systematic covariance
│   ├── desi_dr2_bao.csv                       DESI DR2 BAO observables
│   ├── desi_dr2_bao_cov.npy                   Inter-observable covariance for the 13 BAO points
│   └── desi_dr1_fs_fsigma8.csv                DESI DR1 ShapeFit+BAO compressed fσ_s8 amplitudes at 6 tracer effective redshifts (§6.C)
├── scripts/
│   ├── fit_lcdm.py                            Flat ΛCDM MCMC fit (§5.2)
│   ├── fit_lcos.py                            Λcos MCMC fit; --omega_lambda VALUE (default 0.685, §5.2)
│   ├── fit_wcdm.py                            wCDM MCMC fit (§5.7)
│   ├── fit_clock_exponents.py                 Clock exponent comparison (Appendix A)
│   ├── fit_lcdm_cmb.py                        ΛCDM + CMB distance priors; --non_flat (§5.5)
│   ├── fit_lcos_cmb.py                        Λcos + CMB distance priors; --free_omega_lambda (§5.5)
│   ├── omega_lambda_scan.py                   Aggregate §5.4 Ω_Λ sensitivity table
│   ├── prior_sensitivity.py                   §5.3 prior-reweighting of the baseline Λcos chain
│   ├── bayes_factor.py                        §5.5 Savage-Dickey Bayes factor at the s₀ prior boundary
│   ├── template_bias.py                       Template-bias mocks + Fig. 1 (§4.2)
│   ├── threshold_scan.py                      CPL threshold scan + Fig. 2 (§4.3)
│   ├── make_plots.py                          Λcos corner (Fig. 3) and residuals (Fig. 4)
│   ├── growth.py                              Linear-growth ODE solver for arbitrary E(z); validates against textbook Ω_m(z)^0.55 (§6.C)
│   ├── compute_rsd_chi2.py                    χ²_RSD comparison vs DESI DR1 FS at the SN+BAO best fit (§6.C)
│   ├── make_growth_figures.py                 Fig. 5 (fσ_8 trajectories + DR1 FS data) and Fig. 6 (Ω_m(z) diagnostic) (§6.C)
│   └── _summary.py                            Shared harmonized summary-JSON schema helper
├── results/                                   MCMC chains, post-burn samples, summaries, generated figures
│   ├── lcdm_chain.npy, lcdm_post.csv, lcdm_summary.json, lcdm_corner.png
│   ├── lcos_chain.npy, lcos_post.csv, lcos_summary.json, lcos_corner.{png,pdf}
│   ├── lcos_omegaL_<v>_*.{npy,csv,json,png}   Λcos at alternative Ω_Λ ∈ {0.680, 0.690, 0.700, 0.715} (§5.4)
│   ├── wcdm_chain.npy, wcdm_post.csv, wcdm_summary.json, wcdm_corner.png
│   ├── lcdm_cmb_*.{npy,csv,json,png}          Flat ΛCDM + CMB priors (§5.5)
│   ├── lcdm_cmb_nonflat_*.{npy,csv,json,png}  Non-flat ΛCDM + CMB priors (§5.5)
│   ├── lcos_cmb_*.{npy,csv,json,png}          Λcos at Ω_Λ = 0.685 + CMB priors (§5.5)
│   ├── lcos_cmb_freeOL_*.{npy,csv,json,png}   Λcos with Ω_Λ free + CMB priors (§5.5)
│   ├── clock_exponent_{A,B,C,D}_chain.npy
│   ├── clock_exponent_{A,B,C,D}_postburn.csv
│   ├── clock_exponent_results.csv             Appendix A summary across all four models
│   ├── prior_sensitivity.csv                  §5.3 reweighted-prior medians and 95% upper limits
│   ├── bayes_factor.csv                       §5.5 B_01 across bandwidths for stability
│   ├── template_bias.csv, template_bias.{png,pdf}     §4.2 fits and Fig. 1
│   ├── threshold_scan.csv, threshold_scan.{png,pdf}   §4.3 scan and Fig. 2
│   ├── residuals.{png,pdf}                    Fig. 4
│   ├── growth_LCDM_Om0p315.csv                §6.C growth trajectory at Ω_m = 0.315
│   ├── growth_Lcos_s00p076.csv                §6.C growth trajectory at the Λcos posterior median
│   ├── growth_Lcos_s00p185.csv                §6.C growth trajectory at the 95% upper limit
│   ├── rsd_chi2.csv, rsd_residuals.csv        §6.C χ²_RSD summary and per-tracer pulls
│   ├── fig5_fsigma8.{png,pdf}                 Fig. 5 (fσ_8 + DR1 FS data)
│   └── fig6_omegam_z.{png,pdf}                Fig. 6 (Ω_m(z) diagnostic)
├── tables/
│   ├── clock_exponent_appendix_A_fits.csv     Curated Appendix A reference values
│   └── omega_lambda_scan.csv                  Aggregated §5.4 Ω_Λ sensitivity table
├── figures/
│   ├── fig1_template_bias_overlay.pdf         §4.2: w(z) overlays for CPL/BA/JBP/Polynomial
│   ├── fig2_threshold_scan.pdf                §4.3: recovered (w₀, w_a) vs s₀
│   ├── fig3_lcos_corner.pdf                   §5.2: Λcos posterior in (s₀, H₀r_d, M_B)
│   ├── fig4_hubble_residuals.pdf              §5.2: Pantheon+ binned residuals for ΛCDM and Λcos
│   ├── fig5_fsigma8.pdf                       §6.C: fσ_8(z) trajectories + DESI DR1 FS data
│   └── fig6_omegam_z.pdf                      §6.C: Ω_m(z) diagnostic out to z = 3
└── paper/                                     LaTeX source for the manuscript
    ├── paper.tex                              REVTeX 4.2 single-source LaTeX (preamble + body + bibliography)
    ├── references.bib                         20 entries
    ├── figures/                               Self-contained copies of ../figures/*.pdf
    ├── paper.pdf                              Compiled output (single-column, JCAP submission format)
    ├── Makefile                               Build pipeline (pdflatex + bibtex + pdflatex × 2)
    └── README.md                              Build instructions
```

`figures/` holds the paper-facing PDFs at their published filenames. They are stable copies of the corresponding script outputs in `results/`.

---

## Installation

Python 3.10 or later recommended.

```bash
git clone https://github.com/dmobius3/lambda-cos.git
cd lambda-cos
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```
numpy>=1.24
scipy>=1.10
emcee>=3.1
corner>=2.2
matplotlib>=3.7
h5py>=3.8
pandas>=2.0
```

Total install footprint about 200 MB. The primary SN+BAO MCMC fit takes roughly 5 minutes on a recent laptop (32 walkers, 5000 steps).

---

## Quickstart: reproducing the primary fit

The headline result is the joint Pantheon+ + DESI DR2 BAO Λcos fit (§5.2). All scripts are written to be run from the `scripts/` directory and resolve paths via `../data/` and `../results/`.

```bash
cd scripts
python fit_lcos.py
```

Outputs to `results/`: `lcos_chain.npy`, `lcos_post.csv`, `lcos_summary.json`, `lcos_corner.png`.

For the ΛCDM baseline:

```bash
python fit_lcdm.py
```

For wCDM (§5.7):

```bash
python fit_wcdm.py
```

Reference summary values from the deposited posteriors:

```
Λcos:    s0 median ≈ 0.076  (68% CI: 0.023, 0.143)
         s0 95% UL ≈ 0.185  (flat prior)
         H0 r_d    ≈ 10008  km/s
         M_B       ≈ -19.353
         tau_max   ≈ 46.9
ΛCDM:    Ω_m       ≈ 0.312  (68% CI: 0.304, 0.321)
         H0 r_d    ≈ 10043  km/s
         M_B       ≈ -19.355
         tau_max   ≈ 35.9
wCDM:    Ω_m       ≈ 0.297
         w         ≈ -0.855  (68% CI: -0.89, -0.82)
         Δχ²       ≈ -13.05 vs flat ΛCDM
         ΔAIC      ≈ -11.05
         ΔBIC      ≈  -5.61
         tau_max   ≈ 47.4
```

---

## Reproducing each figure

```bash
cd scripts

# Figure 1 — w(z) recoveries from CPL, BA, JBP, Polynomial
python template_bias.py
# -> results/template_bias.{png,pdf}

# Figure 2 — CPL threshold scan
python threshold_scan.py
# -> results/threshold_scan.{png,pdf}

# Figures 3 and 4 — Λcos corner plot and Pantheon+ residuals
# Requires lcdm_post.csv and lcos_post.csv (run fit_lcdm.py and fit_lcos.py first)
python make_plots.py
# -> results/lcos_corner.{png,pdf}   (Fig. 3)
# -> results/residuals.{png,pdf}     (Fig. 4)

# Figures 5 and 6 — fσ_8(z) with DESI DR1 FS overlay and Ω_m(z) diagnostic
# Requires growth_LCDM_*.csv and growth_Lcos_*.csv (see §6.C reproduction below)
python make_growth_figures.py
# -> results/fig5_fsigma8.{png,pdf}  (Fig. 5)
# -> results/fig6_omegam_z.{png,pdf} (Fig. 6)
```

The paper-facing PDFs in `figures/` (`fig1_template_bias_overlay.pdf`, `fig2_threshold_scan.pdf`, `fig3_lcos_corner.pdf`, `fig4_hubble_residuals.pdf`, `fig5_fsigma8.pdf`, `fig6_omegam_z.pdf`) are stable copies of the corresponding `results/` outputs renamed to match the in-paper figure numbers.

---

## Reproducing the template-bias scan (§4.3)

```bash
cd scripts
python threshold_scan.py
```

Iterates CPL fits across $s_0 \in [0.01, 0.40]$ in steps of $0.01$, recording $(w_0, w_a, \chi^2)$ at each value to `results/threshold_scan.csv` and producing Fig. 2.

The single-$s_0$ mock comparison across all four parameterizations (Table in §4.2) is produced separately by `template_bias.py`, which writes `results/template_bias.csv` and Fig. 1.

---

## Reproducing Appendix A (clock exponent selection)

```bash
cd scripts
python fit_clock_exponents.py
```

Models A (n = 0), B (n = −1), C (n = +1), D (n = −1/2) are fit with the same MCMC setup as the primary Λcos run. Outputs to `results/`:

- `clock_exponent_{A,B,C,D}_chain.npy` — full chains
- `clock_exponent_{A,B,C,D}_postburn.csv` — post-burn samples
- `clock_exponent_results.csv` — summary with one row per model: best-fit parameters, χ² split (SN, BAO, total), Δχ² vs ΛCDM, acceptance fraction

The curated paper-facing values are also deposited at `tables/clock_exponent_appendix_A_fits.csv`.

---

## Reproducing the Ω_Λ sensitivity scan (§5.4)

```bash
cd scripts
python fit_lcos.py --omega_lambda 0.680
python fit_lcos.py --omega_lambda 0.685   # canonical
python fit_lcos.py --omega_lambda 0.690
python fit_lcos.py --omega_lambda 0.700
python fit_lcos.py --omega_lambda 0.715
python omega_lambda_scan.py
```

Outputs to `results/lcos_omegaL_<v>_*.{npy,csv,json,png}` for each non-canonical value (the canonical Ω_Λ = 0.685 writes to `lcos_*` without a suffix). The aggregator reads each summary JSON and produces `tables/omega_lambda_scan.csv` with one row per Ω_Λ (s₀ median, s₀ 95% UL, χ²_min, χ²_SN, χ²_BAO, Δχ² vs ΛCDM baseline, τ_max, acceptance).

---

## Reproducing §5.5 (CMB distance priors)

§5.5 adds compressed Planck 2018 distance priors (R = 1.7502 ± 0.0046, ℓ_A = 301.47 ± 0.09) to the SN+BAO likelihood. Four fits in total:

```bash
cd scripts
python fit_lcdm_cmb.py                       # Flat ΛCDM + CMB priors  (3 params)
python fit_lcdm_cmb.py --non_flat            # Non-flat ΛCDM + CMB     (4 params, Ω_k = 1 - Ω_m - Ω_Λ - Ω_r)
python fit_lcos_cmb.py                       # Λcos at Ω_Λ = 0.685 + CMB priors (3 params)
python fit_lcos_cmb.py --free_omega_lambda   # Λcos with Ω_Λ free + CMB priors  (4 params)
```

Outputs:

- `results/lcdm_cmb_*` and `results/lcdm_cmb_nonflat_*` for the two ΛCDM cases
- `results/lcos_cmb_*` and `results/lcos_cmb_freeOL_*` for the two Λcos cases

Each summary JSON reports the χ² split (SN, BAO, CMB), the best-fit point from a post-MCMC optimizer pass, the integrated autocorrelation time per parameter, the acceptance fraction, and (for the 4-parameter fits) the Ω_Λ posterior quantiles.

The CMB priors are implemented with the standard compressed-prior forms

```
R   = sqrt(Ω_m) * ∫₀^z* dz/E(z)
ℓ_A ≈ π c / (H0 r_d) * ∫₀^z* dz/E(z)        (treating r_d ≈ r_s(z*); ~2% offset for standard cosmology)
```

with z* = 1090 and Ω_r = 9.15 × 10⁻⁵ included in E(z) for the high-z integral.

---

## Reproducing §6.C (linear-growth consistency check)

§6.C compares the linear-growth prediction f σ_8(z) of ΛCDM and Λcos against the DESI DR1 ShapeFit+BAO compressed growth amplitudes (DESI 2024 Paper V, Appendix A, Eqs. A.13–A.24) at six tracer effective redshifts. The Λcos correction enters only through H(z); no perturbation-level parameter is introduced.

```bash
cd scripts

# Step 1: validate the growth ODE solver against textbook ΛCDM
python growth.py --validate
# -> max |Δf/f| ≲ 0.5% vs Ω_m(z)^0.55 over z ∈ [0, 2.4]

# Step 2: compute fσ_8(z) trajectories for ΛCDM and Λcos at two s₀ values
python growth.py --model LCDM --Om 0.315
python growth.py --model Lcos --s0 0.076 --OL 0.685
python growth.py --model Lcos --s0 0.185 --OL 0.685
# -> results/growth_LCDM_Om0p315.csv
# -> results/growth_Lcos_s00p076.csv  (posterior median)
# -> results/growth_Lcos_s00p185.csv  (95% upper limit)

# Step 3: χ²_RSD against DESI DR1 FS at the SN+BAO best fit
python compute_rsd_chi2.py
# -> results/rsd_chi2.csv          (per-model χ²_RSD and Δχ²)
# -> results/rsd_residuals.csv     (per-tracer pulls)

# Step 4: figures
python make_growth_figures.py
# -> results/fig5_fsigma8.{png,pdf}   (Fig. 5)
# -> results/fig6_omegam_z.{png,pdf}  (Fig. 6)
```

Reference values:

```
χ²_RSD (6 tracer bins, diagonal-error consistency check):
  ΛCDM (Ω_m = 0.315)                4.64
  Λcos (s₀ = 0.076, median)         4.68    Δχ² = +0.04
  Λcos (s₀ = 0.185, 95% UL)         4.90    Δχ² = +0.26

Ω_m(z) split at z = 2.3:
  Λcos – ΛCDM at s₀ = 0.185         −2.9%
```

Note on data provenance: DESI DR2 (March 2025) released BAO distances only, not a DR2 full-shape product. The growth comparison therefore uses DR1 FS; the combination of DR2 BAO with DR1 FS follows the collaboration's own precedent (Elbers et al., arXiv:2503.14744; Forero-Sánchez et al., arXiv:2602.18761 for the rigorous joint treatment).

---

## Data sources and provenance

| Dataset | Source | Reference |
|---|---|---|
| Pantheon+ SNe Ia | [pantheonplussh0es.github.io](https://pantheonplussh0es.github.io/) | Brout et al., *Astrophys. J.* **938**, 110 (2022) |
| DESI DR2 BAO | [data.desi.lbl.gov](https://data.desi.lbl.gov/) | DESI Collaboration, arXiv:2503.14738 (2025) |
| DESI DR1 ShapeFit+BAO fσ_s8 | DESI 2024 Paper V, Appendix A | DESI Collaboration, *J. Cosmol. Astropart. Phys.* **2025**, 008, arXiv:2411.12021 |
| Planck 2018 distance priors | Compressed (R, ℓ_A) from Planck VI | Planck Collaboration VI, *Astron. Astrophys.* **641**, A6 (2020) |

The files under `data/` are formatted derivatives of the public sources above, repackaged for direct loading by the fit scripts. No proprietary data is included.

---

## Reproducibility notes

- **Random seeding**: `fit_clock_exponents.py` uses a fixed seed (`RNG_SEED = 12345`). The other MCMC scripts (`fit_lcdm.py`, `fit_lcos.py`, `fit_wcdm.py`) initialize walkers from `np.random.randn` without an explicit seed; small numerical variations between runs are within the posterior thickness and do not change the reported summary values.
- **MCMC configuration**: 32 walkers, 5000 steps, 1000 burn-in across all fit scripts.
- **Numerical accuracy**: distance integrals use SciPy's `cumulative_trapezoid` on a 4000-point grid in z ∈ [0, 2.5] (or up to z\_max ≈ 2.4 from the BAO range). For the clock-exponent run the grid extends to 1.002 × max(z\_data).
- **Working directory**: scripts are run from the `scripts/` subdirectory; paths resolve via `../data/` and `../results/`.
- **Platform**: tested on macOS 14.x and Ubuntu 22.04 with Python 3.11. No GPU required.

---

## License

This repository is released under the MIT License. See `LICENSE` for the full text.

The Pantheon+ and DESI DR2 BAO data products are redistributed under the terms of their original publications; refer to the linked sources above for their license terms.

---

## Contact

- Author: B. Shatto, bshatto.pe@gmail.com
- Issues and questions: please open a [GitHub issue](https://github.com/dmobius3/lambda-cos/issues).
