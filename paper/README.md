# paper.tex

LaTeX source for the paper

> **B. Shatto, "Apparent Phantom Crossing as Template Bias: A Bounded Test Case with Λcos" (2026).**

`paper.tex` is the single source of truth: edit directly.

## Build

```bash
cd paper
make
```

Runs `pdflatex` + `bibtex` + `pdflatex` × 2. `paper.pdf` lands in this directory. `make clean` removes `.aux`, `.log`, `.bbl`, etc.

## Class

REVTeX 4.2 (`\documentclass[aps,prd,reprint,nofootinbib,longbibliography]{revtex4-2}`), the official LaTeX class for Physical Review D. Ships with TeX Live; no class files are bundled.

To switch from two-column reprint to single-column draft format, change `reprint` to `onecolumn` in `paper.tex`. PRD accepts either during review.

## Files

```
paper.tex            Single-source LaTeX (preamble + body + bibliography)
references.bib       17 entries, ref1..ref17, APS apsrev4-2 style
figures/             Self-contained copies of ../figures/
paper.pdf            Compiled output (10 pages, two-column)
Makefile             Build pipeline
```

## Reproducibility

`paper.pdf` numbers are reproducible from the deposited code in `..`:
- §4 (template-bias mocks) → `../scripts/template_bias.py`, `../scripts/threshold_scan.py`
- §5.2 (primary SN+BAO fit) → `../scripts/fit_lcdm.py`, `../scripts/fit_lcos.py`
- §5.3 (prior sensitivity, Table V) → `../scripts/prior_sensitivity.py`
- §5.4 (Ω_Λ scan) → `../scripts/fit_lcos.py --omega_lambda <val>` aggregated by `../scripts/omega_lambda_scan.py`
- §5.5 (CMB priors) → `../scripts/fit_lcdm_cmb.py`, `../scripts/fit_lcos_cmb.py`
- §5.5 (Savage-Dickey Bayes factor) → `../scripts/bayes_factor.py`
- §5.7 (wCDM) → `../scripts/fit_wcdm.py`
- §6.C (linear-growth, fσ_8 vs DESI DR1 FS) → `../scripts/growth.py`, `../scripts/compute_rsd_chi2.py`, `../scripts/make_growth_figures.py`
- Appendix A (clock exponents) → `../scripts/fit_clock_exponents.py`
