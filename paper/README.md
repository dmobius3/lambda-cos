# paper.tex

LaTeX source for the paper

> **B. Shatto, "Apparent Phantom Crossing as Template Bias: A Non-Phantom Test Case with Λcos" (2026).**

Source of truth for the prose lives at
`mode-identity-theory/files/working/files/phantom-crossing-template-artifact.md`;
this directory holds the LaTeX rendering for journal submission.

## Build

```bash
cd paper
make
```

This runs:

1. `cleanup_body.py` to regenerate `body.tex` from `body-raw.tex` (only if `body-raw.tex` is newer)
2. `pdflatex` (first pass, generates the `.aux` file with citation calls)
3. `bibtex` to build the bibliography from `references.bib`
4. `pdflatex` twice more to resolve cross-references and the bibliography

`paper.pdf` lands in this directory.

A `make clean` removes build artifacts (`.aux`, `.log`, `.bbl`, etc.).

## Optional: regenerate from the upstream markdown

If you have the `mode-identity-theory` repo cloned alongside `lambda-cos`:

```bash
make pandoc
make
```

`make pandoc` runs pandoc on the upstream markdown to refresh `body-raw.tex`. The default `make all` does **not** require the markdown source — `body-raw.tex` is committed.

## Class

The paper builds with **REVTeX 4.2** (`\documentclass[aps,prd,reprint,...]{revtex4-2}`), the official LaTeX class for Physical Review D submissions. REVTeX ships with TeX Live; no class files are bundled in this directory.

To switch from the two-column reprint format to single-column draft format, change `reprint` to `onecolumn` in `paper.tex`'s `\documentclass[...]` options. PRD accepts either during review; `reprint` is the final submission format.

## Files

```
paper.tex            Top-level: revtex preamble, title, authors, abstract, \input{body}, \bibliography
body.tex             Generated body content (sections 1–8 + Appendix A)
body-raw.tex         Pandoc output from the upstream markdown (snapshot)
cleanup_body.py      Transformation rules: pandoc -> revtex (cited inline as needed)
references.bib       17 entries, ref1..ref17, matching the numeric ordering in the markdown
figures/             Four figures copied from ../figures/ via `make figures`
paper.pdf            Compiled output (10 pages, two-column)
Makefile             Build pipeline
```

## Reproducibility

Everything in `paper.pdf` is reproducible from the deposited code in `..`:
- Section 4 (template-bias mocks) → `../scripts/template_bias.py`, `../scripts/threshold_scan.py`
- Section 5.2 (primary SN+BAO fit) → `../scripts/fit_lcdm.py`, `../scripts/fit_lcos.py`
- Section 5.4 (Ω_Λ scan) → `../scripts/fit_lcos.py --omega_lambda <val>`, aggregated by `../scripts/omega_lambda_scan.py`
- Section 5.5 (CMB priors) → `../scripts/fit_lcdm_cmb.py`, `../scripts/fit_lcos_cmb.py`
- Section 5.7 (wCDM) → `../scripts/fit_wcdm.py`
- Appendix A (clock exponents) → `../scripts/fit_clock_exponents.py`

Figures 1–4 are produced by `template_bias.py`, `threshold_scan.py`, and `make_plots.py` and live at `../figures/`.
