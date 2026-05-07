PAPER = paper

# Source of truth for the prose lives at
#   ../../mode-identity-theory/files/working/files/phantom-crossing-template-artifact.md
# but body.tex (the cleaned LaTeX rendering) is committed alongside, so a
# fresh clone can build paper.pdf without that upstream repo. Run
# `make pandoc` only if you have the markdown source available and want
# to regenerate body-raw.tex.
MD_PATH = ../../mode-identity-theory/files/working/files/phantom-crossing-template-artifact.md

# Figures live at the repo root (../figures/), copied into paper/figures/
# so the submission tarball is self-contained. Refresh by running
# `make figures` (or just `cp`) after regenerating in ../figures/. PRD
# requires self-contained directories, which is why we copy rather than
# symlink.
LCOS_FIGURES = ../figures
FIGS = figures/fig1_template_bias_overlay.pdf \
       figures/fig2_threshold_scan.pdf \
       figures/fig3_lcos_corner.pdf \
       figures/fig4_hubble_residuals.pdf

.PHONY: all clean figures pandoc

all: $(PAPER).pdf

figures: $(FIGS)
$(FIGS): figures/%: $(LCOS_FIGURES)/%
	@mkdir -p figures
	cp $< $@

# Optional: regenerate body-raw.tex from the upstream markdown. Skipped
# in the default `make all` flow since body-raw.tex is committed.
pandoc:
	pandoc $(MD_PATH) --from markdown --to latex -o body-raw.tex

# body.tex is regenerated from body-raw.tex by cleanup_body.py.
body.tex: body-raw.tex cleanup_body.py
	python3 cleanup_body.py

# Standard pdflatex + bibtex + pdflatex + pdflatex sequence.
# REVTeX's paperNotes.bib mechanism requires the rebuild after a fresh bibtex.
$(PAPER).pdf: $(PAPER).tex body.tex references.bib $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).toc $(PAPER).fls $(PAPER).fdb_latexmk \
	      $(PAPER).synctex.gz paperNotes.bib
