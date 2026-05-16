PAPER = paper

# Figure PDFs in figures/ are copies of ../figures/. Refresh by running
# `make figures` after regenerating in ../figures/.
LCOS_FIGURES = ../figures
FIGS = figures/fig1_template_bias_overlay.pdf \
       figures/fig2_threshold_scan.pdf \
       figures/fig3_lcos_corner.pdf \
       figures/fig4_hubble_residuals.pdf \
       figures/fig5_fsigma8.pdf \
       figures/fig6_omegam_z.pdf

.PHONY: all clean figures

all: $(PAPER).pdf

figures: $(FIGS)
$(FIGS): figures/%: $(LCOS_FIGURES)/%
	@mkdir -p figures
	cp $< $@

# Standard pdflatex + bibtex + pdflatex + pdflatex. paper.tex is the
# single source of truth: edit directly.
$(PAPER).pdf: $(PAPER).tex references.bib $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).toc $(PAPER).fls $(PAPER).fdb_latexmk \
	      $(PAPER).synctex.gz paperNotes.bib
