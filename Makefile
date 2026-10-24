PYTHON := .venv/bin/python
UV := uv
TECTONIC := tectonic
# Python environment with the Splinx package, used only to draw the figures
SPLINX_PYTHON ?= python3

.PHONY: all setup experiments learned figures paper check clean-build

all: experiments learned figures paper check

setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

experiments:
	$(PYTHON) scripts/generate_figures.py

learned:
	$(PYTHON) scripts/train_learned_score.py

figures:
	$(SPLINX_PYTHON) scripts/plot_figures.py

paper:
	mkdir -p output/pdf
	$(TECTONIC) main.tex --outdir output/pdf --keep-intermediates
	mv output/pdf/main.pdf output/pdf/score-geometry-at-vanishing-density.pdf

check:
	$(PYTHON) scripts/check_reproducibility.py

clean-build:
	rm -f output/pdf/main.aux output/pdf/main.bbl output/pdf/main.blg output/pdf/main.log output/pdf/main.out
