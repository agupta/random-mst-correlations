# Makefile for the MST pairwise-correlation paper repository.
#
#   make check       verification suite; nonzero exit on any failure     (~1 min)
#   make check-full  the same at wider ranges                          (~1-2 min)
#   make paper       rebuild paper/main.pdf with a plain pdflatex loop
#   make table       regenerate data/kn-exact-table.csv
#   make sha256      write evidence/SHA256SUMS.txt
#   make verify-sha256 verify every file in the committed manifest
#   make clean       remove LaTeX build by-products
#
# Everything below is standard library Python 3 and plain pdflatex.  There is no
# third-party dependency, no bibtex step and no latexmk step.

PYTHON ?= python3
PDFLATEX ?= pdflatex
TABLE_NMAX ?= 30
# Fix PDF metadata timestamps for this release snapshot so clean builds are
# byte-reproducible.  Override explicitly when preparing a later dated version.
SOURCE_DATE_EPOCH ?= 1785715200
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE := 1
export PYTHONDONTWRITEBYTECODE := 1

CHECKERS := \
	tests/check_paper_claims.py \
	tests/source_check_iter96.py \
	tests/source_check_iter102.py

# In the development repository the integrity boundary is the tracked tree,
# not a hand-maintained subset.  The fallback keeps an exported source package
# usable before its clean Git history is initialized.  The checksum file cannot
# hash itself.  Stage intended new files before running `make sha256` here, so
# `git ls-files` includes them.
MANIFEST := $(shell \
	if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		git ls-files; \
	else \
		find . -type f \
			! -path './paper/*.aux' ! -path './paper/*.log' \
			! -path './paper/*.out' ! -path './paper/*.toc' \
			! -path './paper/*.bbl' ! -path './paper/*.blg' \
			! -path './paper/*.fls' ! -path './paper/*.fdb_latexmk' \
			! -path './paper/*.synctex.gz' \
			! -path './src/__pycache__/*' ! -path './tests/__pycache__/*' \
			| sed 's|^\./||'; \
	fi | grep -v '^evidence/SHA256SUMS.txt$$')

.PHONY: all check check-full paper table sha256 verify-sha256 clean

all: check

# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

# Each checker is standalone, exact (integers and fractions only), standard
# library only, and exits nonzero on any failed check.  `make check` fails as
# soon as one of them fails.
check:
	$(PYTHON) tests/check_paper_claims.py
	$(PYTHON) tests/source_check_iter96.py
	$(PYTHON) tests/source_check_iter102.py
	@echo "make check: all checkers exited 0"

check-full:
	$(PYTHON) tests/check_paper_claims.py --full
	$(PYTHON) tests/source_check_iter96.py --partition-max 34 --shape-max 13 \
		--mc-max 20 --algebra-max 400
	$(PYTHON) tests/source_check_iter102.py
	@echo "make check-full: all checkers exited 0"

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

table:
	$(PYTHON) src/make_table.py $(TABLE_NMAX)

# ---------------------------------------------------------------------------
# manuscript
# ---------------------------------------------------------------------------

paper:
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	@echo "built paper/main.pdf"

# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------

sha256:
	@mkdir -p evidence
	@sha256sum $(MANIFEST) > evidence/SHA256SUMS.txt
	@echo "wrote evidence/SHA256SUMS.txt"

verify-sha256:
	@sha256sum --check evidence/SHA256SUMS.txt

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.bbl \
		paper/*.blg paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	rm -rf src/__pycache__ tests/__pycache__
