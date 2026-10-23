.PHONY: all check figs trace traceability paper clean

all: check

figs:
	python scripts/make_paper_figs.py

trace: traceability
traceability:
	python scripts/build_traceability.py

check: figs traceability
	python scripts/make_partition_table.py
	python scripts/verify_parameter_counts.py
	@echo "All reproduction checks passed successfully."

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode supp.tex

clean:
	rm -rf out/
