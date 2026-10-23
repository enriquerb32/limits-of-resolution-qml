.PHONY: all check figs traceability clean

all: check

figs:
	python scripts/make_paper_figs.py

traceability:
	python scripts/build_traceability.py

check: figs traceability
	python scripts/make_partition_table.py
	python scripts/verify_parameter_counts.py
	@echo "All reproduction checks passed successfully."

clean:
	rm -rf out/
