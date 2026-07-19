.PHONY: test smoke clean

PYTHON ?= python3
PKG = src/volvence_labs

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m volvence_labs.cli run --probe pe-baseline-v0 --profile dev
	PYTHONPATH=src $(PYTHON) -m volvence_labs.cli run --probe r15-rollback-v0 --profile dev
	PYTHONPATH=src $(PYTHON) -m volvence_labs.cli ls

clean:
	rm -rf .labs experiments
