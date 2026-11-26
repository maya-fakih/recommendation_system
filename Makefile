PYTHON ?= python3
VENV := ../recommendation-system-venv
DATA_URL := https://github.com/maya-fakih/recommendation_system/releases/download/course-data-v1/data.zip
DATA_FILES := data/validation.csv data/products.csv data/train_triplets.csv \
	data/validation_triplets.csv data/test.csv data/train.csv data/evaluation.csv \
	data/product_features.npz

.PHONY: install setup_data
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -e .

setup_data:
	@set -eu; staging=$$(mktemp -d); \
	trap 'rm -rf "$$staging"' EXIT; \
	curl --fail --location --silent --show-error --retry 3 "$(DATA_URL)" -o "$$staging/data.zip"; \
	unzip -q "$$staging/data.zip" $(DATA_FILES) -d "$$staging"; \
	mkdir -p data; \
	for file in $(DATA_FILES); do mv "$$staging/$$file" "$$file"; done
