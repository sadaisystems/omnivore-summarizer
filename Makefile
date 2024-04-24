SHELL := /bin/bash
VENV_DIR=osum_venv

run:
	$(VENV_DIR)/bin/python main.py

init: venv uv_sync
	mkdir -p logs

venv:
	if [ ! -d "$(VENV_DIR)" ]; then \
	    echo "Creating virtual environment..."; \
	    uv venv $(VENV_DIR) -p 3.10.12 --seed; \
	fi; \

	source $(VENV_DIR)/bin/activate

uv_compile:
	uv pip compile requirements.in -o requirements.txt

uv_sync:
	uv pip sync requirements.txt

.PHONY: run init venv uv_compile uv_sync
