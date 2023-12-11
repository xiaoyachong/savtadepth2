#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = savta_depth
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create -y --name $(PROJECT_NAME) python=3.7.6 
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	@echo ">>> No conda detected, creating venv environment."
	$(PYTHON_INTERPRETER) -m venv env
	@echo ">>> New virtual env created. Activate with:\nsource env/bin/activate ."
endif

load_requirements:
	@echo ">>> Installing requirements. Make sure your virtual environment is activated."
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

save_requirements:
	@echo ">>> Saving requirements."
	pip list --format=freeze > requirements.txt
