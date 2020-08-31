# import the metadata given in __init__.py
include chebpy/__init__.py
PROJECT_VERSION := ${__version__}

SHELL := /bin/bash

.PHONY: packages help test tag


.DEFAULT: help

help:
	@echo "make test"
		@echo "       Run all tests"
	@echo "make tag"
		@echo "       Make a tag on Github."
	@echo "make server"
		@echo "       Start the Flask server."


test: venv
	$(VENV)/pip install -r ./tests/requirements.txt
	$(VENV)/py.test -cov=chebpy -vv ./tests

tag: test
	git tag -a ${PROJECT_VERSION} -m "new tag"
	git push --tag

include Makefile.venv # All the magic happens here
