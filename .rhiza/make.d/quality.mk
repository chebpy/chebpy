## .rhiza/make.d/quality.mk - Quality and Formatting
# This file provides targets for code quality checks, linting, and formatting.

# Declare phony targets (they don't produce files)
.PHONY: all deptry fmt

##@ Quality and Formatting
all: fmt deptry test docs-coverage security typecheck rhiza-test ## run all CI targets locally

deptry: install-uv ## Run deptry
	@if [ -d ${SOURCE_FOLDER} ]; then \
		$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${SOURCE_FOLDER}; \
	fi

	@if [ -d ${MARIMO_FOLDER} ]; then \
		if [ -d ${SOURCE_FOLDER} ]; then \
			$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${MARIMO_FOLDER} ${SOURCE_FOLDER} --ignore DEP004; \
		else \
		  	$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${MARIMO_FOLDER} --ignore DEP004; \
		fi \
	fi

fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} -p ${PYTHON_VERSION} pre-commit run --all-files
