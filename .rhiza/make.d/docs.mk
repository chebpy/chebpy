## docs.mk - Documentation generation targets
# This file is included by the main Makefile.
# It provides targets for generating API documentation using pdoc
# and building/serving MkDocs documentation sites.

# Declare phony targets (they don't produce files)
.PHONY: mkdocs mkdocs-serve mkdocs-build

# Default output directory for MkDocs (HTML site)
MKDOCS_OUTPUT ?= _mkdocs

# MkDocs config file location
MKDOCS_CONFIG ?= mkdocs.yml

# Additional uvx --with packages to inject into mkdocs-build and mkdocs-serve.
# Projects can extend the package list without editing this template, e.g.:
#   MKDOCS_EXTRA_PACKAGES = --with "mkdocs-graphviz"
MKDOCS_EXTRA_PACKAGES ?=

# The 'mkdocs-build' target builds the MkDocs documentation site.
# 1. Checks if the mkdocs.yml config file exists.
# 2. Cleans up any previous output.
# 3. Builds the static site using mkdocs with material theme.
mkdocs-build:: install-uv ## build MkDocs documentation site
	@printf "${BLUE}[INFO] Building MkDocs site...${RESET}\n"
	@if [ -f "$(MKDOCS_CONFIG)" ]; then \
	  rm -rf "$(MKDOCS_OUTPUT)"; \
	  MKDOCS_OUTPUT_ABS="$$(pwd)/$(MKDOCS_OUTPUT)"; \
	  ${UVX_BIN} --with "mkdocs-material<10.0" --with "pymdown-extensions>=10.0" --with "mkdocs<2.0" $(MKDOCS_EXTRA_PACKAGES) mkdocs build \
	    -f "$(MKDOCS_CONFIG)" \
	    -d "$$MKDOCS_OUTPUT_ABS"; \
	else \
	  printf "${YELLOW}[WARN] $(MKDOCS_CONFIG) not found, skipping MkDocs build${RESET}\n"; \
	fi

# The 'mkdocs-serve' target serves the documentation with live reload.
# Useful for local development and previewing changes.
mkdocs-serve: install-uv ## serve MkDocs site with live reload
	@if [ -f "$(MKDOCS_CONFIG)" ]; then \
	  ${UVX_BIN} --with "mkdocs-material<10.0" --with "pymdown-extensions>=10.0" --with "mkdocs<2.0" $(MKDOCS_EXTRA_PACKAGES) mkdocs serve \
	    -f "$(MKDOCS_CONFIG)"; \
	else \
	  printf "${RED}[ERROR] $(MKDOCS_CONFIG) not found${RESET}\n"; \
	  exit 1; \
	fi

# Convenience alias
mkdocs: mkdocs-serve ## alias for mkdocs-serve
