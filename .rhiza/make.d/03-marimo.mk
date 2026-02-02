## Makefile.marimo - Marimo notebook targets
# This file is included by the main Makefile

# Declare phony targets (they don't produce files)
.PHONY: marimo-validate marimo marimushka

##@ Marimo Notebooks
marimo-validate: install ## validate all Marimo notebooks can run
	@printf "${BLUE}[INFO] Validating all notebooks in ${MARIMO_FOLDER}...${RESET}\n"
	@if [ ! -d "${MARIMO_FOLDER}" ]; then \
	  printf "${YELLOW}[WARN] Directory '${MARIMO_FOLDER}' does not exist. Skipping validation.${RESET}\n"; \
	else \
	  failed=0; \
	  for notebook in ${MARIMO_FOLDER}/*.py; do \
	    if [ -f "$$notebook" ]; then \
	      notebook_name=$$(basename "$$notebook"); \
	      printf "${BLUE}[INFO] Validating $$notebook_name...${RESET}\n"; \
	      if ${UV_BIN} run "$$notebook" > /dev/null 2>&1; then \
	        printf "${GREEN}[SUCCESS] $$notebook_name is valid${RESET}\n"; \
	      else \
	        printf "${RED}[ERROR] $$notebook_name failed validation${RESET}\n"; \
	        failed=$$((failed + 1)); \
	      fi; \
	    fi; \
	  done; \
	  if [ $$failed -eq 0 ]; then \
	    printf "${GREEN}[SUCCESS] All notebooks validated successfully${RESET}\n"; \
	  else \
	    printf "${RED}[ERROR] $$failed notebook(s) failed validation${RESET}\n"; \
	    exit 1; \
	  fi; \
	fi

marimo: install ## fire up Marimo server
	@if [ ! -d "${MARIMO_FOLDER}" ]; then \
	  printf " ${YELLOW}[WARN] Marimo folder '${MARIMO_FOLDER}' not found, skipping start${RESET}\n"; \
	else \
	  ${UV_BIN} run --with marimo marimo edit --no-token --headless "${MARIMO_FOLDER}"; \
	fi

# The 'marimushka' target exports Marimo notebooks (.py files) to static HTML.
# 1. Detects notebooks in the MARIMO_FOLDER.
# 2. Converts them using 'marimushka export'.
# 3. Generates a placeholder index.html if no notebooks are found.
marimushka:: install-uv ## export Marimo notebooks to HTML
	# Clean up previous marimushka output
	rm -rf "${MARIMUSHKA_OUTPUT}";

	@printf "${BLUE}[INFO] Exporting notebooks from ${MARIMO_FOLDER}...${RESET}\n"
	@if [ ! -d "${MARIMO_FOLDER}" ]; then \
	  printf "${YELLOW}[WARN] Directory '${MARIMO_FOLDER}' does not exist. Skipping marimushka.${RESET}\n"; \
	else \
	  mkdir -p "${MARIMUSHKA_OUTPUT}"; \
	  if ! ls "${MARIMO_FOLDER}"/*.py >/dev/null 2>&1; then \
	    printf "${YELLOW}[WARN] No Python files found in '${MARIMO_FOLDER}'.${RESET}\n"; \
	    printf '%s\n' '<html><head><title>Marimo Notebooks</title></head>' \
	      '<body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' \
	      > "${MARIMUSHKA_OUTPUT}/index.html"; \
	  else \
	    CURRENT_DIR=$$(pwd); \
	    OUTPUT_DIR="$$CURRENT_DIR/${MARIMUSHKA_OUTPUT}"; \
	    cd "${MARIMO_FOLDER}" && \
	    UVX_DIR=$$(dirname "$$(command -v uvx || echo "${INSTALL_DIR}/uvx")") && \
	    "${UVX_BIN}" "marimushka>=0.1.9" export --notebooks "." --output "$$OUTPUT_DIR" --bin-path "$$UVX_DIR" && \
	    cd "$$CURRENT_DIR"; \
	  fi; \
	fi
