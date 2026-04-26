## Makefile.marimo - Marimo notebook targets
# This file is included by the main Makefile

# Declare phony targets (they don't produce files)
.PHONY: marimo-validate marimo

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
	      notebook_stem=$$(basename "$$notebook" .py); \
	      artefact_folder="results/$$notebook_stem"; \
	      mkdir -p "$$artefact_folder"; \
	      printf "${BLUE}[INFO] Validating $$notebook_name (artefacts → $$artefact_folder)...${RESET}\n"; \
	      if NOTEBOOK_OUTPUT_FOLDER="$$artefact_folder" ${UV_BIN} run "$$notebook" > /dev/null 2>&1; then \
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
	  ${UV_BIN} run --no-project --with marimo --directory "${MARIMO_FOLDER}" marimo edit --no-token --headless; \
	fi
