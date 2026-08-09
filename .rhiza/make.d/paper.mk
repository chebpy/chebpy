## paper.mk - LaTeX paper compilation targets
# This file is included by the main Makefile

PAPER_DIR ?= docs/paper

.PHONY: paper paper-clean

##@ Paper

paper:: ## compile LaTeX documents in docs/paper to PDF using latexmk
	@printf "${BLUE}[INFO] Checking for latexmk...${RESET}\n"
	@if ! command -v latexmk >/dev/null 2>&1; then \
	  printf "${RED}[ERROR] latexmk not found. Please install a LaTeX distribution (e.g., MacTeX, TeX Live).${RESET}\n"; \
	  exit 1; \
	fi
	@if [ -z "$$(find $(PAPER_DIR) -maxdepth 1 -name '*.tex' 2>/dev/null)" ]; then \
	  printf "${YELLOW}[WARN] No .tex files found in $(PAPER_DIR), skipping.${RESET}\n"; \
	  exit 0; \
	fi
	@if [ -f $(PAPER_DIR)/basanos.tex ]; then \
	  tex_file="basanos.tex"; \
	else \
	  tex_file=$$(find $(PAPER_DIR) -maxdepth 1 -name "*.tex" | head -1 | xargs basename); \
	fi; \
	printf "${BLUE}[INFO] Compiling $$tex_file...${RESET}\n"; \
	cd $(PAPER_DIR) && latexmk -pdf -bibtex -interaction=nonstopmode "$$tex_file" || exit 1; \
	pdf_file="$${tex_file%.tex}.pdf"; \
	printf "${GREEN}[SUCCESS] $(PAPER_DIR)/$$pdf_file${RESET}\n"

paper-clean:: ## remove latexmk build artifacts in docs/paper
	@printf "${BLUE}[INFO] Cleaning paper artifacts...${RESET}\n"
	@cd $(PAPER_DIR) && latexmk -C 2>/dev/null || true
	@printf "${GREEN}[SUCCESS] Cleaned $(PAPER_DIR)${RESET}\n"
