## book.mk - Book-building targets
# This file is included by the main Makefile.
# It provides targets for exporting Marimo notebooks to HTML (marimushka)
# and compiling a companion book (minibook).

# Declare phony targets (they don't produce files)
.PHONY: marimushka mkdocs-build book

# Define a default no-op marimushka target that will be used
# when book/marimo/marimo.mk doesn't exist or doesn't define marimushka
marimushka:: install-uv
	@if [ ! -d "book/marimo" ]; then \
	  printf "${BLUE}[INFO] No Marimo directory found, creating placeholder${RESET}\n"; \
	  mkdir -p "${MARIMUSHKA_OUTPUT}"; \
	  printf '%s\n' '<html><head><title>Marimo Notebooks</title></head>' \
	    '<body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' \
	    > "${MARIMUSHKA_OUTPUT}/index.html"; \
	fi

# Define a default no-op mkdocs-build target that will be used
# when .rhiza/make.d/08-docs.mk doesn't exist or doesn't define mkdocs-build
mkdocs-build:: install-uv
	@if [ ! -f "docs/mkdocs.yml" ]; then \
	  printf "${BLUE}[INFO] No mkdocs.yml found, skipping MkDocs${RESET}\n"; \
	fi

# Default output directory for Marimushka (HTML exports of notebooks)
MARIMUSHKA_OUTPUT ?= _marimushka

# Default output directory for MkDocs
MKDOCS_OUTPUT ?= _mkdocs

# ----------------------------
# Book sections (declarative)
# ----------------------------
# format:
#   name | source index | book-relative index | source dir | book dir

BOOK_SECTIONS := \
  "API|_pdoc/index.html|pdoc/index.html|_pdoc|pdoc" \
  "Coverage|_tests/html-coverage/index.html|tests/html-coverage/index.html|_tests/html-coverage|tests/html-coverage" \
  "Test Report|_tests/html-report/report.html|tests/html-report/report.html|_tests/html-report|tests/html-report" \
  "Notebooks|_marimushka/index.html|marimushka/index.html|_marimushka|marimushka" \
  "Official Documentation|_mkdocs/index.html|docs/index.html|_mkdocs|docs"

##@ Book

# The 'book' target assembles the final documentation book.
# 1. Aggregates API docs, coverage, test reports, notebooks, and MkDocs site into _book.
# 2. Generates links.json to define the book structure.
# 3. Uses 'minibook' to compile the final HTML site.
book:: test docs marimushka mkdocs-build ## compile the companion book
	@printf "${BLUE}[INFO] Building combined documentation...${RESET}\n"
	@rm -rf _book && mkdir -p _book

	@if [ -f "_tests/coverage.json" ]; then \
	  printf "${BLUE}[INFO] Generating coverage badge JSON...${RESET}\n"; \
	  mkdir -p _book/tests; \
	  ${UV_BIN} run python -c "\
import json; \
data = json.load(open('_tests/coverage.json')); \
pct = int(data['totals']['percent_covered']); \
color = 'brightgreen' if pct >= 90 else 'green' if pct >= 80 else 'yellow' if pct >= 70 else 'orange' if pct >= 60 else 'red'; \
badge = {'schemaVersion': 1, 'label': 'coverage', 'message': f'{pct}%', 'color': color}; \
json.dump(badge, open('_book/tests/coverage-badge.json', 'w'))"; \
	  printf "${BLUE}[INFO] Coverage badge JSON:${RESET}\n"; \
	  cat _book/tests/coverage-badge.json; \
	  printf "\n"; \
	else \
	  printf "${YELLOW}[WARN] No coverage.json found, skipping badge generation${RESET}\n"; \
	fi

	@printf "{\n" > _book/links.json
	@first=1; \
	for entry in $(BOOK_SECTIONS); do \
	  name=$${entry%%|*}; \
	  rest=$${entry#*|}; \
	  src_index=$${rest%%|*}; rest=$${rest#*|}; \
	  book_index=$${rest%%|*}; rest=$${rest#*|}; \
	  src_dir=$${rest%%|*}; book_dir=$${rest#*|}; \
	  if [ -f "$$src_index" ]; then \
	    printf "${BLUE}[INFO] Adding $$name...${RESET}\n"; \
	    mkdir -p "_book/$$book_dir"; \
	    cp -r "$$src_dir/"* "_book/$$book_dir"; \
	    if [ $$first -eq 0 ]; then \
	      printf ",\n" >> _book/links.json; \
	    fi; \
	    printf "  \"%s\": \"./%s\"" "$$name" "$$book_index" >> _book/links.json; \
	    first=0; \
	  else \
	    printf "${YELLOW}[WARN] Missing $$name, skipping${RESET}\n"; \
	  fi; \
	done; \
	printf "\n}\n" >> _book/links.json

	@printf "${BLUE}[INFO] Generated links.json:${RESET}\n"
	@cat _book/links.json

	@TEMPLATE_ARG=""; \
	if [ -f "$(BOOK_TEMPLATE)" ]; then \
	  TEMPLATE_ARG="--template $(BOOK_TEMPLATE)"; \
	  printf "${BLUE}[INFO] Using book template $(BOOK_TEMPLATE)${RESET}\n"; \
	fi; \
	if [ -n "$(LOGO_FILE)" ]; then \
	  if [ -f "$(LOGO_FILE)" ]; then \
	    cp "$(LOGO_FILE)" "_book/logo$$(echo $(LOGO_FILE) | sed 's/.*\./\./')"; \
	    printf "${BLUE}[INFO] Copying logo: $(LOGO_FILE)${RESET}\n"; \
	  else \
	    printf "${YELLOW}[WARN] Logo file $(LOGO_FILE) not found, skipping${RESET}\n"; \
	  fi; \
	fi; \
	"$(UVX_BIN)" minibook \
	  --title "$(BOOK_TITLE)" \
	  --subtitle "$(BOOK_SUBTITLE)" \
	  $$TEMPLATE_ARG \
	  --links "$$(python3 -c 'import json;print(json.dumps(json.load(open("_book/links.json"))))')" \
	  --output "_book"

	@touch "_book/.nojekyll"
