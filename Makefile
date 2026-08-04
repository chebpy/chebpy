## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Override template default of 90: CLAUDE.md requires full coverage of src/, with
# genuinely unreachable defensive branches marked `# pragma: no cover` plus a reason.
COVERAGE_FAIL_UNDER = 100

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
