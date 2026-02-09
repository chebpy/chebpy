## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DOCFORMAT=google
DEFAULT_AI_MODEL=gpt-4.1
LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
