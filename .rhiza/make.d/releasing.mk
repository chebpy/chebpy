## .rhiza/make.d/releasing.mk - Releasing and Versioning
# This file provides targets for version bumping and release management.

# Declare phony targets (they don't produce files)
.PHONY: bump release pre-bump post-bump pre-release post-release

# Hook targets (double-colon rules allow multiple definitions)
pre-bump:: ; @:
post-bump:: ; @:
pre-release:: ; @:
post-release:: ; @:

##@ Releasing and Versioning
bump: pre-bump ## bump version
	@if [ -f "pyproject.toml" ]; then \
		$(MAKE) install; \
		PATH="$(abspath ${VENV})/bin:$$PATH" ${UVX_BIN} "rhiza[tools]>=0.8.6" tools bump; \
		printf "${BLUE}[INFO] Updating uv.lock file...${RESET}\n"; \
		${UV_BIN} lock; \
	else \
		printf "${YELLOW}[WARN] No pyproject.toml found, skipping bump${RESET}\n"; \
	fi
	@$(MAKE) post-bump

release: pre-release install-uv ## create tag and push to remote with prompts
	@UV_BIN="${UV_BIN}" /bin/sh ".rhiza/scripts/release.sh"
	@$(MAKE) post-release
