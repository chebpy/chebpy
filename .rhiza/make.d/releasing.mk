## .rhiza/make.d/releasing.mk - Releasing and Versioning
# This file provides targets for version bumping and release management.

# Declare phony targets (they don't produce files)
.PHONY: bump release release-status changelog pre-bump post-bump pre-release post-release

# Hook targets (double-colon rules allow multiple definitions)
pre-bump:: ; @:
post-bump:: ; @:
pre-release:: ; @:
post-release:: ; @:

# DRY_RUN support: pass DRY_RUN=1 to preview changes without applying them
_DRY_RUN_FLAG := $(if $(DRY_RUN),--dry-run,)
# BUMP support: pass BUMP=major|minor|patch to choose the bump type explicitly
# (omit BUMP to select the bump type interactively, the default behaviour)
_BUMP_FLAG := $(if $(BUMP),--bump $(BUMP),)
# PUSH support: pass PUSH=1 to push the tag without the interactive y/N prompt
# (omit PUSH to be prompted before the tag is pushed, the default behaviour)
_PUSH_FLAG := $(if $(PUSH),--push,)
_VERSION=0.7.1

##@ Releasing and Versioning
bump: pre-bump ## bump version of the project (supports DRY_RUN=1, BUMP=major|minor|patch)
	@if [ -f "pyproject.toml" ]; then \
		$(MAKE) install; \
		PATH="$(abspath ${VENV})/bin:$$PATH" ${UVX_BIN} "rhiza-tools>=$(_VERSION)" bump $(_DRY_RUN_FLAG) $(_BUMP_FLAG); \
		if [ -z "$(DRY_RUN)" ]; then \
			printf "${BLUE}[INFO] Checking uv.lock file...${RESET}\n"; \
			${UV_BIN} lock; \
		fi; \
	else \
		printf "${YELLOW}[WARN] No pyproject.toml found, skipping bump${RESET}\n"; \
	fi
	@$(MAKE) post-bump

release: pre-release install-uv ## bump version, create tag and push to trigger the release workflow (supports DRY_RUN=1, BUMP=major|minor|patch, PUSH=1)
	${UVX_BIN} "rhiza-tools>=$(_VERSION)" release $(_DRY_RUN_FLAG) $(_BUMP_FLAG) $(_PUSH_FLAG);
	@$(MAKE) post-release

release-status: ## show release workflow status and latest release information
ifeq ($(FORGE_TYPE),github)
	@{ $(MAKE) --no-print-directory workflow-status; printf "\n"; $(MAKE) --no-print-directory latest-release; } 2>&1 | $${PAGER:-less -R}
else ifeq ($(FORGE_TYPE),gitlab)
	@printf "${YELLOW}[WARN] GitLab detected — release-status is not yet supported for GitLab repositories.${RESET}\n"
	@printf "${BLUE}[INFO] Please check your pipeline status in the GitLab UI.${RESET}\n"
else
	@printf "${RED}[ERROR] Could not detect forge type (.github/workflows/ or .gitlab-ci.yml not found)${RESET}\n"
endif

changelog: install-uv ## generate/update CHANGELOG.md from git history using git-cliff (config: cliff.toml)
	@printf "${BLUE}[INFO] Generating CHANGELOG.md with git-cliff...${RESET}\n"
	@${UVX_BIN} git-cliff --output CHANGELOG.md
	@printf "${GREEN}[OK] CHANGELOG.md updated.${RESET}\n"



