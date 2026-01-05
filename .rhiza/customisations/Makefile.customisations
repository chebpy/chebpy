## Makefile.customisations - User-defined scripts and overrides
# This file is included by the main Makefile

# Declare phony targets
.PHONY: install-extras post-release

##@ Customisations

install-extras:: ## run custom build script (if exists)
	@if [ -x "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh" ]; then \
		printf "${BLUE}[INFO] Running custom build script from customisations folder...${RESET}\n"; \
		"${CUSTOM_SCRIPTS_FOLDER}"/build-extras.sh; \
	elif [ -f "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh" ]; then \
		printf "${BLUE}[INFO] Running custom build script from customisations folder...${RESET}\n"; \
		/bin/sh "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh"; \
	else \
		printf "${BLUE}[INFO] No custom build script in ${CUSTOM_SCRIPTS_FOLDER}, skipping...${RESET}\n"; \
	fi

post-release:: ## perform post-release tasks
	@if [ -x "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh" ]; then \
		printf "${BLUE}[INFO] Running post-release script from customisations folder...${RESET}\n"; \
		"${CUSTOM_SCRIPTS_FOLDER}"/post-release.sh; \
	elif [ -f "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh" ]; then \
		printf "${BLUE}[INFO] Running post-release script from customisations folder...${RESET}\n"; \
		/bin/sh "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh"; \
	else \
		printf "${BLUE}[INFO] No post-release script in ${CUSTOM_SCRIPTS_FOLDER}, skipping...${RESET}\n"; \
	fi
