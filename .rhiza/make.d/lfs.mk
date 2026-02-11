## lfs.mk - Git LFS (Large File Storage) setup
# This file is included by the main Makefile

# Declare phony targets
.PHONY: lfs-install lfs-pull lfs-track lfs-status

##@ Git LFS
lfs-install: ## install git-lfs and configure it for this repository
	@# -----------------------------
	@# Git LFS install (cross-platform)
	@# -----------------------------
	@UNAME_S=$$(uname -s); \
	UNAME_M=$$(uname -m); \
	if [ "$$UNAME_S" = "Darwin" ]; then \
		printf "${BLUE}[INFO] macOS detected ($$UNAME_M)${RESET}\n"; \
		mkdir -p .local/bin .local/tmp; \
		GIT_LFS_VERSION=$$(curl -s https://api.github.com/repos/git-lfs/git-lfs/releases/latest | grep '"tag_name"' | sed 's/.*"v//;s/".*//'); \
		if [ -z "$$GIT_LFS_VERSION" ]; then \
			printf "${RED}[ERROR] Failed to detect git-lfs version${RESET}\n"; \
			exit 1; \
		fi; \
		printf "${BLUE}[INFO] Installing git-lfs v$$GIT_LFS_VERSION${RESET}\n"; \
		if [ "$$UNAME_M" = "arm64" ]; then \
			ARCH_SUFFIX="darwin-arm64"; \
		else \
			ARCH_SUFFIX="darwin-amd64"; \
		fi; \
		DOWNLOAD_URL="https://github.com/git-lfs/git-lfs/releases/download/v$$GIT_LFS_VERSION/git-lfs-$$ARCH_SUFFIX-v$$GIT_LFS_VERSION.zip"; \
		if ! curl -fL -o .local/tmp/git-lfs.zip "$$DOWNLOAD_URL"; then \
			printf "${RED}[ERROR] Failed to download git-lfs v$$GIT_LFS_VERSION for $$ARCH_SUFFIX${RESET}\n"; \
			exit 1; \
		fi; \
		unzip -o -q .local/tmp/git-lfs.zip -d .local/tmp; \
		LFS_BINARY=$$(find .local/tmp -maxdepth 2 -type f -name "git-lfs" -perm +111 2>/dev/null | head -n 1); \
		if [ -z "$$LFS_BINARY" ]; then \
			printf "${RED}[ERROR] Failed to extract git-lfs binary from archive${RESET}\n"; \
			exit 1; \
		fi; \
		cp "$$LFS_BINARY" .local/bin/; \
		chmod +x .local/bin/git-lfs; \
		PATH=$$PWD/.local/bin:$$PATH git-lfs install; \
		rm -rf .local/tmp; \
	elif [ "$$UNAME_S" = "Linux" ]; then \
		printf "${BLUE}[INFO] Linux detected${RESET}\n"; \
		if ! command -v git-lfs >/dev/null 2>&1; then \
			printf "${BLUE}[INFO] Installing git-lfs via apt...${RESET}\n"; \
			if [ "$$(id -u)" -ne 0 ]; then \
				printf "${YELLOW}[WARN] This requires sudo privileges. You may be prompted for your password.${RESET}\n"; \
				sudo apt-get update && sudo apt-get install -y git-lfs || { \
					printf "${RED}[ERROR] Failed to install git-lfs with sudo.${RESET}\n"; \
					exit 1; \
				}; \
			else \
				apt-get update && apt-get install -y git-lfs || { \
					printf "${RED}[ERROR] Failed to install git-lfs.${RESET}\n"; \
					exit 1; \
				}; \
			fi; \
		fi; \
		git lfs install; \
	else \
		printf "${RED}[ERROR] Unsupported OS: $$UNAME_S${RESET}\n"; \
		exit 1; \
	fi

lfs-pull: ## download all git-lfs files for the current branch
	@printf "${BLUE}[INFO] Pulling Git LFS files...${RESET}\n"
	@git lfs pull

lfs-track: ## list all file patterns tracked by git-lfs
	@printf "${BLUE}[INFO] Git LFS tracked patterns:${RESET}\n"
	@git lfs track

lfs-status: ## show git-lfs file status
	@printf "${BLUE}[INFO] Git LFS status:${RESET}\n"
	@git lfs status
