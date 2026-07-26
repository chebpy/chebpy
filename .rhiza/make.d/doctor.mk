## .rhiza/make.d/doctor.mk - Developer prerequisite diagnostics

.PHONY: doctor

##@ Dev
doctor: ## verify local prerequisites and print actionable guidance
	@failed=0; \
	version_ge() { \
		awk -v a="$$1" -v b="$$2" 'BEGIN { \
			split(a, A, /[^0-9]+/); \
			split(b, B, /[^0-9]+/); \
			for (i = 1; i <= 3; i++) { \
				ai = A[i] + 0; \
				bi = B[i] + 0; \
				if (ai > bi) exit 0; \
				if (ai < bi) exit 1; \
			} \
			exit 0; \
		}'; \
	}; \
	check_tool() { \
		tool="$$1"; min="$$2"; install_url="$$3"; version_cmd="$$4"; gnu_required="$$5"; \
		if ! command -v "$$tool" >/dev/null 2>&1; then \
			printf "${RED}[❌]${RESET} %-9s missing — install: %s\n" "$$tool" "$$install_url"; \
			failed=1; \
			return; \
		fi; \
		version="$$(eval "$$version_cmd" 2>/dev/null)"; \
		if [ -z "$$version" ]; then \
			printf "${RED}[❌]${RESET} %-9s unknown version (required ≥ %s)\n" "$$tool" "$$min"; \
			failed=1; \
			return; \
		fi; \
		extra=""; \
		if [ "$$gnu_required" = "gnu" ] && ! make --version 2>/dev/null | grep -q '^GNU Make'; then \
			extra=" (GNU required)"; \
			printf "${RED}[❌]${RESET} %-9s %-8s < %s%s\n" "$$tool" "$$version" "$$min" "$$extra"; \
			failed=1; \
			return; \
		fi; \
		if version_ge "$$version" "$$min"; then \
			if [ "$$gnu_required" = "gnu" ]; then \
				extra=" (GNU required)"; \
			fi; \
			printf "${GREEN}[✅]${RESET} %-9s %-8s ≥ %s%s\n" "$$tool" "$$version" "$$min" "$$extra"; \
		else \
			if [ "$$gnu_required" = "gnu" ]; then \
				extra=" (GNU required)"; \
			fi; \
			printf "${RED}[❌]${RESET} %-9s %-8s < %s%s\n" "$$tool" "$$version" "$$min" "$$extra"; \
			failed=1; \
		fi; \
	}; \
	check_tool "uv" "0.4.0" "https://docs.astral.sh/uv/getting-started/installation/" "uv --version | awk 'NR==1 {print \$$2}'" ""; \
	check_tool "make" "3.8.0" "https://www.gnu.org/software/make/" "make --version | awk 'NR==1 {for (i=1; i<=NF; i++) if (\$$i ~ /^[0-9]+(\\.[0-9]+)+$$/) {print \$$i; exit}}'" "gnu"; \
	check_tool "git" "2.0.0" "https://git-scm.com" "git --version | awk 'NR==1 {print \$$3}'" ""; \
	if [ "$$failed" -ne 0 ]; then \
		printf "\n${YELLOW}[WARN] One or more prerequisites are missing or below minimum version.${RESET}\n"; \
		exit 1; \
	fi
