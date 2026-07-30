## .rhiza/make.d/completions.mk - Install shell tab-completion for make targets
# Copies the bundled completion scripts (.rhiza/completions/) into the user's
# completion directory so `make <TAB>` completes Rhiza targets. See
# .rhiza/completions/README.md for manual setup, aliases, and troubleshooting.

.PHONY: install-completions

# Which shell(s) to install completion for: bash, zsh, or both.
# make cannot reliably detect the login shell (it forces $SHELL to /bin/sh for
# recipes), so this defaults to `both` — installing an unused script is harmless.
# Narrow it with e.g. `make install-completions SHELL_KIND=zsh`.
SHELL_KIND ?= both

##@ Shell Completion
install-completions: ## install shell tab-completion for make targets (SHELL_KIND=bash|zsh|both)
	@src=".rhiza/completions"; \
	if [ ! -d "$$src" ]; then \
	  printf "${RED}[❌]${RESET} $$src not found — run from the project root.\n"; \
	  exit 1; \
	fi; \
	install_bash() { \
	  dest="$${XDG_DATA_HOME:-$$HOME/.local/share}/bash-completion/completions"; \
	  mkdir -p "$$dest"; \
	  cp "$$src/rhiza-completion.bash" "$$dest/make"; \
	  printf "${GREEN}[✓]${RESET} bash: installed to %s\n" "$$dest/make"; \
	  printf "${BLUE}[INFO]${RESET} start a new shell, or run: ${BOLD}source %s${RESET}\n" "$$dest/make"; \
	}; \
	install_zsh() { \
	  dest="$${XDG_DATA_HOME:-$$HOME/.local/share}/zsh/site-functions"; \
	  mkdir -p "$$dest"; \
	  cp "$$src/rhiza-completion.zsh" "$$dest/_make"; \
	  printf "${GREEN}[✓]${RESET} zsh: installed to %s\n" "$$dest/_make"; \
	  printf "${BLUE}[INFO]${RESET} if completion does not activate, add to ~/.zshrc:\n"; \
	  printf "         ${BOLD}fpath=(%s \$$fpath); autoload -U compinit && compinit${RESET}\n" "$$dest"; \
	}; \
	case "$(SHELL_KIND)" in \
	  bash) install_bash ;; \
	  zsh)  install_zsh ;; \
	  both) install_bash; install_zsh ;; \
	  *) printf "${YELLOW}[WARN]${RESET} unknown SHELL_KIND='%s'; installing both (use bash|zsh|both).\n" "$(SHELL_KIND)"; \
	     install_bash; install_zsh ;; \
	esac
