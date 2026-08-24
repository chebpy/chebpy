## Makefile (template-owned) -- synced from rhiza's `core` bundle. Edit it there.
#
# A compatibility shim, not the documented interface -- that is `uv run rhiza-task <task>`.
# It exists for workflows pinned at @v1.3.3 and earlier that still call `make test`, and for
# repos with no Python project to hold the pin.
#
# `uvx rhiza-task shim` used to print this file and each repo owned the copy it printed.
# That put a *template* inside the task runner: the CLI had to know about `local.mk`, the
# `##` help convention and the ./bin/uvx bootstrap, and rhiza then hand-maintained a variant
# of the generator's output anyway. The worse half was the pin below -- the shim wrote the
# version of whichever CLI happened to print it, so moving a repo's gates forward was a
# per-repo hand edit `/rhiza:update` could not make, and every consumer silently lagged.
#
# The template owns the front door instead, the way it owns every other config file, and
# `RHIZA_TASK` travels with the sync -- the property `RHIZA_CHECKS_VERSION` already had: a
# repo synced at a tag runs that tag's gates.
#
# Repo-specific *tasks* go in a `rhiza_task.tasks` entry point, repo-specific *targets* in
# `local.mk`, which core deliberately does not ignore. Nothing goes below the shim: this
# file is synced, so the next `/rhiza:update` overwrites whatever was appended to it.
RHIZA_TASK ?= rhiza-task@1.3.1

# uv cannot be delegated, because uv is what runs the CLI. Prepended so a machine carrying
# an older uv still resolves the pin, exported because task bodies shell out to bare `uv`.
INSTALL_DIR ?= $(abspath ./bin)
UVX ?= $(shell command -v uvx 2>/dev/null || echo $(INSTALL_DIR)/uvx)
export PATH := $(INSTALL_DIR):$(PATH)

# `UV` too, for `local.mk` to reach: the astral installer writes both binaries into the
# same directory, so once $(UVX) exists this does, and the empty recipe both satisfies
# make's remake attempt and keeps the catch-all from forwarding the path as a task name.
UV ?= $(shell command -v uv 2>/dev/null || echo $(INSTALL_DIR)/uv)
$(UV): $(UVX) ;

.DEFAULT_GOAL := help

.PHONY: help

# `rhiza-task list` cannot know about the targets `local.mk` adds, so anything there with
# a `##` comment is listed under them. This is what lets a repo move its own targets out
# of this file without losing them from `make help`.
help: $(UVX)
	@$(UVX) $(RHIZA_TASK) list
	@own=$$(grep -hE '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##/ -- /' -e 's/^/  /'); \
		[ -z "$$own" ] || printf '\nRepo-owned targets:\n%s\n' "$$own"

# Every task, and every typo -- the CLI's "unknown task" error is the backstop. `FORCE` is
# what keeps them phony: .PHONY takes no patterns, but a phony prerequisite is never up to
# date, so `make book` next to a `book/` directory still runs. Recursive `=` because `$@`
# only has a value while make is running the rule.
RHIZA_TASK_GOAL = $@

%: $(UVX) FORCE
	@$(UVX) $(RHIZA_TASK) $(RHIZA_TASK_GOAL)

# A file target, so make's up-to-date check is the idempotence. `$(UVX)` and not
# `$(INSTALL_DIR)/uvx`, because an on-PATH uvx would otherwise be matched by the catch-all
# whose prerequisite is that same file: `make: Circular ... dependency dropped`.
$(UVX):
	@echo "[INFO] uv not found; installing into $(@D)"
	@curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$(@D)" sh >/dev/null

.PHONY: FORCE
FORCE:

# Repo-specific one-offs. An explicit rule beats a pattern rule, so these win.
-include local.mk

# Both are targets make tries to remake, and the catch-all would route that to the CLI.
local.mk: ;
Makefile: ;
