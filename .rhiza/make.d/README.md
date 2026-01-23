# Makefile Cookbook

This directory (`.rhiza/make.d/`) is the designated place for **repository-specific build logic**. Any `.mk` file added here is automatically absorbed by the main Makefile.

Use this cookbook to find copy-paste patterns for common development needs.

## 🥘 Recipes

### 1. Add a Simple Task
**Goal**: Run a script with `make train-model`.

Create `.rhiza/make.d/50-model.mk`:
```makefile
##@ Machine Learning
train: ## Train the model using local data
	@echo "Training model..."
	@uv run python scripts/train.py
```

### 2. Inject Code into Standard Workflows (Hooks)
**Goal**: Apply task after `make sync`.

Create `.rhiza/make.d/90-hooks.mk`:
```makefile
post-sync::
	@echo "Applying something..."
```
*Note: Use double-colons (`::`) for hooks to avoid conflicts.*

### 3. Define Global Variables
**Goal**: Set a default timeout for all test runs.

Create `.rhiza/make.d/01-config.mk`:
```makefile
# Override default timeout (defaults to 60s)
export TEST_TIMEOUT := 120
```

### 4. Create a Private Shortcut
**Goal**: Create a command that only exists on my machine (not committed).

Do not use `.rhiza/make.d/`. Instead, create a `local.mk` in the project root:
```makefile
deploy-dev:
	@./scripts/deploy-to-my-sandbox.sh
```

### 5. Install System Dependencies
**Goal**: Ensure `graphviz` is installed for Marimo notebooks using a hook.

Create `.rhiza/make.d/20-dependencies.mk`:
```makefile
pre-install::
	@if ! command -v dot >/dev/null 2>&1; then \
		echo "Graphviz not found. Installing..."; \
		if command -v brew >/dev/null 2>&1; then \
			brew install graphviz; \
		elif command -v apt-get >/dev/null 2>&1; then \
			sudo apt-get install -y graphviz; \
		else \
			echo "Please install graphviz manually."; \
			exit 1; \
		fi \
	fi
```

---

## ℹ️ Reference

### Execution Order
Files are loaded alphabetically. We use numeric prefixes to ensure dependencies resolve correctly:
- `00-19`: Configuration & Variables
- `20-79`: Custom Tasks & Rules
- `80-99`: Hooks & Lifecycle logic

### Available Hooks
- `pre-install` / `post-install`: Runs around `make install`.
- `pre-sync` / `post-sync`: Runs around repository synchronization.
- `pre-validate` / `post-validate`: Runs around validation checks.
- `pre-release` / `post-release`: Runs around release process.
- `pre-bump` / `post-bump`: Runs around version bumping.
