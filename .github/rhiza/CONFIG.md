# GitHub Configuration

This directory contains the GitHub-specific configuration for the repository.

## Important Documentation

- **[TOKEN_SETUP.md](TOKEN_SETUP.md)** - Instructions for setting up the `PAT_TOKEN` secret required for the SYNC workflow

## Workflows

The repository uses several automated workflows:

- **SYNC** (`workflows/rhiza_sync.yml`) - Synchronizes with the template repository
  - **Requires:** `PAT_TOKEN` secret with `workflow` scope when modifying workflow files
  - See [TOKEN_SETUP.md](TOKEN_SETUP.md) for configuration
- **CI** (`workflows/rhiza_ci.yml`) - Continuous integration tests
- **Pre-commit** (`workflows/rhiza_pre-commit.yml`) - Code quality checks
- **Book** (`workflows/rhiza_book.yml`) - Documentation deployment
- **Release** (`workflows/rhiza_release.yml`) - Package publishing
- **Deptry** (`workflows/rhiza_deptry.yml`) - Dependency checks
- **Marimo** (`workflows/rhiza_marimo.yml`) - Interactive notebooks

## Template Synchronization

This repository is synchronized with the template repository defined in `template.yml`.

The synchronization includes:
- GitHub workflows and actions
- Development tools configuration (`.editorconfig`, `ruff.toml`, etc.)
- Testing infrastructure
- Documentation templates

See `template.yml` for the complete list of synchronized files and exclusions.
