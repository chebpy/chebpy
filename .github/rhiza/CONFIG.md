# GitHub Configuration

This directory contains the GitHub-specific configuration for the repository.

## Important Documentation

- **[TOKEN_SETUP.md](TOKEN_SETUP.md)** - Instructions for setting up the `PAT_TOKEN` secret required for the SYNC workflow

## Workflows

The repository uses several automated workflows:

- **SYNC** (`workflows/sync.yml`) - Synchronizes with the template repository
  - **Requires:** `PAT_TOKEN` secret with `workflow` scope when modifying workflow files
  - See [TOKEN_SETUP.md](TOKEN_SETUP.md) for configuration
- **CI** (`workflows/ci.yml`) - Continuous integration tests
- **Pre-commit** (`workflows/pre-commit.yml`) - Code quality checks
- **Book** (`workflows/book.yml`) - Documentation deployment
- **Release** (`workflows/release.yml`) - Package publishing
- **Deptry** (`workflows/deptry.yml`) - Dependency checks
- **Marimo** (`workflows/marimo.yml`) - Interactive notebooks

## Template Synchronization

This repository is synchronized with the template repository defined in `template.yml`.

The synchronization includes:
- GitHub workflows and actions
- Development tools configuration (`.editorconfig`, `ruff.toml`, etc.)
- Testing infrastructure
- Documentation templates

See `template.yml` for the complete list of synchronized files and exclusions.
