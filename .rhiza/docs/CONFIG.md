# Rhiza Configuration

This directory contains platform-agnostic scripts and utilities for the repository that can be used by GitHub Actions, GitLab CI, or other CI/CD systems.

## Important Documentation

- **[TOKEN_SETUP.md](TOKEN_SETUP.md)** - Instructions for setting up the `PAT_TOKEN` secret required for the SYNC workflow

## Structure

- **scripts/** - Shell scripts for common tasks (book generation, release, etc.)
- **scripts/customisations/** - Repository-specific customization hooks
- **utils/** - Python utilities for version management

GitHub-specific composite actions are located in `.github/rhiza/actions/`.

## Workflows

The repository uses several automated workflows (located in `.github/workflows/`):

- **SYNC** (`rhiza_sync.yml`) - Synchronizes with the template repository
  - **Requires:** `PAT_TOKEN` secret with `workflow` scope when modifying workflow files
  - See [TOKEN_SETUP.md](TOKEN_SETUP.md) for configuration
- **CI** (`rhiza_ci.yml`) - Continuous integration tests
- **Pre-commit** (`rhiza_pre-commit.yml`) - Code quality checks
- **Book** (`rhiza_book.yml`) - Documentation deployment
- **Release** (`rhiza_release.yml`) - Package publishing
- **Deptry** (`rhiza_deptry.yml`) - Dependency checks
- **Marimo** (`rhiza_marimo.yml`) - Interactive notebooks

## Template Synchronization

This repository is synchronized with the template repository defined in `template.yml`.

The synchronization includes:
- GitHub workflows and actions
- Development tools configuration (`.editorconfig`, `ruff.toml`, etc.)
- Testing infrastructure
- Documentation templates

See `template.yml` for the complete list of synchronized files and exclusions.
