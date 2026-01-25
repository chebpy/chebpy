# Rhiza Copilot Instructions

You are working in a project that utilises the `rhiza` framework. Rhiza is a collection of reusable
configuration templates and tooling designed to standardise and streamline modern Python development.

As a Rhiza-based project, this workspace adheres to specific conventions for structure, dependency management, and automation.

## Development Environment

The project uses `make` and `uv` for development tasks.

- **Install Dependencies**: `make install` (installs `uv`, creates `.venv`, installs dependencies)
- **Run Tests**: `make test` (runs `pytest` with coverage)
- **Format Code**: `make fmt` (runs `ruff format` and `ruff check --fix`)
- **Check Dependencies**: `make deptry` (runs `deptry` to check for missing/unused dependencies)
- **Marimo Notebooks**: `make marimo` (starts the Marimo server)
- **Build Documentation**: `make book` (builds the documentation book)

## Project Structure

- `src/`: Source code
- `tests/`: Tests (pytest)
- `assets/`: Static assets
- `book/`: Documentation source
- `docker/`: Docker configuration
- `presentation/`: Presentation slides
- `.rhiza/`: Rhiza-specific scripts and configurations

## Coding Standards

- **Style**: Follow PEP 8. Use `make fmt` to enforce style.
- **Testing**: Write tests in `tests/` using `pytest`. Ensure high coverage.
- **Documentation**: Document code using docstrings.
- **Dependencies**: Manage dependencies in `pyproject.toml`. Use `uv add` to add dependencies.

## Workflow

1.  **Setup**: Run `make install` to set up the environment.
2.  **Develop**: Write code in `src/` and tests in `tests/`.
3.  **Test**: Run `make test` to verify changes.
4.  **Format**: Run `make fmt` before committing.
5.  **Verify**: Run `make deptry` to check dependencies.

## Key Files

- `Makefile`: Main entry point for tasks.
- `pyproject.toml`: Project configuration and dependencies.
- `.devcontainer/bootstrap.sh`: Bootstrap script for dev containers.
