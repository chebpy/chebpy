# DevContainer Configuration

This directory contains the configuration for [GitHub Codespaces](https://github.com/features/codespaces) and [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Contents

- `devcontainer.json`: The primary configuration file defining the development environment.
- `bootstrap.sh`: A script that runs after the container is created to initialize the environment (installing dependencies, setting up tools).

## Features

- **Python Environment**: Pre-configured with Python 3.12.
- **Tools**: Includes `uv` for fast package management and `make` for project tasks.
- **Extensions**: Recommended VS Code extensions for Python development, including Ruff and Marimo.
