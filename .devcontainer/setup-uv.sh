#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Set UV environment variables to avoid prompts and warnings
export UV_VENV_CLEAR=1
export UV_LINK_MODE=copy
# Make UV environment variables persistent for all sessions
echo "export UV_VENV_CLEAR=1" >> ~/.bashrc
echo "export UV_LINK_MODE=copy" >> ~/.bashrc

# Install uv (consider pinning via UV_INSTALL_VERSION or checksum)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure current shell sees uv (installer typically uses ~/.cargo/bin)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Create virtual environment
uv venv

# Sync dependencies if pyproject.toml exists
if [ -f pyproject.toml ]; then
    uv sync --all-extras
fi

# Install marimo
uv pip install marimo

# Initialize pre-commit hooks if configured
if [ -f .pre-commit-config.yaml ]; then
  # uvx runs tools without requiring them in the project deps
  uvx pre-commit install
fi
