#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Use UV_INSTALL_DIR from environment or default to local bin
# In devcontainer, this is set to /home/vscode/.local/bin to avoid conflict with host
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-./bin}"
export UV_BIN="${UV_INSTALL_DIR}/uv"
export UVX_BIN="${UV_INSTALL_DIR}/uvx"

# Only remove existing binaries if we are installing to the default ./bin location
# and we want to force re-installation (e.g. if OS changed)
if [ "$UV_INSTALL_DIR" = "./bin" ]; then
    rm -f "$UV_BIN" "$UVX_BIN"
fi

# Set UV environment variables to avoid prompts and warnings
export UV_VENV_CLEAR=1
export UV_LINK_MODE=copy

# Make UV environment variables persistent for all sessions
echo "export UV_LINK_MODE=copy" >> ~/.bashrc
echo "export UV_VENV_CLEAR=1" >> ~/.bashrc
echo "export PATH=\"$UV_INSTALL_DIR:\$PATH\"" >> ~/.bashrc

# Add to current PATH so subsequent commands can find uv
export PATH="$UV_INSTALL_DIR:$PATH"

make install

# Install Marimo tool for notebook editing
"$UV_BIN" tool install marimo 

# Initialize pre-commit hooks if configured
if [ -f .pre-commit-config.yaml ]; then
  # uvx runs tools without requiring them in the project deps
  "$UVX_BIN" pre-commit install
fi
