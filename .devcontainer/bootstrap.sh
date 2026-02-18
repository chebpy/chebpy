#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Read Python version from .python-version (single source of truth)
if [ -f ".python-version" ]; then
    export PYTHON_VERSION=$(cat .python-version | tr -d '[:space:]')
    echo "Using Python version from .python-version: $PYTHON_VERSION"
fi

# Use INSTALL_DIR from environment or default to local bin
# In devcontainer, this is set to /home/vscode/.local/bin to avoid conflict with host
export INSTALL_DIR="${INSTALL_DIR:-./bin}"
export UV_BIN="${INSTALL_DIR}/uv"
export UVX_BIN="${INSTALL_DIR}/uvx"

# Only remove existing binaries if we are installing to the default ./bin location
# and we want to force re-installation (e.g. if OS changed)
if [ "$INSTALL_DIR" = "./bin" ]; then
    rm -f "$UV_BIN" "$UVX_BIN"
fi

# Set UV environment variables to avoid prompts and warnings
export UV_VENV_CLEAR=1
export UV_LINK_MODE=copy

# Make UV environment variables persistent for all sessions
echo "export UV_LINK_MODE=copy" >> ~/.bashrc
echo "export UV_VENV_CLEAR=1" >> ~/.bashrc
echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.bashrc

# Add to current PATH so subsequent commands can find uv
export PATH="$INSTALL_DIR:$PATH"

make install

# Install Marimo tool for notebook editing
"$UV_BIN" tool install marimo 
