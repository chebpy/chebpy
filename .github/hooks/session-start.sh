#!/bin/bash
set -euo pipefail

# Session Start Hook
# Validates that the environment is correctly set up before the agent begins work.
# The virtual environment should already be activated via copilot-setup-steps.yml.

echo "[copilot-hook] Validating environment..."

# Verify uv is available
if ! command -v uv >/dev/null 2>&1 && [ ! -x "./bin/uv" ]; then
    echo "[copilot-hook] ERROR: uv not found. Run 'make install' to set up the environment."
    exit 1
fi

# Verify virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[copilot-hook] ERROR: .venv not found. Run 'make install' to set up the environment."
    exit 1
fi

# Verify virtual environment is on PATH (activated via copilot-setup-steps.yml)
if ! command -v python >/dev/null 2>&1 || [[ "$(command -v python)" != *".venv"* ]]; then
    echo "[copilot-hook] WARNING: .venv/bin is not on PATH. The agent may not use the correct Python."
fi

echo "[copilot-hook] Environment validated successfully."
