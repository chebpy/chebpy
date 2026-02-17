#!/bin/bash
set -euo pipefail

# Session End Hook
# Runs quality gates after the agent finishes work.

echo "[copilot-hook] Running post-work quality gates..."

echo "[copilot-hook] Formatting code..."
make fmt || {
    echo "[copilot-hook] WARNING: Formatting check failed."
    exit 1
}

echo "[copilot-hook] Running tests..."
make test || {
    echo "[copilot-hook] WARNING: Tests failed."
    exit 1
}

echo "[copilot-hook] All quality gates passed."
