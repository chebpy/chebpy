#!/usr/bin/env bash
# Install dev dependencies from .rhiza/requirements/*.txt files
# This script is used by GitHub Actions workflows to install development dependencies

set -euo pipefail

REQUIREMENTS_DIR=".rhiza/requirements"

if [ ! -d "$REQUIREMENTS_DIR" ]; then
    echo "Warning: Requirements directory $REQUIREMENTS_DIR not found, skipping dev dependencies install" >&2
    exit 0
fi

echo "Installing dev dependencies from $REQUIREMENTS_DIR"

shopt -s nullglob
for req_file in "$REQUIREMENTS_DIR"/*.txt; do
    if [ -f "$req_file" ]; then
        echo "Installing requirements from $req_file"
        uv pip install -r "$req_file"
    fi
done
shopt -u nullglob

echo "Dev dependencies installed successfully"
