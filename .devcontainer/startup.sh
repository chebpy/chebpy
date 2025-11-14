#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/setup-uv.sh"

echo "🚀 Generic Python .devcontainer environment ready!"
echo "🔧 Pre-commit hooks installed for code quality"
echo "Marimo installed"
