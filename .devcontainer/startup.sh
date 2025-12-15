#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/bootstrap.sh"

echo "🚀 Generic Python .devcontainer environment ready!"
echo "🔧 Pre-commit hooks installed for code quality, run 'make fmt' for formatting and linting!"
echo "📓 Marimo installed for notebook editing, run 'make marimo' to start the server!"
