#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Install Task (Taskfile) to user directory
mkdir -p "$HOME/.local/bin"
if ! command -v task >/dev/null 2>&1; then
  # Consider pinning version via TASK_VERSION env or checksum verification
  sh -c "$(curl -fsSL https://taskfile.dev/install.sh)" -- -d -b "$HOME/.local/bin"
fi
export PATH="$HOME/.local/bin:$PATH"
grep -qx 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"