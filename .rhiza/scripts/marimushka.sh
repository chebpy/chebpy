#!/bin/sh
# Export Marimo notebooks in ${MARIMO_FOLDER} to HTML under _marimushka
# This replicates the previous Makefile logic for maintainability and reuse.

set -e

MARIMO_FOLDER=${MARIMO_FOLDER:-book/marimo}
MARIMUSHKA_OUTPUT=${MARIMUSHKA_OUTPUT:-_marimushka}
UV_BIN=${UV_BIN:-./bin/uv}
UVX_BIN=${UVX_BIN:-./bin/uvx}

BLUE="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

printf "%b[INFO] Exporting notebooks from %s...%b\n" "$BLUE" "$MARIMO_FOLDER" "$RESET"

if [ ! -d "$MARIMO_FOLDER" ]; then
  printf "%b[WARN] Directory '%s' does not exist. Skipping marimushka.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  exit 0
fi

# Ensure output directory exists
mkdir -p "$MARIMUSHKA_OUTPUT"

# Discover .py files (top-level only) using globbing; handle no-match case
# Using shell globbing to find all .py files in the notebook folder
# The set command expands the glob pattern; if no files match, the pattern itself is returned
set -- "$MARIMO_FOLDER"/*.py
if [ "$1" = "$MARIMO_FOLDER/*.py" ]; then
  # No Python files found - the glob pattern didn't match any files
  printf "%b[WARN] No Python files found in '%s'.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  # Create a minimal index.html indicating no notebooks
  printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' > "$MARIMUSHKA_OUTPUT/index.html"
  exit 0
fi


CURRENT_DIR=$(pwd)
OUTPUT_DIR="$CURRENT_DIR/$MARIMUSHKA_OUTPUT"

# Resolve UVX_BIN to absolute path if it's a relative path (contains / but doesn't start with /)
# This is necessary because we'll change directory later and need absolute paths
# Case 1: Already absolute (starts with /) - no change needed
# Case 2: Relative path with / (e.g., ./bin/uvx) - convert to absolute
# Case 3: Command name only (e.g., uvx) - leave as-is to search in PATH
case "$UVX_BIN" in
  /*) ;;
  */*) UVX_BIN="$CURRENT_DIR/$UVX_BIN" ;;
  *) ;;
esac

# Resolve UV_BIN to absolute path using the same logic
case "$UV_BIN" in
  /*) ;;
  */*) UV_BIN="$CURRENT_DIR/$UV_BIN" ;;
  *) ;;
esac

# Derive UV_INSTALL_DIR from UV_BIN
# This directory is passed to marimushka so it can find uv for processing notebooks
UV_INSTALL_DIR=$(dirname "$UV_BIN")

# Change to the notebook directory to ensure relative paths in notebooks work correctly
# Marimo notebooks may contain relative imports or file references
cd "$MARIMO_FOLDER"

# Run marimushka export
# - --notebooks: directory containing .py notebooks
# - --output: where to write HTML files
# - --bin-path: where marimushka can find the uv binary for processing
"$UVX_BIN" "marimushka>=0.1.9" export --notebooks "." --output "$OUTPUT_DIR" --bin-path "$UV_INSTALL_DIR"

# Ensure GitHub Pages does not process with Jekyll
# The : command is a no-op that creates an empty file
# .nojekyll tells GitHub Pages to serve files as-is without Jekyll processing
: > "$OUTPUT_DIR/.nojekyll"
