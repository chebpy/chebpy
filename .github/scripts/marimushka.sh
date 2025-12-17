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
set -- "$MARIMO_FOLDER"/*.py
if [ "$1" = "$MARIMO_FOLDER/*.py" ]; then
  printf "%b[WARN] No Python files found in '%s'.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  # Create a minimal index.html indicating no notebooks
  printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' > "$MARIMUSHKA_OUTPUT/index.html"
  exit 0
fi


CURRENT_DIR=$(pwd)
OUTPUT_DIR="$CURRENT_DIR/$MARIMUSHKA_OUTPUT"

# Resolve UVX_BIN to absolute path if it's a relative path (contains / but doesn't start with /)
case "$UVX_BIN" in
  /*) ;;
  */*) UVX_BIN="$CURRENT_DIR/$UVX_BIN" ;;
  *) ;;
esac

# Resolve UV_BIN to absolute path
case "$UV_BIN" in
  /*) ;;
  */*) UV_BIN="$CURRENT_DIR/$UV_BIN" ;;
  *) ;;
esac

# Derive UV_INSTALL_DIR from UV_BIN
UV_INSTALL_DIR=$(dirname "$UV_BIN")

# Change to the notebook directory to ensure relative paths in notebooks work correctly
cd "$MARIMO_FOLDER"

# Run marimushka export
"$UVX_BIN" "marimushka>=0.1.9" export --notebooks "." --output "$OUTPUT_DIR" --bin-path "$UV_INSTALL_DIR"

# Ensure GitHub Pages does not process with Jekyll
: > "$OUTPUT_DIR/.nojekyll"