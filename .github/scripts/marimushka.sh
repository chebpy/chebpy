#!/bin/sh
# Export Marimo notebooks in ${MARIMO_FOLDER} to HTML under _marimushka
# This replicates the previous Makefile logic for maintainability and reuse.

set -e

MARIMO_FOLDER=${MARIMO_FOLDER:-book/marimo}
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
mkdir -p _marimushka

# Discover .py files (top-level only) using globbing; handle no-match case
set -- "$MARIMO_FOLDER"/*.py
if [ "$1" = "$MARIMO_FOLDER/*.py" ]; then
  printf "%b[WARN] No Python files found in '%s'.%b\n" "$YELLOW" "$MARIMO_FOLDER" "$RESET"
  # Create a minimal index.html indicating no notebooks
  printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><p>No notebooks found.</p></body></html>' > _marimushka/index.html
  exit 0
fi

# List files for transparency
py_files="$(printf "%s " "$@")"
printf "%b[INFO] Found Python files: %s%b\n" "$BLUE" "$py_files" "$RESET"

for py_file in "$@"; do
  printf " %b[INFO] Processing %s...%b\n" "$BLUE" "$py_file" "$RESET"
  rel_path=$(echo "$py_file" | sed "s|^$MARIMO_FOLDER/||")
  dir_path=$(dirname "$rel_path")
  base_name=$(basename "$rel_path" .py)
  mkdir -p "_marimushka/$dir_path"
  out_html="_marimushka/$dir_path/$base_name.html"
  # Ensure non-interactive overwrite: remove existing output file if present
  rm -f "$out_html"
  if grep -q "^# /// script" "$py_file"; then
    printf " %b[INFO] Script header detected, using --sandbox flag...%b\n" "$BLUE" "$RESET"
    "$UVX_BIN" marimo export html --sandbox --include-code --output "$out_html" "$py_file"
  else
    printf " %b[INFO] No script header detected, using standard export...%b\n" "$BLUE" "$RESET"
    "$UV_BIN" run marimo export html --include-code --output "$out_html" "$py_file"
  fi
done

# Build a simple index.html linking to all generated HTMLs
printf '<html><head><title>Marimo Notebooks</title></head><body><h1>Marimo Notebooks</h1><ul>' > _marimushka/index.html
find _marimushka -name "*.html" -not -path "*index.html" | sort | while read -r html_file; do
  rel_path=$(echo "$html_file" | sed 's|^_marimushka/||')
  name=$(basename "$rel_path" .html)
  printf '<li><a href="%s">%s</a></li>' "$rel_path" "$name" >> _marimushka/index.html
done
printf '</ul></body></html>' >> _marimushka/index.html

# Ensure GitHub Pages does not process with Jekyll
: > _marimushka/.nojekyll
