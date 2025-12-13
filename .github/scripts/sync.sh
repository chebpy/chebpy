#!/bin/sh
# Sync configuration files from template repository
# - Reads configuration from .github/template.yml
# - Downloads specified files from the template repository
# - Copies them to the current repository
#
# This script is POSIX-sh compatible and provides manual sync capability
# for repositories that use jebel-quant/rhiza as a template.

set -e

BLUE="\033[36m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

TEMPLATE_CONFIG=".github/template.yml"
TEMP_DIR="/tmp/rhiza-sync-$$"

show_usage() {
  printf "Usage: %s [OPTIONS]\n\n" "$0"
  printf "Sync configuration files from the template repository.\n\n"
  printf "Options:\n"
  printf "  -h, --help     Show this help message\n"
  printf "  --dry-run      Show what would be synced without making changes\n\n"
  printf "Configuration:\n"
  printf "  Reads from %s to determine:\n" "$TEMPLATE_CONFIG"
  printf "  - template-repository: Source repository (e.g., 'jebel-quant/rhiza')\n"
  printf "  - template-branch: Branch to sync from (e.g., 'main')\n"
  printf "  - include: Files/directories to sync\n"
  printf "  - exclude: Files/directories to skip (optional)\n\n"
  printf "Example %s:\n" "$TEMPLATE_CONFIG"
  printf "  template-repository: \"jebel-quant/rhiza\"\n"
  printf "  template-branch: \"main\"\n"
  printf "  include: |\n"
  printf "    .github\n"
  printf "    Makefile\n"
  printf "    ruff.toml\n"
}

DRY_RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *)
      printf "%b[ERROR] Unknown option: %s%b\n" "$RED" "$1" "$RESET"
      show_usage
      exit 1
      ;;
  esac
done

# Check if template.yml exists
if [ ! -f "$TEMPLATE_CONFIG" ]; then
  printf "%b[ERROR] Template configuration not found: %s%b\n" "$RED" "$TEMPLATE_CONFIG" "$RESET"
  printf "\nThis repository is not configured for template syncing.\n"
  printf "Create %s with the following content:\n\n" "$TEMPLATE_CONFIG"
  printf "  template-repository: \"jebel-quant/rhiza\"\n"
  printf "  template-branch: \"main\"\n"
  printf "  include: |\n"
  printf "    .github\n"
  printf "    Makefile\n"
  printf "    ruff.toml\n"
  exit 1
fi

# Parse template.yml to extract configuration
# This is a simple parser that works with basic YAML structure
printf "%b[INFO] Reading configuration from %s...%b\n" "$BLUE" "$TEMPLATE_CONFIG" "$RESET"

TEMPLATE_REPO=""
TEMPLATE_BRANCH=""
INCLUDE_LIST=""
EXCLUDE_LIST=""

# Simple YAML parser
in_multiline=""

while IFS= read -r line || [ -n "$line" ]; do
  # Skip comments and empty lines
  case "$line" in
    \#*|"") continue ;;
  esac

  # Check if line starts with spaces by looking at first character
  first_char=$(printf '%s' "$line" | cut -c1)

  if [ "$first_char" = " " ] || [ "$first_char" = "$(printf '\t')" ]; then
    # This is indented content (part of a multiline block)
    if [ -n "$in_multiline" ]; then
      # Remove leading spaces
      item=$(echo "$line" | sed 's/^[[:space:]]*//')
      if [ -n "$item" ]; then
        if [ "$in_multiline" = "include" ]; then
          INCLUDE_LIST="${INCLUDE_LIST}${item}
"
        elif [ "$in_multiline" = "exclude" ]; then
          EXCLUDE_LIST="${EXCLUDE_LIST}${item}
"
        fi
      fi
    fi
  else
    # This is a key line (not indented)
    case "$line" in
      template-repository:*)
        TEMPLATE_REPO=$(echo "$line" | sed 's/^template-repository:[[:space:]]*//' | sed 's/"//g' | sed "s/'//g")
        in_multiline=""
        ;;
      template-branch:*)
        TEMPLATE_BRANCH=$(echo "$line" | sed 's/^template-branch:[[:space:]]*//' | sed 's/"//g' | sed "s/'//g")
        in_multiline=""
        ;;
      include:*)
        in_multiline="include"
        ;;
      exclude:*)
        in_multiline="exclude"
        ;;
    esac
  fi
done < "$TEMPLATE_CONFIG"

# Validate required fields
if [ -z "$TEMPLATE_REPO" ]; then
  printf "%b[ERROR] template-repository not found in %s%b\n" "$RED" "$TEMPLATE_CONFIG" "$RESET"
  exit 1
fi

if [ -z "$TEMPLATE_BRANCH" ]; then
  printf "%b[WARN] template-branch not specified, using 'main'%b\n" "$YELLOW" "$RESET"
  TEMPLATE_BRANCH="main"
fi

if [ -z "$INCLUDE_LIST" ]; then
  printf "%b[ERROR] include list is empty in %s%b\n" "$RED" "$TEMPLATE_CONFIG" "$RESET"
  exit 1
fi

printf "%b[INFO] Template repository: %s%b\n" "$GREEN" "$TEMPLATE_REPO" "$RESET"
printf "%b[INFO] Template branch: %s%b\n" "$GREEN" "$TEMPLATE_BRANCH" "$RESET"
printf "%b[INFO] Files to sync:%b\n" "$GREEN" "$RESET"
echo "$INCLUDE_LIST" | while IFS= read -r item; do
  [ -n "$item" ] && printf "  - %s\n" "$item"
done || true

if [ -n "$EXCLUDE_LIST" ]; then
  printf "%b[INFO] Files to exclude:%b\n" "$YELLOW" "$RESET"
  echo "$EXCLUDE_LIST" | while IFS= read -r item; do
    [ -n "$item" ] && printf "  - %s\n" "$item"
  done || true
fi

if [ -n "$DRY_RUN" ]; then
  printf "\n%b[DRY RUN] No changes will be made%b\n" "$YELLOW" "$RESET"
  exit 0
fi

# Create temporary directory
mkdir -p "$TEMP_DIR"
trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM

# Backup this script to avoid being overwritten during sync
SELF_SCRIPT=".github/scripts/sync.sh"
if [ -f "$SELF_SCRIPT" ]; then
  cp "$SELF_SCRIPT" "$TEMP_DIR/sync.sh.bak"
fi

# Clone the template repository
printf "\n%b[INFO] Cloning template repository...%b\n" "$BLUE" "$RESET"
REPO_URL="https://github.com/${TEMPLATE_REPO}.git"

if ! git clone --depth 1 --branch "$TEMPLATE_BRANCH" "$REPO_URL" "$TEMP_DIR/template" 2>/dev/null; then
  printf "%b[ERROR] Failed to clone template repository from %s%b\n" "$RED" "$REPO_URL" "$RESET"
  exit 1
fi

# Function to check if a file path should be excluded
is_file_excluded() {
  file_path="$1"
  if [ -z "$EXCLUDE_LIST" ]; then
    return 1  # Not excluded (false)
  fi

  while IFS= read -r exclude_item || [ -n "$exclude_item" ]; do
    [ -z "$exclude_item" ] && continue
    if [ "$file_path" = "$exclude_item" ]; then
      return 0  # Is excluded (true)
    fi
  done <<EOF_EXCLUDE_CHECK
$EXCLUDE_LIST
EOF_EXCLUDE_CHECK

  return 1  # Not excluded (false)
}

# Copy files from template to current directory
printf "%b[INFO] Syncing files...%b\n" "$BLUE" "$RESET"

synced_count=0
skipped_count=0
# Track whether .github (containing this script) was synced and whether a direct self update was requested
synced_dotgithub="false"
deferred_self_update="false"

# Use here-document instead of pipeline to avoid subshell
while IFS= read -r item || [ -n "$item" ]; do
  [ -z "$item" ] && continue

  # Check if this item is in the exclude list
  if is_file_excluded "$item"; then
    printf "  %b[SKIP]%b %s (excluded)\n" "$YELLOW" "$RESET" "$item"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  # Defer updating this script to the very end to avoid mid-run overwrite
  if [ "$item" = ".github/scripts/sync.sh" ]; then
    deferred_self_update="true"
    printf "  %b[DEFER]%b %s (will update at end)\n" "$YELLOW" "$RESET" "$item"
    continue
  fi

  src_path="$TEMP_DIR/template/$item"
  dest_path="./$item"

  if [ -e "$src_path" ]; then
    # Create parent directory if needed
    dest_dir=$(dirname "$dest_path")
    mkdir -p "$dest_dir"

    # Copy the file or directory
    if [ -d "$src_path" ]; then
      # Ensure destination directory exists
      mkdir -p "$dest_path"
      # Copy contents of the source directory into the destination directory
      # to avoid nesting (e.g., .github/.github or tests/tests)
      cp -R "$src_path"/. "$dest_path"/
      # Mark if we synced the .github directory so we can safely update sync.sh at the end
      if [ "$item" = ".github" ]; then
        synced_dotgithub="true"
      fi

      # Remove excluded files from the copied directory
      if [ -n "$EXCLUDE_LIST" ]; then
        while IFS= read -r exclude_item || [ -n "$exclude_item" ]; do
          [ -z "$exclude_item" ] && continue
          # Check if the excluded item is a child of the current item
          # e.g., if item=".github" and exclude_item=".github/workflows/docker.yml"
          case "$exclude_item" in
            "$item"/*)
              # This is a nested file that should be excluded
              excluded_file_path="./$exclude_item"
              if [ -e "$excluded_file_path" ]; then
                rm -rf "$excluded_file_path"
                printf "  %b[EXCLUDE]%b %s (removed from synced directory)\n" "$YELLOW" "$RESET" "$exclude_item"
              fi
              ;;
          esac
        done <<EOF_EXCLUDE_NESTED
$EXCLUDE_LIST
EOF_EXCLUDE_NESTED
      fi

      # If we just synced the .github directory, restore this script immediately to avoid mid-run overwrite issues
      if [ "$item" = ".github" ] && [ -f "$TEMP_DIR/sync.sh.bak" ]; then
        cp "$TEMP_DIR/sync.sh.bak" "$SELF_SCRIPT"
      fi
      printf "  %b[SYNC]%b %s (directory contents)\n" "$GREEN" "$RESET" "$item"
    else
      cp "$src_path" "$dest_path"
      printf "  %b[SYNC]%b %s\n" "$GREEN" "$RESET" "$item"
    fi
    synced_count=$((synced_count + 1))
  else
    printf "  %b[WARN]%b %s (not found in template)\n" "$YELLOW" "$RESET" "$item"
    skipped_count=$((skipped_count + 1))
  fi
done <<EOF_INCLUDE
$INCLUDE_LIST
EOF_INCLUDE

# Finalize self-update of sync.sh if applicable
TEMPLATE_SELF_SH="$TEMP_DIR/template/.github/scripts/sync.sh"
if [ -f "$TEMPLATE_SELF_SH" ] && { [ "$deferred_self_update" = "true" ] || [ "$synced_dotgithub" = "true" ]; }; then
  if is_file_excluded ".github/scripts/sync.sh"; then
    printf "  %b[SKIP]%b .github/scripts/sync.sh (excluded from final update)\n" "$YELLOW" "$RESET"
  else
    cp "$TEMPLATE_SELF_SH" "$SELF_SCRIPT"
    chmod +x "$SELF_SCRIPT" 2>/dev/null || true
    printf "  %b[SYNC]%b .github/scripts/sync.sh (finalized)\n" "$GREEN" "$RESET"
  fi
fi

printf "\n%b[INFO] Sync complete!%b\n" "$GREEN" "$RESET"
printf "  Synced: %d files/directories\n" "$synced_count"
if [ "$skipped_count" -gt 0 ]; then
  printf "  Skipped: %d files/directories\n" "$skipped_count"
fi

printf "\n%b[INFO] Review the changes with: git status%b\n" "$BLUE" "$RESET"
printf "%b[INFO] Commit the changes with: git add . && git commit -m 'chore: sync template files'%b\n" "$BLUE" "$RESET"
