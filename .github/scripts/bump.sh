#!/bin/sh
# Version bump script
# - Bumps version in pyproject.toml using uv
# - Optionally commits the changes
# - Optionally pushes the changes
#
# This script is POSIX-sh compatible and follows the style of other scripts
# in this repository. It uses uv to manage version updates.

set -e

UV_BIN=${UV_BIN:-./bin/uv}

BLUE="\033[36m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

# Parse command-line arguments
VERSION=""
TYPE=""
DO_COMMIT=""
COMMIT_MSG=""

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
  printf "%b[ERROR] pyproject.toml not found in current directory%b\n" "$RED" "$RESET"
  exit 1
fi

# Check if uv is available
if [ ! -x "$UV_BIN" ]; then
  printf "%b[ERROR] uv not found at %s. Run 'make install-uv' first.%b\n" "$RED" "$UV_BIN" "$RESET"
  exit 1
fi

# Helper function to prompt user to continue
prompt_continue() {
  local message="$1"
  printf "\n%b[PROMPT] %s Continue? [y/N] %b" "$YELLOW" "$message" "$RESET"
  read -r answer
  case "$answer" in
    [Yy]*)
      return 0
      ;;
    *)
      printf "%b[INFO] Aborted by user%b\n" "$YELLOW" "$RESET"
      exit 0
      ;;
  esac
}

# Helper function to prompt user for yes/no
prompt_yes_no() {
  local message="$1"
  printf "\n%b[PROMPT] %s [y/N] %b" "$YELLOW" "$message" "$RESET"
  read -r answer
  case "$answer" in
    [Yy]*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Function: Bump version
do_bump() {
  # Get current version
  CURRENT_VERSION=$("$UV_BIN" version --short 2>/dev/null || echo "unknown")

  # Calculate next versions
  NEXT_PATCH=$("$UV_BIN" version --bump patch --dry-run --short 2>/dev/null)
  NEXT_MINOR=$("$UV_BIN" version --bump minor --dry-run --short 2>/dev/null)
  NEXT_MAJOR=$("$UV_BIN" version --bump major --dry-run --short 2>/dev/null)

  printf "%bSelect bump type (Current: %s):%b\n" "$BLUE" "$CURRENT_VERSION" "$RESET"
  printf "1) patch (%b%s -> %s%b)\n" "$YELLOW" "$CURRENT_VERSION" "$NEXT_PATCH" "$RESET"
  printf "2) minor (%b%s -> %s%b)\n" "$YELLOW" "$CURRENT_VERSION" "$NEXT_MINOR" "$RESET"
  printf "3) major (%b%s -> %s%b)\n" "$YELLOW" "$CURRENT_VERSION" "$NEXT_MAJOR" "$RESET"
  printf "4) Enter specific version\n"
  printf "Enter choice [1-4]: "
  read -r choice
  case "$choice" in
    1) TYPE="patch" ;;
    2) TYPE="minor" ;;
    3) TYPE="major" ;;
    4)
      printf "Enter version: "
      read -r VERSION
      # Strip 'v' prefix if present
      VERSION=$(echo "$VERSION" | sed 's/^v//')
      ;;
    *)
      printf "%b[ERROR] Invalid choice%b\n" "$RED" "$RESET"
      exit 1
      ;;
  esac

  # Get current branch
  CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  if [ -z "$CURRENT_BRANCH" ]; then
    printf "%b[ERROR] Could not determine current branch%b\n" "$RED" "$RESET"
    exit 1
  fi

  # Determine default branch
  DEFAULT_BRANCH=$(git remote show origin | grep 'HEAD branch' | cut -d' ' -f5)
  if [ -z "$DEFAULT_BRANCH" ]; then
    printf "%b[ERROR] Could not determine default branch from remote%b\n" "$RED" "$RESET"
    exit 1
  fi

  # Warn if not on default branch
  if [ "$CURRENT_BRANCH" != "$DEFAULT_BRANCH" ]; then
    printf "%b[WARN] You are on branch '%s' but the default branch is '%s'%b\n" "$YELLOW" "$CURRENT_BRANCH" "$DEFAULT_BRANCH" "$RESET"
    printf "%b[WARN] Releases are typically created from the default branch.%b\n" "$YELLOW" "$RESET"
    prompt_continue "Proceed with version bump on '$CURRENT_BRANCH'?"
  fi

  printf "%b[INFO] Current version: %s%b\n" "$BLUE" "$CURRENT_VERSION" "$RESET"

  # Determine the new version using uv version with --dry-run first
  if [ -n "$TYPE" ]; then
    printf "%b[INFO] Bumping version using: %s%b\n" "$BLUE" "$TYPE" "$RESET"
    NEW_VERSION=$("$UV_BIN" version --bump "$TYPE" --dry-run --short 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$NEW_VERSION" ]; then
      printf "%b[ERROR] Failed to calculate new version with type: %s%b\n" "$RED" "$TYPE" "$RESET"
      exit 1
    fi
  else
    # Validate the version format by having uv try it with --dry-run
    if ! "$UV_BIN" version "$VERSION" --dry-run >/dev/null 2>&1; then
      printf "%b[ERROR] Invalid version format: %s%b\n" "$RED" "$VERSION" "$RESET"
      printf "uv rejected this version. Please use a valid semantic version.\n"
      exit 1
    fi
    NEW_VERSION="$VERSION"
  fi

  printf "%b[INFO] New version will be: %s%b\n" "$BLUE" "$NEW_VERSION" "$RESET"

  TAG="v$NEW_VERSION"

  # Check if tag already exists
  if git rev-parse "$TAG" >/dev/null 2>&1; then
    printf "%b[ERROR] Tag '%s' already exists locally%b\n" "$RED" "$TAG" "$RESET"
    exit 1
  fi

  if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    printf "%b[ERROR] Tag '%s' already exists on remote%b\n" "$RED" "$TAG" "$RESET"
    exit 1
  fi

  # Check for uncommitted changes
  if [ -n "$(git status --porcelain)" ]; then
    printf "%b[ERROR] You have uncommitted changes:%b\n" "$RED" "$RESET"
    git status --short
    printf "\n%b[ERROR] Please commit or stash your changes before bumping version.%b\n" "$RED" "$RESET"
    exit 1
  fi

  # Update version in pyproject.toml using uv
  printf "%b[INFO] Updating pyproject.toml...%b\n" "$BLUE" "$RESET"
  if [ -n "$TYPE" ]; then
    if ! "$UV_BIN" version --bump "$TYPE" >/dev/null 2>&1; then
      printf "%b[ERROR] Failed to bump version using 'uv version --bump %s'%b\n" "$RED" "$TYPE" "$RESET"
      exit 1
    fi
  else
    if ! "$UV_BIN" version "$NEW_VERSION" >/dev/null 2>&1; then
      printf "%b[ERROR] Failed to set version using 'uv version %s'%b\n" "$RED" "$NEW_VERSION" "$RESET"
      exit 1
    fi
  fi

  # Verify the update
  UPDATED_VERSION=$("$UV_BIN" version --short 2>/dev/null)
  if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
    printf "%b[ERROR] Version update failed. Expected %s but got %s%b\n" "$RED" "$NEW_VERSION" "$UPDATED_VERSION" "$RESET"
    exit 1
  fi

  printf "%b[SUCCESS] 🚀 Version bumped: %s -> %s in pyproject.toml%b\n" "$GREEN" "$CURRENT_VERSION" "$NEW_VERSION" "$RESET"

  # Handle commit if requested
  if [ -z "$DO_COMMIT" ]; then
    if prompt_yes_no "Commit changes?"; then
      DO_COMMIT="true"
    fi
  fi

  if [ -n "$DO_COMMIT" ]; then
    # Set commit message
    if [ -z "$COMMIT_MSG" ]; then
      COMMIT_MSG="chore: bump version to $NEW_VERSION"
    fi

    printf "%b[INFO] Committing version change...%b\n" "$BLUE" "$RESET"
    git add pyproject.toml
    git add uv.lock 2>/dev/null || true  # In case uv modifies the lock file
    git commit -m "$COMMIT_MSG"

    printf "%b[SUCCESS] Version committed with message: '%s'%b\n" "$GREEN" "$COMMIT_MSG" "$RESET"

    # Prompt for push
    if prompt_yes_no "Push changes?"; then
      # Check if branch exists on remote
      if git ls-remote --exit-code --heads origin "$CURRENT_BRANCH" >/dev/null 2>&1; then
        git push origin "$CURRENT_BRANCH"
        printf "%b[SUCCESS] Pushed to origin/%s%b\n" "$GREEN" "$CURRENT_BRANCH" "$RESET"
      else
        printf "%b[WARN] Branch '%s' does not exist on remote%b\n" "$YELLOW" "$CURRENT_BRANCH" "$RESET"
        if prompt_yes_no "Publish branch?"; then
          git push -u origin "$CURRENT_BRANCH"
          printf "%b[SUCCESS] Published branch to origin/%s%b\n" "$GREEN" "$CURRENT_BRANCH" "$RESET"
        fi
      fi
    fi
  else
    printf "%b[INFO] Version bumped but not committed%b\n" "$BLUE" "$RESET"
    printf "%b[INFO] Next steps:%b\n" "$BLUE" "$RESET"
    printf "  1. Commit changes: git commit -a -m 'chore: bump version to %s'\n" "$NEW_VERSION"
    printf "  2. Push changes: git push origin %s\n" "$CURRENT_BRANCH"
  fi
}

do_bump
