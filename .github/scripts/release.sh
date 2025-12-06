#!/bin/sh
# Version bump and tag creation script for releases
# - Two-command release process: bump (with optional commit), release (tag + push with prompts)
# - Validates the version format using uv
# - Updates pyproject.toml with the new version using uv
# - Creates a git tag and pushes it to trigger the release workflow
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
COMMAND=""
DO_COMMIT=""
COMMIT_MSG=""

show_usage() {
  printf "Usage: %s [OPTIONS] [COMMAND]\n\n" "$0"
  printf "Commands:\n"
  printf "  bump           Bump version (updates pyproject.toml, optionally commits)\n"
  printf "  release        Create tag and push to remote (with prompts)\n\n"
  printf "Options (for bump):\n"
  printf "  --type TYPE    Bump version semantically (major, minor, patch, alpha, beta, rc, etc.)\n"
  printf "  --version VER  Set explicit version number\n"
  printf "  --commit       Commit the version changes\n"
  printf "  --message MSG  Custom commit message (default: 'chore: bump version to X.Y.Z')\n"
  printf "  -h, --help     Show this help message\n\n"
  printf "Examples:\n"
  printf "  %s bump --type patch                    (bump version, don't commit)\n" "$0"
  printf "  %s bump --type minor --commit           (bump and commit with default message)\n" "$0"
  printf "  %s bump --version 1.2.3 --commit --message 'Release v1.2.3' (bump and commit with custom message)\n" "$0"
  printf "  %s release                              (create tag and push with prompts)\n" "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --type)
      if [ -z "$2" ]; then
        printf "%b[ERROR] --type requires a value%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      TYPE="$2"
      shift 2
      ;;
    --version)
      if [ -z "$2" ]; then
        printf "%b[ERROR] --version requires a value%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      VERSION="$2"
      shift 2
      ;;
    --commit)
      DO_COMMIT="true"
      shift
      ;;
    --message)
      if [ -z "$2" ]; then
        printf "%b[ERROR] --message requires a value%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      COMMIT_MSG="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    bump|release)
      if [ -n "$COMMAND" ]; then
        printf "%b[ERROR] Multiple commands provided%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      COMMAND="$1"
      shift
      ;;
    -*)
      printf "%b[ERROR] Unknown option: %s%b\n" "$RED" "$1" "$RESET"
      show_usage
      exit 1
      ;;
    *)
      printf "%b[ERROR] Unknown argument: %s%b\n" "$RED" "$1" "$RESET"
      show_usage
      exit 1
      ;;
  esac
done

# Strip 'v' prefix if present in explicit version
if [ -n "$VERSION" ]; then
  VERSION=$(echo "$VERSION" | sed 's/^v//')
fi

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

# Function: Bump version
do_bump() {
  # Validate that either version or bump was provided
  if [ -z "$VERSION" ] && [ -z "$TYPE" ]; then
    printf "%b[ERROR] No version or bump type specified for bump command%b\n" "$RED" "$RESET"
    printf "Use --type TYPE or --version VER\n"
    show_usage
    exit 1
  fi

  if [ -n "$VERSION" ] && [ -n "$TYPE" ]; then
    printf "%b[ERROR] Cannot specify both --version and --type%b\n" "$RED" "$RESET"
    show_usage
    exit 1
  fi

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

  # Get current version
  CURRENT_VERSION=$("$UV_BIN" version --short 2>/dev/null || echo "unknown")
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

  printf "%b[SUCCESS] Version bumped to %s in pyproject.toml%b\n" "$GREEN" "$NEW_VERSION" "$RESET"

  # Handle commit if requested
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
    printf "%b[INFO] Next step: Run 'make release' to create tag and push%b\n" "$BLUE" "$RESET"
  else
    printf "%b[INFO] Version bumped but not committed%b\n" "$BLUE" "$RESET"
    printf "%b[INFO] Next steps:%b\n" "$BLUE" "$RESET"
    printf "  1. Commit changes: git commit -a -m 'chore: bump version to %s'\n" "$NEW_VERSION"
    printf "  2. Run 'make release' to create tag and push\n"
  fi
}

# Function: Release - create tag and push (with prompts)
do_release() {
  # Get the current version from pyproject.toml
  CURRENT_VERSION=$("$UV_BIN" version --short 2>/dev/null)
  if [ -z "$CURRENT_VERSION" ]; then
    printf "%b[ERROR] Could not determine version from pyproject.toml%b\n" "$RED" "$RESET"
    exit 1
  fi

  TAG="v$CURRENT_VERSION"

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
    prompt_continue "Proceed with release from '$CURRENT_BRANCH'?"
  fi

  printf "%b[INFO] Current version: %s%b\n" "$BLUE" "$CURRENT_VERSION" "$RESET"
  printf "%b[INFO] Tag to create: %s%b\n" "$BLUE" "$TAG" "$RESET"

  # Check if there are uncommitted changes
  if [ -n "$(git status --porcelain)" ]; then
    printf "%b[ERROR] You have uncommitted changes:%b\n" "$RED" "$RESET"
    git status --short
    printf "\n%b[ERROR] Please commit or stash your changes before releasing.%b\n" "$RED" "$RESET"
    exit 1
  fi

  # Check if branch is up-to-date with remote
  printf "%b[INFO] Checking remote status...%b\n" "$BLUE" "$RESET"
  git fetch origin >/dev/null 2>&1
  UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)
  if [ -z "$UPSTREAM" ]; then
    printf "%b[ERROR] No upstream branch configured for %s%b\n" "$RED" "$CURRENT_BRANCH" "$RESET"
    exit 1
  fi
  
  LOCAL=$(git rev-parse @)
  REMOTE=$(git rev-parse "$UPSTREAM")
  BASE=$(git merge-base @ "$UPSTREAM")
  
  if [ "$LOCAL" != "$REMOTE" ]; then
    if [ "$LOCAL" = "$BASE" ]; then
        printf "%b[ERROR] Your branch is behind '%s'. Please pull changes.%b\n" "$RED" "$UPSTREAM" "$RESET"
        exit 1
    elif [ "$REMOTE" = "$BASE" ]; then
        printf "%b[WARN] Your branch is ahead of '%s'.%b\n" "$YELLOW" "$UPSTREAM" "$RESET"
        printf "Unpushed commits:\n"
        git log --oneline --graph --decorate "$UPSTREAM..HEAD"
        prompt_continue "Push changes to remote before releasing?"
        git push origin "$CURRENT_BRANCH"
    else
        printf "%b[ERROR] Your branch has diverged from '%s'. Please reconcile.%b\n" "$RED" "$UPSTREAM" "$RESET"
        exit 1
    fi
  fi

  # Check if tag already exists locally
  if git rev-parse "$TAG" >/dev/null 2>&1; then
    printf "%b[WARN] Tag '%s' already exists locally%b\n" "$YELLOW" "$TAG" "$RESET"
    prompt_continue "Tag exists. Skip tag creation and proceed to push?"
    SKIP_TAG_CREATE="true"
  fi

  # Check if tag already exists on remote
  if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    printf "%b[ERROR] Tag '%s' already exists on remote%b\n" "$RED" "$TAG" "$RESET"
    printf "The release for version %s has already been published.\n" "$CURRENT_VERSION"
    exit 1
  fi

  # Step 1: Create the tag (if it doesn't exist)
  if [ -z "$SKIP_TAG_CREATE" ]; then
    printf "\n%b=== Step 1: Create Tag ===%b\n" "$BLUE" "$RESET"
    printf "Creating signed tag '%s' for version %s\n" "$TAG" "$CURRENT_VERSION"
    prompt_continue ""
    
    git tag -s "$TAG" -m "Release $TAG"
    printf "%b[SUCCESS] Tag '%s' created locally%b\n" "$GREEN" "$TAG" "$RESET"
  fi

  # Step 2: Push the tag to remote
  printf "\n%b=== Step 2: Push Tag to Remote ===%b\n" "$BLUE" "$RESET"
  printf "Pushing tag '%s' to origin will trigger the release workflow.\n" "$TAG"
  
  # Show what commits are in this tag
  LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
  if [ -n "$LAST_TAG" ] && [ "$LAST_TAG" != "$TAG" ]; then
    COMMIT_COUNT=$(git rev-list "$LAST_TAG..$TAG" --count 2>/dev/null || echo "0")
    printf "Commits since %s: %s\n" "$LAST_TAG" "$COMMIT_COUNT"
  fi
  
  prompt_continue ""
  
  git push origin "refs/tags/$TAG"
  
  REPO_URL=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
  printf "\n%b[SUCCESS] Release tag %s pushed to remote!%b\n" "$GREEN" "$TAG" "$RESET"
  printf "%b[INFO] The release workflow will now be triggered automatically.%b\n" "$BLUE" "$RESET"
  printf "%b[INFO] Monitor progress at: https://github.com/%s/actions%b\n" "$BLUE" "$REPO_URL" "$RESET"
}

# Main execution logic
case "$COMMAND" in
  bump)
    do_bump
    ;;
  release)
    do_release
    ;;
  "")
    printf "%b[ERROR] No command specified%b\n" "$RED" "$RESET"
    show_usage
    exit 1
    ;;
  *)
    printf "%b[ERROR] Unknown command: %s%b\n" "$RED" "$COMMAND" "$RESET"
    show_usage
    exit 1
    ;;
esac
