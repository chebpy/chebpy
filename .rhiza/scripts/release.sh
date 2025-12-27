#!/bin/sh
# Release script
# - Creates a git tag based on the current version in pyproject.toml
# - Pushes the tag to remote to trigger the release workflow
# - Performs checks (branch, upstream status, clean working tree)
#
# This script is POSIX-sh compatible and follows the style of other scripts
# in this repository. It uses uv to read the current version.

set -e

UV_BIN=${UV_BIN:-./bin/uv}

BLUE="\033[36m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

# Parse command-line arguments
show_usage() {
  printf "Usage: %s [OPTIONS]\n\n" "$0"
  printf "Description:\n"
  printf "  Create tag and push to remote (with prompts)\n\n"
  printf "Options:\n"
  printf "  -h, --help     Show this help message\n\n"
  printf "Examples:\n"
  printf "  %s                                      (create tag and push with prompts)\n" "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
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
  # This prevents releasing from an out-of-sync branch which could miss commits or conflict
  printf "%b[INFO] Checking remote status...%b\n" "$BLUE" "$RESET"
  git fetch origin >/dev/null 2>&1
  # Get the upstream tracking branch (e.g., origin/main)
  UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null)
  if [ -z "$UPSTREAM" ]; then
    printf "%b[ERROR] No upstream branch configured for %s%b\n" "$RED" "$CURRENT_BRANCH" "$RESET"
    exit 1
  fi
  
  # Compare local, remote, and merge-base commits to determine sync status
  # LOCAL: current commit on local branch
  # REMOTE: current commit on remote tracking branch
  # BASE: most recent common ancestor between local and remote
  LOCAL=$(git rev-parse @)
  REMOTE=$(git rev-parse "$UPSTREAM")
  BASE=$(git merge-base @ "$UPSTREAM")
  
  # Use git revision comparison to detect branch status
  if [ "$LOCAL" != "$REMOTE" ]; then
    if [ "$LOCAL" = "$BASE" ]; then
        # Local is behind remote (need to pull)
        printf "%b[ERROR] Your branch is behind '%s'. Please pull changes.%b\n" "$RED" "$UPSTREAM" "$RESET"
        exit 1
    elif [ "$REMOTE" = "$BASE" ]; then
        # Local is ahead of remote (need to push)
        printf "%b[WARN] Your branch is ahead of '%s'.%b\n" "$YELLOW" "$UPSTREAM" "$RESET"
        printf "Unpushed commits:\n"
        git log --oneline --graph --decorate "$UPSTREAM..HEAD"
        prompt_continue "Push changes to remote before releasing?"
        git push origin "$CURRENT_BRANCH"
    else
        # Branches have diverged (need to merge or rebase)
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
    printf "Creating tag '%s' for version %s\n" "$TAG" "$CURRENT_VERSION"
    prompt_continue ""
    
    # Check if GPG signing is configured for git commits/tags
    # If user.signingkey is set or commit.gpgsign is true, create a signed tag
    # Signed tags provide cryptographic verification of release authenticity
    if git config --get user.signingkey >/dev/null 2>&1 || [ "$(git config --get commit.gpgsign)" = "true" ]; then
      printf "%b[INFO] GPG signing is enabled. Creating signed tag.%b\n" "$BLUE" "$RESET"
      git tag -s "$TAG" -m "Release $TAG"
    else
      printf "%b[INFO] GPG signing is not enabled. Creating unsigned tag.%b\n" "$BLUE" "$RESET"
      git tag -a "$TAG" -m "Release $TAG"
    fi
    printf "%b[SUCCESS] Tag '%s' created locally%b\n" "$GREEN" "$TAG" "$RESET"
  fi

  # Step 2: Push the tag to remote
  printf "\n%b=== Step 2: Push Tag to Remote ===%b\n" "$BLUE" "$RESET"
  printf "Pushing tag '%s' to origin will trigger the release workflow.\n" "$TAG"
  
  # Show what commits are in this tag compared to the last tag
  # This helps users understand what changes are included in the release
  LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
  if [ -n "$LAST_TAG" ] && [ "$LAST_TAG" != "$TAG" ]; then
    # Count commits between last tag and current tag
    COMMIT_COUNT=$(git rev-list "$LAST_TAG..$TAG" --count 2>/dev/null || echo "0")
    printf "Commits since %s: %s\n" "$LAST_TAG" "$COMMIT_COUNT"
  fi
  
  prompt_continue ""
  
  # Push only the specific tag (not all tags) to trigger the release workflow
  git push origin "refs/tags/$TAG"
  
  # Extract repository name from remote URL for constructing GitHub Actions link
  # Converts git@github.com:user/repo.git or https://github.com/user/repo.git to user/repo
  REPO_URL=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
  printf "\n%b[SUCCESS] Release tag %s pushed to remote!%b\n" "$GREEN" "$TAG" "$RESET"
  printf "%b[INFO] The release workflow will now be triggered automatically.%b\n" "$BLUE" "$RESET"
  printf "%b[INFO] Monitor progress at: https://github.com/%s/actions%b\n" "$BLUE" "$REPO_URL" "$RESET"
}

# Main execution logic
do_release
