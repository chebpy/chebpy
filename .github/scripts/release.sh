#!/bin/sh
# Version bump and tag creation script for releases
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
BUMP=""
DRY_RUN=""
BRANCH=""

show_usage() {
  printf "Usage: %s [OPTIONS] VERSION|--bump BUMP_TYPE\n\n" "$0"
  printf "Options:\n"
  printf "  --bump TYPE    Bump version semantically (major, minor, patch, alpha, beta, rc, etc.)\n"
  printf "  --branch REF   Branch or ref to tag (default: current default branch)\n"
  printf "  --dry-run      Show what would be done without making changes\n"
  printf "  -h, --help     Show this help message\n\n"
  printf "Examples:\n"
  printf "  %s 1.2.3\n" "$0"
  printf "  %s --dry-run 1.2.3\n" "$0"
  printf "  %s v1.2.3                (the 'v' prefix will be stripped)\n" "$0"
  printf "  %s --bump patch          (bump patch version)\n" "$0"
  printf "  %s --bump minor          (bump minor version)\n" "$0"
  printf "  %s --bump major          (bump major version)\n" "$0"
  printf "  %s --branch main 1.2.3   (tag specific branch)\n" "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --bump)
      if [ -z "$2" ]; then
        printf "%b[ERROR] --bump requires a value%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      BUMP="$2"
      shift 2
      ;;
    --branch)
      if [ -z "$2" ]; then
        printf "%b[ERROR] --branch requires a value%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      BRANCH="$2"
      shift 2
      ;;
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
      if [ -z "$VERSION" ] && [ -z "$BUMP" ]; then
        VERSION="$1"
      else
        printf "%b[ERROR] Multiple version arguments provided%b\n" "$RED" "$RESET"
        show_usage
        exit 1
      fi
      shift
      ;;
  esac
done

# Validate that either version or bump was provided
if [ -z "$VERSION" ] && [ -z "$BUMP" ]; then
  printf "%b[ERROR] No version or bump type specified%b\n" "$RED" "$RESET"
  show_usage
  exit 1
fi

if [ -n "$VERSION" ] && [ -n "$BUMP" ]; then
  printf "%b[ERROR] Cannot specify both VERSION and --bump%b\n" "$RED" "$RESET"
  show_usage
  exit 1
fi

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

# Determine target branch and default branch
DEFAULT_BRANCH=$(git remote show origin | grep 'HEAD branch' | cut -d' ' -f5)
if [ -z "$DEFAULT_BRANCH" ]; then
  printf "%b[ERROR] Could not determine default branch from remote%b\n" "$RED" "$RESET"
  exit 1
fi

if [ -z "$BRANCH" ]; then
  BRANCH="$DEFAULT_BRANCH"
  printf "%b[INFO] Using default branch: %s%b\n" "$BLUE" "$BRANCH" "$RESET"
else
  printf "%b[INFO] Using specified branch: %s%b\n" "$BLUE" "$BRANCH" "$RESET"
  if [ "$BRANCH" != "$DEFAULT_BRANCH" ]; then
    printf "%b[WARN] Target branch '%s' differs from default branch '%s'%b\n" "$YELLOW" "$BRANCH" "$DEFAULT_BRANCH" "$RESET"
    printf "%b[WARN] Releases are typically created from the default branch.%b\n" "$YELLOW" "$RESET"
    if [ -z "$DRY_RUN" ]; then
      printf "Continue with branch '%s'? [y/N] " "$BRANCH"
      read -r answer
      case "$answer" in
        [Yy]*)
          ;;
        *)
          printf "%b[INFO] Aborted by user%b\n" "$YELLOW" "$RESET"
          exit 1
          ;;
      esac
    fi
  fi
fi

# Verify branch exists
if ! git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  printf "%b[ERROR] Branch 'origin/%s' does not exist%b\n" "$RED" "$BRANCH" "$RESET"
  exit 1
fi

# Get current version
CURRENT_VERSION=$("$UV_BIN" version --short 2>/dev/null || echo "unknown")
printf "%b[INFO] Current version: %s%b\n" "$BLUE" "$CURRENT_VERSION" "$RESET"

# Determine the new version using uv version with --dry-run first
if [ -n "$BUMP" ]; then
  printf "%b[INFO] Bumping version using: %s%b\n" "$BLUE" "$BUMP" "$RESET"
  NEW_VERSION=$("$UV_BIN" version --bump "$BUMP" --dry-run --short 2>/dev/null)
  if [ $? -ne 0 ] || [ -z "$NEW_VERSION" ]; then
    printf "%b[ERROR] Failed to calculate new version with bump type: %s%b\n" "$RED" "$BUMP" "$RESET"
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
  printf "%b[WARN] You have uncommitted changes:%b\n" "$YELLOW" "$RESET"
  git status --short
  printf "\n%b[WARN] These changes will be included in the release commit.%b\n" "$YELLOW" "$RESET"
  if [ -z "$DRY_RUN" ]; then
    printf "Continue? [y/N] "
    read -r answer
    case "$answer" in
      [Yy]*)
        ;;
      *)
        printf "%b[INFO] Aborted by user%b\n" "$YELLOW" "$RESET"
        exit 1
        ;;
    esac
  fi
fi

if [ -n "$DRY_RUN" ]; then
  # Get repository info for the dry run message
  REPO_URL=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
  
  printf "\n%b[DRY RUN] Would perform the following actions:%b\n" "$YELLOW" "$RESET"
  printf "  1. Checkout and update branch '%s'\n" "$BRANCH"
  printf "  2. Update version in pyproject.toml from %s to %s\n" "$CURRENT_VERSION" "$NEW_VERSION"
  printf "  3. Git commit: 'chore: bump version to %s'\n" "$NEW_VERSION"
  printf "  4. Create git tag: %s\n" "$TAG"
  printf "  5. Push tag to origin\n"
  printf "  6. Trigger release workflow at: https://github.com/%s/actions\n" "$REPO_URL"
  printf "\n%b[DRY RUN] No changes made.%b\n" "$YELLOW" "$RESET"
  exit 0
fi

# Checkout and update the target branch
printf "%b[INFO] Checking out branch %s...%b\n" "$BLUE" "$BRANCH" "$RESET"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# Update version in pyproject.toml using uv
printf "%b[INFO] Updating version in pyproject.toml...%b\n" "$BLUE" "$RESET"
if [ -n "$BUMP" ]; then
  if ! "$UV_BIN" version --bump "$BUMP" >/dev/null 2>&1; then
    printf "%b[ERROR] Failed to bump version using 'uv version --bump %s'%b\n" "$RED" "$BUMP" "$RESET"
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

printf "%b[SUCCESS] Updated version to %s%b\n" "$GREEN" "$NEW_VERSION" "$RESET"

# Commit the version change
printf "%b[INFO] Committing version change...%b\n" "$BLUE" "$RESET"
git add pyproject.toml
git add uv.lock  # In case uv modifies the lock file, which it will do for the current version update
git commit -m "chore: bump version to $NEW_VERSION"

# Push the commit to the branch
printf "%b[INFO] Pushing commit to %s...%b\n" "$BLUE" "$BRANCH" "$RESET"
git push origin "$BRANCH"

# Create the tag
printf "%b[INFO] Creating tag %s...%b\n" "$BLUE" "$TAG" "$RESET"
git tag -a "$TAG" -m "Release $TAG"

# Push the tag
printf "%b[INFO] Pushing tag to origin...%b\n" "$BLUE" "$RESET"
git push origin "$TAG"

REPO_URL=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
printf "\n%b[SUCCESS] Release tag %s created and pushed!%b\n" "$GREEN" "$TAG" "$RESET"
printf "%b[INFO] The release workflow will now be triggered automatically.%b\n" "$BLUE" "$RESET"
printf "%b[INFO] Monitor progress at: https://github.com/%s/actions%b\n" "$BLUE" "$REPO_URL" "$RESET"
