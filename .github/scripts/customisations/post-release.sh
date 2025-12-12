#!/bin/bash
# This file is part of the tschm/.config-templates repository
# (https://github.com/tschm/.config-templates).
#
# Optional hook script for post-release actions
#
# Purpose: This script is called automatically after a release is created.
#          Use it to perform additional tasks after the release process completes
#          (e.g., notifications, cleanup, triggering deployments).
#
# When it runs:
#   - Called by: make release (after release tag is created and pushed)
#   - Environment: GitHub Actions runner or local development machine
#
# How to use:
#   1. Add your custom post-release commands below
#   2. Make sure the script is executable: chmod +x .github/scripts/customisations/post-release.sh
#   3. Commit to your repository
#
# Examples:
#   - Send notifications (Slack, email, etc.)
#   - Trigger deployment workflows
#   - Update external documentation sites
#   - Clean up temporary release artifacts
#
# Note: If you customize this file in your repository, add it to the exclude list
#       in template.yml to prevent it from being overwritten by template updates:
#       exclude: |
#         .github/scripts/customisations/post-release.sh
#

set -euo pipefail

echo "Running post-release.sh..."

# Add your custom post-release commands here

# Example: Send notification
# if command -v curl &> /dev/null && [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
#     echo "Sending notification to Slack..."
#     curl -X POST -H 'Content-type: application/json' \
#         --data '{"text":"New release created!"}' \
#         "${SLACK_WEBHOOK_URL}"
# fi

echo "Post-release setup complete."
