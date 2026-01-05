#!/bin/bash
# This file is part of the jebel-quant/rhiza repository
# (https://github.com/jebel-quant/rhiza).
#
# Optional hook script for installing extra dependencies
#
# Purpose: This script is called automatically during the install phase and before
#          building documentation. Use it to install additional system packages or
#          dependencies that your project needs (e.g., graphviz for diagrams).
#
# When it runs:
#   - Called by: make install-extras (during setup phase)
#   - Also runs before: make test, make book, make docs
#   - Environment: GitHub Actions runner or local development machine
#
# How to use:
#   1. Add your custom installation commands below
#   2. Make sure the script is executable: chmod +x .rhiza/scripts/customisations/build-extras.sh
#   3. Commit to your repository
#
# Examples:
#   - Install system packages: apt-get, brew, yum, etc.
#   - Install optional dependencies for documentation
#   - Download or build tools needed by your project
#
# Note: If you customize this file in your repository, add it to the exclude list
#       in action.yml to prevent it from being overwritten by template updates:
#       exclude: |
#         .rhiza/scripts/customisations/build-extras.sh
#

set -euo pipefail

echo "Running build-extras.sh..."

# Add your custom installation commands here

# Example: graphviz
# Good practice to check if already installed.
#
# if ! command -v dot &> /dev/null; then
#     echo "graphviz not found, installing..."
#     sudo apt-get update && sudo apt-get install -y graphviz
# else
#     echo "graphviz is already installed, skipping installation."
# fi

echo "Build extras setup complete."
