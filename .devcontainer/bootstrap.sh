#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Error handler with descriptive messages and remediation steps
error_with_recovery() {
    local step="$1"
    local error_msg="$2"
    local remediation="$3"
    
    echo "❌ ERROR: $step failed"
    echo "   Details: $error_msg"
    echo "   💡 Suggested fix: $remediation"
    return 1
}

# Read Python version from .python-version (single source of truth)
if [ -f ".python-version" ]; then
    export PYTHON_VERSION=$(cat .python-version | tr -d '[:space:]')
    echo "✓ Using Python version from .python-version: $PYTHON_VERSION"
else
    error_with_recovery \
        "Python version detection" \
        ".python-version file not found" \
        "Ensure .python-version file exists in repository root"
fi

# Use INSTALL_DIR from environment or default to local bin
# In devcontainer, this is set to /home/vscode/.local/bin to avoid conflict with host
export INSTALL_DIR="${INSTALL_DIR:-./bin}"
export UV_BIN="${INSTALL_DIR}/uv"
export UVX_BIN="${INSTALL_DIR}/uvx"

# Only remove existing binaries if we are installing to the default ./bin location
# and we want to force re-installation (e.g. if OS changed)
if [ "$INSTALL_DIR" = "./bin" ]; then
    rm -f "$UV_BIN" "$UVX_BIN"
fi

# Set UV environment variables to avoid prompts and warnings
export UV_VENV_CLEAR=1
export UV_LINK_MODE=copy

# Make UV environment variables persistent for all sessions
echo "export UV_LINK_MODE=copy" >> ~/.bashrc
echo "export UV_VENV_CLEAR=1" >> ~/.bashrc
echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.bashrc

# Add to current PATH so subsequent commands can find uv
export PATH="$INSTALL_DIR:$PATH"

# Install dependencies with recovery options
echo "📦 Installing project dependencies..."
if ! make install; then
    error_with_recovery \
        "Dependency installation" \
        "make install failed" \
        "Try: 1) Check internet connectivity, 2) Manually run 'make install' to see detailed errors, 3) Check disk space with 'df -h'"
fi
echo "✓ Dependencies installed successfully"

# Install Marimo tool for notebook editing with fallback
echo "📓 Installing Marimo notebook tool..."
if ! "$UV_BIN" tool install marimo 2>/dev/null; then
    echo "⚠️  WARNING: Marimo installation failed (non-critical)"
    echo "   You can manually install later with: uv tool install marimo"
    echo "   Continuing with bootstrap..."
else
    echo "✓ Marimo installed successfully"
fi

# Initialize pre-commit hooks if configured with fallback
if [ -f .pre-commit-config.yaml ]; then
    echo "🔧 Setting up pre-commit hooks..."
    if ! "$UVX_BIN" pre-commit install 2>/dev/null; then
        echo "⚠️  WARNING: Pre-commit hook installation failed (non-critical)"
        echo "   You can manually install later with: uvx pre-commit install"
        echo "   Continuing with bootstrap..."
    else
        echo "✓ Pre-commit hooks configured successfully"
    fi
fi

echo "✅ Bootstrap completed successfully!"
