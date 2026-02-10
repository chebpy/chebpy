# DevContainer Configuration

This directory contains the configuration for [GitHub Codespaces](https://github.com/features/codespaces) and [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers).

## Contents

- `devcontainer.json`: The primary configuration file defining the development environment.
- `bootstrap.sh`: Post-create script that initializes the environment (installing dependencies, setting up tools).

## Python Version

The Python version is controlled by the `.python-version` file in the repository root (single source of truth).

**How it works:**
1. The devcontainer uses a base Python image (3.12)
2. `bootstrap.sh` reads `.python-version` and exports `PYTHON_VERSION`
3. `make install` uses UV to create a venv with the exact Python version specified
4. UV automatically downloads the correct Python version if needed

No manual setup required - UV handles Python version management!

## What's Configured

The `.devcontainer` setup provides:

- 🐍 **Python** runtime environment
- 🔧 **UV Package Manager** - Fast Python package installer and resolver
- ⚡ **Makefile** - For running project workflows
- 🧪 **Pre-commit Hooks** - Automated code quality checks
- 📊 **Marimo Integration** - Interactive notebook support with VS Code extension
- 🔍 **Python Development Tools** - Pylance, Python extension, and optimized settings
- 🚀 **Port Forwarding** - Port 8080 for development servers
- 🔐 **SSH Agent Forwarding** - Full Git functionality with your host SSH keys

## Usage

### In VS Code

1. Install the "Dev Containers" extension
2. Open the repository in VS Code
3. Click "Reopen in Container" when prompted
4. The environment will automatically set up with all dependencies

### In GitHub Codespaces

1. Navigate to the repository on GitHub
2. Click the green "Code" button
3. Select "Codespaces" tab
4. Click "Create codespace on main" (or your branch)
5. Your development environment will be ready in minutes

The dev container automatically runs the initialization script that:

- Installs UV package manager
- Configures the Python virtual environment
- Installs project dependencies
- Sets up pre-commit hooks

## Publishing Devcontainer Images

The repository includes workflows for building and publishing devcontainer images:

### CI Validation

The **DEVCONTAINER** workflow automatically validates that your devcontainer builds successfully:
- Triggers on changes to `.devcontainer/**` files or the workflow itself
- Builds the image without publishing (`push: never`)
- Works on pushes to any branch and pull requests
- Gracefully skips if no `.devcontainer/devcontainer.json` exists

## VS Code Dev Container SSH Agent Forwarding

Dev containers launched locally via VS code
are configured with SSH agent forwarding
to enable seamless Git operations:

- **Mounts your SSH directory** - Your `~/.ssh` folder is mounted into the container
- **Forwards SSH agent** - Your host's SSH agent is available inside the container
- **Enables Git operations** - Push, pull, and clone using your existing SSH keys
- **Works transparently** - No additional setup required in VS Code dev containers

## Troubleshooting

Common issues and solutions when using this configuration template.

---

### SSH authentication fails on macOS when using devcontainer

**Symptom**: When building or using the devcontainer on macOS, Git operations (pull, push, clone) fail with SSH authentication errors, even though your SSH keys work fine on the host.

**Cause**: macOS SSH config often includes `UseKeychain yes`, which is a macOS-specific directive. When the devcontainer mounts your `~/.ssh` directory, other platforms (Linux containers) don't recognize this directive and fail to parse the SSH config.

**Solution**: Add `IgnoreUnknown UseKeychain` to the top of your `~/.ssh/config` file on your Mac:

```ssh-config
# At the top of ~/.ssh/config
IgnoreUnknown UseKeychain

Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_rsa
```

This tells SSH clients on non-macOS platforms to ignore the `UseKeychain` directive instead of failing.

**Reference**: [Stack Overflow solution](https://stackoverflow.com/questions/75613632/trying-to-ssh-to-my-server-from-the-terminal-ends-with-error-line-x-bad-configu/75616369#75616369)
