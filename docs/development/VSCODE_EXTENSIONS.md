# VSCode Extensions

This document describes the VSCode extensions that are automatically configured when using the [Dev Container](.devcontainer/) or [GitHub Codespaces](https://github.com/features/codespaces) environment.

## Extension Overview

The project configures extensions in `.devcontainer/devcontainer.json` to provide a complete, integrated development experience. All extensions are automatically installed when you open the project in a Dev Container or Codespace.

## Configured Extensions

### Python Development

#### [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (`ms-python.python`)
**Purpose**: Core Python language support for VS Code

**Features**:
- IntelliSense (code completion and suggestions)
- Linting and debugging
- Code formatting with multiple formatters
- Environment management
- Jupyter notebook support

**Why included**: Essential for any Python development work. Provides the foundation for all Python-related features in VS Code.

#### [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) (`ms-python.vscode-pylance`)
**Purpose**: Fast, feature-rich Python language server

**Features**:
- Type checking and type information
- Auto-imports and import sorting
- Semantic highlighting
- Advanced IntelliSense with type stubs
- Code navigation (go to definition, find references)

**Why included**: Provides superior performance and accuracy compared to the default Python language server. Essential for modern type-aware Python development.

### Marimo Notebooks

#### [Marimo](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo) (`marimo-team.vscode-marimo`)
**Purpose**: Official Marimo notebook support

**Features**:
- Edit and run Marimo notebooks in VS Code
- Integrated Marimo server management
- Syntax highlighting for Marimo cells
- Live preview of notebook output

**Why included**: This project uses [Marimo notebooks](https://marimo.io/) for interactive documentation (see `book/` directory). This extension enables editing notebooks directly in VS Code.

#### [Marimo VSCode](https://marketplace.visualstudio.com/items?itemName=marimo-ai.marimo-vscode) (`marimo-ai.marimo-vscode`)
**Purpose**: Enhanced Marimo integration

**Features**:
- Additional Marimo cell execution features
- Improved notebook rendering
- Cell management utilities

**Why included**: Complements the official Marimo extension with additional features for working with interactive notebooks.

### Linting and Formatting

#### [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (`charliermarsh.ruff`)
**Purpose**: Lightning-fast Python linter and formatter

**Features**:
- Real-time linting with instant feedback
- Automatic code formatting
- Import sorting and organization
- Hundreds of linting rules covering common Python issues
- Replaces Flake8, isort, and Black with a single tool

**Why included**: This project uses [Ruff](https://github.com/astral-sh/ruff) as its primary linter and formatter (configured in `ruff.toml`). The extension provides real-time feedback as you code.

#### [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) (`tamasfe.even-better-toml`)
**Purpose**: Enhanced TOML language support

**Features**:
- Syntax highlighting for TOML files
- Validation and error detection
- IntelliSense for `pyproject.toml` schema
- Formatting and organization

**Why included**: Python projects use TOML extensively (`pyproject.toml`, `ruff.toml`, etc.). This extension makes editing configuration files easier and prevents syntax errors.

### Build Tools

#### [Makefile Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.makefile-tools) (`ms-vscode.makefile-tools`)
**Purpose**: Makefile integration and task running

**Features**:
- Syntax highlighting for Makefiles
- Run make targets from VS Code
- Target discovery and organization
- Build problem matching
- IntelliSense for make targets

**Why included**: This project uses a comprehensive [Makefile](../Makefile) as its primary task runner (`make install`, `make test`, `make fmt`, etc.). This extension makes it easy to discover and run available tasks.

### AI Assistance

#### [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=github.copilot) (`github.copilot`)
**Purpose**: AI-powered code completion

**Features**:
- Context-aware code suggestions
- Whole-line and function completions
- Multi-language support
- Learning from public code patterns

**Why included**: Accelerates development with intelligent code suggestions. Particularly useful for boilerplate code, test generation, and exploring new APIs.

#### [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=github.copilot-chat) (`github.copilot-chat`)
**Purpose**: AI-powered conversational coding assistant

**Features**:
- Natural language code explanations
- Code refactoring suggestions
- Interactive debugging help
- Documentation generation
- Test case generation

**Why included**: Provides an interactive AI assistant for code understanding, debugging, and documentation. Complements Copilot's inline suggestions with conversational help.

#### [Claude Code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code) (`anthropic.claude-code`)
**Purpose**: Claude AI integration for VS Code

**Features**:
- Chat with Claude directly in VS Code
- Code analysis and suggestions
- Documentation and explanation generation
- Multi-turn conversations about code

**Why included**: Offers an alternative AI assistant with different strengths. Claude excels at detailed explanations, complex refactoring, and understanding large codebases.

## Extension Configuration

Extensions are configured in `.devcontainer/devcontainer.json` and are automatically installed when:
- Opening the repository in a Dev Container (VS Code)
- Creating a GitHub Codespace
- Rebuilding the container

## Customization

### Adding Extensions

To add additional extensions to your development environment, edit `.devcontainer/devcontainer.json`:

```json
{
    "customizations": {
        "vscode": {
            "extensions": [
                // ... existing extensions ...
                "your-publisher.your-extension"
            ]
        }
    }
}
```

### Removing Extensions

To remove unwanted extensions, simply delete the corresponding line from the `extensions` array in `.devcontainer/devcontainer.json`.

### Local Extension Preferences

If you prefer different extensions for local development, you can:
1. Install them manually in VS Code (they won't affect others)
2. Create a `.vscode/extensions.json` file for team recommendations
3. Use VS Code profiles to separate personal preferences from project requirements

## Manual Installation (Non-Container Development)

If you're not using the Dev Container but want the same extensions:

1. Open the Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Run "Extensions: Configure Recommended Extensions (Workspace)"
3. Copy the extension IDs from `.devcontainer/devcontainer.json`
4. Or install each extension individually from the VS Code marketplace

Alternatively, create `.vscode/extensions.json` with:

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "marimo-team.vscode-marimo",
        "marimo-ai.marimo-vscode",
        "charliermarsh.ruff",
        "tamasfe.even-better-toml",
        "ms-vscode.makefile-tools",
        "github.copilot-chat",
        "github.copilot",
        "anthropic.claude-code"
    ]
}
```

## Related Documentation

- [DevContainer Configuration](DEVCONTAINER.md) - Dev container setup and usage
- [Makefile Customisation](../.rhiza/make.d/README.md) - Task automation and customization
- [Marimo Documentation](MARIMO.md) - Interactive notebooks and Marimo integration
- [Quick Reference](../guides/QUICK_REFERENCE.md) - Common development tasks

## Extension Requirements

Most extensions work out-of-the-box with no additional configuration. However:

- **AI extensions** (Copilot, Claude) require active subscriptions or API keys
- **Python extensions** require the Python virtual environment (created by `make install`)
- **Marimo extensions** require Marimo to be installed (included in project dependencies)

Run `make install` to ensure all project dependencies are available for the extensions.
