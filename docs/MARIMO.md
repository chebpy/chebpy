# Marimo Notebooks

This directory contains interactive [Marimo](https://marimo.io/) notebooks for the Rhiza project.

## Available Notebooks

### 📊 rhiza.py - Marimo Feature Showcase

A comprehensive demonstration of Marimo's most useful features, including:

- **Interactive UI Elements**: Sliders, dropdowns, text inputs, checkboxes, and multiselect
- **Reactive Programming**: Automatic cell updates when dependencies change
- **Data Visualisation**: Interactive plots using Plotly
- **DataFrames**: Working with Pandas data
- **Layout Components**: Columns, tabs, and accordions for organised content
- **Forms**: Dictionary-based forms for collecting user input
- **Rich Text**: Markdown and LaTeX support for documentation
- **Advanced Features**: Callouts, collapsible accordions, and more

This notebook is perfect for:
- Learning Marimo's capabilities
- Understanding reactive programming in notebooks
- Seeing real examples of interactive UI components
- Getting started with Marimo in your own projects

## Running the Notebooks

### Using the Makefile

From the repository root:

```bash
make marimo
```

This will start the Marimo server and open all notebooks in the `book/marimo` directory.

### Running a Specific Notebook

To run a single notebook:

```bash
marimo edit book/marimo/rhiza.py
```

### Using uv (Recommended)

The notebooks include inline dependency metadata, making them self-contained:

```bash
uv run book/marimo/rhiza.py
```

This will automatically install the required dependencies and run the notebook.

## Notebook Structure

Marimo notebooks are **pure Python files** (`.py`), not JSON. This means:

- ✅ Easy version control with Git
- ✅ Standard code review workflows  
- ✅ No hidden metadata
- ✅ Compatible with all Python tools

Each notebook includes inline metadata that specifies its dependencies:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.18.4",
#     "numpy>=1.24.0",
# ]
# ///
```

## Configuration

Marimo is configured in `pyproject.toml` to properly import the local package:

```toml
[tool.marimo.runtime]
pythonpath = ["src"]
```

## CI/CD Integration

The `.github/workflows/marimo.yml` workflow automatically:

1. Discovers all `.py` files in this directory
2. Runs each notebook in a fresh environment
3. Verifies that notebooks can bootstrap themselves
4. Ensures reproducibility

This guarantees that all notebooks remain functional and up-to-date.

## Creating New Notebooks

To create a new Marimo notebook:

1. Create a new `.py` file in this directory:
   ```bash
   marimo edit book/marimo/my_notebook.py
   ```

2. Add inline metadata at the top:
   ```python
   # /// script
   # requires-python = ">=3.11"
   # dependencies = [
   #     "marimo==0.18.4",
   #     # ... other dependencies
   # ]
   # ///
   ```

3. Start building your notebook with cells

4. Test it runs in a clean environment:
   ```bash
   uv run book/marimo/my_notebook.py
   ```

5. Commit and push - the CI will validate it automatically

## Learn More

- **Marimo Documentation**: [https://docs.marimo.io/](https://docs.marimo.io/)
- **Example Gallery**: [https://marimo.io/examples](https://marimo.io/examples)
- **Community Discord**: [https://discord.gg/JE7nhX6mD8](https://discord.gg/JE7nhX6mD8)

## Tips

- **Reactivity**: Remember that cells automatically re-run when their dependencies change
- **Pure Python**: Edit notebooks in any text editor, not just Marimo's UI
- **Git-Friendly**: Notebooks diff and merge like regular Python files
- **Self-Contained**: Use inline metadata to make notebooks reproducible
- **Interactive**: Take advantage of Marimo's rich UI components for better user experience

---

*Happy exploring with Marimo! 🚀*
