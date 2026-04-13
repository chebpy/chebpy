# Installation

## From PyPI

The recommended way to install ChebPy:

```bash
pip install chebfun
```

## From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/chebpy/chebpy.git
cd chebpy
make install
```

This uses [uv](https://docs.astral.sh/uv/) to:

1. Install the correct Python version (specified in `.python-version`)
2. Create a virtual environment
3. Install all project dependencies

## Requirements

- **Python** >= 3.11
- **NumPy** >= 2.4.1
- **Matplotlib** >= 3.10.0

Development additionally requires ruff, pytest, marimo, and other tools
managed automatically by `make install`.

## Verifying the Installation

```bash
make test
```

Or in a Python session:

```python
import chebpy
print(chebpy.__version__)
```
