<div align="center">

# ChebPy

### A Python implementation of Chebfun

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE.rst)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/chebfun.svg)](https://pypi.org/project/chebfun/)

![Github](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=white)
![MAC OS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![CodeFactor](https://www.codefactor.io/repository/github/chebpy/chebpy/badge)](https://www.codefactor.io/repository/github/chebpy/chebpy)

[![CI](https://github.com/chebpy/chebpy/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/chebpy/chebpy/actions/workflows/rhiza_ci.yml)
[![MARIMO](https://github.com/chebpy/chebpy/actions/workflows/rhiza_marimo.yml/badge.svg?event=push)](https://github.com/chebpy/chebpy/actions/workflows/rhiza_marimo.yml)
[![DEVCONTAINER](https://github.com/chebpy/chebpy/actions/workflows/rhiza_devcontainer.yml/badge.svg?event=push)](https://github.com/chebpy/chebpy/actions/workflows/rhiza_devcontainer.yml)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/chebpy/chebpy)

**🔬 Numerical computing with Chebyshev series approximations**

Symbolic-numeric computation with functions

</div>

ChebPy is a Python implementation of [Chebfun](http://www.chebfun.org/), bringing the power of Chebyshev polynomial approximations to Python. It allows you to work with functions as first-class objects, performing operations like differentiation, integration, and root-finding with machine precision accuracy.
---

## Table of Contents

- [✨ Features](#-features)
- [📥 Installation](#-installation)
- [🛠️ Development](#️-development)
- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)
- [📄 License](#-license)
- [👥 Contributing](#-contributing)

---

## ✨ Features

> **Work with functions as easily as numbers**

- 🔢 **Function Approximation**: Automatic Chebyshev polynomial approximation of smooth functions
- 📐 **Calculus Operations**: Differentiation, integration, and root-finding with machine precision
- 📊 **Plotting**: Beautiful function visualizations with matplotlib integration
- 🧮 **Arithmetic**: Add, subtract, multiply, and compose functions naturally
- 🎯 **Adaptive**: Automatically determines optimal polynomial degree for given tolerance
- 🔗 **Interoperability**: Works seamlessly with NumPy and SciPy ecosystems

---

## 📥 Installation

### Using pip (recommended)

```bash
pip install chebpy
```

### From source (development)

```bash
git clone https://github.com/chebpy/chebpy.git
cd chebpy
pip install -e .
```

> **Note**: Use `-e` flag for editable installation during development

## 🛠️ Development

> **For contributors and advanced users**

ChebPy uses modern Python development tools for a smooth developer experience:

```bash
# 📦 Install development dependencies
make install

# 🧪 Run tests with coverage
make test

# ✨ Format and lint code
make fmt
make lint

# 📓 Start interactive notebooks
make marimo

# 🔍 View test coverage report
make coverage
```

### Development Tools

- **Testing**: pytest with coverage reporting
- **Formatting**: ruff for code formatting and linting
- **Notebooks**: marimo for interactive development
- **Task Management**: Taskfile for build automation

## Quick Start

<div align="center">
  <img src="docs/notebooks/chebpy-readme-image1.png" alt="ChebPy Example" width="80%">
</div>


This figure was generated with the following simple ChebPy code:

```python
import numpy as np
from chebpy import chebfun

# Create functions as chebfuns on interval [0, 10]
f = chebfun(lambda x: np.sin(x**2) + np.sin(x)**2, [0, 10])
g = chebfun(lambda x: np.exp(-(x-5)**2/10), [0, 10])

# Find intersection points
roots = (f - g).roots()

# Plot both functions and mark intersections
ax = f.plot(label='f(x) = sin(x²) + sin²(x)')
g.plot(ax=ax, label='g(x) = exp(-(x-5)²/10)')
ax.plot(roots, f(roots), 'ro', markersize=8, label='Intersections')
ax.legend()
ax.grid(True, alpha=0.3)
```

```result

```

### More Examples

```python
# Differentiation and integration
f = chebfun(lambda x: np.exp(x) * np.sin(x), [-1, 1])
df_dx = f.diff()          # Derivative
integral = f.sum()        # Definite integral

# Root finding
g = chebfun(lambda x: x**3 - 2*x - 5, [-3, 3])
roots = g.roots()         # All roots in the domain
```

### Convolution

Convolve two functions to produce a new Chebfun on the summed domain:

```python
from chebpy import chebfun
import numpy as np

f = chebfun(lambda x: np.exp(-x**2), [-1, 1])
g = chebfun(lambda x: np.where(np.abs(x) < 0.5, 1.0, 0.0), [-1, 1])

h = f.conv(g)        # h(x) = ∫ f(t) g(x−t) dt, a Chebfun on [−2, 2]
h.plot()
```

### Quasimatrices

Stack functions as columns of an ∞×n matrix and use continuous
linear algebra — QR, SVD, least-squares:

```python
from chebpy import Quasimatrix, chebfun

x = chebfun("x")
A = Quasimatrix([1, x, x**2, x**3, x**4, x**5])

Q, R = A.qr()             # QR factorisation → Legendre polynomials
U, S, V = A.svd()         # Singular value decomposition

f = chebfun(lambda t: np.exp(t) * np.sin(6 * t), [-1, 1])
c = A.solve(f)            # Least-squares polynomial fit
f_approx = A @ c          # Reconstruct as a Chebfun
```

### Gaussian Process Regression

Fit a GP to scattered data and get the posterior mean and variance
back as Chebfuns — ready for differentiation, integration, and root-finding:

```python
from chebpy import gpr
import numpy as np

rng = np.random.default_rng(1)
x_obs = np.sort(-2 + 4 * rng.random(10))
y_obs = np.sin(np.exp(x_obs))

f_mean, f_var = gpr(x_obs, y_obs, domain=[-2, 2])

f_mean.plot()                     # Posterior mean (a Chebfun)
extrema = f_mean.diff().roots()   # Local extrema via calculus
integral = f_mean.sum()           # Definite integral
```

```result

```
---

## Documentation

- 🚀 **[Codespaces](https://codespaces.new/chebpy/chebpy)**: Try ChebPy in your browser
- 📚 **[Documentation](https://chebpy.github.io/chebpy)**: Full documentation, user guide, and API reference

---

## 📄 License

ChebPy is licensed under the **3-Clause BSD License**.

📜 See the full license in the [LICENSE.rst](LICENSE.rst) file.

---

## 👥 Contributing

**We welcome contributions!** 🎉

Whether you're fixing bugs, adding features, or improving documentation, your help makes ChebPy better for everyone.

### Quick Start for Contributors

1. 🍴 **Fork** the repository
2. 🌿 **Create** your feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. ✨ **Make** your changes and add tests
4. 🧪 **Test** your changes
   ```bash
   make test
   ```
5. 📝 **Commit** your changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. 🚀 **Push** to your branch
   ```bash
   git push origin feature/amazing-feature
   ```
7. 🎯 **Open** a Pull Request

### Resources

- 📋 [Contributing Guide](CONTRIBUTING.md)
- 🤝 [Code of Conduct](CODE_OF_CONDUCT.md)
- 🐛 [Issue Tracker](https://github.com/chebpy/chebpy/issues)

### Acknowledgments 🙏

- [Jebel-Quant/rhiza](https://github.com/Jebel-Quant/rhiza) for standardised CI/CD templates and project tooling


---

<div align="center">

**Made with ❤️ by the ChebPy community**

⭐ *If you find ChebPy useful, please consider giving it a star!* ⭐

</div>
