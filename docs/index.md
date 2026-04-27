# ChebPy

**A Python implementation of Chebfun — numerical computing with Chebyshev series approximations.**

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](https://github.com/chebpy/chebpy/blob/main/LICENSE.rst)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/chebfun.svg)](https://pypi.org/project/chebfun/)

---

ChebPy is a Python implementation of [Chebfun](http://www.chebfun.org/), bringing the power of Chebyshev polynomial approximations to Python. It allows you to work with functions as first-class objects, performing operations like differentiation, integration, and root-finding with machine precision accuracy.

## Features

- **Function Approximation** — Automatic Chebyshev polynomial approximation of smooth functions
- **Periodic Functions** — Fourier-based approximation via `trigfun` for smooth periodic functions
- **Infinite Intervals** — Functions on $[a, \infty)$, $(-\infty, b]$ or the full real line via `CompactFun`
- **Calculus Operations** — Differentiation, integration, and root-finding with machine precision
- **Plotting** — Beautiful function visualisations with matplotlib integration
- **Arithmetic** — Add, subtract, multiply, and compose functions naturally
- **Adaptive** — Automatically determines optimal polynomial degree for given tolerance
- **Convolution** — Convolve two Chebfun objects to produce a new function
- **Quasimatrices** — Continuous linear algebra via QR, SVD, and least-squares
- **Gaussian Process Regression** — GP posteriors returned as Chebfun objects
- **Interoperability** — Works seamlessly with NumPy and SciPy ecosystems

## Quick Example

```python
import numpy as np
from chebpy import chebfun

# Create a function
f = chebfun(lambda x: np.sin(x**2) + np.sin(x)**2, [0, 10])

# Differentiate, integrate, find roots
df = f.diff()
integral = f.sum()
roots = f.roots()
```

## Getting Started

Head to the [Getting Started](user/quickstart.md) guide for a hands-on introduction, or explore the [API Reference](api.md) for full documentation.
