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

## Quickstart

Head to the [Quickstart](user/quickstart.md) guide for a hands-on introduction, or explore the [API Reference](api.md) for full documentation.

## Acknowledgments

ChebPy is a direct port of the **Chebfun project** led by
[Nick Trefethen](https://people.maths.ox.ac.uk/trefethen/) and the Chebfun
development team at the University of Oxford. The mathematical design,
algorithms, and naming conventions used here are adaptations of their
decades of open scholarship, most notably:

- The original [MATLAB Chebfun](https://www.chebfun.org/) system
  ([github.com/chebfun/chebfun](https://github.com/chebfun/chebfun)).
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013 (extended edition 2019).
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
  [*Chebfun Guide*](https://www.chebfun.org/docs/guide/), Pafnuty
  Publications, 2014.

We are grateful for their generosity in making this body of work freely
available; any errors in translation or adaptation are ours alone.
