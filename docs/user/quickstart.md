# Getting Started

This guide walks you through the basics of ChebPy in a few minutes.

## Installation

```bash
pip install chebfun
```

Or install from source for development:

```bash
git clone https://github.com/chebpy/chebpy.git
cd chebpy
make install
```

## Creating a Chebfun

The `chebfun` function is the main entry point. Pass it a callable, and ChebPy will
adaptively approximate it with Chebyshev polynomials:

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi])
```

You can also create common objects with shorthand:

```python
x = chebfun('x')          # identity function on [-1, 1]
c = chebfun(3.14)          # constant function
empty = chebfun()           # empty Chebfun
```

## Evaluating

Evaluate at a point or an array of points:

```python
f(0.5)
f(np.linspace(-1, 1, 100))
```

## Basic Operations

```python
f = chebfun(lambda x: np.exp(x), [-1, 1])

# Calculus
df = f.diff()              # derivative
F = f.cumsum()             # indefinite integral
integral = f.sum()         # definite integral

# Root-finding
g = chebfun(lambda x: np.cos(10 * x), [-1, 1])
roots = g.roots()

# Arithmetic
h = f + g
h = f * g
h = f ** 2
```

## Plotting

```python
import matplotlib.pyplot as plt

f = chebfun(lambda x: np.sin(10 * x) * np.exp(-x), [-1, 1])
f.plot()
plt.show()
```

## What's Next?

- [User Guide](intro.md) — deeper introduction to ChebPy concepts
- [API Reference](../api.md) — full function and class documentation
- [Marimo Notebooks](../notebooks/1_introduction.html) — interactive explorations
