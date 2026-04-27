# Infinite Intervals

ChebPy supports functions defined on semi-infinite intervals $[a, \infty)$,
$(-\infty, b]$ or the full real line $(-\infty, \infty)$ through the
`CompactFun` class.

## Usage

Pass a domain containing `np.inf` or `-np.inf` to `chebfun`:

```python
import numpy as np
from chebpy import chebfun

# A doubly-infinite Gaussian
f = chebfun(lambda x: np.exp(-x**2), [-np.inf, np.inf])

# Decaying oscillation on [0, ∞)
g = chebfun(lambda x: np.sin(10 * x) * np.exp(-x), [0, np.inf])

# Standard operations work transparently
g.sum()       # definite integral over [0, ∞)
g.diff()      # derivative
g.roots()     # roots within numerical support
```

Pieces with infinite endpoints are automatically constructed as `CompactFun`
objects, while interior pieces remain ordinary `Bndfun`s:

```python
p = chebfun(lambda x: np.exp(-x**2), [-np.inf, -2.0, 0.0, 3.0, np.inf])
print([type(piece).__name__ for piece in p.funs])
# ['CompactFun', 'Bndfun', 'Bndfun', 'CompactFun']
```

## How it works

`CompactFun` represents a function whose **logical** interval may extend to
$\pm\infty$, but whose **numerical support** — the set of points where
the function is non-negligible relative to tolerance — is finite. At
construction time, ChebPy probes the function geometrically outward from
the finite anchor to discover the storage interval, then represents the
function as a standard Chebyshev expansion on that interval. Outside the
storage interval the function is reported as identically zero.

This is a deliberate departure from MATLAB Chebfun's `@unbndfun`, which
applies a rational change of variables $[a, \infty) \to [-1, 1]$. The
numerical-support approach has two consequences:

- **Decaying functions** (Gaussians, $e^{-x}$, $1/\Gamma(x+1)$) are handled
  cleanly and accurately, and benefit directly from the existing
  Hale–Townsend Legendre convolution machinery.
- **Heavy-tailed or non-decaying functions** — $1/(1+x^2)$, $\tanh(x)$,
  $1/(1+|x|)$ — are explicitly **refused** with a
  `CompactFunConstructionError` rather than silently approximated.

## What gets refused

```python
from chebpy.exceptions import CompactFunConstructionError

for f in [
    lambda x: 1.0 / (np.pi * (1.0 + x * x)),   # Cauchy density: O(1/x²) decay
    lambda x: 1.0 / (1.0 + np.abs(x)),          # O(1/x) decay
    lambda x: np.tanh(x - 1.0),                 # does not decay
]:
    try:
        chebfun(f, [-np.inf, np.inf])
    except CompactFunConstructionError as err:
        print(err)
```

## Convolution

Because each `CompactFun` lives on a finite storage interval, convolution
works directly. For example, convolving the standard Gaussian density
with itself yields $\mathcal{N}(0, 2)$:

```python
pdf = chebfun(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
              [-np.inf, np.inf])
pdf2 = pdf.conv(pdf)
pdf2.sum()    # ≈ 1
pdf2(0.0)     # ≈ 1 / sqrt(4π)
```

## See also

- The [Infinite Intervals notebook](../../notebooks/9_infinite_intervals.html)
  for a tour of worked examples adapted from §9.1 of the Chebfun guide.
- The [Convolution](convolution.md) page for finite-interval convolution.
