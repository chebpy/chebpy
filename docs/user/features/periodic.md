# Periodic Functions

For smooth periodic functions, ChebPy provides Fourier-based approximation
through `Trigtech` — the trigonometric analogue of the default `Chebtech`
representation. The two technologies share the same `Onefun → Smoothfun`
interface, so periodic Chebfuns interoperate transparently with the rest of
ChebPy.

## The `trigfun` constructor

`trigfun` is the explicit user-facing entry point for periodic functions.
It mirrors `chebfun` exactly but always uses `Trigtech` as the underlying
approximation technology.

```python
import numpy as np
from chebpy import trigfun

f = trigfun(lambda x: np.cos(np.pi * x), [-1, 1])
g = trigfun(lambda x: np.sin(2 * np.pi * x))
```

The user is responsible for ensuring that `f` is smooth and periodic on the
given domain. There is no automatic detection — periodicity is opted into
explicitly. This mirrors MATLAB Chebfun's `chebfun(f, 'trig')` while keeping
the Python API unambiguous.

### Special constructors

The same shorthand forms as `chebfun` are supported:

```python
trigfun()               # empty Chebfun
trigfun(3.14)            # constant function
trigfun(lambda x: np.sin(np.pi * x), n=16)   # fixed number of Fourier modes
```

## Why use `trigfun`?

For a smooth periodic function, a Fourier series typically requires far
fewer coefficients than a Chebyshev series of comparable accuracy. As a
result `trigfun` approximations are more compact and cheaper to evaluate,
differentiate, and integrate.

```python
import numpy as np
from chebpy import chebfun, trigfun

f = lambda x: np.cos(8 * np.pi * x) + np.sin(3 * np.pi * x)

fc = chebfun(f, [-1, 1])
ft = trigfun(f, [-1, 1])

len(fc)   # Chebyshev degree
len(ft)   # number of Fourier modes
```

## Calculus and arithmetic

All standard Chebfun operations work on periodic Chebfuns:

```python
f = trigfun(lambda x: np.sin(np.pi * x), [-1, 1])

df = f.diff()        # spectral differentiation in Fourier space
F = f.cumsum()       # antiderivative
total = f.sum()      # ≈ 0
```

## Periodic Gaussian process regression

The `gpr` interface accepts a `trig=True` flag that produces a posterior
backed by `Trigtech`. See the
[Gaussian Process Regression notebook](../../notebooks/6_gaussian_process.html)
for a worked example.

## See also

- [`Trigtech` API reference](../../api.md)
- The [Function Approximation](approximation.md) page for the default
  Chebyshev-based construction.

## References

- G. B. Wright, M. Javed, H. Montanelli, and L. N. Trefethen,
  [*Extension of Chebfun to periodic functions*](https://epubs.siam.org/doi/abs/10.1137/141001007),
  SIAM J. Sci. Comput., 37 (2015), pp. C554–C573.
- L. N. Trefethen and J. A. C. Weideman,
  [*The exponentially convergent trapezoidal rule*](https://epubs.siam.org/doi/10.1137/130932132),
  SIAM Review, 56 (2014), pp. 385–458.
