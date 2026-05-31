# Arithmetic

Chebfun objects support natural arithmetic operations, producing new Chebfun
objects as results.

## Basic Operations

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi])
g = chebfun(lambda x: np.cos(x), [-np.pi, np.pi])

h = f + g       # addition
h = f - g       # subtraction
h = f * g       # multiplication
h = f / g       # division (where g != 0)
h = f ** 2      # power
h = -f          # negation
```

## Scalar Operations

```python
h = f + 1       # add a constant
h = 2 * f       # scalar multiplication
h = f / 3       # scalar division
```

## NumPy Universal Functions

Many NumPy ufuncs work with Chebfun objects:

```python
h = np.sin(f)
h = np.exp(f)
h = np.abs(f)
```

## Norms and Comparisons

```python
print(f.norm())          # L2 norm
print(np.max(f))         # maximum value
print(np.min(f))         # minimum value
```

## Tangential Contacts in Pointwise Max/Min

Pointwise `maximum` and `minimum` split at roots of the difference between
the two inputs.  If the curves only touch, the root is numerically
ill-conditioned and a rootfinder may report a close pair of roots around the
contact point.  The following example illustrates this case:

```python
import numpy as np
from chebpy import chebfun

x = chebfun("x", [-2, 3])
f = np.sin(3 * x)
g = -np.sin(x)

(f - g).roots()
# Some BLAS/FFT/platform combinations report roots close to:
# [-pi/2, -pi/2, 0, pi/2, pi/2]
```

The contacts at `+/- pi/2` do not change which branch is active, so
`f.maximum(g)` filters them as switch points and only splits at the true
crossing near `0`.  The script
`docs/examples/tangential_maximum.py` visualises both the historical
duplicate-root failure mode and the fixed result.

## References

- Z. Battles and L. N. Trefethen,
  [*An extension of MATLAB to continuous functions and operators*](https://epubs.siam.org/doi/10.1137/S1064827503430126),
  SIAM J. Sci. Comput., 25 (2004), pp. 1743–1770.
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
  [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
  Pafnuty Publications, 2014.
