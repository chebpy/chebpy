# Function Approximation

ChebPy automatically approximates smooth functions with Chebyshev polynomials to
machine precision.

## Adaptive Construction

Pass any callable to `chebfun` and ChebPy determines the optimal polynomial degree:

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.exp(np.sin(x)), [-5, 5])
print(len(f))  # polynomial degree chosen automatically
```

## Fixed-Length Construction

Specify the number of points explicitly with the `n` parameter:

```python
f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi], n=32)
```

## Special Constructors

```python
# Identity function
x = chebfun('x')

# Constant function
c = chebfun(3.14)

# Piecewise-constant function
from chebpy import pwc
f = pwc(domain=[-2, -1, 0, 1, 2], values=[-1, 0, 1, 2])
```

## Multi-Interval Functions

ChebPy can represent functions with breakpoints as piecewise Chebyshev expansions:

```python
f = chebfun(lambda x: np.abs(x), [-1, 0, 1])
```

## Accuracy and Preferences

Adaptive construction samples the function on Chebyshev grids of length
`2**k + 1`.  It converts the samples to Chebyshev coefficients and stops when
the coefficient tail can be chopped below a tolerance.  The default tolerance is
roughly machine epsilon:

```python
from chebpy import UserPreferences

prefs = UserPreferences()
prefs.eps       # default: numpy.finfo(float).eps
prefs.maxpow2   # default: 16, so the largest grid has 65537 points
```

Use the preferences object to change these defaults:

```python
from chebpy import UserPreferences

prefs = UserPreferences()
prefs.eps = 1e-12
prefs.maxpow2 = 17

# Restore one setting, or all settings:
prefs.reset("eps")
prefs.reset()
```

Preferences are global.  For temporary changes, use the object as a context
manager:

```python
from chebpy import UserPreferences, chebfun

prefs = UserPreferences()

with prefs as local_prefs:
    local_prefs.eps = 1e-12
    f = chebfun(lambda x: x**2)

# eps is restored here
```

`eps` is a construction tolerance, not a certified maximum error.  It controls
the coefficient chopping test used during construction, but the final pointwise
error also depends on smoothness, conditioning, floating-point roundoff, and
whether the function is resolved on the chosen interval.  In practice, validate
sensitive approximations against extra sample points:

```python
import numpy as np
from chebpy import chebfun

def g(x):
    return 0.3 + 0.02*x + abs(x)**1.8

f = chebfun(g, [-1, 1])
x_test = np.linspace(-1, 1, 1001)
err_est = np.max(np.abs(g(x_test) - f(x_test)))
```

The example above is deliberately difficult at `x = 0`: `abs(x)**1.8` is not
twice differentiable there.  A single global polynomial therefore converges only
algebraically, and its largest interpolation error can occur near the cusp even
though the function is continuous.  If you know the location of the nonsmooth
point, add it as a breakpoint:

```python
f = chebfun(g, [-1, 0, 1])
```

Breakpoints let ChebPy approximate each smooth piece separately, which is
usually a better remedy than forcing a larger global polynomial.

If construction reaches `maxpow2`, ChebPy emits a warning similar to:

```text
The Chebtech constructor did not converge: using 65537 points. The
representation may be under-resolved; consider increasing
UserPreferences().maxpow2 or adding breakpoints.
```

Treat this as a resolution warning.  The returned object contains the largest
sampled interpolant, but ChebPy has not seen coefficient decay strong enough to
declare it resolved.  For a smooth but highly oscillatory function, increasing
`maxpow2` may be appropriate.  For a nonsmooth function, add breakpoints at
known kinks, jumps, or singularities.  For noisy or discontinuous data, a
polynomial interpolant may not be the right model.

## Chebyshev Points

Use `chebpts` to get the Chebyshev interpolation points and barycentric weights:

```python
from chebpy import chebpts

pts, wts = chebpts(16)              # 16 points on [-1, 1]
pts, wts = chebpts(16, [0, 3])      # 16 points on [0, 3]
```

## References

- Z. Battles and L. N. Trefethen,
  [*An extension of MATLAB to continuous functions and operators*](https://epubs.siam.org/doi/10.1137/S1064827503430126),
  SIAM J. Sci. Comput., 25 (2004), pp. 1743–1770.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013 (extended edition 2019).
- J. L. Aurentz and L. N. Trefethen,
  [*Chopping a Chebyshev series*](https://dl.acm.org/doi/10.1145/2998442),
  ACM Trans. Math. Softw., 43 (2017), Article 33.
- R. Pachón, R. B. Platte, and L. N. Trefethen,
  [*Piecewise-smooth chebfuns*](https://academic.oup.com/imajna/article/30/4/898/659725),
  IMA J. Numer. Anal., 30 (2010), pp. 898–916.
