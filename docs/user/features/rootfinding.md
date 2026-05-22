# Root-Finding

ChebPy finds all roots of a function in its domain using a recursive
subdivision algorithm based on the colleague matrix.

## Finding Roots

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.cos(10 * x), [-1, 1])
roots = f.roots()
print(roots)   # all zeros of cos(10x) in [-1, 1]
```

## Finding Extrema

Local extrema are the roots of the derivative:

```python
f = chebfun(lambda x: np.sin(x**2), [0, 5])
extrema = f.diff().roots()
```

## Intersection of Two Functions

Find where two functions meet by taking roots of their difference:

```python
f = chebfun(lambda x: np.sin(x), [0, 10])
g = chebfun(lambda x: np.cos(x), [0, 10])
crossings = (f - g).roots()
```

## Options

By default, roots are not sorted. Use `UserPreferences` to control behaviour:

```python
from chebpy import UserPreferences

prefs = UserPreferences()
prefs.sortroots = True
```

## References

- I. J. Good,
  [*The colleague matrix, a Chebyshev analogue of the companion matrix*](https://academic.oup.com/qjmath/article/12/1/61/1573665),
  Quart. J. Math. Oxford, 12 (1961), pp. 61–68.
- J. P. Boyd,
  [*Computing zeros on a real interval through Chebyshev expansion and polynomial rootfinding*](https://epubs.siam.org/doi/10.1137/S0036142901398325),
  SIAM J. Numer. Anal., 40 (2002), pp. 1666–1682.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013 (extended edition 2019), ch. 18.
