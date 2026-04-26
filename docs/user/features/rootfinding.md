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
