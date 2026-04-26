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
