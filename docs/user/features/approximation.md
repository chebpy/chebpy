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

## Chebyshev Points

Use `chebpts` to get the Chebyshev interpolation points and barycentric weights:

```python
from chebpy import chebpts

pts, wts = chebpts(16)              # 16 points on [-1, 1]
pts, wts = chebpts(16, [0, 3])      # 16 points on [0, 3]
```
