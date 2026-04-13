# Convolution

Convolve two Chebfun objects to produce a new Chebfun on the summed domain.

## Usage

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.exp(-x**2), [-1, 1])
g = chebfun(lambda x: np.where(np.abs(x) < 0.5, 1.0, 0.0), [-1, 1])

h = f.conv(g)   # h is a Chebfun on [-2, 2]
h.plot()
```

The convolution is defined as:

$$h(x) = \int f(t)\, g(x - t)\, dt$$

The domain of the result is $[a_f + a_g, \, b_f + b_g]$ where $[a_f, b_f]$ and
$[a_g, b_g]$ are the domains of `f` and `g` respectively.

## Implementation

Convolution is computed by converting to Legendre coefficients, exploiting the
linearisation property of Legendre polynomials, and converting back to Chebyshev
form.
