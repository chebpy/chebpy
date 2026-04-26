# Calculus

ChebPy provides differentiation, integration, and cumulative sums with
spectral accuracy.

## Differentiation

```python
import numpy as np
from chebpy import chebfun

f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi])
df = f.diff()       # first derivative
d2f = f.diff(2)     # second derivative
```

Differentiation is performed on the Chebyshev coefficients in $O(n)$ time.

## Definite Integration

```python
integral = f.sum()  # ∫ f(x) dx over the domain
```

Integration uses Clenshaw–Curtis quadrature and is accurate to machine precision.

## Indefinite Integration (Cumulative Sum)

```python
F = f.cumsum()      # F(x) = ∫_{a}^{x} f(t) dt
```

The result is a new Chebfun whose degree is one higher than the input.

## Example: Verify the Fundamental Theorem

```python
f = chebfun(lambda x: np.exp(x), [-1, 1])
F = f.cumsum()
df = F.diff()

# df should agree with f to machine precision
print(np.max(np.abs(f(np.linspace(-1, 1, 100)) - df(np.linspace(-1, 1, 100)))))
```
