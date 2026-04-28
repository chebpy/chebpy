# Plotting

ChebPy integrates with matplotlib for function visualisation.

## Basic Plotting

```python
import numpy as np
import matplotlib.pyplot as plt
from chebpy import chebfun

f = chebfun(lambda x: np.sin(10 * x) * np.exp(-x), [-1, 1])
f.plot()
plt.show()
```

## Plotting on Axes

Pass an existing `Axes` object to overlay multiple functions:

```python
fig, ax = plt.subplots()
f = chebfun(lambda x: np.sin(x), [0, 10])
g = chebfun(lambda x: np.cos(x), [0, 10])

f.plot(ax=ax, label='sin(x)')
g.plot(ax=ax, label='cos(x)')
ax.legend()
plt.show()
```

## Marking Roots

```python
f = chebfun(lambda x: np.sin(x**2) + np.sin(x)**2, [0, 10])
g = chebfun(lambda x: np.exp(-(x - 5)**2 / 10), [0, 10])

ax = f.plot(label='f')
g.plot(ax=ax, label='g')

roots = (f - g).roots()
ax.plot(roots, f(roots), 'ro', markersize=8, label='Intersections')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## References

- J. D. Hunter,
  [*Matplotlib: A 2D Graphics Environment*](https://doi.org/10.1109/MCSE.2007.55),
  Computing in Science & Engineering, 9 (2007), pp. 90–95.
