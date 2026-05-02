# Gaussian Process Regression

ChebPy implements Gaussian process regression (GPR) and returns the posterior
mean and variance as Chebfun objects — ready for differentiation, integration,
and root-finding.

## Usage

```python
import numpy as np
from chebpy import gpr

rng = np.random.default_rng(1)
x_obs = np.sort(-2 + 4 * rng.random(10))
y_obs = np.sin(np.exp(x_obs))

f_mean, f_var = gpr(x_obs, y_obs, domain=[-2, 2])
```

## Working with Results

Because the posterior is a Chebfun, all standard operations apply:

```python
f_mean.plot()                      # plot the posterior mean
extrema = f_mean.diff().roots()    # local extrema
integral = f_mean.sum()            # definite integral
```

## Options

```python
f_mean, f_var = gpr(
    x_obs, y_obs,
    domain=[-2, 2],
    sigma=1.0,            # signal standard deviation
    length_scale=0.5,     # kernel length scale
    noise=0.01,           # observation noise
)
```

Set `n_samples` to draw random realisations from the posterior:

```python
f_mean, f_var = gpr(x_obs, y_obs, domain=[-2, 2], n_samples=5)
# f_var is a Quasimatrix with 5 sample columns
```

## References

- C. E. Rasmussen and C. K. I. Williams,
  [*Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/),
  MIT Press, 2006.
- S. Filip, A. Javeed, and L. N. Trefethen,
  [*Smooth random functions, random ODEs, and Gaussian processes*](https://doi.org/10.1137/17M1161853),
  SIAM Review, 61 (2019), 185–205.
