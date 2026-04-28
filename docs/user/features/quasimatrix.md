# Quasimatrices

A quasimatrix is an $\infty \times n$ matrix whose columns are Chebfun objects
defined on the same domain. This enables continuous analogues of linear algebra.

## Creating a Quasimatrix

```python
import numpy as np
from chebpy import Quasimatrix, chebfun

x = chebfun('x')
A = Quasimatrix([1, x, x**2, x**3, x**4, x**5])
```

## QR Factorisation

The QR factorisation of a quasimatrix produces orthonormal columns (analogous
to Legendre polynomials for the monomial basis):

```python
Q, R = A.qr()
```

## SVD

```python
U, S, V = A.svd()
```

## Least-Squares Approximation

Solve $A c \approx f$ in the least-squares sense:

```python
f = chebfun(lambda t: np.exp(t) * np.sin(6 * t), [-1, 1])
c = A.solve(f)
f_approx = A @ c   # reconstruct as a Chebfun
```

## Polynomial Fitting

The `polyfit` convenience function fits a polynomial of given degree:

```python
from chebpy import polyfit

f = chebfun(lambda x: np.exp(x), [-1, 1])
p = polyfit(f, 5)   # degree-5 polynomial least-squares fit
```

## References

- L. N. Trefethen,
  [*Householder triangularization of a quasimatrix*](https://academic.oup.com/imajna/article/30/4/887/659727),
  IMA J. Numer. Anal., 30 (2010), pp. 887–897.
- Z. Battles and L. N. Trefethen,
  [*An extension of MATLAB to continuous functions and operators*](https://epubs.siam.org/doi/10.1137/S1064827503430126),
  SIAM J. Sci. Comput., 25 (2004), pp. 1743–1770.
- A. Townsend and L. N. Trefethen,
  [*Continuous analogues of matrix factorizations*](https://royalsocietypublishing.org/doi/10.1098/rspa.2014.0585),
  Proc. Roy. Soc. A, 471 (2015), 20140585.
