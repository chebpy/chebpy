# Feature Matrix: Chebfun ↔ ChebPy

This page is a side-by-side reference of MATLAB
[**Chebfun**](https://www.chebfun.org/) features and their status in
**ChebPy**. ChebPy adapts a focused subset of Chebfun's univariate
machinery to Python; many of Chebfun's larger subsystems (2D/3D, ODEs,
chebops, GUI, etc.) are out of scope.

Legend: ✅ supported · 🟡 partial / different shape · ❌ not implemented.

## Core representation

| Capability | Chebfun (MATLAB) | ChebPy | Notes |
| --- | --- | --- | --- |
| Smooth functions on $[-1, 1]$ via Chebyshev-T expansion | `chebtech2` | `Chebtech` | Chebyshev points of the second kind. |
| Functions on bounded $[a, b]$ | `bndfun` | `Bndfun` | Affine wrap of `Chebtech`. |
| Piecewise (multi-domain) functions | `chebfun` | `Chebfun` | Multi-piece domains supported. |
| Periodic (trigonometric) functions | `chebfun(..., 'trig')` / `trigtech` | `trigfun` / `Trigtech` | Fourier-based, equispaced grid. |
| Infinite / semi-infinite intervals | `chebfun(..., [-Inf, Inf])` | `CompactFun` | 🟡 ChebPy uses an explicit `CompactFun` wrapper rather than `unbndfun`; same compactification idea. |
| Endpoint singularities | `chebfun(..., 'blowup', 'on')` / `singfun` | `Singfun` | 🟡 ChebPy uses the Adcock–Richardson slit-strip map (no exponent prescription needed). |
| Quasimatrices | `chebfun` array (column quasimatrix) | `Quasimatrix` | Continuous columns; QR / SVD / `\`. |
| User preferences | `chebfunpref` | `UserPreferences` | Tolerance, max length, splitting. |

## Construction

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| Adaptive construction from a callable | `chebfun(@f)` | `chebfun(f)` | |
| Fixed-length construction | `chebfun(@f, n)` | `chebfun(f, n=n)` | |
| Construction from coefficients | `chebfun(coeffs, 'coeffs')` | `Chebtech.initcoeffs` | Lower-level entry point. |
| Construction on a domain | `chebfun(@f, [a, b])` | `chebfun(f, [a, b])` | |
| Splitting (automatic breakpoints) | `'splitting', 'on'` | breakpoints passed explicitly | 🟡 ChebPy currently expects breakpoints to be supplied by the user. |
| Identity / linear function | `chebfun('x')` | `Chebfun.initidentity` | |
| Constant function | `chebfun(c)` | `Chebfun.initconst` / `pwc` | `pwc` for piecewise constants. |
| Equispaced data input | `chebfun(values, 'equi')` | ❌ | Use `polyfit` from samples instead. |

## Arithmetic and elementary operations

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| `+`, `-`, `*`, `/`, `**` | ✅ | ✅ | Operator overloads on `Chebfun`. |
| Unary `-`, `abs`, `sign` | ✅ | ✅ | |
| `floor`, `ceil` | ✅ | ✅ | |
| `min(f, g)`, `max(f, g)` | ✅ | ✅ | `f.minimum(g)`, `f.maximum(g)`. |
| Composition `f(g)` | ✅ | ✅ | `__call__` accepts a `Chebfun`. |
| NumPy ufuncs (`sin`, `exp`, `log`, …) | n/a | ✅ | Via `np.<ufunc>(f)`. |
| Complex-valued functions | ✅ | ✅ | `f.real`, `f.imag`, `f.iscomplex`. |
| `restrict` to a sub-interval | ✅ | ✅ | |
| Translate (shift domain) | ✅ | ✅ | `f.translate(d)`. |
| Vector-valued (array-valued) chebfuns | ✅ | 🟡 | Use `Quasimatrix` for column collections. |

## Calculus

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| Definite integral | `sum(f)` | `f.sum()` | |
| Indefinite integral | `cumsum(f)` | `f.cumsum()` | |
| Derivative | `diff(f)` | `f.diff()` | Higher-order via repeated calls. |
| Norms ($L^2$, $L^1$, $L^\infty$) | `norm(f, p)` | `f.norm(p)` | |
| Inner product | `f' * g` | `f.dot(g)` | |
| Convolution | `conv(f, g)` | `f.conv(g)` | Hale–Townsend Legendre convolution. |
| Cross-correlation | `xcorr` | ❌ | |
| Antiderivative with constant | `cumsum(f) + c` | same | |

## Rootfinding and extrema

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| Real roots in domain | `roots(f)` | `f.roots()` | Colleague-matrix recursion. |
| Complex roots | `roots(f, 'complex')` | ❌ | |
| Maximum / minimum value & location | `max(f)`, `min(f)` | `f.maximum()`, `f.minimum()` | |
| `argmax` / `argmin` | ✅ | 🟡 | Returned together with the extreme value. |

## Plotting

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| Plot the function | `plot(f)` | `f.plot()` | matplotlib backend. |
| Plot Chebyshev coefficients | `plotcoeffs(f)` | `f.plotcoeffs()` | |
| Domain / breakpoint markers | ✅ | ✅ | |
| 3D / surface plots | `plot3`, `surf` | ❌ | (No 2D representation in ChebPy.) |

## Linear algebra (quasimatrices)

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| Quasimatrix construction | column array of chebfuns | `Quasimatrix([f1, f2, …])` | |
| QR factorisation | `qr(A)` | `Q, R = A.qr()` | Householder triangularisation. |
| SVD | `svd(A)` | `U, S, V = A.svd()` | |
| Polynomial least-squares fit | `polyfit` | `polyfit` | Top-level helper. |
| Backslash `A \ b` | ✅ | 🟡 | Use SVD/QR-based solve for now. |

## Gaussian process regression

| Capability | Chebfun | ChebPy | Notes |
| --- | --- | --- | --- |
| GP posterior as a function object | n/a | `gpr(...)` returns a `Chebfun` | ChebPy-only; no Chebfun analogue. |

## Out of scope (Chebfun features not in ChebPy)

| Chebfun feature | Status in ChebPy |
| --- | --- |
| Linear operators / `chebop` | ❌ |
| ODE BVP / IVP solvers (`solvebvp`, `bvp4c` glue) | ❌ |
| Time-dependent PDEs (`pde15s`, `spin`, `spin2`, `spin3`, `spinsphere`) | ❌ |
| Automatic differentiation | ❌ |
| `Chebgui` graphical interface | ❌ |
| 2D functions on rectangles (`Chebfun2`) | ❌ |
| 3D functions (`Chebfun3`) | ❌ |
| Spherical / polar geometries (`Spherefun`, `Diskfun`) | ❌ |
| Rational approximation (`aaa`, `minimax`, `remez`, `cf`, `chebpade`, `padeapprox`, `ratinterp`) | ❌ |
| Equispaced-data construction (`'equi'` flag) | ❌ |
| Carathéodory–Féjer (`cf`) | ❌ |
| `dirac` / distributional features | ❌ |
| Lebesgue functions / constants (`lebesgue`) | ❌ |
| Examples gallery (`cheb.gallery`) | ❌ |

## See also

- The [User Guide](user/intro.md) for narrative documentation of the
  features that *are* implemented.
- The [About](about.md) page for the history of Chebfun and the
  contributors whose work ChebPy adapts.
- The [API Reference](api.md) for the full Python API.
- [chebfun.org](https://www.chebfun.org/) for the canonical MATLAB
  Chebfun project.
