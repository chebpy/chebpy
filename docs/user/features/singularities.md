# Endpoint Singularities

ChebPy supports functions with branch-type singularities at one or both endpoints
of a bounded interval — square roots, fractional powers, weak logarithms — through
the [`Singfun`][chebpy.singfun.Singfun] class.

## Usage

Pass a `sing=` hint to `chebfun()` to flag which endpoint(s) carry a singularity:

```python
import numpy as np
from chebpy import chebfun

# Left endpoint: f(x) = sqrt(x) on [0, 1]
f = chebfun(np.sqrt, [0.0, 1.0], sing="left")

# Right endpoint: f(x) = sqrt(1 - x) on [0, 1]
g = chebfun(lambda x: np.sqrt(1.0 - x), [0.0, 1.0], sing="right")

# Both endpoints: f(x) = sqrt(x * (1 - x)) on [0, 1]
h = chebfun(lambda x: np.sqrt(x * (1.0 - x)), [0.0, 1.0], sing="both")

# Definite integrals reach machine precision
float(f.sum())   # 2/3
float(h.sum())   # pi/8
```

The optional `alpha` keyword controls the clustering strength of the underlying
exponential map (default `alpha=1.0`); larger values cluster nodes more aggressively
near the singular endpoint and trade resolution for fewer coefficients.

## How it works

A `Singfun` stores the function as a standard Chebyshev expansion in a *transformed*
variable $t \in [-1, 1]$, related to the logical variable $x \in [a, b]$ by a
non-affine map that clusters nodes super-exponentially near the singular endpoint.
For a left singularity ChebPy uses the Adcock–Richardson "single-slit" map

$$
m(t) = a + (b - a) \, \exp\!\bigl(-\alpha\,(1 - t)/(1 + t)\bigr),
$$

which exponentially compresses $t \approx -1$ onto $x \approx a$. The composition
$f(m(t))$ is analytic in a Bernstein ellipse around $[-1, 1]$ and is therefore
resolved to spectral accuracy by ordinary Chebyshev interpolation. A symmetric
"double-slit" map is used for two-sided singularities.

The recipe is from Adcock & Richardson, *A higher-order generalisation of the
Adcock–Hale recipe for endpoint singularities*, [arXiv:1305.2643](https://arxiv.org/abs/1305.2643).

## Multi-piece domains

Pass a domain with interior breakpoints to control where the singular pieces live:

```python
# Only the leftmost piece is a Singfun; the rest are ordinary Bndfuns.
p = chebfun(np.sqrt, [0.0, 0.3, 1.0], sing="left")
print([type(piece).__name__ for piece in p.funs])
# ['Singfun', 'Bndfun']
```

## Accuracy ceiling near non-zero clustered endpoints

Pointwise accuracy of a `Singfun` is bounded below by the floating-point spacing
of $x$ at the clustered samples:

$$
|f(x_{\text{exact}}) - f(x_{\text{float}})| \;\approx\; |f'(x)| \cdot \mathrm{ulp}(x).
$$

For $f(x) = \sqrt{x}$ clustered at $x = 0$, subnormals give effectively unbounded
relative resolution and the result reaches full machine precision. For
$f(x) = \sqrt{1 - x}$ clustered at $x = 1$, $\mathrm{ulp}(1) \approx 2.22 \times 10^{-16}$
limits pointwise accuracy to roughly $10^{-10}$. For weak singularities like
$(1-x)^{0.1}$ the floor is much higher and the adaptive constructor will fail to
converge.

**Integrals are unaffected.** The bad samples sit at $t \to \pm 1$ where the
Jacobian $m'(t) \to 0$ super-exponentially, suppressing the noise below machine
epsilon. So `f.sum()` reaches full precision even when pointwise evaluation does
not.

| `f(x)`           | `sing=`  | typical pointwise accuracy |
|------------------|----------|----------------------------|
| `sqrt(x)`        | `"left"`  | machine precision          |
| `sqrt(1-x)`      | `"right"` | $\sim 10^{-10}$              |
| `sqrt(x(1-x))`   | `"both"`  | $\sim 10^{-10}$              |
| `(1-x)**0.3`     | `"right"` | $\sim 10^{-9}$ (may not converge) |
| `(1-x)**0.1`     | `"right"` | $\sim 10^{-7}$ (may not converge) |

If you need high pointwise accuracy near $x = 1$ for a known weak singularity,
factor it out analytically before constructing the `Singfun`.

## Mixed-piece arithmetic

Operations between a `Singfun` piece and a regular `Bndfun` piece on the same
interval reconstruct the result on the singular representation:

```python
from chebpy.singfun import Singfun
from chebpy.bndfun import Bndfun
from chebpy.utilities import Interval

s = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left")
b = Bndfun.initfun_adaptive(lambda x: x * x, Interval(0.0, 1.0))
r = s + b   # type(r) is Singfun
```

The same logic applies to `Chebfun`-level arithmetic on multi-piece domains.

## What does not close: `conv`

The Hale–Townsend Legendre convolution algorithm, and the Gauss–Legendre
fallback, both assume an affine map between the logical and reference
variables. The Adcock–Richardson clustering map breaks this assumption, so
`conv()` refuses any `Chebfun` that contains a `Singfun` piece:

```python
f = chebfun(np.sqrt, [0.0, 1.0], sing="left")
g = chebfun(1.0, [0.0, 1.0])
f.conv(g)   # raises NotImplementedError
```

A future release may provide an opt-in helper that recasts the singular
pieces into a piecewise `Bndfun` representation (closing under `conv` at
the cost of accuracy near the endpoints).  Until then, `conv` simply
refuses.

## Restriction

`Singfun.restrict(subinterval)` decides automatically what representation to use:

| `subinterval` relationship                            | Result type |
|-------------------------------------------------------|-------------|
| equals `self.interval`                                | `self`        |
| shares the clustered endpoint                         | `Singfun`     |
| purely interior (excludes the clustered endpoint(s))  | `Bndfun`      |

A two-sided `Singfun` restricted to a half-interval becomes a one-sided `Singfun`
of the appropriate side.

## Differentiation is deferred

`Singfun.diff()` is **not implemented** in the current release: the chain rule
introduces a $1/m'(t)$ factor that blows up at the clustered endpoint, and the
single-slit map cannot resolve it without further work.

## See also

- The [Endpoint Singularities notebook](../../notebooks/10_endpoint_singularities.html)
  for a worked tour with plots.
- The [Infinite Intervals](infinite-intervals.md) page — the corresponding
  *non*-singular construction for unbounded domains.

## References

- B. Adcock and M. Richardson,
  [*A higher-order generalisation of the Adcock–Hale recipe for endpoint
  singularities*](https://arxiv.org/abs/1305.2643), 2013.
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
  [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
  Pafnuty Publications, 2014, ch. 9.
