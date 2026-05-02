# Endpoint Singularities

ChebPy supports functions with branch-type singularities at one or both endpoints
of a bounded interval — square roots, fractional powers, weak logarithms — through
the [`Singfun`][chebpy.singfun.Singfun] class.

## Usage

Pass a `sing=` hint to `chebfun()` to flag which endpoint(s) carry a singularity:

```python
import numpy as np
from chebpy import chebfun

# Left endpoint: f(x) = x log(x) on [0, 1] (derivative blows up at x = 0)
f = chebfun(lambda x: x * np.log(x), [0.0, 1.0], sing="left")

# Right endpoint: f(x) = sqrt(1 - x) on [0, 1]
g = chebfun(lambda x: np.sqrt(1.0 - x), [0.0, 1.0], sing="right")

# Both endpoints: f(x) = sqrt(x * (1 - x)) on [0, 1]
h = chebfun(lambda x: np.sqrt(x * (1.0 - x)), [0.0, 1.0], sing="both")

# Definite integrals reach machine precision
float(f.sum())   # -1/4 (up to a tiny endpoint-gap correction; see below)
float(h.sum())   # pi/8
```

The optional `params` keyword bundles the two map parameters as a frozen
[`MapParams`][chebpy.maps.MapParams] dataclass:

```python
from chebpy import MapParams, chebfun

f = chebfun(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(L=8.0, alpha=1.0))
```

- `alpha > 0` is the strip half-width of the underlying conformal map; smaller
  `alpha` clusters nodes more aggressively near the singular endpoint.
- `L > 0` truncates the (otherwise semi-/bi-infinite) paper map to a finite
  reference interval; with the default `L = 8.0` the image of $[-1, 1]$ falls
  short of the clustered endpoint(s) by a `gap < 1e-10` (invisible at working
  precision). Smaller `L` shrinks the resolved interval visibly but improves
  the convergence rate of the mapped Chebyshev expansion.

## How it works

A `Singfun` stores the function as a standard Chebyshev expansion in a *transformed*
variable $t \in [-1, 1]$, related to the logical variable $x \in [a, b]$ by a
non-affine map $m$ that clusters nodes super-exponentially near the singular
endpoint.

### One-sided: the slit-strip map $\varphi_S$

For a single endpoint singularity (`sing="left"` or `sing="right"`) ChebPy
uses the Adcock–Richardson semi-infinite slit-strip map $\varphi_S$.
Writing $s = L(t-1)/2 \in [-L, 0]$ for the affine pre-scaling and

$$
u_\alpha(s) = \frac{\alpha}{\pi} \log(1 + e^{\pi(s + \gamma)/\alpha}),
$$

with

$$
\gamma = \tfrac{\alpha}{\pi}\,\log(e^{\pi/\alpha} - 1),
$$

the forward map for `sing="left"` is

$$
m(t) = a + (b - a)\, u_\alpha(L(t - 1)/2),
$$

and for `sing="right"` it is the reflection
$m(t) = b - (b - a)\, u_\alpha(L(-t - 1)/2)$.
The shift $\gamma$ is chosen so that $u_\alpha(0) = 1$, i.e. the smooth
endpoint maps exactly to $b$ (resp. $a$); the truncation $L$ leaves a small
`gap = (b-a) * u_α(-L)` near the clustered endpoint, which shrinks
super-exponentially with $L$.

### Two-sided: the two-slit-strip map $\psi_S$

For `sing="both"` ChebPy uses the paper's infinite two-slit-strip map
$\psi_S$. With $s = L\,t \in [-L, L]$,

$$
v_\alpha(s) = \frac{\alpha}{\pi} \left[ \log(1 + e^{\pi(s + 1/2)/\alpha}) - \log(1 + e^{\pi(s - 1/2)/\alpha}) \right],
$$

so that $v_\alpha(0) = 1/2$ and $v_\alpha(\pm\infty) = (1\pm 1)/2$. The
forward map is

$$
m(t) = a + (b - a)\, v_\alpha(L\,t),
$$

clustering at both endpoints simultaneously and leaving a (symmetric)
`gap` at each end.

### Why the map works

The composition $f \circ m$ is analytic in a Bernstein ellipse around
$[-1, 1]$ and is resolved to spectral accuracy by ordinary Chebyshev
interpolation. The recipe is from Adcock & Richardson, *New exponential
variable transform methods for functions with endpoint singularities*,
[SIAM J. Numer. Anal. 52(4), 1887–1912 (2014)](https://doi.org/10.1137/130920460)
([arXiv:1305.2643](https://arxiv.org/abs/1305.2643)).

!!! note "Historical note and a mnemonic"
    The basic construction of the one-sided slit-strip map $\varphi_S$
    was suggested to the authors by Nick Trefethen, who — when reviewing a
    draft of the paper — also offered a useful mnemonic for telling the two
    maps apart: $\varphi_S$ is *one*-sided because it has *one* tip (a
    single slit), while $\psi_S$ is *two*-sided because it has *two* tips
    (a pair of slits).

## Multi-piece domains

Pass a domain with interior breakpoints to control where the singular pieces live:

```python
# Only the leftmost piece is a Singfun; the rest are ordinary Bndfuns.
p = chebfun(np.sqrt, [0.0, 0.3, 1.0], sing="left")
print([type(piece).__name__ for piece in p.funs])
# ['Singfun', 'Bndfun']
```

## The endpoint gap

With finite `L` the image of `formap([-1, 1])` is a sub-interval of `[a, b]`
that falls short of the clustered endpoint(s) by
`map.gap = (b - a) * map.gap_unit`. With the default `L = 8.0`, `alpha = 1.0`
this shortfall is below `1e-10` and is invisible at working precision; the
integral computed by `f.sum()` is therefore $c\cdot(b-a-\text{gap})$ rather
than $c\cdot(b-a)$ for a constant $c$. Increasing `L` shrinks the gap further;
decreasing `L` (closer to the paper's empirical optimum $L\sim 1$) widens it
in exchange for a faster-converging series.

## Tuning `MapParams(L, alpha)`

The two parameters control three quantities — coefficient count, pointwise
accuracy near the clustered endpoint, and the size of the unresolved `gap`.
Defaults `L = 8.0`, `alpha = 1.0` are a safe choice for the canonical
algebraic singularities $x^p$ with $p \in [1/4, 1]$.

### `alpha` — clustering strength

Smaller $\alpha$ packs more nodes into the immediate neighbourhood of the
singular endpoint; larger $\alpha$ relaxes the clustering.

| `alpha` | Best for                                                         |
|---------|------------------------------------------------------------------|
| `~0.5`  | Strong singularities $f \sim (x-a)^p$ with $p \lesssim 0.25$, weak logs |
| `1.0` (default) | Square roots and most algebraic singularities; matches the paper's empirical optimum |
| `~2.0`  | Mild singularities such as $(x-a)^{0.8}$, or smooth functions where you only want a hint of clustering |

### `L` — truncation length

`L` controls how far into the paper's semi-/bi-infinite strip ChebPy
samples, and equivalently the `gap` left at the clustered endpoint.

| `L`     | `gap` (at `alpha=1`) | Use case                                       |
|---------|----------------------|------------------------------------------------|
| `1`–`2` | $10^{-2}$–$10^{-1}$  | Paper's optimum for fastest coefficient decay; integrals / interior values only |
| `8.0` (default) | $< 10^{-10}$ | Indistinguishable from a closed map at working precision |
| `16`–`20` | $< 10^{-20}$       | Vanishing gap; useful when evaluating *at* the clustered endpoint matters |

### Rules of thumb

- **Adaptive constructor fails to converge?** Decrease `alpha` first
  (try `0.5`), then increase `L`.
- **Want fewer coefficients and only need integrals?** Decrease `L`
  toward `1`–`2`. Coefficient decay improves noticeably; the gap is
  visible but `f.sum()` still benefits from the analytic integrand.
- **Need pointwise accuracy at the clustered endpoint?** Increase `L`,
  but check that float64 ulp at the clustered $x$ isn't already the
  dominant error (see the next section).

The table below is reproduced in the [Endpoint Singularities
notebook](../../notebooks/10_endpoint_singularities.html) and shows the
trade-off concretely for $f(x) = \sqrt{x}$ on $[0, 1]$ (sweep on a 401-point
interior grid):

| `alpha` | `L`  | size  | `gap`        | max interior err |
|---------|------|-------|--------------|------------------|
| 0.5     | 1    | 31    | $1.1\!\times\!10^{-1}$ | $3.0\!\times\!10^{-1}$ |
| 0.5     | 8    | 173   | $1.3\!\times\!10^{-20}$ | machine ε       |
| 1.0     | 1    | 22    | $2.1\!\times\!10^{-1}$ | $4.3\!\times\!10^{-1}$ |
| 1.0     | 8    | 96    | $8.6\!\times\!10^{-11}$ | machine ε       |
| 2.0     | 8    | 55    | $8.5\!\times\!10^{-6}$ | machine ε       |
| 2.0     | 16   | 80    | $3.0\!\times\!10^{-11}$ | machine ε       |

Note how $\alpha = 2$ at $L = 8$ resolves $\sqrt{x}$ in *fewer* coefficients
(55 vs. 96) than the default; that is the right setting if you have a
collection of mild singularities and care about size. Conversely, the strong
case $\alpha = 0.5$ pays roughly $2\times$ in coefficients but is the safer
choice for $(x-a)^{0.1}$.

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
variables. The slit-strip clustering map breaks this assumption, so
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
  *New exponential variable transform methods for functions with endpoint
  singularities*, [SIAM Journal on Numerical Analysis 52(4),
  pp. 1887–1912 (2014)](https://doi.org/10.1137/130920460);
  [arXiv:1305.2643](https://arxiv.org/abs/1305.2643).
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
  [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
  Pafnuty Publications, 2014, ch. 9.
