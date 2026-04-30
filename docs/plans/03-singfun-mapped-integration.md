# Plan: Singfun-via-Mapped-Representation — Endpoint Singularities on `[a, b]`

> **Status (April 2026):** Phases 1–5 v1 are complete. The implemented
> maps follow the paper faithfully:
> [`SingleSlitMap`][chebpy.maps.SingleSlitMap] is the semi-infinite
> slit-strip map $\varphi_S$ and
> [`DoubleSlitMap`][chebpy.maps.DoubleSlitMap] is the infinite two-slit
> strip map $\psi_S$. Both are parameterised by a single
> [`MapParams(L, alpha)`][chebpy.maps.MapParams] dataclass; with the
> default `L = 8.0` the truncation gap at the clustered endpoint is
> below `1e-10`. The placeholder map names `SlitMap` / `DslitMap` and
> the `alpha`/`beta` parameter pair used below are superseded by these.
>
> Reference: Adcock & Richardson, *New exponential variable transform methods
> for functions with endpoint singularities*, SIAM J. Numer. Anal. 52(4),
> 1887–1912, 2014, doi:10.1137/130920460; arXiv:1305.2643. We adopt the
> exponential / double-exponential transforms `m: [-1, 1] → [a, b]` that
> cluster grid points towards the endpoints, so that a function with
> branch-point endpoint behaviour (e.g. `√x` on `[0, 1]`, `log(1−x)` on
> `[−1, 1]`) becomes analytic in the new variable and is resolved by a
> standard `Chebtech` (or, in the periodic case, `Trigtech`) on `[−1, 1]`.

## Summary

Introduce `Singfun`, a new `Classicfun` subclass whose mapping from the
storage variable `y ∈ [−1, 1]` to the logical variable `x ∈ [a, b]` is a
**non-affine, endpoint-clustering conformal map** rather than the affine
`Interval` map used by `Bndfun`. The `Onefun` payload (a standard
`Chebtech` / `Trigtech`) is unchanged. Closed operations: evaluation,
plotting, addition / multiplication / division (via reconstruction),
differentiation, definite and indefinite integration, rootfinding.
Operations that **do not close** in the mapped representation —
convolution, composition with a non-monotone outer function, and
`restrict` to interior subintervals — are explicitly handled by
falling back to a piecewise / standard `Bndfun` representation, or by
raising a clear `NotImplementedError` with a documented escape hatch.

This is the natural follow-on to plan 02 (`CompactFun` for unbounded
intervals): both deliberately depart from MATLAB Chebfun's `singfun` /
`unbndfun` rational-map machinery in favour of a representation that
slots cleanly into the existing `Classicfun → Onefun` plumbing.

## What we are NOT doing in this PR

1. **Algebraic-singularity `'exps'` mode** (the MATLAB Chebfun route, where
   `f(x) = (x−a)^α (b−x)^β · g(x)` with `g` smooth and `α, β` stored as
   metadata). That representation closes under more operations but
   requires bespoke arithmetic. Possible future addition.
2. **Interior singularities.** Only endpoint singularities at `a` and/or `b`
   are in scope. Interior singularities should be handled by splitting the
   `Domain` so they fall on a breakpoint.
3. **Convolution between two `Singfun`s, or `Singfun ⋆ Bndfun`.** Refused
   in v1; falls back to adaptive reconstruction as a `Chebfun` of `Bndfun`
   pieces (see "Closure failures" below).
4. **Mapped `Trigtech` payloads with a non-affine `m`.** Periodicity in
   `y` does not generally imply periodicity in `x` under a non-affine
   map; for v1 the only `Trigtech`-backed `Singfun` is the trivial affine
   degenerate case (i.e. equivalent to `Bndfun`).
5. **Heavy / fat-tailed densities on unbounded intervals.** Out of scope;
   that would combine plan 02 and this plan.

## Background

For `f` analytic on `(a, b)` with branch-type behaviour at the endpoints,
the affine map `t ↦ x = ½(a+b) + ½(b−a)t` does **not** make `f`
analytic in `t`: the branch singularity remains, just shifted to
`t = ±1`, and Chebyshev coefficients decay only algebraically.

The exponential / double-exponential transforms of Adcock–Richardson
introduce a smooth bijection `m: (−1, 1) → (a, b)` whose derivative
`m'(±1)` vanishes at a controlled rate. The composition `f ∘ m` is then
analytic in a Bernstein ellipse around `[−1, 1]`, and standard Chebyshev
interpolation in `t` recovers root-exponential or full geometric
convergence depending on the chosen map and parameters.

The two map families from the paper we will support:

| Tag        | Map (informal)                                     | Use case                          |
|------------|----------------------------------------------------|------------------------------------|
| `slit`     | Single exponential clustering at one endpoint      | `√x` at `x=0`, smooth elsewhere    |
| `dslit`    | Double exponential clustering at both endpoints    | `√(x(1−x))`, `log(1−x²)`, etc.     |

(Names are placeholders; the paper uses different conventions. Final
naming TBD during implementation.)

## Design

### Class hierarchy

```
Fun (abstract)
└── Classicfun (abstract — affine OR non-affine map + Onefun)
    ├── Bndfun        # affine map, finite [a,b]
    ├── CompactFun    # affine map, finite storage, unbounded logical
    └── Singfun       # NON-affine map, finite [a,b]   ← new
```

### Map abstraction

Today `Classicfun` stores `self._interval: Interval`, and the affine map
lives on `Interval` as `formap` / `invmap` / `drvmap`. To introduce a
non-affine map without disturbing `Bndfun` / `CompactFun`, factor the
map interface out of `Interval`:

1. New protocol `IntervalMap` (in `src/chebpy/utilities.py`) with methods
   `formap(y)`, `invmap(x)`, `drvmap(y)`, plus metadata `support` (a
   `(float, float)` tuple) and a `kind: str` tag for repr / dispatch.
2. `Interval` becomes the canonical affine implementation of
   `IntervalMap`. Existing call sites — which all do
   `interval.formap / .invmap / .drvmap` — keep working unchanged.
3. New concrete classes `SlitMap(a, b, *, side, alpha)` and
   `DslitMap(a, b, *, alpha, beta)` implement `IntervalMap` with the
   exponential transforms. They carry the parameters needed to
   reconstruct `formap` / `invmap` / `drvmap` analytically.
4. `Classicfun.__init__` is widened to accept any `IntervalMap`. It
   already only uses the three mapping methods.

### `Singfun` class

`src/chebpy/singfun.py`:

- Inherits from `Classicfun`.
- Stores a `Slit`/`Dslit` map alongside the `Onefun`.
- `__call__(x)` is unchanged from `Classicfun`: `y = map.invmap(x); return onefun(y)`.
- `support` returns `(a, b)` (finite). Logical and storage intervals
  coincide; the non-trivial part is the **map**, not the support.
- `_rebuild(onefun)` preserves the map (mirrors the `CompactFun`
  override pattern).

### Adaptive constructor

`Singfun.initfun_adaptive(f, interval, *, sing=...)`:

- `sing` is one of `"none"`, `"left"`, `"right"`, `"both"`, or a
  parameter tuple. If omitted, attempt automatic detection by sampling
  `f` near the endpoints and fitting the local exponent (off by
  default; opt-in via `prefs.detect_endpoint_singularities`).
- Builds the appropriate `IntervalMap`, then calls the standard
  `Chebtech.initfun(lambda y: f(map.formap(y)), …)` adaptive loop
  unchanged.

The adaptive resolution loop in `Chebtech` is **untouched**; all the
new code does is choose a different `formap`.

### Calculus

- `cumsum(f) = ∫_a^x f`: substitute `dx = m'(t) dt`, build a `Chebtech`
  from `(f ∘ m) · m'` on `[−1, 1]`, antidifferentiate it there, and wrap
  the result in a fresh `Singfun` with the **same** map. (`m'` may be
  zero at `±1` but the product is integrable by construction; this is
  precisely the Adcock–Richardson convergence guarantee.)
- `sum(f) = cumsum(f)(b)`. Computed in `t`-space via
  `Chebtech.sum` after weighting by `m'`.
- `diff(f)`: `f'(x) = (f ∘ m)'(t) / m'(t)` is the natural chain-rule
  formula, but `m'(±1) → 0` makes this numerically singular at the
  endpoints — which is **correct** behaviour for a function with an
  endpoint branch singularity. Implementation: compute the `Chebtech`
  derivative in `t` and store the result in a new `Singfun` whose map
  is the same; treat the `1/m'` factor as deferred (i.e. it appears in
  evaluation, not in the stored coefficients). Spelled out in
  the implementation notes below.

### Roots

Inherit from `Classicfun`: roots in `t`, push back through `formap`.
Endpoint clustering of the map turns multiple-root behaviour at the
endpoint into a benign zero-coefficient sequence in `t`-space, so
existing rootfinding works.

### Arithmetic

- `Singfun + Singfun` with **the same map**: add the underlying
  `Chebtech` coefficients directly. No reconstruction.
- `Singfun + Singfun` with different maps, or `Singfun + Bndfun`: pick
  the more singular operand's map (or the most refined one if both are
  singular at the same endpoint), then reconstruct the sum adaptively
  in that map.
- Multiplication / division: same rule. Division by something vanishing
  at an endpoint may legitimately introduce a new singularity; if the
  result fails to resolve in the chosen map, raise
  `BadFunctionLengthWarning` as the standard adaptive loop does today.

## Closure failures and fallbacks

The headline question from the analysis: **what doesn't close?**

| Operation                   | Closes in `Singfun`? | v1 behaviour                                    |
|-----------------------------|----------------------|--------------------------------------------------|
| eval, plot, abs, max/min    | yes                  | inherit                                          |
| `+`, `−`, `×`, `÷`          | yes (via rebuild)    | reconstruct on chosen map                        |
| `diff`                      | yes                  | implicit `1/m'` factor                           |
| `cumsum`, `sum`             | yes                  | weighted Chebtech antiderivative in `t`          |
| `roots`                     | yes                  | inherit                                          |
| `restrict([c, d])`, interior| **no**               | return `Bndfun`/standard chebfun on `[c, d]`     |
| `restrict([a, c])` boundary | yes                  | new `Singfun` with rescaled map                  |
| composition `g(f(x))`       | **no**               | reconstruct adaptively; result is generic `Fun`  |
| `conv(f, g)` involving any  | **no**               | refuse with `NotImplementedError("singular ⋆")` |
| Singfun                     |                      | + suggest using `chebfun(f).conv(g)` after       |
|                             |                      | adaptive recasting                               |

The `restrict` and `compose` fallbacks are not error paths — they
return a perfectly usable `Chebfun` built by adaptive reconstruction,
and the user almost never needs to know which map was used.

The `conv` refusal is the one user-visible limitation. The current
release simply raises `NotImplementedError`; an opt-in
`chebpy.recast(f, target="bndfun")` helper that forces piecewise
reconstruction (closing under `conv` at the cost of accuracy near the
endpoints) is **deferred to a follow-up PR**.

## Accuracy ceiling: ulp-limited sampling near singularities

Once Phase 3 was working we observed an apparent accuracy plateau on
two-sided `sqrt(x*(1-x))` and on weak singularities `(1-x)**p` with
`p < 0.5`. This is **not** a defect of the map or the Chebtech — it is
a fundamental floor coming from float64 spacing of `x` near the
clustered endpoint.

**Mechanism.** The adaptive constructor builds a Chebtech of the
composition `F(t) = f(m(t))`. As `t -> +/-1`, `m(t) -> b` (or `a`)
super-exponentially. The user's `f` is evaluated at `x = m(t)`, but
near `x = b != 0` the floating-point spacing is `ulp(b) ~ 2^-52 * |b|`.
Every sample of `f(x)` therefore carries an absolute error bounded
below by

$$
|f(x_{\text{exact}}) - f(x_{\text{float}})| \approx |f'(x)| \cdot \mathrm{ulp}(x).
$$

For `f(x) = (1-x)**p` clustered near `x=1` this gives

$$
\mathrm{abs\ err} \sim p \, (1-x)^{p-1} \cdot 2^{-52},
$$

which **diverges** as `1-x -> 0` whenever `p < 1`. The Chebtech faithfully
represents the noisy samples, so `adaptive` cannot push the tail below
this floor and either plateaus or fails to converge at all.

Empirically (alpha = 1, sing = "right" or "both", grid avoiding 1e-3 of
the boundary):

| `f(x)`           | size  | max pointwise err |
|------------------|-------|-------------------|
| `sqrt(x)`        | 155   | 3.3e-16 *         |
| `sqrt(1-x)`      |  76   | 5.6e-11           |
| `sqrt(x*(1-x))`  | 249   | 3.9e-11           |
| `(1-x)**0.3`     | 65537 | 6.8e-10 (no conv) |
| `(1-x)**0.1`     | 65537 | 7.6e-7  (no conv) |
| `(1-x)**0.01`    | 65537 | 1.8e-5  (no conv) |

\* `sqrt(x)` clustered at `x=0` is the lucky case: subnormals give
effectively unbounded relative resolution, so `ulp(x)/sqrt(x)` stays
small.

**Integrals are unaffected.** The bad samples sit at `t -> +/-1` where
`m'(t) -> 0` super-exponentially. The integrand `f(m(t)) * m'(t)`
multiplies the noisy bits by something below `eps`, so `sum()` reaches
machine precision even when pointwise evaluation does not. This is why
`test_sum_two_sided` legitimately uses `atol = 1e-13` while
`test_initfun_adaptive_resolves_two_sided_singularity` uses
`atol = 1e-10`.

**Implications for the user docs (Phase 5).**

1. Document the floor: pointwise accuracy is `~ |f'(x)| * ulp(x)` at
   the clustered samples, not `eps`.
2. Note that the recipe is well suited to integrals, moments, roots,
   and plots, but a user wanting machine-precision pointwise values
   near a non-zero clustered endpoint must supply `f` already pulled
   back to `t`-space (a future `initfun_in_tspace(f_t, ...)`
   constructor would expose this directly).
3. Warn explicitly that `(1-x)**p` for small `p` will fail to resolve;
   suggest factoring out the singularity analytically when possible.

## Scope

| Area | Detail |
|------|--------|
| **New module** | `src/chebpy/singfun.py` — `Singfun(Classicfun)` |
| **New module** | `src/chebpy/maps.py` — `IntervalMap` protocol, `SlitMap`, `DslitMap` (or co-locate in `utilities.py`) |
| **New tests** | `tests/test_singfun.py` parallel of `tests/test_bndfun.py` |
| **New tests** | `tests/test_maps.py` — round-trip `formap ∘ invmap = id`, `drvmap` correctness, parameter sweeps |
| **New user API** | `chebfun(f, [a, b], sing="left"|"right"|"both")` returns a `Singfun`-backed `Chebfun` |
| **New user API** | `chebpy.recast(f, target=...)` for opt-in conversion to standard representation **(deferred to follow-up PR)** |
| **Modified — utilities** | Refactor `Interval` to implement a new `IntervalMap` protocol; no behaviour change for existing callers |
| **Modified — classicfun** | Type-relax `_interval` annotation to `IntervalMap`; verify no code path assumes affine |
| **Modified — chebfun** | Detect singular pieces in `conv`; raise the documented error |
| **Modified — settings** | `prefs.detect_endpoint_singularities` (default `False`); `prefs.singfun_default_alpha` |
| **Modified — exports** | `src/chebpy/__init__.py` — export `Singfun`, map classes |
| **Modified — docs** | New `docs/user/features/singularities.md`; update `docs/notebooks/` with a sqrt-on-[0,1] example |

## Implementation phases

### Phase 1 — map refactor (no behaviour change)

1. Introduce `IntervalMap` protocol; make `Interval` an explicit
   implementer.
2. Add typing throughout `Classicfun` so `self._interval: IntervalMap`.
3. Run the full existing test suite; nothing should change.

**Exit criterion:** all existing tests pass unchanged; `Bndfun` and
`CompactFun` use the protocol without code changes.

### Phase 2 — map implementations

1. Implement `SlitMap` and `DslitMap` per the paper, with parameters
   `alpha`, `beta`.
2. Unit tests: round-trip identities, derivative correctness via finite
   differences, monotonicity, endpoint limits.

**Exit criterion:** `tests/test_maps.py` green; coverage of the
parameter space documented.

### Phase 3 — `Singfun` class

1. `Singfun(Classicfun)` with constructors mirroring `Bndfun`.
2. Reuse all of `Classicfun`'s `__call__`, `_rebuild`, arithmetic.
3. Adaptive constructor wraps `Chebtech.initfun` with a mapped target.

**Exit criterion:** `Singfun(lambda x: np.sqrt(x), [0, 1], sing="left")`
resolves to ≤ ~30 coefficients at `eps`; `sum`, `diff`, `cumsum`,
`roots` all match analytic answers within tolerance.

### Phase 4 — `Chebfun`-level integration

1. `chebfun()` accepts the `sing=` kwarg.
2. Mixed-piece arithmetic between `Bndfun` and `Singfun` reconstructs
   in the more singular map.
3. `conv` detects `Singfun` operands and raises with a helpful message.
4. `restrict` to interior subintervals returns a standard `Bndfun`.

**Exit criterion:** the integration tests in `tests/test_chebfun.py`
pick up new mixed-piece cases; user-guide examples run.

### Phase 5 — docs & marimo notebook

1. New marimo notebook `docs/notebooks/10_endpoint_singularities.py`
   reproducing the paper's headline figures (resolution power, decay
   of Chebyshev coefficients with and without the map).
2. User-guide page `docs/user/features/singularities.md`.
3. Cross-reference from `docs/plans/02-compactfun-integration.md`.

## Open questions

1. **Automatic detection of endpoint exponents.** Off by default in v1,
   but worth prototyping: estimate `α` by sampling `log|f(a+ε)|` vs
   `log ε` on a geometric mesh near the endpoint. Risk: false positives
   on benign functions. Decision: implement, gate behind a setting,
   evaluate cost.
2. **Should `Singfun` be a separate `Onefun` rather than a separate
   `Classicfun`?** The mathematical argument cuts both ways. Choosing
   `Classicfun` here because the `Onefun` interface is "function on
   `[−1, 1]`" and the new payload genuinely is just a standard
   `Chebtech` on `[−1, 1]` — the novelty is the **map**, which is
   `Classicfun`'s responsibility.
3. **Storage for two-sided maps where only one side is singular.**
   `SlitMap` covers it directly; we could equivalently express this
   with `DslitMap(α=k, β=0)`. Keep both for clarity; document
   equivalence.
4. **Interaction with `Quasimatrix` columns.** Mixed-singularity
   columns should still concatenate; the per-column representation is
   independent. No change expected, but add a regression test.

## Risks

- **Phase 1 typing churn.** The `IntervalMap` protocol must not
  accidentally tighten contracts that `Interval` currently honours
  loosely (e.g. accepting both scalar and array inputs). Mitigate by
  not adding any runtime checks; the protocol is purely structural.
- **`diff` cancellation near endpoints.** Computing `1/m'` near `±1`
  is numerically delicate. Mitigate by deferring the `1/m'` factor to
  evaluation time, never storing differentiated coefficients with the
  factor baked in.
- **User confusion about `conv` refusal.** The error message names
  `Singfun` and links to the docs page; the deferred `recast()` helper
  is intended to provide a future opt-in path.

## Acceptance criteria

- `chebfun(np.sqrt, [0, 1], sing="left")` resolves at `eps` with a
  small coefficient count and matches `sqrt` pointwise to `~10·eps`.
- `(f.cumsum())(1) == 2/3` for the same example, to `~10·eps`.
- All existing tests still pass; new test modules ≥ 90% line coverage.
- The marimo notebook runs end-to-end in CI.
- The plan-02 `CompactFun` workflows are untouched (regression tests
  green).
