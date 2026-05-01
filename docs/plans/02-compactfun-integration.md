# Plan: CompactFun Integration — Light-Tailed Functions on (−∞, ∞)

> **Status update (plan 02b).** Two of the v1 limitations recorded below
> have since been lifted by the follow-up plan
> [`02b-compactfun-tail-constants.md`](02b-compactfun-tail-constants.md):
> non-zero asymptotic constants (`tanh`, sigmoids) are now representable
> via `(tail_left, tail_right)` metadata, and `cumsum` now closes inside
> `CompactFun` (carrying the integral as a right-tail constant) rather
> than returning a bounded `Chebfun`. Convolution against operands with
> non-zero tails is still refused — but now with a clear
> `DivergentIntegralError`. Read this document for the core design;
> consult plan 02b for the tail-constant extension.

## Summary

Add support for representing functions on `(−∞, ∞)`, `(−∞, b]`, and `[a, ∞)`
that decay rapidly enough to have **finite numerical support** at machine
precision. The new `CompactFun` class is a sibling of `Bndfun` under
`Classicfun`: it stores a `Onefun` (Chebtech) on a discovered finite storage
interval `[a, b]` exactly like `Bndfun` does, and adds a separate *logical*
interval (which may have `±inf` endpoints) that is what the user sees. The
function reports zero outside `[a, b]`. From the user's perspective the
function lives on the whole (semi-)real line; internally everything reduces
to existing bounded machinery.

This is a deliberate departure from MATLAB Chebfun's `@unbndfun` (which uses a
rational change of variables to map `(−∞, ∞)` onto `[−1, 1]`). The
rational-map approach is left for a possible future PR; this plan covers the
practical-use majority case and — critically — gives us **convolution on
`ℝ`** for free by reusing the existing `_conv_legendre` machinery.

> **Ref (deliberately not followed):** MATLAB Chebfun's `@unbndfun`
> [`@unbndfun/createMap.m`](https://github.com/chebfun/chebfun/blob/master/%40unbndfun/createMap.m).
> See [Chebfun guide chapter 9.1](https://www.chebfun.org/docs/guide/guide09.html)
> for the user-level behaviour we partially mirror.

## What we are NOT doing in this PR

These are explicitly out of scope and deferred to follow-up plans:

1. **Heavy / fat tails** (Cauchy `1/(1 + x²)`, Pareto, Lévy stable, Student-t
   with low df, anything that does not decay below `tol · vscale` in a
   finite, modestly-sized window). A separate plan will introduce the
   appropriate machinery; do not attempt a logarithmic-mesh workaround here.
2. **Non-zero asymptotic constants** (`tanh`, sigmoids, step-like functions
   that approach `±1` at infinity). The numerical-support model is only
   sound when both tail limits are zero; non-zero asymptotes break closure
   under multiplication and are deferred.
3. **Singularities / poles / `'exps'` mode.** Independent feature.
4. **The rational-map `Unbndfun` class.** Possible future addition; not
   required by anything in this plan.
5. **Convolution involving non-zero asymptotes** (mathematically ill-posed
   on `ℝ`).

The scope is intentionally narrow: integrable, light-tailed functions whose
numerical support comfortably fits in a single `Bndfun`.

## Background

For an integrable `f: ℝ → ℝ` with `lim_{x → ±∞} f(x) = 0` and rapid decay,
define the numerical support at tolerance `ε`:

$$
\text{numsupp}_\varepsilon(f) \;=\; [a, b] \quad \text{such that} \quad
|f(x)| < \varepsilon\,\|f\|_\infty \;\;\forall\, x \notin [a, b].
$$

For Gaussians, exponentials, sub-exponentials, and any density with
super-algebraic decay, `numsupp` is finite and modest in width (e.g.
`[−40, 40]` for a unit Gaussian at `tol = 1e-15`).

Inside `[a, b]`, `f` is approximated by an ordinary `Onefun` (Chebtech) via
the `Classicfun` mapping plumbing — exactly the same machinery `Bndfun`
uses. Outside `[a, b]`, `f` is reported as identically zero. From the
user's perspective:

- `f(x)` evaluates the underlying `Onefun` for `x ∈ [a, b]`, and returns `0`
  otherwise.
- `sum(f) = ∫_{-∞}^{∞} f = ∫_a^b f` is computed by the existing
  `Classicfun.sum` (truncation error bounded by `ε · (b − a)`).
- `diff(f)`, `roots(f)` reduce to the inherited `Classicfun` operations,
  with the result extended by `0` outside `[a, b]`.
- `cumsum(f)` is the one operation that does not close (the result has a
  non-zero right-tail asymptote `∫f`); it returns a bounded `Chebfun` on
  `[a, b]` for v1.
- `conv(f, g)` reuses the existing `_conv_legendre` machinery on the
  underlying `Onefun`s, with result interval `[a_f + a_g, b_f + b_g]`.
  This is the headline win.

## Scope

| Area | Detail |
|------|--------|
| **New module** | `src/chebpy/compactfun.py` — `CompactFun(Classicfun)` |
| **New tests** | `tests/test_compactfun.py` parallel of `tests/test_bndfun.py` |
| **New user API** | `chebfun(f, [-inf, inf])` / `[a, inf]` / `[-inf, b]` returns a `CompactFun`-backed `Chebfun` |
| **Modified — utilities** | `src/chebpy/utilities.py` — relax `Domain` to allow `±inf` only at the outermost breakpoints; teach `generate_funs` to dispatch to `CompactFun` for unbounded intervals |
| **Modified — chebfun** | `src/chebpy/chebfun.py` — `conv` accepts `CompactFun` pieces; display; mixed-piece sums |
| **Modified — settings** | `src/chebpy/settings.py` — `numsupp_tol` (default `prefs.eps`), `numsupp_max_width` (refusal threshold for v1, default `1e6`) |
| **Modified — exports** | `src/chebpy/__init__.py` — export `CompactFun` |

## Design decisions

1. **`CompactFun` is a sibling of `Bndfun` under `Classicfun`.** It reuses
   the existing `Classicfun` plumbing (`Onefun` + storage `Interval`,
   evaluation via `interval.invmap`, calculus via the standard Jacobian)
   exactly as `Bndfun` does. The only thing that's new is a separate
   *logical* interval that is what the user sees and what arithmetic with
   other `CompactFun`s validates against. For `Bndfun` the logical and
   storage intervals coincide; for `CompactFun` they differ when the
   logical interval is unbounded.

2. **The logical interval may be unbounded; the storage interval is always
   finite.** Two attributes:
   - `self._logical_interval`: what the user sees (e.g. `[-inf, inf]`).
   - `self._interval`: storage interval `[a, b] = numsupp_ε(f)`, inherited
     from `Classicfun`.
   - `self.onefun` is the standard `Onefun` (Chebtech) on `[−1, 1]`,
     inherited from `Classicfun`.

3. **Numerical-support discovery** is done once at construction time by
   probing `f(±10^k)` for `k = 0, 1, 2, ...` until `|f| < tol · vscale_so_far`,
   then refining via bisection. The probing budget is bounded
   (`prefs.numsupp_max_probes`, default 30). If `tol` is not reached within
   `prefs.numsupp_max_width`, construction fails with a clear error pointing
   to the future "heavy tails" facility.

4. **Closure under operations is preserved by recomputing numerical support
   when needed.**
   - `f + g`: union of supports, then re-trim to the new numerical support
     of the sum (since cancellation may shrink it; addition cannot enlarge
     support beyond the union).
   - `f * g` (pointwise): intersection of supports (product is zero where
     either factor is zero); re-trim is unnecessary in the common case.
   - `α · f`: support unchanged (assuming `α ≠ 0`).
   - `f.diff()`: support unchanged at construction-tolerance level.
   - `f.cumsum()`: result has **non-zero left or right asymptotic value**
     (the integral). `cumsum` is therefore the one operation that does NOT
     close in `CompactFun` — it returns a regular `Chebfun` on the bounded
     interval `[a, b]` (supplemented with constant pieces if the user wants
     `(−∞, ∞)`). For v1 we return the bounded `Chebfun` and document the
     restriction.

5. **`conv` is implemented via existing `_conv_legendre`** on the
   underlying `Onefun`s. Result storage interval `[a_f + a_g, b_f + b_g]`.
   Returned as a `CompactFun` whose logical interval is the union of the
   inputs' logical intervals (typically `(−∞, ∞)`).

6. **No tech choice for `CompactFun` storage; always Chebtech.** Trigtech is
   meaningless here (no periodicity).

7. **Construction always begins with a finite-domain probe.** We do not
   trust the user-supplied `±inf` literally; we always discover numerical
   support and instantiate the underlying `Onefun` on the discovered finite
   storage interval. The logical interval keeps the user's requested `±inf`
   for the lifetime of the object.

8. **Minimal overrides.** Because `CompactFun` reuses the `Classicfun`
   machinery, only a small set of methods need to be overridden:
   - `__call__` (return `0` outside the storage interval)
   - `support` (returns the logical interval, not the storage one)
   - `endvalues` (returns `0` at any `±inf` endpoint)
   - `__repr__` (displays the logical interval)
   - `cumsum` (returns a bounded `Chebfun` on `[a, b]`; documented
     restriction)
   - `restrict` and `translate` (need to update both intervals
     consistently)
   Everything else — `coeffs`, `size`, `vscale`, `iscomplex`, `+`, `-`,
   `*`, `**`, `diff`, `roots`, `sum`, `simplify`, plotting — is inherited
   from `Classicfun` unchanged.

## API surface

### User-facing

```python
from numpy import inf
from chebpy import chebfun

# Light-tailed, integrable, decays to zero on both sides → CompactFun
f = chebfun(lambda x: np.exp(-x**2) / np.sqrt(np.pi), [-inf, inf])
f.sum()            # ≈ 1
f(0.0)             # 1/√π
f(100.0)           # 0   (outside numerical support)
f.support          # [-inf, inf]    (logical)
f.numerical_support  # [-a, +a]    (discovered)

# Convolution: this is the headline feature
g = chebfun(lambda x: np.exp(-x**2) / np.sqrt(np.pi), [-inf, inf])
h = f.conv(g)      # CompactFun, ≈ exp(-x²/2)/√(2π)
h.sum()            # ≈ 1

# Heavy tail: explicit refusal in v1
chebfun(lambda x: 1 / (1 + x**2), [-inf, inf])
# ConstructorError: numerical support not contained in [-1e6, 1e6] at
# tolerance 2.22e-16; heavy-tailed inputs are not supported in this
# release.

# Non-zero asymptote: explicit refusal in v1
chebfun(lambda x: np.tanh(x), [-inf, inf])
# ConstructorError: function does not decay to zero at ±∞; non-zero
# asymptotic limits are not supported in this release.
```

### New internal classes / functions

| Symbol | Location | Purpose |
|---|---|---|
| `CompactFun(Classicfun)` | `compactfun.py` | New sibling of `Bndfun`; adds a logical (possibly unbounded) interval on top of the standard `Classicfun` storage |
| `CompactFun.initfun_adaptive(f, interval)` | `compactfun.py` | Adaptive constructor; runs numsupp discovery, then calls the inherited `Classicfun.initfun_adaptive` on the discovered storage interval |
| `CompactFun.initfun_fixedlen(f, interval, n)` | `compactfun.py` | Fixed-length variant (numsupp discovery still runs) |
| `CompactFun.initconst(c, interval)` | `compactfun.py` | Constant on unbounded interval — only `c == 0` allowed (else error) |
| `CompactFun.numerical_support` | `compactfun.py` | Property returning the storage interval `[a, b]` |
| `_discover_numsupp(f, interval, tol, max_width)` | `compactfun.py` | Probe-and-bisect helper |

## Integration steps

1. **Foundations — `utilities.py`**
   - Relax `Domain` to accept `±inf` at the outermost breakpoints (interior
     breakpoints must remain finite).
   - Update `generate_funs` to inspect each interval: if both endpoints are
     finite, dispatch to `Bndfun.initfun_*` as today; if either endpoint is
     `±inf`, dispatch to `CompactFun.initfun_*`.
   - Update `check_funs` and `compute_breakdata` so that `±inf` outer
     endpoints don't trigger the existing "must average endpoint values"
     code (we report `0` at `±inf`).

2. **`CompactFun` class — `compactfun.py`**
   - Subclass `Classicfun`, sibling of `Bndfun`. Add
     `self._logical_interval` alongside the inherited `self._interval`
     (storage) and `self.onefun`.
   - Override `__init__(onefun, storage_interval, logical_interval)` to
     accept the logical interval; default it to the storage interval if
     not supplied (so a `CompactFun` constructed with a finite logical
     interval behaves identically to a `Bndfun`).
   - Override `__call__(x)`: zero outside the storage interval; inside,
     delegate to the inherited evaluation path (`self.interval.invmap` +
     `self.onefun(y)`).
   - Override `support` (returns logical interval), `endvalues` (returns
     `0` at any `±inf` endpoint, evaluates normally at finite endpoints),
     `__repr__`.
   - Override `cumsum` to return a bounded `Chebfun` on the storage
     interval; document the restriction.
   - Override `restrict` and `translate` to keep the logical and storage
     intervals consistent.
   - Implement `_discover_numsupp(f, interval, tol, max_width)`:
     - Probe at `x = ±10^k` for `k = 0..K` (with sign chosen by which side
       of the interval is unbounded).
     - Track a running `vscale = max |f(probe)|`.
     - Identify the largest `k` with `|f| > tol · vscale`; set the outer
       endpoint a small multiple beyond that.
     - Bisect to refine the boundary.
     - Verify by sampling within (catch missed bumps): if any sample inside
       `[a, b]` exceeds `vscale`, expand and retry; if budget exhausted,
       error.
     - For semi-infinite intervals, only the unbounded side is discovered;
       the finite endpoint is kept verbatim.

3. **`Chebfun` integration — `chebfun.py`**
   - `Chebfun.conv` currently dispatches on `Bndfun` only. Extend it to
     handle the case where one or both `funs` are `CompactFun`s: the
     existing equal-width / piecewise paths apply directly to the
     underlying `Onefun`s and storage intervals; wrap the result as a
     `CompactFun` whose logical interval is `[−inf, +inf]` (or the
     appropriate semi-infinite combination).
   - Update display so unbounded pieces print the logical interval
     (e.g. `[ -inf, +inf ]    13`).
   - Sums / arithmetic across a chebfun with mixed bounded and `CompactFun`
     pieces: the existing piece-by-piece machinery should work because
     `CompactFun` exposes the same `Classicfun` interface; only operations
     that compare `support` need to be aware that two `CompactFun`s with
     differing storage but matching logical intervals are compatible.

   **Piecewise `conv` involving `CompactFun` pieces.** A user-supplied
   piecewise density on `ℝ` (e.g.
   `chebfun(f, domain=[-inf, -2, 0, 3, +inf])`) yields a chebfun whose
   outer pieces are `CompactFun` and interior pieces are `Bndfun`.
   Convolving two such chebfuns must produce a result whose breakpoints are
   the pairwise sums of the input breakpoints; with `±inf` endpoints these
   sums collapse correctly to `[-inf, ..., +inf]` thanks to IEEE arithmetic
   (`-inf + finite = -inf`, etc.). The required adjustments are:
   - **Equal-width fast path:** add a guard that skips the fast path when
     either piece is a `CompactFun` (logical width `inf`); fall through to
     the general piecewise path.
   - **General `_conv_piecewise`:** when computing integration limits for a
     sub-interval that touches a `CompactFun` piece, use the piece's
     **storage interval** (i.e. its discovered numerical support), not its
     logical interval. This is consistent with `__call__`, which reports
     zero outside the storage interval.
   - **Output piece dispatch:** an output piece is wrapped as `CompactFun`
     iff at least one of its breakpoints is `±inf`; otherwise as `Bndfun`.
     Numerical support of each `CompactFun` output piece is discovered from
     the constructed `Onefun` exactly as in standalone construction.

4. **Settings — `settings.py`**
   - `prefs.numsupp_tol = prefs.eps` (default tolerance for numerical
     support discovery; users can tighten/loosen).
   - `prefs.numsupp_max_width = 1e6` (refusal threshold for v1).
   - `prefs.numsupp_max_probes = 30` (probing budget).

5. **Exports — `__init__.py`**
   - Export `CompactFun`.

6. **Tests — `tests/test_compactfun.py`**
   - **Construction**: Gaussian, exp(−x), exp(−|x|), bump function — verify
     discovered support is sensible and `f(x_far)` returns `0`.
   - **Round-trip evaluation**: `f(x) ≈ ground_truth(x)` for `x` inside and
     outside numerical support.
   - **Calculus**: `sum` against analytic answers (`∫e^{−x²}dx = √π`,
     `∫_0^∞ e^{−x}dx = 1`, etc.); `diff` round-trip
     (`f.diff().cumsum()` ≈ `f − f(a)`).
   - **Convolution** (the headline test):
     - Two unit Gaussians convolve to `N(0, 2)` density.
     - Two unit-rate exponentials on `[0, ∞)` convolve to a Gamma(2, 1)
       density.
     - `f.conv(g).sum() == f.sum() * g.sum()` for any two `CompactFun`s.
   - **Algebra**: `CompactFun + CompactFun`, `CompactFun + scalar`,
     `CompactFun * CompactFun` (product of two Gaussians).
   - **Error paths**:
     - `1/(1+x²)` on `[-inf, inf]`: refuse with heavy-tail message.
     - `tanh(x)` on `[-inf, inf]`: refuse with non-zero-asymptote message.
     - `initconst(c≠0, [-inf, inf])`: refuse.
   - **Mixed-domain `Chebfun`**: not in v1 (interior breakpoints must remain
     finite, and a chebfun cannot mix `CompactFun` pieces with bounded
     pieces in the same object except as outermost segments — defer until
     there's a use case).

## Out of scope (deferred)

- Heavy-tailed distributions (Cauchy, Pareto, etc.).
- Non-zero asymptotic limits (`tanh`-like).
- The rational-map `Unbndfun` (Chebfun's `@unbndfun`).
- `cumsum` returning a `CompactFun` (would need either non-zero asymptote
  support or a sentinel right-tail constant).
- Mixed `Chebfun`s combining bounded and unbounded segments.
- `'splitting on'`, `blowup`, `'exps'`.

## Risks & open questions

- **Numerical-support discovery may miss bumps far from origin.** Mitigation:
  the verification step samples within the discovered support and expands
  if any sample exceeds `vscale`. This is heuristic but matches Chebfun's
  general philosophy.
- **`f * g` pointwise can have larger numerical support than expected if
  the factors' tails interact.** Closure is preserved (product is bounded
  by both factors), but the discovered support of the product may need
  re-discovery. We will recompute numsupp on the product result.
- **Refusing heavy tails is a UX choice.** We must produce a clear,
  actionable error message rather than silently truncating. The error
  should mention that a future release will handle these cases.
- **`cumsum` not closing under `CompactFun` is unfortunate** but
  unavoidable without asymptotic-constants metadata. Documenting clearly
  that `cumsum` returns a bounded `Chebfun` is acceptable for v1.

## Dependencies

- Builds on existing `Bndfun`, `Chebtech`, and `_conv_legendre` machinery.
- Independent of any trigtech / singfun / unbndfun work.

## See also

- [`docs/plans/03-singfun-mapped-integration.md`](03-singfun-mapped-integration.md)
  — companion plan adding `Singfun` for endpoint-singular functions on
  bounded intervals.  `CompactFun` and `Singfun` are siblings under
  `Classicfun`: the former handles unbounded support with rapid decay;
  the latter handles bounded support with branch-type endpoint behaviour.
