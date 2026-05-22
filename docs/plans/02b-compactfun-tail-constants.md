# Plan: CompactFun Tail-Constants Extension — Non-Zero Asymptotes

## Summary

Extend `CompactFun` (introduced in
[02-compactfun-integration.md](02-compactfun-integration.md)) to support
functions whose limits at `±∞` are **non-zero constants**: `tanh`, sigmoids,
smoothed step functions, and any superposition of a decaying density with a
piecewise-constant background. The extension adds two scalar metadata fields
`tail_left`, `tail_right` to `CompactFun` and propagates them through
arithmetic, calculus, and evaluation. The `(0, 0)` special case reproduces
plan 02 exactly, so this is a pure superset — no behavioural change for
existing `CompactFun` users.

The headline observation is that an additive skeleton

$$
f(x) \;=\; b(x) \;+\; r(x), \qquad r(x) \xrightarrow{|x|\to\infty} 0
$$

with `b(x)` a closed-form bridge between `tail_left` and `tail_right` makes
the residual `r` a standard zero-tail `CompactFun`. We do **not** store `b`
explicitly — it is folded into the `Onefun` sample values during
construction. What we store is just the pair `(tail_left, tail_right)` plus
the standard `Onefun` on the discovered storage interval. This keeps the
representation minimal and makes operator rules algebraic on the tail pair:

| Op | New `tail_left` | New `tail_right` |
|---|---|---|
| `α · f` | `α · L` | `α · R` |
| `f + g` | `L_f + L_g` | `R_f + R_g` |
| `f · g` | `L_f · L_g` | `R_f · R_g` |
| `f.diff()` | `0` | `0` |
| `f.cumsum()` | finite iff `L = 0` | finite iff `R = 0` |
| `conv(f, g)` | refused unless both decay (zero tails on both sides for at least one operand) | as left |

## Relationship to plan 02

Plan 02 ships first and assumes `tail_left = tail_right = 0`. This plan
relaxes that assumption. After landing 02b:

- **Refusal removed:** the explicit "non-zero asymptote" refusal in
  `CompactFun` construction is replaced by a probe path that detects tail
  constants.
- **No API break:** existing zero-tail `CompactFun` instances continue to
  behave identically; defaults for `tail_left`, `tail_right` are `0.0`.
- **Out-of-scope items in plan 02 that stay out of scope here:** heavy
  tails, the rational-map `Unbndfun`, `'exps'` / singularities, mixed
  `Chebfun`s combining bounded and unbounded segments.

## What we are NOT doing in this PR

1. **Heavy / fat tails** — still deferred (plan 02's exclusion stands).
2. **Convolution with non-decaying signals** — mathematically ill-posed on
   `ℝ`; we refuse with a clear error.
3. **Singularities / `'exps'`.** A multiplicative skeleton `f = s · r` is
   the natural fit there; that is a separate plan.
4. **Bridge-shape user customisation.** The bridge is an internal
   implementation detail. Users see only `(tail_left, tail_right)`.

## Background

For `f: ℝ → ℝ` with `lim_{x→−∞} f(x) = L` and `lim_{x→+∞} f(x) = R`
(both finite, possibly equal, possibly zero), pick a smooth bridge
`b(x; L, R, w, x₀)` satisfying

$$
b(-\infty) = L, \qquad b(+\infty) = R, \qquad b - L,\; b - R \;\;\text{decay super-exponentially}.
$$

The natural choice is the error-function bridge

$$
b(x) \;=\; \tfrac{L+R}{2} \;+\; \tfrac{R-L}{2}\,\operatorname{erf}\!\left(\frac{x - x_0}{w}\right)
$$

with `(x₀, w)` chosen from the numsupp probe so that the residual
`r(x) = f(x) − b(x)` decays to zero within the storage interval at
tolerance `tol`. The `erf` choice is favoured because:

- super-exponential tails ⇒ `r` is genuinely zero at machine precision
  outside a modest window;
- monotone, infinitely smooth ⇒ no artificial features introduced into the
  residual;
- centred and symmetric ⇒ `(x₀, w)` are the only free parameters.

Crucially `b` is **not stored**. Once `r` is approximated by a `Onefun` on
the discovered storage interval `[a, b_storage]`, evaluation outside that
interval is `tail_left` (left of `a`) or `tail_right` (right of `b_storage`).
Inside the storage interval, evaluation is `b(x) + r(x)` — but since both
were folded into the `Onefun` at construction time, this is just
`onefun(invmap(x))`. The bridge is purely a construction-time fitting
device.

## Design decisions

1. **Two scalars are enough metadata.** The full bridge shape `(x₀, w)` is
   a fitting choice that affects only the construction quality of `r`; it
   does not need to be persisted. Two scalars `(tail_left, tail_right)`
   carry all the algebraic information operators need.

2. **The `(0, 0)` case is identical to plan 02.** No code path for zero-tail
   `CompactFun` changes. New code is gated on
   `tail_left != 0.0 or tail_right != 0.0`.

3. **Bridge family is `erf` only in v1.** A single bridge family keeps the
   construction code small. Future work may add `tanh`-bridge or
   `½(1+x/√(1+x²))`-bridge variants for badly-conditioned cases.

4. **Closure under `+`, `−`, `*`, `α·` is algebraic on the tail pair.**
   The new `Onefun` of the result is constructed by re-sampling `f_op_g`
   on the union/intersection of storage intervals after subtracting the
   new combined bridge — exactly the existing "recompute numsupp on the
   result" step generalised. No new spectral machinery is needed.

5. **`sum` honours mathematical truth.** If either tail is non-zero on its
   unbounded side, `sum` raises `DivergentIntegralError`. This is correct:
   `tanh` is not Lebesgue-integrable on `ℝ`. Differences of sigmoids that
   share asymptotes (residual tails are zero after subtraction) integrate
   normally — and the metadata model handles this automatically.

6. **`cumsum` extends naturally.** For `[a, ∞)` the antiderivative `F` is
   finite at `+∞` iff `tail_right = 0`; if so, `F` is itself a
   `CompactFun` with `tail_left_F = F(a)` and `tail_right_F = ∫f`. This
   removes the plan-02 caveat that `cumsum` does not close: with tail
   metadata it does close, provided the integrability condition holds.

7. **`conv` refuses non-decaying inputs.** Convolution of two functions
   with non-zero asymptotes on the same side diverges. We refuse with a
   clear `DivergentIntegralError`. Mixed cases (one operand zero-tailed,
   one not) are *also* refused in v1; revisit if a use case appears.

8. **Detection at construction.** The numsupp probe is extended: instead
   of looking for `|f| < tol · vscale`, it checks whether `f(±10^k)`
   *converges to a constant*. If it does, the constant becomes the tail
   value and bridge fitting kicks in.

## API surface

### User-facing

```python
from numpy import inf, tanh
from chebpy import chebfun

# Sigmoid: tail_left = -1, tail_right = +1
f = chebfun(tanh, [-inf, inf])
f.tail_left, f.tail_right     # (-1.0, 1.0)
f(-1e6)                       # -1.0   (returned from tail_left)
f(0.0)                        # 0.0    (from onefun)
f(1e6)                        # +1.0   (returned from tail_right)
f.support                     # [-inf, inf]

# Algebraic closure: difference of two sigmoids has zero tails
g = chebfun(lambda x: tanh(x - 5), [-inf, inf])
h = f - g
h.tail_left, h.tail_right     # (0.0, 0.0)
h.sum()                       # well-defined, finite

# Sum of a non-decaying function diverges (correctly)
f.sum()
# DivergentIntegralError: integrand has non-zero asymptotic limit
# tail_left=-1.0, tail_right=+1.0; integral over (-inf, inf) diverges.

# Convolution refuses
f.conv(g)
# DivergentIntegralError: convolution requires at least one operand to
# decay to zero at ±inf; got tail_left=-1.0, tail_right=+1.0 for self.
```

### New / changed internal symbols

| Symbol | Location | Purpose |
|---|---|---|
| `CompactFun.tail_left, .tail_right` | `compactfun.py` | Two new scalar attributes; default `0.0` |
| `_detect_tail_constants(f, side, tol, max_probes)` | `compactfun.py` | Replaces / extends `_discover_numsupp` |
| `_erf_bridge(L, R, x0, w)` | `compactfun.py` | Constructs the bridge function used during sampling |
| `CompactFun._combine_tails(other, op)` | `compactfun.py` | Helper used by arithmetic operators |
| `DivergentIntegralError` | `exceptions.py` | New (or reuse existing if present) |

## Integration steps

1. **Construction — `compactfun.py`**
   - Add `tail_left`, `tail_right` parameters to `__init__`, defaulting to
     `0.0`. Persist on the instance.
   - Extend the numsupp probe to detect convergence to a constant on each
     unbounded side: probe `f(σ · 10^k)` for `k = 0..K` (`σ = ±1` per
     side); if `f(σ · 10^k) − f(σ · 10^{k+1})` falls below
     `tol · max(1, |f|)`, the limit is the running mean of the last few
     probes — that is the detected tail constant.
   - If the detected constant is `0` (within tolerance), fall through to
     the plan-02 zero-tail path unchanged.
   - Otherwise: pick a bridge centre `x₀` (zero of `f − ½(L+R)` if it
     exists in the probed range, else `0`) and width `w` (smallest scale
     at which `|f − b| < tol · max(|L|, |R|, 1)`); fit the residual
     `r(x) = f(x) − b(x)` with the existing `Classicfun.initfun_adaptive`
     on the discovered storage interval `[a, b_storage]`.
   - Store the *bridged* `Onefun` directly: i.e. sample `f` (not `r`) on
     `[a, b_storage]`. The bridge is only a fitting device for choosing
     `(a, b_storage)`; the `Onefun` itself represents `f` on its storage
     interval as usual. This means inside the storage interval evaluation
     is unchanged from plan 02; only outside-interval evaluation differs.

2. **Evaluation — `__call__`**
   - For `x` outside the storage interval:
     - if `x < a` and logical-left is `-inf`: return `tail_left`;
     - if `x > b_storage` and logical-right is `+inf`: return
       `tail_right`;
     - else (finite logical endpoint nearby): unchanged from plan 02
       (return `0`, or whatever plan 02 specifies).
   - Inside the storage interval: unchanged.

3. **`endvalues`**
   - Return `(tail_left, tail_right)` where the corresponding logical
     endpoint is `±inf`; otherwise the existing finite-endpoint values.

4. **Arithmetic — `compactfun.py`**
   - `__neg__`: negate both tails.
   - `__add__`, `__sub__`: tails add/subtract; storage interval becomes the
     union; rebuild `Onefun` of `f ± g` by resampling on the union (same
     as plan 02's "recompute numsupp on the result").
   - `__mul__`: tails multiply; storage interval becomes the union (not
     intersection — the product `(L_f + r_f)(L_g + r_g)` is non-trivial
     wherever either `r_f` or `r_g` is non-trivial); rebuild.
   - Scalar `α · f`: tails scale by `α`; storage interval unchanged.
   - **Optimisation:** if both operand tails are zero, dispatch to the
     plan-02 fast paths (intersection-based product etc.) without
     rebuilding bridges.

5. **Calculus**
   - `diff`: result tails are `(0, 0)` (derivative of a constant is zero).
     Storage interval inherited.
   - `cumsum`:
     - For `[-inf, +inf]`: requires `tail_left = 0`; result has
       `tail_left_F = 0`, `tail_right_F = ∫f` (using the existing `sum`
       on the storage interval).
     - For `[a, +inf]`: same condition on `tail_right`-driven divergence
       at `+∞` (must be `0` for the integral to be finite); result is
       `CompactFun` with `tail_right_F = ∫f`.
     - For `[-inf, b]`: symmetric.
     - When the divergence condition fails, raise
       `DivergentIntegralError` (do not silently truncate).
   - `sum`:
     - Raise `DivergentIntegralError` if `tail_left ≠ 0` and logical-left
       is `-inf`, or `tail_right ≠ 0` and logical-right is `+inf`.
     - Otherwise inherit `Classicfun.sum` on the storage interval.

6. **Convolution**
   - `conv` checks both operands' tails. If either operand has any
     non-zero tail on an unbounded side, raise `DivergentIntegralError`
     with a message pointing the user to the algebraic-closure escape
     hatch (subtract matching sigmoids first, convolve the residual).
   - All zero-tail paths inherit plan 02 behaviour unchanged.

7. **Display — `__repr__`**
   - Show tail constants when non-zero, e.g.:
     `CompactFun([-inf, +inf], n=42, tails=(-1.0, +1.0))`.
   - When both tails are zero, fall back to plan 02's repr.

8. **Settings — `settings.py`**
   - Reuse `prefs.numsupp_tol`, `prefs.numsupp_max_probes`,
     `prefs.numsupp_max_width` from plan 02.
   - Optional: `prefs.tail_detect_tol = prefs.numsupp_tol` for the
     constant-convergence test (default to `numsupp_tol` if unset).

9. **Exceptions — `exceptions.py`**
   - Add `DivergentIntegralError(ValueError)` if not already present.

10. **Tests — `tests/test_compactfun_tails.py`** (new file, parallel of
    `test_compactfun.py` from plan 02)
    - **Construction**: `tanh`, `1/(1+exp(-x))`, smoothed step
      `½(1+erf(x/w))`, constant offsets `c + Gaussian` — verify detected
      `(tail_left, tail_right)` matches analytical limits to `tol`.
    - **Evaluation**: `f(±1e10)` returns the tail constants exactly (no
      Onefun extrapolation).
    - **Algebra**:
      - `tanh(x) − tanh(x − a)` has zero tails and finite `sum` (equals
        `2a` analytically — useful sanity check).
      - `α · tanh(x)` has tails `(±α)`.
      - Product `tanh(x) · sech²(x)` has tails `(0, 0)` (because
        `sech² → 0`); the product machinery should recover this.
    - **Calculus**:
      - `tanh(x).diff()` = `sech²(x)` with zero tails; round-trip via
        `cumsum` recovers `tanh(x) − tanh(a)`.
      - `cumsum` of a Gaussian on `[-inf, +inf]` is an `erf`-shaped
        `CompactFun` with `tail_left = 0`, `tail_right = √π`.
      - `sum(tanh)` raises `DivergentIntegralError`.
    - **Convolution refusal**: `tanh.conv(Gaussian)` raises with a clear
      message; subtracting matched sigmoids first then convolving works.
    - **Backward compatibility**: every test in `test_compactfun.py`
      (plan 02) still passes unchanged.

## Out of scope (deferred)

- Multiplicative skeletons (singfun-style `'exps'`). Different feature.
- Tail-aware convolution via Fourier/principal-value tricks.
- User-pluggable bridge families.
- Heavy tails, rational-map `Unbndfun`, mixed bounded/unbounded
  `Chebfun`s.

## Risks & open questions

- **Bridge-fitting robustness.** Functions with very slow approach to
  their asymptote (`1/log(x)`-like) will be misclassified as non-converging
  by the probe and refused as heavy-tailed. This is acceptable in v1 —
  any user hitting it is in heavy-tail territory.
- **Tail detection on noisy callables.** If the user passes a numerically
  noisy function (e.g. a Monte Carlo estimator), the constant-convergence
  test may be fooled. Mitigation: the test uses a generous tolerance
  (`prefs.tail_detect_tol`, default = `numsupp_tol`) and requires three
  consecutive probes to agree.
- **Equal-but-non-zero tails.** `f(x) = c + Gaussian` has
  `tail_left = tail_right = c`. The bridge degenerates to a constant `c`
  in this case; the residual is just the Gaussian. This is the simplest
  case and should be the smoke test.
- **Conv refusal feels strict.** Some users might reasonably want
  `tanh.conv(narrow_kernel)` interpreted as "convolve the smooth part".
  We refuse to keep semantics honest; the documented escape hatch is to
  subtract a matched sigmoid first.

## Dependencies

- Builds directly on plan 02 (`CompactFun`, numsupp discovery,
  `_conv_legendre` reuse).
- Independent of trigtech, singfun, heavy-tail, and `Unbndfun` work.

## Migration notes

After 02b lands, two lines in plan 02's text become inaccurate and should
be removed or amended:

- The "non-zero asymptote" entry in §"What we are NOT doing in this PR".
- The `cumsum` design decision claiming `cumsum` does not close — with
  tail metadata it does close, subject to the standard integrability
  condition.

These are documentation-only follow-ups; no code in plan 02 changes.
