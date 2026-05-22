# Plan: Trigtech Integration (PR 1/4)

## Summary

Add Fourier-based (trigonometric) function approximation to chebpy via a new `trigtech` module. This is the analogue of MATLAB Chebfun's `@trigtech` class and provides the low-level representation for smooth periodic functions on `[-1, 1]`.

> **Ref:** MATLAB Chebfun `@trigtech` — see [`@trigtech/trigtech.m`](https://github.com/chebfun/chebfun/blob/master/%40trigtech/trigtech.m)

## Scope

| Area | Detail |
|------|--------|
| **New module** | `src/chebpy/trigtech.py` (~40 KB in PR #231) |
| **New tests** | `tests/trigtech/` directory (mirrors existing `tests/chebtech/` structure) |
| **New user API** | `src/chebpy/api.py` — add `trigfun(...)` as the explicit entry point for periodic functions |
| **Modified modules** | `src/chebpy/classicfun.py` — support Trigtech-backed funs |
| | `src/chebpy/chebfun.py` — periodic chebfun construction & display |
| | `src/chebpy/__init__.py` — export `Trigtech` and `trigfun` |

## API Surface (chebpy-style)

The `Trigtech` class should follow the existing `Chebtech` contract defined in `onefun.py`. The public user-facing entry point is `trigfun(f, domain)`, mirroring `chebfun(f, domain)` but always using `Trigtech` as the underlying tech — no automatic detection.

Key methods and properties:

| Method / Property | Description | MATLAB equivalent |
|---|---|---|
| `trigfun(f, domain)` | **User-facing constructor** for periodic functions (explicit, no auto-detection) | `chebfun(f, 'trig')` |
| `Trigtech.initfun(f, n)` | Low-level constructor from callable or coefficients | `trigtech(op)` |
| `Trigtech.coeffs` | Fourier coefficient array | `f.coeffs` |
| `Trigtech.values` | Function values at equispaced points | `f.values` |
| `Trigtech.isperiodic` | Always returns `True` | `f.isPeriodicTech` |
| `Trigtech.prolong(n)` | Resample to `n` points | `prolong(f, n)` |
| `Trigtech.simplify()` | Chop trailing Fourier coefficients | `simplify(f)` |
| `__add__`, `__mul__`, etc. | Arithmetic, following chebpy operator overloading pattern | `+`, `.*`, etc. |
| `diff()` | Differentiation via Fourier multiplier | `diff(f)` |
| `cumsum()` | Integration via Fourier multiplier | `cumsum(f)` |
| `roots()` | Root-finding (via colleague matrix or conversion to Chebyshev) | `roots(f)` |
| `plotcoeffs()` | Plot Fourier coefficients | `plotcoeffs(f)` |

### Differences from MATLAB Chebfun

- chebpy uses **NumPy FFT** (not FFTW); performance characteristics differ.
- chebpy's class hierarchy has `Onefun → Smoothfun → {Chebtech, Trigtech}` rather than MATLAB's tech-switching on the preference object.
- Coefficient ordering: chebpy should store coefficients in **standard NumPy FFT order** and provide a `._coeffs_to_plotorder()` helper for display.

## Integration Steps

1. Add `src/chebpy/trigtech.py` implementing the `Trigtech(Smoothfun)` class.
2. Add `trigfun(f, domain)` to `src/chebpy/api.py` as the explicit user-facing constructor for periodic functions; wire it through `Classicfun` / `Bndfun` using `Trigtech` as the tech (no heuristic detection in `Onefun.initfun`).
3. Update `Classicfun` / `Bndfun` to accept and propagate a `tech=Trigtech` argument through construction.
4. Update `Chebfun` display to indicate when a piece is backed by `Trigtech`.
5. Add `tests/trigtech/` with construction, arithmetic, calculus, and edge-case tests mirroring `tests/chebtech/`.
6. Export `Trigtech` and `trigfun` from `__init__.py`.

## Dependencies

- None. This PR is self-contained and can land before any chebop work.

## Design Decisions

- **Periodic detection is explicit.** There is no heuristic in `Onefun.initfun`. Users call `trigfun(f, domain)` to get a `Trigtech`-backed fun. This keeps the API unambiguous and mirrors the MATLAB `chebfun(f, 'trig')` pattern while being idiomatic for chebpy.
- **Coefficient ordering follows NumPy-native (FFT) order.** Coefficients are stored in the order returned by `numpy.fft.fft` / `numpy.fft.rfft`. A `._coeffs_to_plotorder()` helper re-orders them to human-readable (DC-centred) order for `plotcoeffs()` and other display methods.
