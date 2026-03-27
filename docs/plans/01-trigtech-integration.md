# Plan: Trigtech Integration (PR 1/4)

## Summary

Add Fourier-based (trigonometric) function approximation to chebpy via a new `trigtech` module. This is the analogue of MATLAB Chebfun's `@trigtech` class and provides the low-level representation for smooth periodic functions on `[-1, 1]`.

> **Ref:** MATLAB Chebfun `@trigtech` — see [`@trigtech/trigtech.m`](https://github.com/chebfun/chebfun/blob/master/%40trigtech/trigtech.m)

## Scope

| Area | Detail |
|------|--------|
| **New module** | `src/chebpy/trigtech.py` (~40 KB in PR #231) |
| **New tests** | `tests/trigtech/` directory (mirrors existing `tests/chebtech/` structure) |
| **Modified modules** | `src/chebpy/onefun.py` — factory dispatch to select `Trigtech` for periodic functions |
| | `src/chebpy/classicfun.py` — support Trigtech-backed funs |
| | `src/chebpy/chebfun.py` — periodic chebfun construction & display |
| | `src/chebpy/__init__.py` — export `Trigtech` |

## API Surface (chebpy-style)

The `Trigtech` class should follow the existing `Chebtech` contract defined in `onefun.py`. Key methods and properties:

| Method / Property | Description | MATLAB equivalent |
|---|---|---|
| `Trigtech.initfun(f, n)` | Construct from callable or coefficients | `trigtech(op)` |
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
2. Update `Onefun` factory in `onefun.py` to dispatch to `Trigtech` when the function is flagged as periodic.
3. Update `Classicfun` / `Bndfun` to propagate the periodic flag through construction.
4. Update `Chebfun` to support periodic construction (`chebfun(f, 'trig')`-style in MATLAB; in chebpy, via a `kind='trigtech'` or `periodic=True` kwarg).
5. Add `tests/trigtech/` with construction, arithmetic, calculus, and edge-case tests mirroring `tests/chebtech/`.
6. Export `Trigtech` from `__init__.py`.

## Dependencies

- None. This PR is self-contained and can land before any chebop work.

## Open Questions

- Should periodic detection be automatic (via `Onefun.initfun` heuristic) or purely explicit?
- Coefficient storage convention: match MATLAB's or use NumPy-native ordering?
