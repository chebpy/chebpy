# Architecture

This page describes how the `chebpy` source is organised: the two runtime type
hierarchies, the layered module import graph and its direction rule, where the
`Chebfun` implementation lives, and which files this repository owns versus
those synced from the [Rhiza](https://github.com/jebel-quant/rhiza) template.

## Type hierarchies

`chebpy` has **two** abstract hierarchies that meet in `Classicfun`. One
describes a function on the reference interval `[-1, 1]`; the other places such
a function onto an arbitrary interval `[a, b]`.

```text
Onefun (abstract)                 a function on the reference interval [-1, 1]
└─ Smoothfun (abstract)
   ├─ Chebtech                    Chebyshev series      (aperiodic functions)
   └─ Trigtech                    Fourier series        (periodic functions)

Fun (abstract)
└─ Classicfun (abstract)          maps [-1, 1] ↔ [a, b]; wraps one Onefun
   ├─ Bndfun                      bounded interval, affine map
   ├─ CompactFun                  (semi-)infinite interval via numerical-support truncation
   └─ Singfun                     endpoint singularities via a non-affine clustering map
```

A `Classicfun` **holds an `Onefun`** (its representation on `[-1, 1]`) together
with the interval or map that places it on `[a, b]`. The choice of `Onefun`
subclass is orthogonal to the choice of `Classicfun` subclass: e.g. a `Bndfun`
usually wraps a `Chebtech`, but a periodic piece wraps a `Trigtech`.

### The `Chebfun` container

`Chebfun` is the user-facing class. It is **not** part of the `Fun` hierarchy;
it is a container holding an ordered array of `Fun` pieces over a `Domain`
(one piece per sub-interval between breakpoints). Piecewise operations
(arithmetic, calculus, root-finding, `conv`, …) fan out over the pieces.

```text
Chebfun ──holds──▶ [ Fun, Fun, … ]      one piece per sub-interval
                     each Fun ──holds──▶ Onefun on [-1, 1]
```

## Module layering

Modules are organised into layers. **A module imports only from strictly lower
layers** (plus, in a few noted cases, siblings in its own layer). Lower layers
never import upper layers at module scope — see
[Import direction](#import-direction-and-deferred-imports) for the one audited
exception.

```text
L7  High-level API / apps   api · quasimatrix · gpr
L6  Container + impl         chebfun · _convolution · _pointwise ·
                             _singular_construction · _ufuncs
L5  Classicfun pieces        bndfun · compactfun · singfun
L4  Interval-mapped base     classicfun
L3  Reference reps (Onefun)  chebtech · trigtech
L2  Numerical kernels        algorithms · chebyshev
L1  Primitives               utilities · plotting · smoothfun
L0  Foundations              settings · exceptions · decorators · maps · fun · onefun
```

`chebpy/__init__.py` sits above everything as the package facade, re-exporting
the public surface (`chebfun`, `chebpts`, `pwc`, `trigfun`, `CompactFun`,
`Singfun`, `Trigtech`, `Quasimatrix`, `gpr`, …).

Selected module dependencies (top-level imports):

| Module | Imports |
|--------|---------|
| `utilities` | `decorators`, `exceptions`, `settings` |
| `algorithms` | `decorators`, `settings`, `utilities` |
| `chebtech` / `trigtech` | `algorithms`*/`decorators`, `plotting`, `settings`, `smoothfun`, `utilities` |
| `classicfun` | `chebtech`, `trigtech`, `fun`, `decorators`, `exceptions`, `plotting`, `settings`, `utilities` |
| `bndfun` / `compactfun` / `singfun` | `classicfun` (+ `exceptions`/`settings`/`utilities`/`maps`) |
| `chebfun` | `bndfun`, `_ufuncs`, `decorators`, `exceptions`, `plotting`, `settings`, `utilities` |
| `api` | `algorithms`, `bndfun`, `chebfun`, `settings`, `utilities` |
| `gpr` | `algorithms`, `chebfun`, `quasimatrix`, `settings` |

*`trigtech` does not import `algorithms` at module scope; it defers that import
(see below).

## The `Chebfun` implementation modules

`Chebfun` is large, so its self-contained algorithms live in dedicated modules.
The public `Chebfun` methods are **thin, documented wrappers** that delegate:

| Module | Provides | Fronted by |
|--------|----------|------------|
| `_convolution.py` | Hale–Townsend / Gauss–Legendre convolution | `Chebfun.conv` |
| `_pointwise.py` | root-splitting step ops | `Chebfun.abs`, `sign`, `ceil`, `floor`, `maximum`, `minimum` |
| `_singular_construction.py` | endpoint-singularity piece construction | the `sing=` Chebfun constructors |
| `_ufuncs.py` | registers elementwise NumPy ufunc methods | `Chebfun.sin`, `exp`, … |

These modules operate on a `Chebfun` passed in as an argument (constructing
results via `f.__class__(...)`) and reference `Chebfun` only under
`TYPE_CHECKING`, so importing them never creates a runtime cycle.

## Import direction and deferred imports

The layering rule is enforced by keeping upward references out of module scope.
The module-scope import graph is acyclic — no module imports a higher layer,
even via a function-local import. Function-local (deferred) imports are used
only for three reasons, none of which reach upward:

1. **Typing only** — `_convolution`, `_pointwise`, `_ufuncs` import `Chebfun`
   under `TYPE_CHECKING`.
2. **Breaking sibling cycles** — e.g. `singfun`/`compactfun` import `bndfun`
   inside `restrict`; `trigtech` imports `chebtech` inside `roots`.
3. **Lazy/optional heavy paths** — `chebfun` imports its implementation modules
   (`_convolution`, `_pointwise`, `_singular_construction`) inside the relevant
   methods.

### Dependency injection instead of an upward import

The one place a low-level module would otherwise reach up is
`utilities.generate_funs()`, which builds an unbounded piece with a
`CompactFun` (L5) constructor. Rather than import `compactfun` from `utilities`
(L1), the constructor is **injected** by the caller: `chebfun` passes the
matching `CompactFun` classmethod as `generate_funs(..., compact_constructor=…)`.
This keeps `utilities` free of any dependency on `compactfun` (resolved in
[#417](https://github.com/chebpy/chebpy/issues/417)).

## Source ownership: this repo vs Rhiza

This project syncs its development infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The
machine-readable list of synced files is the `files:` block of
`.rhiza/template.lock`; that file is authoritative. In summary:

**Owned by this repository** — edit here:

- `src/chebpy/**` — the library
- `tests/**` — chebpy's own test suite (note: `.rhiza/tests/**` is Rhiza's)
- `pyproject.toml`, `README.md`, `mkdocs.yml`, `cliff.toml`
- `docs/**` *except* `docs/mkdocs-base.yml`
- project notebooks

**Managed by Rhiza** — do **not** edit in place; change upstream in Rhiza or via
`.rhiza/template.yml`, then re-sync:

- `Makefile` and `.rhiza/make.d/*.mk`, plus everything under `.rhiza/**`
- `.github/workflows/rhiza_*.yml`
- `.pre-commit-config.yaml`, `ruff.toml`, `pytest.ini`, `.bandit`, `.editorconfig`
- `docs/mkdocs-base.yml`
- `.devcontainer/*` and `.claude/commands/rhiza_*.md`

When scoring or reviewing the repo, attribute gaps in Rhiza-managed files to the
template (fix upstream), and gaps in the owned files to this project.
