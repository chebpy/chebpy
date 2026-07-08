# CLAUDE.md

Guidance for working in this repository. `chebpy` is a Python implementation of
Chebfun: functions are represented by piecewise polynomial (Chebyshev) or
trigonometric (Fourier) approximations and manipulated like numbers.

## Where things live

Full detail is in [`docs/development/architecture.md`](docs/development/architecture.md).
In brief, `src/chebpy/` is organised into import layers (lower never imports
upper at module scope):

```text
api · quasimatrix · gpr                     high-level API / applications
chebfun (+ _convolution, _pointwise,        piecewise container + its impl modules
         _singular_construction, _ufuncs)
bndfun · compactfun · singfun               Classicfun pieces
classicfun                                  interval-mapped base ([-1,1] ↔ [a,b])
chebtech · trigtech                         reference-interval representations (Onefun)
algorithms · chebyshev                      numerical kernels
utilities · plotting · smoothfun            primitives
settings · exceptions · decorators ·        foundations (no intra-package imports)
  maps · fun · onefun
```

Two runtime type hierarchies meet at `Classicfun`: `Onefun → Smoothfun →
{Chebtech, Trigtech}` (a function on `[-1, 1]`) and `Fun → Classicfun →
{Bndfun, CompactFun, Singfun}` (that function placed on `[a, b]`). `Chebfun` is
a container of `Fun` pieces, not part of the `Fun` hierarchy. `Chebfun`'s public
methods are thin wrappers that delegate to the `_convolution` / `_pointwise` /
`_singular_construction` / `_ufuncs` implementation modules.

The module-scope import graph is acyclic (a DAG): no module imports a higher
layer, even via a deferred import. Where a low-level module would otherwise
reach up — `utilities.generate_funs()` building an unbounded piece with a
`CompactFun` constructor — the constructor is injected by the caller rather than
imported (resolved in [#417](https://github.com/chebpy/chebpy/issues/417)).

## Source ownership: this repo vs Rhiza

Development infrastructure is synced from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The
authoritative list of synced files is the `files:` block of
`.rhiza/template.lock`.

- **Owned here (edit freely):** `src/chebpy/**`, `tests/**`, `pyproject.toml`,
  `README.md`, `mkdocs.yml`, `cliff.toml`, `docs/**` except `docs/mkdocs-base.yml`.
- **Rhiza-managed (do not edit in place):** `Makefile` and `.rhiza/**`,
  `.github/workflows/rhiza_*.yml`, `.pre-commit-config.yaml`, `ruff.toml`,
  `pytest.ini`, `.bandit`, `.editorconfig`, `docs/mkdocs-base.yml`,
  `.devcontainer/*`, `.claude/commands/rhiza_*.md`. Change these upstream in
  Rhiza (or via `.rhiza/template.yml`) and re-sync, rather than editing the
  synced artifact.

Note: `tests/**` is chebpy's own suite; `.rhiza/tests/**` belongs to Rhiza.

## Working conventions

- **Use `make` targets; do not invoke `.venv/bin/...` directly.** Key gates:
  `make fmt` (pre-commit: ruff, markdownlint, bandit, …), `make typecheck`
  (`ty` + `mypy --strict`), `make docs-coverage` (interrogate, 100%),
  `make deptry`, `make security`, `make validate` (template drift), `make test`.
- **Tests must stay at 100% coverage** and 100% docstring coverage; genuinely
  unreachable defensive branches are marked `# pragma: no cover` with a reason.
- Type-check clean under `mypy --strict`; keep public signatures typed.
