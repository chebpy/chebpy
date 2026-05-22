"""Marimo notebook demonstrating periodic (Fourier) function representations in ChebPy."""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebfun]
# path = "../.."
# editable = true
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from chebpy import chebfun, trigfun

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Periodic Representations

    For smooth periodic functions on a bounded interval, ChebPy offers a
    Fourier-based representation via `Trigtech` — the trigonometric analogue
    of the default `Chebtech`.  The user-facing entry point is
    [`trigfun`](../api.md), which mirrors `chebfun` exactly but always uses
    `Trigtech` as the underlying technology.

    Two design points are worth noting up-front:

    - **Periodicity is opted into explicitly** — there is no automatic
      detection.  This mirrors MATLAB Chebfun's `chebfun(f, 'trig')` while
      keeping the Python API unambiguous.
    - **`Trigtech` shares the `Onefun → Smoothfun` interface** with
      `Chebtech`, so a `trigfun`-backed Chebfun supports all the usual
      operations (`diff`, `cumsum`, `sum`, `roots`, arithmetic, plotting).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A first example: $\cos(\pi x)$ on $[-1, 1]$

    A simple period-2 cosine is captured exactly by a single Fourier mode:
    """)
    return


@app.cell
def _():
    f = trigfun(lambda x: np.cos(np.pi * x), [-1, 1])
    print(f)
    print(f"len(f)            = {len(f)}")
    print(f"f(0.0)            = {f(0.0):+.15f}   (expected  1)")
    print(f"f(0.5)            = {f(0.5):+.15f}   (expected  0)")
    print(f"f(1.0)            = {f(1.0):+.15f}   (expected -1)")
    return (f,)


@app.cell
def _(f):
    _fig, _ax = plt.subplots()
    f.plot(ax=_ax)
    _ax.set_xlabel("x")
    _ax.set_title(r"$f(x) = \cos(\pi x)$ via $\mathrm{trigfun}$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compactness vs. Chebyshev

    For a smooth periodic target, a Fourier series typically requires far
    fewer coefficients than a Chebyshev series of comparable accuracy.
    Compare the lengths of two representations of the same function:
    """)
    return


@app.cell
def _():
    def target(x):
        return np.cos(8 * np.pi * x) + np.sin(3 * np.pi * x)

    fc = chebfun(target, [-1, 1])
    ft = trigfun(target, [-1, 1])
    print(f"chebfun length   = {len(fc):4d}   (Chebyshev coefficients)")
    print(f"trigfun length   = {len(ft):4d}   (Fourier coefficients)")
    _xx = np.linspace(-1.0, 1.0, 4001)
    _err_c = float(np.max(np.abs(fc(_xx) - target(_xx))))
    _err_t = float(np.max(np.abs(ft(_xx) - target(_xx))))
    print(f"max |error|      chebfun = {_err_c:.2e}")
    print(f"max |error|      trigfun = {_err_t:.2e}")
    return fc, ft


@app.cell
def _(fc, ft):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    fc.plotcoeffs(ax=_axes[0])
    _axes[0].set_title(f"Chebyshev coefficients (n = {len(fc)})")
    ft.plotcoeffs(ax=_axes[1])
    _axes[1].set_title(f"Fourier coefficients (n = {len(ft)})")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculus in Fourier space

    Differentiation and integration are spectral operations on the Fourier
    coefficients.  The result of `diff` or `cumsum` is itself a `trigfun`-
    backed Chebfun:
    """)
    return


@app.cell
def _():
    g = trigfun(lambda x: np.sin(np.pi * x), [-1, 1])
    dg = g.diff()
    print(f"sum(g)         = {g.sum():+.15f}   (expected 0)")
    print(f"dg(0.0)        = {dg(0.0):+.15f}   (expected π = {np.pi:.15f})")
    print(f"piece tech     = {type(dg.funs[0].onefun).__name__}")
    return dg, g


@app.cell
def _(dg, g):
    _fig, _ax = plt.subplots()
    g.plot(ax=_ax, label=r"$g(x) = \sin(\pi x)$")
    dg.plot(ax=_ax, label=r"$g'(x) = \pi \cos(\pi x)$")
    _ax.set_xlabel("x")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Roots of a periodic function

    `roots` works as usual; for $\sin(2\pi x)$ on $[-1, 1]$ we expect roots
    at $\{-1, -0.5, 0, 0.5, 1\}$:
    """)
    return


@app.cell
def _():
    h = trigfun(lambda x: np.sin(2 * np.pi * x), [-1, 1])
    print("roots:", np.array2string(h.roots(), precision=12, suppress_small=True))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A non-trivial periodic target

    The function $f(x) = e^{\sin(\pi x)}$ is smooth and exactly 2-periodic on
    $[-1, 1]$.  Its Fourier coefficients decay geometrically:
    """)
    return


@app.cell
def _():
    fp = trigfun(lambda x: np.exp(np.sin(np.pi * x)), [-1, 1])
    print(fp)
    print(f"len(fp)        = {len(fp)}")
    print(f"sum(fp)        = {fp.sum():.15f}   (≈ 2 * I_0(1) = {2.0 * 1.2660658777520084:.15f})")
    return (fp,)


@app.cell
def _(fp):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    fp.plot(ax=_axes[0])
    _axes[0].set_xlabel("x")
    _axes[0].set_title(r"$f(x) = e^{\sin(\pi x)}$")
    fp.plotcoeffs(ax=_axes[1])
    _axes[1].set_title("Fourier coefficient magnitudes")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When *not* to use `trigfun`

    `trigfun` assumes the target is smooth **and** periodic on the requested
    domain — i.e. $f(a) = f(b)$ along with all relevant derivatives.  If
    the target violates periodicity, the Fourier series suffers from the
    Gibbs phenomenon and convergence stalls.  The example below — a
    non-periodic linear function fit on $[-1, 1]$ as if it were periodic —
    illustrates the failure mode:
    """)
    return


@app.cell
def _():
    bad = trigfun(lambda x: x, [-1, 1])
    _xx = np.linspace(-1.0, 1.0, 4001)
    _err = float(np.max(np.abs(bad(_xx) - _xx)))
    print(f"len(bad)        = {len(bad)}")
    print(f"max |bad - x|   = {_err:.2e}    (large — Gibbs)")
    return (bad,)


@app.cell
def _(bad):
    _xx = np.linspace(-1.0, 1.0, 2001)
    _fig, _ax = plt.subplots()
    _ax.plot(_xx, _xx, color="C0", linewidth=2, label="$x$ (truth)")
    _ax.plot(_xx, bad(_xx), color="C1", linewidth=1.2, label="trigfun fit")
    _ax.set_xlabel("x")
    _ax.set_title("Non-periodic target: Gibbs oscillations near the endpoints")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - The [Periodic Functions](../user/features/periodic.md) feature page for
      a written summary of the `trigfun` API.
    - The [Gaussian Process Regression notebook](6_gaussian_process.html)
      for `gpr(..., trig=True)`, which produces a periodic posterior backed
      by `Trigtech`.
    """)
    return


if __name__ == "__main__":
    app.run()
