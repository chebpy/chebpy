"""Marimo notebook demonstrating CompactFun support for (semi-)infinite intervals."""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebfun]
# path = "../.."
# editable = true
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()

with app.setup:
    import math

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from chebpy import chebfun
    from chebpy.exceptions import CompactFunConstructionError
    from chebpy.settings import _preferences as prefs

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Infinite Intervals

    This notebook revisits §9.1 of the
    [Chebfun guide](https://www.chebfun.org/docs/guide/guide09.html), adapted to ChebPy's
    `CompactFun` machinery.  In MATLAB Chebfun, functions on $(-\infty, \infty)$ or
    $[a, \infty)$ are represented via a rational change of variables that maps the
    unbounded domain onto $[-1, 1]$.

    ChebPy takes a deliberately different route.  A `CompactFun` is a `Classicfun`
    on a finite *storage interval* $[a', b']$ where the function is **numerically
    nonzero**, and is reported as identically zero outside that interval.  The
    user-facing logical interval may still extend to $\pm\infty$.  This approach
    has two consequences for the examples below:

    - Functions that decay rapidly to zero (Gaussians, decaying exponentials) are
      handled cleanly and accurately.
    - Functions approaching a **non-zero asymptote** ($\tanh$, logistic
      sigmoids, …) are also supported: the asymptotic limits are detected
      automatically and stored as `tail_left` / `tail_right` metadata on the
      `CompactFun`.  Outside the storage interval the function evaluates to
      its tail constants rather than to zero.
    - Functions that genuinely fail the model — heavy algebraic tails like
      $1/(1+x^2)$, slowly-decaying oscillations, or non-convergent tails — are
      still explicitly refused with a `CompactFunConstructionError`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A decaying oscillation on $[0, \infty)$

    The Chebfun guide opens with $f(x) = 0.75 + \sin(10x)/e^x$ on $[0, \infty)$.
    The function approaches the non-zero asymptote $0.75$ as $x \to \infty$, so
    the probe records `tail_right = 0.75` automatically — the whole expression
    fits inside a single `CompactFun`:
    """)
    return


@app.cell
def _():
    f_total = chebfun(lambda x: 0.75 + np.sin(10 * x) * np.exp(-x), [0, np.inf])
    print(f_total)
    print("")
    print(f"tail_right = {f_total.funs[0].tail_right:.15f}   (expected 0.75)")
    return (f_total,)


@app.cell(hide_code=True)
def _(f_total):
    # Find the maximum of 0.75 + sin(10x)/exp(x) on [0, ∞) via critical points.
    _crit = f_total.diff().roots()
    _a, _b = f_total.funs[0].numerical_support
    _candidates = np.concatenate([_crit, [_a, _b]])
    _values = f_total(_candidates)
    _idx = int(np.argmax(_values))
    print(f"argmax (in storage window) = {_candidates[_idx]:.15f}")
    print(f"max value                  = {_values[_idx]:.15f}")
    return


@app.cell
def _(f_total):
    _fig, _ax = plt.subplots()
    with prefs:
        prefs.N_plot = 10001
        f_total.plot(ax=_ax)
    _ax.set_xlim(0.0, 8)
    _ax.set_xlabel("x")
    _ax.set_title(r"$f(x) = 0.75 + \sin(10x)/e^x$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## $1/\Gamma(x+1)$ on $[0, \infty)$

    Reciprocal-factorial decays super-exponentially, so it is a textbook
    `CompactFun` candidate:
    """)
    return


@app.cell
def _():
    _rgamma = np.frompyfunc(lambda y: math.exp(-math.lgamma(y + 1.0)), 1, 1)
    g = chebfun(lambda x: np.asarray(_rgamma(x), dtype=float), [0, np.inf])
    print(g)
    print(f"sum(g) = {g.sum():.15f}")
    return (g,)


@app.cell
def _(g):
    _b = float(g.funs[0].numerical_support[1])
    _fig, _ax = plt.subplots()
    g.plot(ax=_ax)
    _ax.set_xlim(0.0, _b)
    _ax.set_xlabel("x")
    _ax.set_title(r"$g(x) = 1/\Gamma(x+1)$ on $[0, \infty)$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A doubly-infinite Gaussian

    The bread-and-butter case for `CompactFun`: the standard Gaussian on
    $(-\infty, \infty)$.
    """)
    return


@app.cell
def _():
    h = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
    print(h)
    print()
    print(f"sum(h)  = {h.sum():.15f}")
    print(f"√π      = {np.sqrt(np.pi):.15f}")
    return (h,)


@app.cell
def _(h):
    _fig, _ax = plt.subplots()
    h.plot(ax=_ax)
    _ax.set_xlim(-6.0, 6.0)
    _ax.set_xlabel("x")
    _ax.set_title(r"$h(x) = e^{-x^2}$ on $(-\infty, \infty)$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convolution: density of the sum of two independent Gaussians

    Because each `CompactFun` lives on a finite storage interval, the existing
    Hale–Townsend Legendre convolution machinery applies directly.  Convolving the
    standard Gaussian density with itself produces $\mathcal{N}(0, 2)$:
    """)
    return


@app.cell
def _():
    def _f(x):
        return np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)

    pdf1 = chebfun(_f, [-np.inf, np.inf])
    pdf2 = pdf1.conv(pdf1)

    print(pdf1)
    print()
    print(pdf2)
    return pdf1, pdf2


@app.cell(hide_code=True)
def _(pdf1, pdf2):
    print(f"sum(pdf1)        = {pdf1.sum():.15f}  (expected 1)")
    print(f"sum(pdf1*pdf1)   = {pdf2.sum():.15f}  (expected 1)")
    _expected = 1.0 / np.sqrt(4.0 * np.pi)
    print()
    print(f"(pdf1*pdf1)(0)   = {pdf2(0.0):.15f}")
    print(f"1/sqrt(4*pi)     = {_expected:.15f}")
    return


@app.cell
def _(pdf1, pdf2):
    _fig, _ax = plt.subplots()
    pdf1.plot(ax=_ax, label=r"$\phi(x)$ — $\mathcal{N}(0,1)$")
    pdf2.plot(ax=_ax, label=r"$\phi \star \phi$ — $\mathcal{N}(0,2)$")
    _ax.set_xlim(-6.0, 6.0)
    _ax.set_xlabel("x")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sum of two exponentials: Gamma$(2, 1)$

    Convolving $e^{-x}$ on $[0, \infty)$ with itself yields the $\mathrm{Gamma}(2,1)$
    density $x \, e^{-x}$:
    """)
    return


@app.cell(hide_code=True)
def _():
    expo = chebfun(lambda x: np.exp(-x), [0, np.inf])
    gamma2 = expo.conv(expo)
    print(f"sum(expo*expo)        = {gamma2.sum():.15f}  (expected 1)")
    print(f"(expo*expo)(1)        = {gamma2(1.0):.15f}")
    print(f"1*exp(-1)             = {np.exp(-1.0):.15f}")
    print()
    print(f"piece types: {[type(_p).__name__ for _p in gamma2.funs]}")
    return expo, gamma2


@app.cell
def _(expo, gamma2):
    _fig, _ax = plt.subplots()
    expo.plot(ax=_ax, label=r"$e^{-x}$")
    gamma2.plot(ax=_ax, label=r"$x \, e^{-x}$")
    _ax.set_xlim(0.0, 12.0)
    _ax.set_xlabel("x")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What gets refused

    A central design choice of `CompactFun` is to **honestly refuse** inputs that
    cannot be represented under the numerical-support model.  These are exactly
    the cases where MATLAB Chebfun's rational map silently produces a result of
    questionable accuracy:

    - The Cauchy density $\dfrac{1}{\pi(1+x^2)}$ — heavy tails, only $O(1/x^2)$ decay.
    - $\dfrac{1}{1+|x|}$ — even heavier tails, $O(1/x)$ decay.
    - $\sin(x)$ — non-convergent oscillation at $\pm\infty$.

    Each of the following raises a `CompactFunConstructionError`:
    """)
    return


@app.cell(hide_code=True)
def _():
    _refused = []
    for _label, _f in [
        ("Cauchy 1/(π(1+x²))", lambda x: 1.0 / (np.pi * (1.0 + x * x))),
        ("1/(1+|x|)", lambda x: 1.0 / (1.0 + np.abs(x))),
        ("sin(x)", lambda x: np.sin(x)),
    ]:
        try:
            chebfun(_f, [-np.inf, np.inf])
            _refused.append((_label, "ACCEPTED (unexpected)"))
        except CompactFunConstructionError as _err:
            _refused.append((_label, str(_err).split(";")[0]))
    for _label, _message in _refused:
        print(f"{_label:25s} -> {_message}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sigmoid-like inputs: non-zero tail constants

    Functions that approach **finite, non-zero** asymptotic limits at $\pm\infty$
    — $\tanh$, the logistic sigmoid, smoothed step functions — are supported via
    `(tail_left, tail_right)` metadata on the `CompactFun`.  The constants are
    detected automatically by the same probe that locates the numerical-support
    window, so the user does not need to pass them explicitly.

    Outside the storage interval the function evaluates to its tail constants
    rather than to zero, and arithmetic with scalars or other `CompactFun`s
    propagates the tails (e.g. $-f$ flips both, $\alpha f$ scales them, and
    $f + c$ shifts them).
    """)
    return


@app.cell
def _():
    tanh = chebfun(np.tanh, [-np.inf, np.inf])
    tanh
    return (tanh,)


@app.cell(hide_code=True)
def _(tanh):
    _piece = tanh.funs[0]
    print(f"  tail_left   =  {_piece.tail_left:+.15f}")
    print(f"  tail_right  =  {_piece.tail_right:+.15f}")
    print()
    print(f"tanh(-1e10)  ->  {tanh(-1e10):+.15f}   (returns tail_left)")
    print(f"tanh(0.0)    ->  {tanh(0.0):+.15f}")
    print(f"tanh(+1e10)  ->  {tanh(+1e10):+.15f}   (returns tail_right)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Algebraic operations that would diverge on an unbounded domain are still
    refused — but now with a `DivergentIntegralError` that names the offending
    operation, rather than silently returning a wrong number:

    - `f.sum()` on a function with a non-zero tail on an infinite side.
    - `f.cumsum()` on the same (the antiderivative is unbounded).
    - `f.conv(g)` whenever either operand has a non-zero tail.

    These errors point users at the natural escape hatch: subtract a matched
    sigmoid first so the residual has zero tails, operate on the residual, then
    re-add the sigmoid analytically.
    """)
    return


@app.cell
def _(tanh):
    from chebpy.exceptions import DivergentIntegralError

    try:
        tanh.sum()
    except DivergentIntegralError as _err:
        print(f"t.sum() refused: {_err}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixed piecewise: finite breakpoints with infinite endpoints

    `chebfun(f, [-inf, a₁, …, a_k, +inf])` produces a `Chebfun` whose two outer
    pieces are `CompactFun` and whose interior pieces are ordinary `Bndfun`s:
    """)
    return


@app.cell(hide_code=True)
def _():
    p = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, -2.0, 0.0, 3.0, np.inf])
    print(p)
    print()
    print(f"piece types: {[type(_piece).__name__ for _piece in p.funs]}")
    print()
    print(f"sum(p)  = {p.sum():.15f}")
    print(f"√π      = {np.sqrt(np.pi):.15f}")
    return (p,)


@app.cell
def _(p):
    _fig, _ax = plt.subplots()
    p.plot(ax=_ax)
    for _bp in p.breakpoints[1:-1]:
        _ax.axvline(float(_bp), color="grey", linestyle="--", linewidth=0.8)
    _ax.set_xlim(-6.0, 6.0)
    _ax.set_xlabel("x")
    _ax.set_title("Piecewise Gaussian on $[-\\infty, -2, 0, 3, +\\infty]$")
    _fig
    return


if __name__ == "__main__":
    app.run()
