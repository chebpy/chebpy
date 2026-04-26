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

__generated_with = "0.23.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

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
    - Functions that **don't** decay to zero — heavy tails like $1/(1+x^2)$,
      slowly-decaying oscillations, or functions approaching a non-zero asymptote
      like $\tanh$ — are explicitly refused with a
      `CompactFunConstructionError`.  These cases are out of scope for `v1`.
    """)
    return


@app.cell
def _():
    from chebpy import chebfun
    from chebpy.exceptions import CompactFunConstructionError

    return CompactFunConstructionError, chebfun


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A decaying oscillation on $[0, \infty)$

    The Chebfun guide opens with $f(x) = 0.75 + \sin(10x)/e^x$ on $[0, \infty)$.
    Below we represent the decaying part $\sin(10x)/e^x$ as a `CompactFun` on
    $[0, \infty)$ and add the constant $0.75$ separately:
    """)
    return


@app.cell
def _(chebfun):
    f_decay = chebfun(lambda x: np.sin(10 * x) * np.exp(-x), [0, np.inf])
    _a, _b = f_decay.funs[0].numerical_support
    print(f"Numerical support: [{_a:.3f}, {_b:.3f}]")
    print(f"Logical support:   {tuple(f_decay.funs[0].support)}")
    print(f"Length:            {f_decay.funs[0].size}")
    return (f_decay,)


@app.cell
def _(f_decay):
    # Find the maximum of 0.75 + sin(10x)/exp(x) on [0, ∞) via critical points.
    f_total = f_decay + 0.75
    _crit = f_total.diff().roots()
    _a, _b = f_decay.funs[0].numerical_support
    _candidates = np.concatenate([_crit, [_a, _b]])
    _values = f_total(_candidates)
    _idx = int(np.argmax(_values))
    print(f"argmax (in storage window) = {_candidates[_idx]:.15f}")
    print(f"max value                  = {_values[_idx]:.15f}")
    return (f_total,)


@app.cell
def _(f_total):
    _a, _b = f_total.funs[0].numerical_support
    _xs = np.linspace(0.0, max(_b, 6.0), 800)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, f_total(_xs))
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
def _(chebfun):
    import math

    # Use 1/Γ(x+1) = exp(-lgamma(x+1)) so the probe at large x underflows to 0
    # cleanly rather than overflowing math.gamma.
    _rgamma = np.frompyfunc(lambda y: math.exp(-math.lgamma(y + 1.0)), 1, 1)
    g = chebfun(lambda x: np.asarray(_rgamma(x), dtype=float), [0, np.inf])
    print(g)
    print(f"sum(g) = {g.sum():.15f}")
    return (g,)


@app.cell
def _(g):
    _a, _b = g.funs[0].numerical_support
    _xs = np.linspace(0.0, _b, 600)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, g(_xs))
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
def _(chebfun):
    h = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
    print(h)
    print(f"sum(h)  = {h.sum():.15f}")
    print(f"√π      = {np.sqrt(np.pi):.15f}")
    return (h,)


@app.cell
def _(h):
    _xs = np.linspace(-6.0, 6.0, 600)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, h(_xs))
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
def _(chebfun):
    pdf = chebfun(lambda x: np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi), [-np.inf, np.inf])
    pdf2 = pdf.conv(pdf)
    print(f"sum(pdf)        = {pdf.sum():.15f}  (expected 1)")
    print(f"sum(pdf*pdf)    = {pdf2.sum():.15f}  (expected 1)")
    _expected = 1.0 / np.sqrt(4.0 * np.pi)
    print(f"(pdf*pdf)(0)    = {pdf2(0.0):.15f}")
    print(f"1/sqrt(4*pi)    = {_expected:.15f}")
    return pdf, pdf2


@app.cell
def _(pdf, pdf2):
    _xs = np.linspace(-6.0, 6.0, 600)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, pdf(_xs), label=r"$\phi(x)$ — $\mathcal{N}(0,1)$")
    _ax.plot(_xs, pdf2(_xs), label=r"$\phi \star \phi$ — $\mathcal{N}(0,2)$")
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


@app.cell
def _(chebfun):
    expo = chebfun(lambda x: np.exp(-x), [0, np.inf])
    gamma2 = expo.conv(expo)
    print(f"sum(expo*expo)        = {gamma2.sum():.15f}  (expected 1)")
    print(f"(expo*expo)(1)        = {gamma2(1.0):.15f}")
    print(f"1*exp(-1)             = {np.exp(-1.0):.15f}")
    print(f"piece types: {[type(_p).__name__ for _p in gamma2.funs]}")
    return expo, gamma2


@app.cell
def _(expo, gamma2):
    _xs = np.linspace(0.0, 12.0, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, expo(_xs), label=r"$e^{-x}$")
    _ax.plot(_xs, gamma2(_xs), label=r"$x \, e^{-x}$")
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
    - $\tanh(x-1)$ — does not decay; approaches $\pm 1$.

    Each of the following raises a `CompactFunConstructionError`:
    """)
    return


@app.cell
def _(CompactFunConstructionError, chebfun):
    _refused = []
    for _label, _f in [
        ("Cauchy 1/(π(1+x²))", lambda x: 1.0 / (np.pi * (1.0 + x * x))),
        ("1/(1+|x|)", lambda x: 1.0 / (1.0 + np.abs(x))),
        ("tanh(x-1)", lambda x: np.tanh(x - 1.0)),
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
    ## Mixed piecewise: finite breakpoints with infinite endpoints

    `chebfun(f, [-inf, a₁, …, a_k, +inf])` produces a `Chebfun` whose two outer
    pieces are `CompactFun` and whose interior pieces are ordinary `Bndfun`s:
    """)
    return


@app.cell
def _(chebfun):
    p = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, -2.0, 0.0, 3.0, np.inf])
    print(p)
    print(f"piece types: {[type(_piece).__name__ for _piece in p.funs]}")
    print(f"sum(p)  = {p.sum():.15f}")
    print(f"√π      = {np.sqrt(np.pi):.15f}")
    return (p,)


@app.cell
def _(p):
    _xs = np.linspace(-6.0, 6.0, 600)
    _fig, _ax = plt.subplots()
    _ax.plot(_xs, p(_xs))
    for _bp in p.breakpoints[1:-1]:
        _ax.axvline(float(_bp), color="grey", linestyle="--", linewidth=0.8)
    _ax.set_xlabel("x")
    _ax.set_title("Piecewise Gaussian on $[-\\infty, -2, 0, 3, +\\infty]$")
    _fig
    return


if __name__ == "__main__":
    app.run()
