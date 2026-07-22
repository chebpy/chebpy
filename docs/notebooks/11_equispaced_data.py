"""Marimo notebook demonstrating equifun for equispaced sample data in ChebPy."""

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
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from chebpy import equifun

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Equispaced Sample Data

    Most of ChebPy's constructors expect a *callable* — a function that can be
    evaluated at Chebyshev points.  Sometimes, though, all you have is a vector
    of values sampled on an **equispaced** grid that includes both endpoints of
    the interval (measured data, the output of another simulation, a lookup
    table).  Interpolating equispaced data with a single high-degree polynomial
    is famously unstable: it triggers the **Runge phenomenon**, and no fast,
    stable scheme can do better in general (Platte, Trefethen & Kuijlaars,
    2011).

    ChebPy's [`equifun`](../api.md) sidesteps this by first fitting a
    **Floater–Hormann barycentric rational interpolant** through the samples and
    then adaptively representing that smooth interpolant as an ordinary Chebfun.
    The result behaves like any other Chebfun — you can differentiate,
    integrate, find roots, and plot it.

    This mirrors MATLAB Chebfun's `chebfun(values, 'equi')` constructor, whose
    `funqui` routine implements the same Floater–Hormann fit and adaptive
    degree selection.  ChebPy keeps it as a standalone `equifun` factory (in the
    spirit of `trigfun`) rather than overloading `chebfun`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A first example

    Given values of a smooth function on an equispaced grid, `equifun`
    reconstructs a Chebfun that interpolates the samples.  The domain defaults
    to the preference domain $[-1, 1]$; pass a two-element `[a, b]` to place the
    grid on any bounded interval.
    """)
    return


@app.cell
def _():
    nodes = np.linspace(0.0, 2.0 * np.pi, 17)
    values = np.sin(nodes) + 0.25 * np.cos(3.0 * nodes)
    f = equifun(values, [0.0, 2.0 * np.pi])
    print(f)
    print(f"len(f)            = {len(f)}")
    print(f"max |f(node) - y| = {float(np.max(np.abs(f(nodes) - values))):.2e}")
    return f, nodes, values


@app.cell
def _(f, nodes, values):
    _xx = np.linspace(0.0, 2.0 * np.pi, 500)
    _fig, _ax = plt.subplots()
    _ax.plot(_xx, f(_xx), color="C0", linewidth=2, label="equifun")
    _ax.plot(nodes, values, "o", color="C1", label="samples")
    _ax.set_xlabel("x")
    _ax.set_title("Chebfun through equispaced samples")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Runge phenomenon

    The classic cautionary tale is Runge's function
    $r(x) = 1/(1 + 25x^2)$ on $[-1, 1]$.  Interpolating it by a single
    polynomial through equispaced nodes diverges wildly near the endpoints as
    the node count grows.  The Floater–Hormann fit used by `equifun` stays
    bounded and converges.
    """)
    return


@app.cell
def _():
    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    _nodes = np.linspace(-1.0, 1.0, 25)
    fh = equifun(runge(_nodes))

    _coeffs = np.polyfit(_nodes, runge(_nodes), _nodes.size - 1)
    _xx = np.linspace(-1.0, 1.0, 2001)
    _err_fh = float(np.max(np.abs(fh(_xx) - runge(_xx))))
    _err_poly = float(np.max(np.abs(np.polyval(_coeffs, _xx) - runge(_xx))))
    print(f"equifun (Floater-Hormann)  max |error| = {_err_fh:.2e}")
    print(f"degree-24 polynomial       max |error| = {_err_poly:.2e}")
    return fh, runge


@app.cell
def _(fh, runge):
    _nodes = np.linspace(-1.0, 1.0, 25)
    _coeffs = np.polyfit(_nodes, runge(_nodes), _nodes.size - 1)
    _xx = np.linspace(-1.0, 1.0, 2001)

    _fig, _ax = plt.subplots()
    _ax.plot(_xx, runge(_xx), color="C0", linewidth=2, label="$r(x) = 1/(1+25x^2)$")
    _ax.plot(_xx, fh(_xx), color="C1", linewidth=1.5, label="equifun")
    _ax.plot(_xx, np.polyval(_coeffs, _xx), color="C3", linewidth=1.2, label="degree-24 polynomial")
    _ax.plot(_nodes, runge(_nodes), "k.", label="samples")
    _ax.set_ylim(-1.0, 2.0)
    _ax.set_xlabel("x")
    _ax.set_title("Equispaced interpolation of Runge's function")
    _ax.legend(loc="upper right", fontsize="small")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convergence under grid refinement

    As the equispaced grid is refined, the reconstruction error decreases
    steadily — the opposite of the polynomial's divergence.  Note that the
    length of the resulting Chebfun stays modest: the Floater–Hormann
    interpolant is smooth, so its adaptive Chebyshev representation is compact.
    """)
    return


@app.cell
def _(runge):
    _xx = np.linspace(-1.0, 1.0, 4001)
    print(f"{'samples':>8}  {'len(f)':>7}  {'max |error|':>12}")
    print("-" * 32)
    for _n in (17, 33, 65, 129, 257):
        _nodes = np.linspace(-1.0, 1.0, _n)
        _f = equifun(runge(_nodes))
        _err = float(np.max(np.abs(_f(_xx) - runge(_xx))))
        print(f"{_n:>8d}  {len(_f):>7d}  {_err:>12.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculus on reconstructed data

    Because `equifun` returns an ordinary Chebfun, the full toolbox applies.
    Here we recover a derivative and a definite integral directly from sampled
    values of $g(x) = \sin(\pi x)$ on $[-1, 1]$:
    """)
    return


@app.cell
def _():
    _nodes = np.linspace(-1.0, 1.0, 33)
    g = equifun(np.sin(np.pi * _nodes))
    dg = g.diff()
    print(f"sum(g)   = {float(g.sum()):+.12f}   (expected 0)")
    print(f"dg(0.0)  = {float(dg(0.0)):+.12f}   (expected pi = {np.pi:.12f})")
    return dg, g


@app.cell
def _(dg, g):
    _xx = np.linspace(-1.0, 1.0, 500)
    _fig, _ax = plt.subplots()
    _ax.plot(_xx, g(_xx), color="C0", linewidth=2, label=r"$g(x) = \sin(\pi x)$")
    _ax.plot(_xx, dg(_xx), color="C1", linewidth=2, label=r"$g'(x) = \pi \cos(\pi x)$")
    _ax.set_xlabel("x")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Small-sample edge cases

    `equifun` degrades gracefully when there is very little data:

    - a **single** value produces the corresponding constant Chebfun, and
    - **two** values produce the straight line joining them.

    Both real and complex sample vectors are supported.
    """)
    return


@app.cell
def _():
    const = equifun([2.5], [2.0, 5.0])
    line = equifun([-3.0, 3.0])
    cplx = equifun([1.0 + 1.0j, 0.0, -1.0 + 1.0j])
    print(f"single sample : f(3.0) = {float(const(3.0)):.3f}   (constant)")
    print(f"two samples   : f(0.0) = {float(line(0.0)):.3f}   (linear midpoint)")
    print(f"complex data  : f(0.0) = {complex(cplx(0.0)):.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## See also

    - The [Function Approximation](../user/features/approximation.md) feature
      page for a written summary of `equifun`.
    - MATLAB Chebfun's `chebfun(values, 'equi')` constructor and its `funqui`
      routine, which ChebPy's implementation follows.

    ## References

    - M. S. Floater and K. Hormann,
      *Barycentric rational interpolation with no poles and high rates of
      approximation*,
      [Numer. Math. 107, 315–331 (2007)](https://doi.org/10.1007/s00211-007-0093-y).
    - R. B. Platte, L. N. Trefethen, and A. B. J. Kuijlaars,
      *Impossibility of fast stable approximation of analytic functions from
      equispaced samples*,
      [SIAM Rev. 53(2), 308–318 (2011)](https://doi.org/10.1137/090774707).
    - T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
      [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
      Pafnuty Publications, 2014.
    """)
    return


if __name__ == "__main__":
    app.run()
