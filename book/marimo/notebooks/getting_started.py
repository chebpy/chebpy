"""Marimo notebook providing an introduction to ChebPy functionality and basic usage examples."""

# /// script
# dependencies = ["marimo==0.18.4", "chebpy", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebpy]
# path = "../../.."
# editable = true
# ///

import marimo

__generated_with = "0.14.16"
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


@app.cell
def _():
    from chebpy import chebfun

    return (chebfun,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""The function ``chebfun`` behaves in essentially the same way as its MATLAB counterpart.
        A good way to begin is to type:"""
    )
    return


@app.cell
def _(chebfun):
    x = chebfun("x", [0, 10])
    x
    return (x,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    What's happened here is we've instantiated a numerical representation of the identity function
    on the interval `[0,10]` and assigned this to a computer variable `x`. This particular representation
    has length 2, meaning that it is a degree one polynomial defined via two degrees of freedom
    (as you would expect of a linear function).

    An intuitive set of composition-like operations can now be performed. For instance here is
    the specification of a function `f` that oscillates with two modes:
    """
    )
    return


@app.cell
def _(x):
    f = np.sin(x) + np.sin(5 * x)
    f
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""The zeros of f can be computed via `roots`, which behind the scenes is implemented
        via a recursive subdivision algorithm in which a number of Colleague Matrix eigenvalue
        sub-problems are solved:"""
    )
    return


@app.cell
def _(f):
    r = f.roots()
    r
    return (r,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    By default ChebPy computations are accurate to machine precision, or approximately fifteen digits
    in double-precision arithmetic (see also the `UserPrefs` interface [here](./implementation.ipynb)).

    We can verify this for the computed roots of `f` by typing:
    """
    )
    return


@app.cell
def _(f, r):
    f(r)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The function and its roots can be plotted together as follows:""")
    return


@app.cell
def _(f, r):
    _ax = f.plot(linewidth=3)
    _ax.plot(r, f(r), ".r", markersize=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Calculus operations are natively possible with Chebfun objects.
        For example here is the derivative and indefinite integral of `f`:"""
    )
    return


@app.cell
def _(f):
    df = f.diff()
    if_ = f.cumsum()
    f.plot(linewidth=3)
    df.plot(linewidth=3)
    if_.plot(linewidth=3)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    One can verify analytically that the exact value of the definite integral here is `1.2 - cos(10) - 0.2cos(50)`.

    This matches our numerical integral (via Clenshaw-Curtis quadrature), which is computable in ChebPy
    via the `sum` command.
    """
    )
    return


@app.cell
def _(f):
    i_ana = 1.2 - np.cos(10) - 0.2 * np.cos(50)
    i_num = f.sum()
    print(f"analytical : I={i_ana}")
    print(f"    ChebPy : I={i_num}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Discontinuities

    Chebfun is capable of handling certain classes of mathematical nonsmoothness. For example, here we compute
    the pointwise maximum of two functions, which results in a 'piecewise-smooth' concatenation of twelve
    individual pieces (in Chebfun & ChebPy terminology this is a collection of 'Funs'). The breakpoints
    between the pieces (Funs) have been determined by ChebPy in the background by solving the corresponding
    root-finding problem.
    """
    )
    return


@app.cell
def _(f, x):
    g = x / 5 - 1
    h = f.maximum(g)
    h
    return g, h


@app.cell(hide_code=True)
def _():
    mo.md(r"""Here's a plot of both `f` and `g`, and their maximum, `h`:""")
    return


@app.cell
def _(f, g, h):
    _fig, _ax = plt.subplots()
    f.plot(ax=_ax, linewidth=3, linestyle="--", label="f")
    g.plot(ax=_ax, linewidth=3, linestyle="--", label="g")
    h.plot(ax=_ax, linewidth=3, label="max(f, g)")
    _ax.set_ylim([-2.5, 2.5])
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""The function `h` is a further Chebfun representation (Chebfun operations such as this are closures)
        and thus the same set of operations can be applied as normal. Here for instance is the exponential
        of `h` and its integral:"""
    )
    return


@app.cell
def _(h):
    np.exp(h).plot(linewidth=3)
    plt.show()
    print(f"integral: {np.exp(h).sum()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Probability distributions

    Here's a further example, this time related to statistics. We consider the following Chebfun representation
    of the standardised Gaussian distribution, using a sufficiently wide interval as to facilitate a machine-precision
    representation. On this occasion we utlilise a slightly different (but still perfectly valid) approach to
    construction whereby we supply the function handle (in this case, a Python lambda, but more generally any
    object in possession of a `__call__` attribute) together with the interval of definition.
    """
    )
    return


@app.cell
def _(chebfun):
    def gaussian(x):
        """Calculate the standard Gaussian probability density function.

        Args:
            x: Input value or array

        Returns:
            The value of the standard Gaussian PDF at x
        """
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

    pdf = chebfun(gaussian, [-15, 15])
    _ax = pdf.plot(linewidth=3)
    _ax.set_ylim([-0.05, 0.45])
    _ax.set_title("Standard Gaussian distribution (mean  0, variance 1)")
    plt.show()
    return (pdf,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""The integral of any probability density function should be unity,
        and this is the case for our numerical approximation:"""
    )
    return


@app.cell
def _(pdf):
    print(f"integral : {pdf.sum()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Suppose we wish to generate quantiles of the distribution. This can be achieved as follows.
    First we form the cumulative distribution function, computed as the indefinite integral
    (`cumsum`) of the density:
    """
    )
    return


@app.cell
def _(pdf):
    cdf = pdf.cumsum()
    _ax = cdf.plot(linewidth=3)
    _ax.set_ylim([-0.1, 1.1])
    plt.show()
    return (cdf,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Then it is simply a case of utilising the `roots` command to determine the standardised score
        (sometimes known as 'z-score') corresponding to the quantile of interest. For example:"""
    )
    return


@app.cell
def _(cdf):
    print("quantile    z-score ")
    print("--------------------")
    for quantile in np.arange(0.1, 0.0, -0.01):
        roots = (cdf - quantile).roots()
        print(f"  {quantile * 100:2.0f}%       {roots[0]:+5.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Other distributional properties are also computable. Here's how we can compute the first four
        normalised and centralised moments (Mean, Variance, Skew, Kurtosis):"""
    )
    return


@app.cell
def _(pdf):
    x_1 = pdf.x
    m1 = (pdf * x_1).sum()
    m2 = (pdf * (x_1 - m1) ** 2).sum()
    m3 = (pdf * (x_1 - m1) ** 3).sum() / m2**1.5
    m4 = (pdf * (x_1 - m1) ** 4).sum() / m2**2
    print(f"    mean = {m1:+.15f}")
    print(f"variance = {m2:+.15f}")
    print(f"    skew = {m3:+.15f}")
    print(f"kurtosis = {m4:+.15f}")
    return


if __name__ == "__main__":
    app.run()
