"""Marimo notebook demonstrating convolution of Chebfuns in ChebPy."""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebfun]
# path = "../../.."
# editable = true
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.function(hide_code=True)
def mark_breakpoints(chebfun, ax=None, **kwargs):
    """Mark breakpoints of a Chebfun on the given axes."""
    ax = ax or plt.gca()
    opts = {"color": "k", "marker": "o", "markersize": 6, "zorder": 5, "linestyle": "none"}
    opts.update(kwargs)
    bps = chebfun.breakpoints
    vals = chebfun(bps)
    ax.plot(bps, vals, **opts)
    return ax


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Convolution of Chebfuns

    ChebPy supports the **convolution** of two Chebfuns via the `.conv` method.

    Given two functions $f$ on $[a, b]$ and $g$ on $[c, d]$, their convolution is:

    $$
    (f \star g)(x) = \int f(t)\, g(x - t)\, dt
    $$

    The result is a piecewise Chebfun on $[a+c,\; b+d]$ whose breakpoints are the
    pairwise sums of the breakpoints of $f$ and $g$.  Both inputs may be
    **piecewise** (an arbitrary number of smooth pieces).

    For single-piece inputs of equal width, the fast Hale–Townsend algorithm is used.
    For general piecewise inputs, each output sub-interval is constructed via
    Gauss–Legendre quadrature.

    > N. Hale and A. Townsend, "An algorithm for the convolution of Legendre series",
    > *SIAM J. Sci. Comput.*, 36(3), A1207–A1220, 2014.
    """)
    return


@app.cell(hide_code=True)
def _():
    from chebpy.chebfun import Chebfun

    return (Chebfun,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 1: The triangle function

    The simplest example is $f = g = 1$ on $[-1, 1]$.  The convolution is the
    well-known **triangle function**:

    $$
    (1 \star 1)(x) = \max(0,\; 2 - |x|)
    $$

    which is supported on $[-2, 2]$ and peaks at $x = 0$ with value $2$.
    """)
    return


@app.cell
def _(Chebfun):
    ones = Chebfun.initconst(1.0, [-1, 1])
    triangle = ones.conv(ones)
    triangle
    return (triangle,)


@app.cell(hide_code=True)
def _(triangle):
    triangle.plot(linewidth=3)
    mark_breakpoints(triangle)
    plt.title("Triangle function: 1 ★ 1")
    plt.xlabel("x")
    plt.ylabel("(1 ★ 1)(x)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(triangle):
    mo.md(rf"""
    We can verify the expected values at a few key points:

    - $(1 \star 1)(0) = {float(triangle(0.0)):.6f}$ (expected $2$)
    - $(1 \star 1)(-1) = {float(triangle(-1.0)):.6f}$ (expected $1$)
    - $(1 \star 1)(1) = {float(triangle(1.0)):.6f}$ (expected $1$)
    - $(1 \star 1)(\pm 2) = 0$ (ends of support)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 2: Smooth convolution — $\sin \star \cos$

    Convolution of smooth functions produces a smooth result.  Let's compute
    $(\sin \star \cos)(x)$ on $[-1, 1]$ and plot the inputs alongside the output.
    """)
    return


@app.cell
def _(Chebfun):
    f_sin = Chebfun.initfun_adaptive(np.sin)
    f_cos = Chebfun.initfun_adaptive(np.cos)
    h_sincos = f_sin.conv(f_cos)
    h_sincos
    return f_cos, f_sin, h_sincos


@app.cell(hide_code=True)
def _(f_cos, f_sin, h_sincos):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("Input functions")
    f_sin.plot(ax=_ax1, linewidth=3)
    f_cos.plot(ax=_ax1, linewidth=3)
    mark_breakpoints(f_sin, _ax1, color="C0")
    mark_breakpoints(f_cos, _ax1, color="C1")
    _ax1.legend(["sin", "cos"])
    _ax1.set_xlabel("x")

    _ax2.set_title("sin ★ cos")
    h_sincos.plot(ax=_ax2, linewidth=3, color="C2")
    mark_breakpoints(h_sincos, _ax2)
    _ax2.set_xlabel("x")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 3: Commutativity

    Convolution is commutative: $f \star g = g \star f$.  Let's verify this
    numerically by comparing $\sin \star \cos$ with $\cos \star \sin$.
    """)
    return


@app.cell(hide_code=True)
def _(f_cos, f_sin, h_sincos):
    h_cossin = f_cos.conv(f_sin)
    xs = np.linspace(-1.8, 1.8, 200)
    err_commut = np.max(np.abs(h_sincos(xs) - h_cossin(xs)))
    print(f"‖sin★cos − cos★sin‖∞ = {err_commut:.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 4: Self-convolution of $e^x$

    We can also convolve a function with itself.  Here is $e^x \star e^x$:
    """)
    return


@app.cell
def _(Chebfun):
    f_exp = Chebfun.initfun_adaptive(np.exp)
    h_exp = f_exp.conv(f_exp)
    h_exp
    return f_exp, h_exp


@app.cell(hide_code=True)
def _(f_exp, h_exp):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("exp(x) on [-1, 1]")
    f_exp.plot(ax=_ax1, linewidth=3)
    mark_breakpoints(f_exp, _ax1)
    _ax1.set_xlabel("x")

    _ax2.set_title("exp ★ exp")
    h_exp.plot(ax=_ax2, linewidth=3, color="C3")
    mark_breakpoints(h_exp, _ax2)
    _ax2.set_xlabel("x")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 5: Linearity

    Convolution is **linear**: $(af + bg) \star h = a(f \star h) + b(g \star h)$.

    Let's verify with $a = 2$, $b = -3$, $f = \sin$, $g = \cos$, $h = e^x$:
    """)
    return


@app.cell(hide_code=True)
def _(Chebfun, f_cos, f_sin):
    a_coeff, b_coeff = 2.0, -3.0
    h_rhs = Chebfun.initfun_adaptive(np.exp)

    lhs_fun = (a_coeff * f_sin + b_coeff * f_cos).conv(h_rhs)
    rhs_fun = a_coeff * f_sin.conv(h_rhs) + b_coeff * f_cos.conv(h_rhs)

    xs_lin = np.linspace(-1.8, 1.8, 200)
    err_linearity = np.max(np.abs(lhs_fun(xs_lin) - rhs_fun(xs_lin)))
    print(f"‖(2sin − 3cos) ★ exp − (2(sin★exp) − 3(cos★exp))‖∞ = {err_linearity:.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 6: B-spline construction by repeated convolution

    The `conv` method handles **piecewise Chebfuns** with an arbitrary number
    of smooth pieces.  A beautiful illustration is the construction of
    **cardinal B-splines** via repeated convolution of the box function
    $B_0 = \mathbf{1}_{[-1/2,\,1/2]}$:

    $$
    B_0 = \mathbf{1}_{[-\tfrac12,\tfrac12]}, \qquad
    B_n = B_0 \star B_{n-1}, \quad n = 1, 2, 3, \ldots
    $$

    Each convolution increases smoothness by one order — $B_0$ is $C^{-1}$
    (discontinuous), $B_1$ is $C^0$ (the hat/tent function), $B_2$ is $C^1$
    (quadratic B-spline), and $B_3$ is $C^2$ (cubic B-spline).  The support
    grows by 1 at each step.

    *Cf.* the MATLAB Chebfun example
    [BSplineConv](https://www.chebfun.org/examples/approx/BSplineConv.html).
    """)
    return


@app.cell(hide_code=True)
def _(Chebfun):
    B0 = Chebfun.initconst(1.0, [-0.5, 0.5])
    B1 = B0.conv(B0)
    B2 = B0.conv(B1)
    B3 = B0.conv(B2)

    _splines = [B0, B1, B2, B3]
    _titles = [
        "$B_0$: box  (1 piece)",
        "$B_1$: hat / linear  (2 pieces)",
        "$B_2$: quadratic  (3 pieces)",
        "$B_3$: cubic  (4 pieces)",
    ]
    _colors = ["C0", "C1", "C2", "C3"]

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    for _ax, _b, _t, _c in zip(_axes.flat, _splines, _titles, _colors, strict=False):
        _b.plot(ax=_ax, linewidth=3, color=_c)
        mark_breakpoints(_b, _ax, color=_c)
        _ax.set_title(_t)
        _ax.set_xlabel("x")
        _ax.set_xlim(-2.5, 2.5)
        _ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.show()
    return B0, B1, B2, B3


@app.cell(hide_code=True)
def _(B0, B1, B2, B3):
    mo.md(rf"""
    | Spline | Pieces | Support | Continuity | Peak at 0 |
    |--------|--------|---------|------------|-----------|
    | $B_0$  | {B0.funs.size} | $[-1/2,\; 1/2]$ | $C^{{-1}}$ | {float(B0(0.0)):.4f} |
    | $B_1$  | {B1.funs.size} | $[-1,\; 1]$ | $C^0$ | {float(B1(0.0)):.4f} |
    | $B_2$  | {B2.funs.size} | $[-3/2,\; 3/2]$ | $C^1$ | {float(B2(0.0)):.4f} |
    | $B_3$  | {B3.funs.size} | $[-2, 2]$ | $C^2$ | {float(B3(0.0)):.4f} |

    Each B-spline integrates to $1$ and the peak value decreases as the
    support widens.  The breakpoints (marked with dots) are at the integers and
    half-integers — exactly the pairwise sums of the input breakpoints, as
    expected from the convolution theorem.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 7: Boundary behaviour

    The convolution of two functions on $[-1, 1]$ always vanishes at the
    endpoints $x = \pm 2$ of the result domain, because the overlap region
    between $f$ and the shifted copy of $g$ becomes empty.
    """)
    return


@app.cell(hide_code=True)
def _(h_exp):
    _a, _b = float(h_exp.domain[0]), float(h_exp.domain[-1])
    print(f"(exp★exp)({_a}) = {float(h_exp(_a)):.2e}")
    print(f"(exp★exp)({_b}) = {float(h_exp(_b)):.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example 8: Probability — PDF of a sum of independent random variables

    A classic application of convolution is in **probability theory**.  If $X$
    and $Y$ are independent continuous random variables with PDFs $f_X$ and
    $f_Y$, then the PDF of $Z = X + Y$ is the convolution $f_Z = f_X \star f_Y$.

    ### Uniform + Uniform → Triangular

    Let $X, Y \sim \mathrm{Uniform}(0, 1)$, so $f_X = f_Y = 1$ on $[0, 1]$.
    The sum $Z = X + Y$ has the **triangular distribution** on $[0, 2]$, peaking
    at $z = 1$.
    """)
    return


@app.cell(hide_code=True)
def _(Chebfun):
    pdf_uniform = Chebfun.initconst(1.0, [0, 1])
    pdf_sum_uniform = pdf_uniform.conv(pdf_uniform)

    pdf_sum_uniform.plot(linewidth=3)
    mark_breakpoints(pdf_sum_uniform)
    plt.title("PDF of $Z = X + Y$,  $X, Y \\sim \\mathrm{Uniform}(0,1)$")
    plt.xlabel("z")
    plt.ylabel("$f_Z(z)$")
    plt.show()
    return (pdf_sum_uniform,)


@app.cell(hide_code=True)
def _(pdf_sum_uniform):
    _total_area = pdf_sum_uniform.sum()
    _peak = float(pdf_sum_uniform(1.0))
    mo.md(rf"""
    As expected, the result integrates to 1 (total probability =
    ${_total_area:.15f}$) and peaks at $z = 1$ with $f_Z(1) = {_peak:.6f}$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Quadratic PDF + Quadratic PDF

    For a less trivial example, consider two independent random variables each
    with PDF $f(x) = 6x(1 - x)$ on $[0, 1]$ (the $\mathrm{Beta}(2, 2)$
    distribution, a symmetric bell on $[0, 1]$).

    The PDF of their sum — the **Irwin–Hall-like** distribution — is obtained
    by convolution and lives on $[0, 2]$.
    """)
    return


@app.cell(hide_code=True)
def _(Chebfun):
    pdf_beta22 = Chebfun.initfun_adaptive(lambda x: 6.0 * x * (1.0 - x), [0, 1])
    pdf_sum_beta = pdf_beta22.conv(pdf_beta22)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("$f_X(x) = 6x(1-x)$  (Beta(2,2) PDF)")
    pdf_beta22.plot(ax=_ax1, linewidth=3)
    mark_breakpoints(pdf_beta22, _ax1)
    _ax1.set_xlabel("x")
    _ax1.set_ylabel("$f_X(x)$")

    _ax2.set_title("PDF of $Z = X + Y$")
    pdf_sum_beta.plot(ax=_ax2, linewidth=3, color="C1")
    mark_breakpoints(pdf_sum_beta, _ax2)
    _ax2.set_xlabel("z")
    _ax2.set_ylabel("$f_Z(z)$")

    plt.tight_layout()
    plt.show()
    return (pdf_sum_beta,)


@app.cell(hide_code=True)
def _(pdf_sum_beta):
    _area = pdf_sum_beta.sum()
    mo.md(rf"""
    Again the total area is ${_area:.15f} \approx 1$, confirming a valid PDF.
    Note how convolving two bell-shaped distributions produces a smoother, more
    concentrated bell — an illustration of the **Central Limit Theorem** in action.
    """)
    return


if __name__ == "__main__":
    app.run()
