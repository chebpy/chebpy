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

__generated_with = "0.20.4"
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
    # Convolution of Chebfuns

    ChebPy supports the **convolution** of two Chebfuns via the `.conv` method.

    Given two functions $f$ and $g$ defined on $[a, b]$, their convolution is:

    $$
    (f \star g)(x) = \int_a^b f(t)\, g(x - t)\, dt
    $$

    The result is a piecewise Chebfun on $[2a, 2b]$ with a breakpoint at $a + b$.

    The implementation uses the Hale–Townsend algorithm, which converts to Legendre
    coefficients, performs the convolution analytically, and converts back to Chebyshev.

    > N. Hale and A. Townsend, "An algorithm for the convolution of Legendre series",
    > *SIAM J. Sci. Comput.*, 36(3), A1207–A1220, 2014.
    """)
    return


@app.cell
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


@app.cell
def _(triangle):
    triangle.plot(linewidth=3)
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


@app.cell
def _(f_cos, f_sin, h_sincos):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("Input functions")
    f_sin.plot(ax=_ax1, linewidth=3)
    f_cos.plot(ax=_ax1, linewidth=3)
    _ax1.legend(["sin", "cos"])
    _ax1.set_xlabel("x")

    _ax2.set_title("sin ★ cos")
    h_sincos.plot(ax=_ax2, linewidth=3, color="C2")
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


@app.cell
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


@app.cell
def _(f_exp, h_exp):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("exp(x) on [-1, 1]")
    f_exp.plot(ax=_ax1, linewidth=3)
    _ax1.set_xlabel("x")

    _ax2.set_title("exp ★ exp")
    h_exp.plot(ax=_ax2, linewidth=3, color="C3")
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


@app.cell
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
    ## Example 6: Convolution on a general interval

    The `conv` method works on any interval $[a, b]$, not just $[-1, 1]$.
    The result lives on $[2a, 2b]$.

    For example, $f = g = 1$ on $[0, 1]$ produces a triangle on $[0, 2]$:
    """)
    return


@app.cell
def _(Chebfun):
    ones_01 = Chebfun.initconst(1.0, [0, 1])
    tri_01 = ones_01.conv(ones_01)
    tri_01
    return (tri_01,)


@app.cell
def _(tri_01):
    tri_01.plot(linewidth=3)
    plt.title("1 ★ 1 on [0, 1] → triangle on [0, 2]")
    plt.xlabel("x")
    plt.show()
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


@app.cell
def _(h_exp):
    _a, _b = float(h_exp.domain[0]), float(h_exp.domain[-1])
    print(f"(exp★exp)({_a}) = {float(h_exp(_a)):.2e}")
    print(f"(exp★exp)({_b}) = {float(h_exp(_b)):.2e}")
    return


# ---------------------------------------------------------------------------
# Example 8: Probability — sum of independent random variables
# ---------------------------------------------------------------------------


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


@app.cell
def _(Chebfun):
    pdf_uniform = Chebfun.initconst(1.0, [0, 1])
    pdf_sum_uniform = pdf_uniform.conv(pdf_uniform)

    pdf_sum_uniform.plot(linewidth=3)
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


@app.cell
def _(Chebfun):
    pdf_beta22 = Chebfun.initfun_adaptive(lambda x: 6.0 * x * (1.0 - x), [0, 1])
    pdf_sum_beta = pdf_beta22.conv(pdf_beta22)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _ax1.set_title("$f_X(x) = 6x(1-x)$  (Beta(2,2) PDF)")
    pdf_beta22.plot(ax=_ax1, linewidth=3)
    _ax1.set_xlabel("x")
    _ax1.set_ylabel("$f_X(x)$")

    _ax2.set_title("PDF of $Z = X + Y$")
    pdf_sum_beta.plot(ax=_ax2, linewidth=3, color="C1")
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
