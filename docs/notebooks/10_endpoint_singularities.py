"""Marimo notebook demonstrating Singfun support for endpoint singularities."""

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
    import warnings

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from chebpy import chebfun
    from chebpy.bndfun import Bndfun
    from chebpy.singfun import Singfun
    from chebpy.utilities import Interval

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Endpoint Singularities

    This notebook tours ChebPy's `Singfun` machinery for functions with
    branch-type singularities at one or both endpoints of a bounded interval.
    The underlying technology is the Adcock-Richardson exponential
    variable transform (arXiv:1305.2643), which clusters Chebyshev nodes
    super-exponentially near the singular endpoint so that the
    transformed function is analytic and resolves to spectral accuracy.

    Square-root and other algebraic singularities — $\sqrt{x}$,
    $(1-x)^{1/2}$, $\sqrt{x(1-x)}$ — are the canonical use cases.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A one-sided square root

    The simplest example: $f(x) = \sqrt{x}$ on $[0, 1]$, with the
    singularity at the left endpoint.  Pass `sing="left"` to `chebfun`:
    """)
    return


@app.cell
def _():
    sqrt_left = chebfun(np.sqrt, [0.0, 1.0], sing="left")
    print(sqrt_left)
    print(f"piece type    : {type(sqrt_left.funs[0]).__name__}")
    print(f"size          : {sqrt_left.funs[0].size} coefficients")
    print(f"sum (= 2/3)   : {float(sqrt_left.sum()):.16f}")
    print(f"            ref {2.0 / 3.0:.16f}")
    return (sqrt_left,)


@app.cell
def _(sqrt_left):
    _fig, _ax = plt.subplots()
    sqrt_left.plot(ax=_ax)
    _ax.set_xlabel("x")
    _ax.set_title(r"$f(x) = \sqrt{x}$ on $[0, 1]$ (sing='left')")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A two-sided singularity

    With `sing="both"`, the symmetric "double-slit" map clusters nodes
    at *both* endpoints.  The integrand $\sqrt{x(1-x)}$ is the textbook
    example with closed-form integral $\pi/8$:
    """)
    return


@app.cell
def _():
    sqrt_two_sided = chebfun(lambda x: np.sqrt(x * (1.0 - x)), [0.0, 1.0], sing="both")
    print(f"size : {sqrt_two_sided.funs[0].size} coefficients")
    print(f"sum  : {float(sqrt_two_sided.sum()):.16f}")
    print(f"  ref: {np.pi / 8.0:.16f}")
    return (sqrt_two_sided,)


@app.cell
def _(sqrt_two_sided):
    _fig, _ax = plt.subplots()
    sqrt_two_sided.plot(ax=_ax)
    _ax.set_xlabel("x")
    _ax.set_title(r"$\sqrt{x(1-x)}$ on $[0, 1]$ (sing='both')")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Resolution power vs. plain Chebyshev

    A standard `Bndfun` (no map) cannot resolve a square root with
    machine precision: the adaptive constructor refuses to converge and
    emits a `did not converge` warning.  The `Singfun` construction
    reaches full machine precision with a small coefficient count.
    """)
    return


@app.cell
def _():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bndfun_sqrt = Bndfun.initfun_adaptive(np.sqrt, Interval(0.0, 1.0))
    singfun_sqrt = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)

    _grid = np.linspace(0.001, 0.999, 401)
    _err_bnd = float(np.max(np.abs(bndfun_sqrt(_grid) - np.sqrt(_grid))))
    _err_sng = float(np.max(np.abs(singfun_sqrt(_grid) - np.sqrt(_grid))))
    print(f"Bndfun   : size={bndfun_sqrt.size:5d}, max-err={_err_bnd:.2e}")
    print(f"Singfun  : size={singfun_sqrt.size:5d}, max-err={_err_sng:.2e}")
    return bndfun_sqrt, singfun_sqrt


@app.cell
def _(bndfun_sqrt, singfun_sqrt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(np.arange(bndfun_sqrt.size), np.abs(bndfun_sqrt.coeffs) + 1e-20, label="Bndfun (no map)")
    _ax.semilogy(np.arange(singfun_sqrt.size), np.abs(singfun_sqrt.coeffs) + 1e-20, label="Singfun (sing='left')")
    _ax.set_xlabel("coefficient index")
    _ax.set_ylabel("|coefficient|")
    _ax.set_title(r"Chebyshev coefficient decay for $\sqrt{x}$ on $[0,1]$")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Accuracy ceiling near non-zero clustered endpoints

    Pointwise accuracy is bounded below by the float64 spacing of $x$ at
    the clustered samples, scaled by the local Lipschitz constant of $f$:

    $$ |f(x_{\text{exact}}) - f(x_{\text{float}})| \;\approx\; |f'(x)| \cdot \mathrm{ulp}(x). $$

    For $\sqrt{x}$ clustered at $x = 0$ the subnormals give effectively
    unbounded relative resolution.  For $\sqrt{1-x}$ clustered at $x=1$
    the floor is around $10^{-10}$.  Weak singularities $(1-x)^p$ with
    small $p$ can fail to converge entirely.
    """)
    return


@app.cell
def _():
    accuracy_cases = [
        ("sqrt(x)", np.sqrt, "left", 1.0),
        ("sqrt(1-x)", lambda x: np.sqrt(1.0 - x), "right", 1.0),
        ("sqrt(x(1-x))", lambda x: np.sqrt(x * (1.0 - x)), "both", 1.0),
    ]
    for _label, _fn, _side, _alpha in accuracy_cases:
        _s = Singfun.initfun_adaptive(_fn, [0.0, 1.0], sing=_side, alpha=_alpha)
        _xx = np.linspace(0.001, 0.999, 401)
        _err = float(np.max(np.abs(_s(_xx) - _fn(_xx))))
        print(f"{_label:14s} sing={_side:5s} size={_s.size:4d}  max-err={_err:.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The integrals are unaffected.  The bad samples sit at $t \to \pm 1$
    where the Jacobian $m'(t) \to 0$ super-exponentially, suppressing
    noise below machine epsilon — so `sum()` reaches full precision
    even where pointwise evaluation does not.

    ## Convolution refuses Singfun pieces

    The Hale-Townsend Legendre convolution algorithm assumes an affine
    map between logical and reference variables.  The Adcock-Richardson
    clustering map breaks this assumption, so `Chebfun.conv` refuses
    `Singfun` operands:
    """)
    return


@app.cell
def _():
    conv_lhs = chebfun(np.sqrt, [0.0, 1.0], sing="left")
    conv_rhs = chebfun(lambda x: 1.0 + 0 * x, [0.0, 1.0])
    try:
        conv_lhs.conv(conv_rhs)
    except NotImplementedError as err:
        print(str(err))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixed-piece arithmetic

    Operations between a `Singfun` and a `Bndfun` on the same interval
    reconstruct the result on the singular representation.  This means a
    user assembling expressions like $\sqrt{x} + x^2$ does not have to
    think about which operand is singular:
    """)
    return


@app.cell
def _():
    mixed_singular = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left")
    mixed_smooth = Bndfun.initfun_adaptive(lambda x: x * x, Interval(0.0, 1.0))
    mixed_sum = mixed_singular + mixed_smooth
    print(f"Singfun + Bndfun is a {type(mixed_sum).__name__}")
    _grid = np.linspace(0.001, 0.999, 21)
    _err = float(np.max(np.abs(mixed_sum(_grid) - (np.sqrt(_grid) + _grid * _grid))))
    print(f"max-err vs reference: {_err:.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Restriction

    `Singfun.restrict` chooses its result type by inspecting the
    subinterval:

    - sharing the clustered endpoint -> `Singfun`
    - purely interior -> `Bndfun` (the function is analytic there).
    """)
    return


@app.cell
def _():
    restrict_source = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
    restrict_left = restrict_source.restrict([0.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restrict_interior = restrict_source.restrict([0.2, 0.8])
    print(f"restrict([0.0, 0.5]) -> {type(restrict_left).__name__}")
    print(f"restrict([0.2, 0.8]) -> {type(restrict_interior).__name__}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## References

    - B. Adcock and M. Richardson,
      [*A higher-order generalisation of the Adcock-Hale recipe for endpoint
      singularities*](https://arxiv.org/abs/1305.2643), 2013.
    - T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
      [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
      Pafnuty Publications, 2014, ch. 9.
    """)
    return


if __name__ == "__main__":
    app.run()
