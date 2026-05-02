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
    from chebpy.maps import MapParams
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
    The underlying technology is the Adcock-Richardson slit-strip
    variable transform (Adcock & Richardson, *SIAM J. Numer. Anal.*
    52(4), 1887–1912, 2014; arXiv:1305.2643), which clusters
    Chebyshev nodes super-exponentially near the singular endpoint(s)
    so that the transformed function is analytic and resolves to
    spectral accuracy.

    Square-root and other algebraic singularities — $\sqrt{x}$,
    $(1-x)^{1/2}$, $\sqrt{x(1-x)}$ — are the canonical use cases.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The map families

    Both clustering maps act through a smooth bijection
    $m : [-1, 1] \to [a, b]$ whose derivative vanishes
    super-exponentially at the clustered endpoint(s). The `Onefun`
    payload then sees the analytic function $f \circ m$ and resolves
    it to spectral accuracy. Both maps are parameterised by
    `MapParams(L, alpha)`: $\alpha > 0$ is the strip half-width and
    $L > 0$ truncates the (otherwise semi-/bi-infinite) underlying
    paper map.

    ### One-sided (`sing="left"`, `sing="right"`)

    With $s = L(t - 1)/2$ and the shift
    $\gamma = (\alpha/\pi)\log(e^{\pi/\alpha} - 1)$ chosen so that the
    smooth endpoint is hit exactly,

    $$
    u_\alpha(s) = \frac{\alpha}{\pi} \log(1 + e^{\pi(s + \gamma)/\alpha}), \qquad
    m(t) = a + (b - a)\, u_\alpha(L(t - 1)/2).
    $$

    For `sing="right"` the analogous reflection
    $m(t) = b - (b - a)\, u_\alpha(L(-t - 1)/2)$ is used.

    ### Two-sided (`sing="both"`)

    With $s = L\,t$,

    $$
    v_\alpha(s) = \frac{\alpha}{\pi} \left[ \log(1 + e^{\pi(s + 1/2)/\alpha}) - \log(1 + e^{\pi(s - 1/2)/\alpha}) \right], \qquad
    m(t) = a + (b - a)\, v_\alpha(L\,t),
    $$

    so $v_\alpha(0) = 1/2$ and $v_\alpha(\pm\infty) = (1\pm 1)/2$,
    clustering at both endpoints simultaneously.

    With finite $L$ the image of $m$ falls short of the clustered
    endpoint(s) by a small `gap`; with the default $L = 8$ this is
    below $10^{-10}$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A one-sided logarithmic singularity

    The simplest example: $f(x) = x\log x$ on $[0, 1]$, with the
    derivative singularity at the left endpoint.  Pass `sing="left"` to
    `chebfun`:
    """)
    return


@app.cell
def _():
    xlogx_left = chebfun(lambda x: x * np.log(x), [0.0, 1.0], sing="left")
    print(xlogx_left)
    print()
    print(f"piece type    : {type(xlogx_left.funs[0]).__name__}")
    print(f"size          : {xlogx_left.funs[0].size} coefficients")
    print(f"sum (= -1/4)  : {float(xlogx_left.sum()):.16f}")
    print(f"            ref {-1.0 / 4.0:.16f}")
    return (xlogx_left,)


@app.cell
def _(xlogx_left):
    _fig, _ax = plt.subplots()
    xlogx_left.plot(ax=_ax)
    _ax.set_xlabel("x")
    _ax.set_title(r"$f(x) = x\log x$ on $[0, 1]$ (sing='left')")
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
    singfun_sqrt = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))

    _grid = np.linspace(0.001, 0.999, 401)
    _err_bnd = float(np.max(np.abs(bndfun_sqrt(_grid) - np.sqrt(_grid))))
    _err_sng = float(np.max(np.abs(singfun_sqrt(_grid) - np.sqrt(_grid))))
    print(f"Bndfun   : size={bndfun_sqrt.size:5d}, max-err={_err_bnd:.2e}")
    print(f"Singfun  : size={singfun_sqrt.size:5d}, max-err={_err_sng:.2e}")
    return bndfun_sqrt, singfun_sqrt


@app.cell
def _(bndfun_sqrt, singfun_sqrt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _ax1.semilogy(np.arange(bndfun_sqrt.size), np.abs(bndfun_sqrt.coeffs) + 1e-20, color="C0")
    _ax1.set_xlabel("coefficient index")
    _ax1.set_ylabel("|coefficient|")
    _ax1.set_title(f"Bndfun (no map): {bndfun_sqrt.size} coeffs")
    _ax2.semilogy(np.arange(singfun_sqrt.size), np.abs(singfun_sqrt.coeffs) + 1e-20, color="C1")
    _ax2.set_xlabel("coefficient index")
    _ax2.set_title(f"Singfun (sing='left'): {singfun_sqrt.size} coeffs")
    _fig.suptitle(r"Chebyshev coefficient decay for $\sqrt{x}$ on $[0,1]$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tuning the map: `MapParams(L, alpha)`

    The two parameters trade off three things — coefficient count,
    pointwise accuracy near the clustered endpoint, and the size of
    the unresolved `gap`. Defaults `L = 8.0`, `alpha = 1.0` are a
    safe choice for the canonical algebraic singularities $x^p$ with
    $p \in [1/4, 1]$ on a bounded interval; the cells below show
    when and how to depart from them.

    **`alpha` (strip half-width).** Smaller $\alpha$ clusters nodes
    more aggressively near the singular endpoint, which helps for
    *stronger* singularities (e.g. $(1-x)^{0.1}$) at the cost of more
    Chebyshev coefficients. Larger $\alpha$ relaxes the clustering —
    fewer coefficients, but weak/log singularities may stop
    converging.

    - $\alpha \approx 0.5$ — strong singularities,
      $f \sim (x-a)^p$ with $p \lesssim 0.25$, or weak logarithms.
    - $\alpha = 1.0$ (default) — square roots and most algebraic
      cases; matches the paper's empirical optimum.
    - $\alpha \approx 2.0$ — gentle singularities, e.g.
      $(x-a)^{0.8}$, or smooth functions where you only want a hint
      of clustering.

    **`L` (truncation length).** $L$ controls how far into the
    paper's semi-/bi-infinite strip we sample, and equivalently the
    `gap` left at the clustered endpoint. With the default $L = 8$
    the gap is below $10^{-10}$ — invisible at working precision.
    Smaller $L$ widens the gap visibly; larger $L$ shrinks it
    super-exponentially but also moves samples closer to the
    endpoint where float64 ulp's bound pointwise accuracy.

    - $L = 1$–$2$ — the paper's empirical optimum for fastest
      coefficient decay; gap is $\sim 10^{-2}$–$10^{-1}$, fine
      when only the integral or interior values matter.
    - $L = 8$ (default) — gap below $10^{-10}$, indistinguishable
      from a closed map at working precision.
    - $L = 16$–$20$ — vanishing gap; useful when evaluating *at*
      the clustered endpoint matters (e.g. boundary conditions),
      assuming the float64 ulp ceiling at the clustered $x$ is not
      your real limit.

    **Rules of thumb.**

    - If the adaptive constructor fails to converge, *decrease*
      $\alpha$ first, then increase $L$.
    - If you want fewer coefficients and only need integrals,
      *decrease* $L$ (toward $1$–$2$); coefficient decay improves
      noticeably.
    - If pointwise accuracy at the clustered endpoint matters,
      *increase* $L$ but check that ulp$(x_a)$ is not the dominant
      error.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The cell below sweeps `alpha` and `L` independently for
    $f(x) = \sqrt{x}$ on $[0, 1]$, reporting the adaptive
    coefficient count, the resulting `gap`, and the max pointwise
    error on a uniform interior grid.
    """)
    return


@app.cell
def _():
    print(f"{'alpha':>6}  {'L':>5}  {'size':>6}  {'gap':>10}  {'max-err':>10}")
    print("-" * 46)
    _grid = np.linspace(0.001, 0.999, 401)
    for _alpha in (0.5, 1.0, 2.0):
        for _L in (1.0, 4.0, 8.0, 16.0):
            _s = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(L=_L, alpha=_alpha))
            _err = float(np.max(np.abs(_s(_grid) - np.sqrt(_grid))))
            print(f"{_alpha:>6.2f}  {_L:>5.1f}  {_s.size:>6d}  {_s.map.gap:>10.2e}  {_err:>10.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Reading the table: holding $\alpha$ fixed, decreasing $L$
    typically *reduces* coefficient count (faster decay, the paper's
    main result) at the price of a visible `gap`. Holding $L$ fixed,
    decreasing $\alpha$ pulls samples closer to the singularity —
    sometimes the only way to get a stubborn weak-power case to
    converge.
    """)
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
        _s = Singfun.initfun_adaptive(_fn, [0.0, 1.0], sing=_side, params=MapParams(alpha=_alpha))
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
    map between logical and reference variables.  The slit-strip
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
    _grid = np.linspace(0.0001, 0.9999, 21)
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
    restrict_source = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
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
      *New exponential variable transform methods for functions with
      endpoint singularities*,
      [SIAM J. Numer. Anal. 52(4), 1887–1912 (2014)](https://doi.org/10.1137/130920460);
      [arXiv:1305.2643](https://arxiv.org/abs/1305.2643).
    - T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
      [*Chebfun Guide*](https://www.chebfun.org/docs/guide/),
      Pafnuty Publications, 2014, ch. 9.
    """)
    return


if __name__ == "__main__":
    app.run()
