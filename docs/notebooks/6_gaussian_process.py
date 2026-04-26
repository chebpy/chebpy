"""Marimo notebook demonstrating Gaussian process regression with ChebPy."""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebfun]
# path = "../.."
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Gaussian Process Regression with ChebPy

    This notebook demonstrates the ``gpr`` function in ChebPy, which performs
    **Gaussian process regression** and returns the posterior mean, variance,
    and (optionally) random samples as Chebfun objects.

    Because the outputs are Chebfuns, you can immediately differentiate,
    integrate, find roots, and compose them with the full ChebPy toolkit.

    > C. E. Rasmussen & C. K. I. Williams, *Gaussian Processes for Machine
    > Learning*, MIT Press, 2006.

    > S. Filip, A. Javeed, and L. N. Trefethen, Smooth random functions,
    > random ODEs, and Gaussian processes, *SIAM Review*, 61 (2019), 185–205.
    """)
    return


@app.cell(hide_code=True)
def _():
    from chebpy import gpr

    return (gpr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Basic usage

    We start by sampling a smooth function $f(x) = \sin(e^x)$ at a handful
    of random points on $[-2, 2]$ and then asking ``gpr`` to recover it.

    The squared exponential kernel

    $$
    k(x, x') = \sigma^2 \exp\!\Bigl(-\frac{(x-x')^2}{2\ell^2}\Bigr)
    $$

    is used by default. The signal variance $\sigma$ defaults to
    $\max|y|$ and the length scale $\ell$ is selected automatically by
    maximising the log marginal likelihood.
    """)
    return


@app.cell(hide_code=True)
def _():
    rng = np.random.default_rng(1)
    n = 10
    x_obs = np.sort(-2 + 4 * rng.random(n))
    y_obs = np.sin(np.exp(x_obs))
    return rng, x_obs, y_obs


@app.cell
def _(gpr, x_obs, y_obs):
    f_mean, f_var = gpr(x_obs, y_obs, domain=[-2, 2])
    f_mean
    return f_mean, f_var


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The posterior mean is a Chebfun — let's plot it together with the data and
    a ±2 standard-deviation band computed from the variance Chebfun.
    """)
    return


@app.cell(hide_code=True)
def _(f_mean, f_var, x_obs, y_obs):
    _tt = np.linspace(-2, 2, 500)
    _mu = f_mean(_tt)
    _sd = np.sqrt(np.maximum(f_var(_tt), 0.0))

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1], sharex=True)

    _ax1.fill_between(_tt, _mu - 2 * _sd, _mu + 2 * _sd, alpha=0.25, label="±2σ")
    _ax1.plot(_tt, _mu, linewidth=2, label="mean")
    _ax1.plot(x_obs, y_obs, "ok", markersize=8)
    _ax1.set_ylabel("y")
    _ax1.legend()
    _ax1.set_title("GPR — noiseless observations")

    _ax2.fill_between(_tt, _sd**2, alpha=0.35, color="C1")
    _ax2.plot(_tt, _sd**2, linewidth=1.5, color="C1")
    _ax2.set_xlabel("x")
    _ax2.set_ylabel("variance")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Posterior samples

    Pass ``n_samples`` to draw independent realisations from the posterior.
    Each sample is a Chebfun column in a ``Quasimatrix``.
    """)
    return


@app.cell
def _(gpr, x_obs, y_obs):
    f2_mean, f2_var, samples = gpr(x_obs, y_obs, domain=[-2, 2], n_samples=10)
    samples
    return f2_mean, f2_var, samples


@app.cell(hide_code=True)
def _(f2_mean, f2_var, samples, x_obs, y_obs):
    _tt = np.linspace(-2, 2, 500)
    _mu = f2_mean(_tt)
    _sd = np.sqrt(np.maximum(f2_var(_tt), 0.0))

    _fig, _ax = plt.subplots()
    _ax.fill_between(_tt, _mu - 2 * _sd, _mu + 2 * _sd, alpha=0.15, color="C0")
    for _k in range(samples.shape[1]):
        _s = samples[:, _k]
        _ax.plot(_tt, _s(_tt), color="0.65", linewidth=0.8)
    _ax.plot(_tt, _mu, linewidth=2, label="posterior mean")
    _ax.plot(x_obs, y_obs, "ok", markersize=8, label="observations")
    _ax.legend()
    _ax.set_title("GPR — ten posterior samples")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Noisy observations

    Real data is rarely exact. The ``noise`` parameter specifies the standard
    deviation $\sigma_y$ of i.i.d. Gaussian observation noise. The kernel
    is augmented on the diagonal:

    $$
    k'(x, x') = k(x, x') + \sigma_y^2\,\delta_{xx'}
    $$

    With noise, the posterior mean **smooths** the data rather than
    interpolating it, and the variance never reaches zero.
    """)
    return


@app.cell(hide_code=True)
def _(gpr, rng, x_obs, y_obs):
    y_noisy = y_obs + 0.15 * rng.standard_normal(len(y_obs))
    fn_mean, fn_var = gpr(x_obs, y_noisy, domain=[-2, 2], noise=0.15)
    return fn_mean, fn_var, y_noisy


@app.cell(hide_code=True)
def _(fn_mean, fn_var, x_obs, y_noisy):
    _tt = np.linspace(-2, 2, 500)
    _mu = fn_mean(_tt)
    _sd = np.sqrt(np.maximum(fn_var(_tt), 0.0))

    _fig, _ax = plt.subplots()
    _ax.fill_between(_tt, _mu - 2 * _sd, _mu + 2 * _sd, alpha=0.25, label="±2σ")
    _ax.plot(_tt, _mu, linewidth=2, label="posterior mean")
    _ax.plot(x_obs, y_noisy, "ok", markersize=8, label="noisy data")
    _ax.legend()
    _ax.set_title("GPR — noisy observations (σ_y = 0.15)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Chebfun calculus on the GP output

    Because the posterior mean is a Chebfun, all standard operations are
    available. Here we differentiate the mean, find its roots (zero
    crossings of the derivative = local extrema of the mean), and integrate.
    """)
    return


@app.cell
def _(f_mean):
    df = f_mean.diff()
    df
    return (df,)


@app.cell(hide_code=True)
def _(df, f_mean):
    extrema = df.roots()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: mean with extrema marked
    _tt = np.linspace(-2, 2, 500)
    _ax1.plot(_tt, f_mean(_tt), linewidth=2)
    _ax1.plot(extrema, f_mean(extrema), "or", markersize=8)
    _ax1.set_title("posterior mean & local extrema")
    _ax1.set_xlabel("x")

    # right: derivative
    _ax2.plot(_tt, df(_tt), linewidth=2, color="C1")
    _ax2.axhline(0, color="k", linewidth=0.5)
    _ax2.set_title("derivative of posterior mean")
    _ax2.set_xlabel("x")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(f_mean):
    integral = f_mean.sum()
    return (integral,)


@app.cell(hide_code=True)
def _(f_mean, integral):
    mo.md(rf"""
    The definite integral of the posterior mean over $[-2, 2]$ is

    $$
    \int_{{-2}}^{{2}} \mu(x)\,dx \approx {float(integral):.6f}
    $$

    and the function is represented by a Chebfun of length **{len(f_mean)}**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Periodic kernel

    For data from a periodic process, pass ``trig=True``. This uses the
    periodic squared exponential kernel:

    $$
    k(x,x') = \sigma^2 \exp\!\Bigl(-\frac{2}{\ell^2}\sin^2\!\Bigl(\frac{\pi(x-x')}{P}\Bigr)\Bigr)
    $$

    where $P$ is the period (the length of the domain).
    """)
    return


@app.cell
def _(gpr):
    x_per = np.linspace(0, 2 * np.pi, 15, endpoint=False)
    y_per = np.sin(x_per) + 0.3 * np.cos(3 * x_per)
    fp_mean, fp_var = gpr(x_per, y_per, domain=[0, 2 * np.pi], trig=True)
    return fp_mean, fp_var, x_per, y_per


@app.cell(hide_code=True)
def _(fp_mean, fp_var, x_per, y_per):
    _tt = np.linspace(0, 2 * np.pi, 500)
    _mu = fp_mean(_tt)
    _sd = np.sqrt(np.maximum(fp_var(_tt), 0.0))

    _fig, _ax = plt.subplots()
    _ax.fill_between(_tt, _mu - 2 * _sd, _mu + 2 * _sd, alpha=0.25)
    _ax.plot(_tt, _mu, linewidth=2, label="periodic GP mean")
    _ax.plot(x_per, y_per, "ok", markersize=8, label="data")
    _ax.legend()
    _ax.set_title("GPR — periodic kernel")
    _ax.set_xlabel("x")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Effect of the length scale

    The length scale $\ell$ controls how far correlations extend. A small
    $\ell$ fits local wiggles; a large $\ell$ produces a smoother fit.
    Here we compare three hand-picked values against the automatically
    selected one.
    """)
    return


@app.cell
def _(gpr, x_obs, y_obs):
    ls_values = [0.2, 0.5, 2.0]
    gp_by_ls = {}
    for _ls in ls_values:
        _fm, _fv = gpr(x_obs, y_obs, domain=[-2, 2], length_scale=_ls)
        gp_by_ls[_ls] = _fm
    return gp_by_ls, ls_values


@app.cell(hide_code=True)
def _(f_mean, gp_by_ls, ls_values, x_obs, y_obs):
    _tt = np.linspace(-2, 2, 500)

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for _ax, _ls in zip(_axes, ls_values, strict=False):
        _ax.plot(_tt, gp_by_ls[_ls](_tt), linewidth=2, label=f"ℓ = {_ls}")
        _ax.plot(_tt, f_mean(_tt), "--", linewidth=1, color="0.5", label="auto ℓ")
        _ax.plot(x_obs, y_obs, "ok", markersize=6)
        _ax.set_title(f"ℓ = {_ls}")
        _ax.legend(fontsize=10)
    _axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    | Parameter | Meaning | Default |
    |---|---|---|
    | ``sigma`` | Signal variance $\sigma$ | $\max|y|$ |
    | ``length_scale`` | Length scale $\ell$ | Maximise log marginal likelihood |
    | ``noise`` | Observation noise $\sigma_y$ | $0$ (exact interpolation) |
    | ``trig`` | Periodic kernel | ``False`` |
    | ``n_samples`` | Posterior draws | $0$ |

    All outputs are Chebfun objects — differentiate, integrate, find
    roots, evaluate, and compose them freely.
    """)
    return


if __name__ == "__main__":
    app.run()
