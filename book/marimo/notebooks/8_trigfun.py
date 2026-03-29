"""Marimo notebook demonstrating trigfun and Trigtech for periodic function approximation."""

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
    # Periodic Functions with `trigfun`

    ChebPy supports Fourier (trigonometric) approximation of smooth periodic functions
    via the `trigfun` constructor and the underlying `Trigtech` class.

    Whereas `chebfun` uses Chebyshev polynomials, `trigfun` uses a truncated Fourier
    series — ideal for periodic signals such as $\sin$, $\cos$ and their compositions.

    Coefficients are stored in NumPy FFT order and the approximation is constructed
    adaptively until the high-frequency modes decay to machine precision.
    """)
    return


@app.cell(hide_code=True)
def _():
    from chebpy import trigfun

    return (trigfun,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Construction

    Create a `trigfun` representation of $\sin(2\pi x)$ on $[-1, 1]$.

    The adaptive algorithm needs only a handful of Fourier modes since the function
    is an exact trigonometric polynomial.
    """)
    return


@app.cell
def _(trigfun):
    f = trigfun(lambda x: np.sin(2 * np.pi * x))
    f
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `trigfun` returns a standard `Chebfun` object, so the full `Chebfun` API is available.
    The number of Fourier modes used is shown in the representation above.
    """)
    return


@app.cell
def _(f):
    _ax = f.plot(linewidth=3)
    _ax.set_title(r"$f(x) = \sin(2\pi x)$")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A more complex periodic function

    Now try a function that requires more modes:
    $g(x) = \cos(4\pi x) + 0.5\sin(10\pi x)$.
    """)
    return


@app.cell
def _(trigfun):
    g = trigfun(lambda x: np.cos(4 * np.pi * x) + 0.5 * np.sin(10 * np.pi * x))
    g
    return (g,)


@app.cell
def _(g):
    _ax = g.plot(linewidth=3)
    _ax.set_title(r"$g(x) = \cos(4\pi x) + 0.5\sin(10\pi x)$")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculus

    `trigfun` objects support `diff` (derivative) and `cumsum` (antiderivative)
    via exact Fourier differentiation in the frequency domain.

    The derivative of $\sin(2\pi x)$ is $2\pi\cos(2\pi x)$.
    """)
    return


@app.cell
def _(f):
    df = f.diff()
    _ax = f.plot(linewidth=3, label=r"$f = \sin(2\pi x)$")
    df.plot(ax=_ax, linewidth=3, linestyle="--", label=r"$f' = 2\pi\cos(2\pi x)$")
    _ax.legend(fontsize=12)
    _ax.set_title("Function and its derivative")
    plt.show()
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Definite integral

    The integral of $\sin(2\pi x)$ over $[-1, 1]$ is exactly zero.
    """)
    return


@app.cell
def _(f):
    print(f"∫ sin(2πx) dx  =  {float(f.sum()):.2e}   (exact: 0)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Root-finding

    Roots of a `trigfun` are found by resampling on Chebyshev points and delegating
    to the standard companion-matrix root-finder.

    The roots of $\sin(2\pi x)$ on $[-1, 1]$ are $\{-1, -0.5, 0, 0.5, 1\}$.
    """)
    return


@app.cell
def _(f):
    roots = f.roots()
    print("roots:", roots)
    _ax = f.plot(linewidth=3)
    _ax.plot(roots, f(roots), ".r", markersize=12, label="roots")
    _ax.axhline(0, color="k", linewidth=0.8)
    _ax.legend()
    plt.show()
    return (roots,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Arithmetic

    `trigfun` objects support standard arithmetic — addition, subtraction,
    multiplication — all inherited from the `Chebfun` class.
    """)
    return


@app.cell
def _(trigfun):
    h1 = trigfun(lambda x: np.sin(2 * np.pi * x))
    h2 = trigfun(lambda x: np.cos(2 * np.pi * x))
    h_sum = h1 + h2
    h_prod = h1 * h2
    _ax = h_sum.plot(linewidth=3, label=r"$\sin + \cos$")
    h_prod.plot(ax=_ax, linewidth=3, linestyle="--", label=r"$\sin \times \cos$")
    _ax.legend(fontsize=12)
    _ax.set_title("Sum and product of trigfuns")
    plt.show()
    return h1, h2, h_prod, h_sum


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Custom domain

    `trigfun` accepts an arbitrary domain $[a, b]$.  Here we approximate $\sin(x)$
    on $[0, 2\pi]$.
    """)
    return


@app.cell
def _(trigfun):
    f_2pi = trigfun(lambda x: np.sin(x), [0, 2 * np.pi])
    _ax = f_2pi.plot(linewidth=3)
    _ax.set_title(r"$\sin(x)$ on $[0,\, 2\pi]$")
    plt.show()
    print(f"  size : {f_2pi.funs[0].onefun.size}")
    print(f"∫ sin(x) dx = {float(f_2pi.sum()):.2e}  (exact: 0)")
    return (f_2pi,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fourier coefficient spectrum

    Access the underlying `Trigtech` object to inspect the Fourier coefficients.
    The `_coeffs_to_plotorder()` method returns the DC-centred (fftshift) ordering
    suitable for plotting.
    """)
    return


@app.cell
def _(trigfun):
    f_spec = trigfun(lambda x: np.cos(4 * np.pi * x) + 0.5 * np.sin(10 * np.pi * x))
    tech = f_spec.funs[0].onefun
    plot_coeffs = tech._coeffs_to_plotorder()
    n = tech.size
    freqs = np.arange(-(n // 2), n - n // 2)
    _fig, _ax = plt.subplots()
    _ax.semilogy(freqs, np.abs(plot_coeffs) + 1e-18, "o-", markersize=5)
    _ax.set_xlabel("Frequency")
    _ax.set_ylabel("|coefficient|")
    _ax.set_title("Fourier coefficient spectrum (DC-centred)")
    plt.show()
    return (f_spec,)


if __name__ == "__main__":
    app.run()
