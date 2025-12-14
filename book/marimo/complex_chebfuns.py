"""Marimo notebook demonstrating complex-valued Chebfun functionality in ChebPy."""

# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "chebpy",
# ]
#
# [tool.uv.sources]
# chebpy = { path = "../..", editable=true }
#
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    matplotlib.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Complex Chebfuns

    As of `v0.4.0` ChebPy supports complex variable representations.
    This makes it extremely convenient to perform certain computations in the complex plane.
    """
    )
    return


@app.cell
def _():
    from chebpy import chebfun

    return (chebfun,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    For example here is how we can plot a series of "Bernstein ellipses" - important objects
    in the convergence theory of Chebyshev series approximations for analytic functions.
    They are computed as transformations of the scaled complex unit circle under the Joukowsky map:
    """
    )
    return


@app.cell
def _(chebfun):
    x = chebfun("x", [-1, 1])
    z = np.exp(2 * np.pi * 1j * x)

    def joukowsky(z):
        """Apply the Joukowsky transformation to a complex number.

        Args:
            z: Complex input value

        Returns:
            The transformed complex value
        """
        return 0.5 * (z + 1 / z)

    for rho in np.arange(1.1, 2, 0.1):
        ellipse = joukowsky(rho * z)
        ellipse.plot(linewidth=2)
    plt.show()
    return (ellipse,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Per the first line of the above code segment, each of these ellipses is a complex-valued function
    of the real variable `x` defined on `[-1, 1]`. It is trivial to extract the real and imaginary components
    and plot these on the `x` domain, which we do for the last (largest) ellipse in the sequence as follows:
    """
    )
    return


@app.cell
def _(ellipse):
    _fig, _ax = plt.subplots()
    ellipse.real().plot(linewidth=3, label="real")
    ellipse.imag().plot(linewidth=3, label="imag")
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""Here is an example of using ChebPy to perform a contour integral calculation
        (replicating Trefethen & Hale's
        [example](https://www.chebfun.org/examples/complex/KeyholeContour.html)):"""
    )
    return


@app.cell
def _(chebfun):
    def keyhole(r, r_outer, e):
        """Create a keyhole contour in the complex plane.

        Args:
            r: Inner radius of the keyhole
            r_outer: Outer radius of the keyhole
            e: Half-height of the keyhole slit

        Returns:
            Four chebfun segments forming the keyhole contour
        """
        v = [-r_outer + e * 1j, -r + e * 1j, -r - e * 1j, -r_outer - e * 1j]
        s = chebfun("x", [0, 1])
        z0 = v[0] + (v[1] - v[0]) * s
        z1 = v[1] * v[2] ** s / v[1] ** s
        z2 = v[2] + s * (v[3] - v[2])
        z3 = v[3] * v[0] ** s / v[3] ** s
        return (z0, z1, z2, z3)

    z0, z1, z2, z3 = keyhole(r=0.2, r_outer=2, e=0.1)
    _fig, _ax = plt.subplots()
    kwds = dict(color="b", linewidth=3)
    z0.plot(ax=_ax, **kwds)
    z1.plot(ax=_ax, **kwds)
    z2.plot(ax=_ax, **kwds)
    z3.plot(ax=_ax, **kwds)
    _ax.plot([-4, 0], [0, 0], color="r", linewidth=2, linestyle="-")
    _ax.axis("equal")
    _ax.set_xlim([-2.2, 2.2])
    plt.show()
    return z0, z1, z2, z3


@app.cell(hide_code=True)
def _():
    mo.md(r"""We then perform the numerical integration as follows, obtaining a typically high-accuracy result.""")
    return


@app.cell
def _(z0, z1, z2, z3):
    def f(x):
        """Calculate the product of log(x) and tanh(x).

        Args:
            x: Input value or array

        Returns:
            log(x) * tanh(x)
        """
        return np.log(x) * np.tanh(x)

    def contour_integral(z, f):
        """Calculate the contour integral of a function along a path.

        Args:
            z: Complex path as a chebfun
            f: Function to integrate

        Returns:
            The value of the contour integral
        """
        integral = f(z) * z.diff()
        return integral.sum()

    y0 = np.sum([contour_integral(z, f) for z in (z0, z1, z2, z3)])  # numerical integral
    y1 = 4j * np.pi * np.log(np.pi / 2)  # exact value

    print(f"   y0 = {y0:+.15f}")
    print(f"   y1 = {y1:+.15f}")
    print(f"y0-y1 = {y0 - y1:+.15f}")
    return


if __name__ == "__main__":
    app.run()
