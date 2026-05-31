"""Marimo notebook demonstrating Crouzeix ratios with ChebPy."""

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

    from chebpy import fov, polyvalm

    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Crouzeix's Theorem

    For a square matrix $A$ and a polynomial $p$, Crouzeix's theorem bounds the
    matrix polynomial norm by the maximum of $|p|$ on the field of values

    \[
    W(A) = \{x^*Ax : x^*x = 1\}.
    \]

    This notebook mirrors the Chebfun example with NumPy matrices and ChebPy
    curves: `fov(A)` builds a periodic complex Chebfun for the boundary of
    $W(A)$, and `polyvalm` evaluates polynomial coefficients at a matrix.
    """)
    return


@app.cell
def _():
    def grcar(n):
        """Return the Grcar matrix of order n."""
        matrix = np.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1.0
            if i > 0:
                matrix[i, i - 1] = -1.0
            for j in range(i + 1, min(n, i + 4)):
                matrix[i, j] = 1.0
        return matrix

    def polyval_chebfun(coeffs, z):
        """Evaluate a polynomial with highest-first coefficients at a Chebfun."""
        result = 0 * z + coeffs[0]
        for coeff in coeffs[1:]:
            result = result * z + coeff
        return result

    def chebfun_max_abs(z, samples=4000):
        """Estimate max(abs(z)) from a dense sample of a Chebfun."""
        theta = np.linspace(z.support[0], z.support[-1], samples + 1)
        return float(np.max(np.abs(z(theta))))

    def crouzeix_ratio(matrix, coeffs, samples=8000):
        """Compute norm(p(A), 2) / max(abs(p(z))) for z on boundary W(A)."""
        boundary = fov(matrix)
        numerator = float(np.linalg.norm(polyvalm(coeffs, matrix), 2))
        denominator = chebfun_max_abs(polyval_chebfun(coeffs, boundary), samples=samples)
        return numerator / denominator, boundary

    return chebfun_max_abs, crouzeix_ratio, grcar, polyval_chebfun


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Grcar Matrix

    The Grcar matrix is a useful nonnormal test problem.  Rotating it by a
    complex phase moves the eigenvalues and field of values in the complex
    plane without changing the underlying geometry.
    """)
    return


@app.cell
def _(grcar):
    G = np.exp(0.6j) * grcar(32)
    W_G = fov(G)
    eig_G = np.linalg.eigvals(G)
    return G, W_G, eig_G


@app.cell
def _(W_G, eig_G):
    _fig, _ax = plt.subplots()
    W_G.plot(ax=_ax, color="C0", linewidth=2, label=r"$\partial W(A)$")
    _ax.scatter(eig_G.real, eig_G.imag, color="C3", s=22, zorder=3, label="eigenvalues")
    _ax.set_aspect("equal", adjustable="box")
    _ax.set_xlabel("real")
    _ax.set_ylabel("imag")
    _ax.set_title("Rotated Grcar matrix")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A Sharp 2-by-2 Example

    For the Jordan block

    \[
    A = \begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}
    \]

    and $p(z)=z$, the field of values is the disk of radius $1/2$.  The
    Crouzeix ratio is exactly $2$.
    """)
    return


@app.cell
def _(chebfun_max_abs, polyval_chebfun):
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    p_jordan = np.array([1.0, 0.0])
    W_J = fov(J, n=64)
    pW_J = polyval_chebfun(p_jordan, W_J)
    jordan_ratio = np.linalg.norm(polyvalm(p_jordan, J), 2) / chebfun_max_abs(pW_J)
    print(f"||p(A)||_2       = {np.linalg.norm(polyvalm(p_jordan, J), 2):.12f}")
    print(f"max_W(A) |p(z)| = {chebfun_max_abs(pW_J):.12f}")
    print(f"ratio           = {jordan_ratio:.12f}")
    return J, W_J, jordan_ratio


@app.cell
def _(W_J):
    _fig, _ax = plt.subplots()
    W_J.plot(ax=_ax, linewidth=2)
    _ax.set_aspect("equal", adjustable="box")
    _ax.set_xlabel("real")
    _ax.set_ylabel("imag")
    _ax.set_title("Field of values of the 2-by-2 Jordan block")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A Seeded Random Matrix and Polynomial

    The same workflow applies to a larger matrix and a higher-degree polynomial.
    The denominator is computed by evaluating the scalar polynomial on the
    Chebfun boundary curve.
    """)
    return


@app.cell
def _(crouzeix_ratio):
    rng = np.random.default_rng(62)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    coeffs = rng.standard_normal(7) + 1j * rng.standard_normal(7)
    random_ratio, W_A = crouzeix_ratio(A, coeffs)
    print(f"degree          = {coeffs.size - 1}")
    print(f"ratio           = {random_ratio:.12f}")
    return A, W_A, coeffs, random_ratio


@app.cell
def _(A, W_A):
    _fig, _ax = plt.subplots()
    W_A.plot(ax=_ax, color="C0", linewidth=2, label=r"$\partial W(A)$")
    _eig = np.linalg.eigvals(A)
    _ax.scatter(_eig.real, _eig.imag, color="C3", s=32, zorder=3, label="eigenvalues")
    _ax.set_aspect("equal", adjustable="box")
    _ax.set_xlabel("real")
    _ax.set_ylabel("imag")
    _ax.set_title("Random nonnormal matrix")
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
