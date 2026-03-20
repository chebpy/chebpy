"""Marimo notebook: Quasimatrices and Continuous Linear Algebra.

Demonstrates how ChebPy extends standard matrix operations to the continuous
setting via quasimatrices — matrices whose columns are functions rather than
vectors of numbers.
"""

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
    ---

    **Reference.**  The ideas and examples in this notebook are drawn from
    Chapter 6 ("Quasimatrices and Least-Squares") of the
    [Chebfun Guide](https://www.chebfun.org/docs/guide/guide06.html)
    by L. N. Trefethen (2009, revised 2019).
    """)
    return


@app.cell(hide_code=True)
def _():
    from chebpy import Quasimatrix, chebfun, polyfit

    return Quasimatrix, chebfun, polyfit


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Quasimatrices and Continuous Linear Algebra

    Numerical linear algebra operates on matrices — rectangular arrays of
    numbers.  But many of the same ideas (QR, SVD, least-squares, norms)
    make perfect sense when the "rows" are continuous: each column is a
    *function* rather than a finite vector of samples.

    A **quasimatrix** is exactly this object: an $\infty \times n$ matrix
    whose $n$ columns are chebfuns on a common interval.  ChebPy's
    `Quasimatrix` class lets you manipulate them with familiar linear
    algebra syntax.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building a quasimatrix

    The simplest example: stack the first six monomials
    $1, x, x^2, \ldots, x^5$ on $[-1,1]$ into a single object.
    """)
    return


@app.cell
def _(Quasimatrix, chebfun):
    x = chebfun("x")
    A = Quasimatrix([1, x, x**2, x**3, x**4, x**5])
    print("shape:", A.shape)
    print("A(0.5, 2) =", A[0.5, 2])  # x² evaluated at 0.5
    return A, x


@app.cell
def _(A):
    _fig, _ax = plt.subplots()
    A.plot(ax=_ax)
    _ax.set_ylim(-1.1, 1.1)
    _ax.grid(True)
    _ax.set_title("Columns of A = [1, x, x², x³, x⁴, x⁵]")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Because integration is the continuous analogue of summation, the column
    sums of $A$ are the integrals $\int_{-1}^{1} x^k \, dx$:
    """)
    return


@app.cell
def _(A):
    print(A.sum())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inner products and the Gram matrix

    The inner product of two column chebfuns is
    $\langle f, g \rangle = \int_{-1}^{1} f(x)\,g(x)\,dx$.
    For example, $\langle x^2, x^4 \rangle = 2/7$:
    """)
    return


@app.cell
def _(A):
    ip = A[:, 2].dot(A[:, 4])
    print(f"⟨x², x⁴⟩ = {ip:.15f}  (exact = {2 / 7:.15f})")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The Gram matrix $G = A^\top\!A$ collects all pairwise inner products
    into a single $n \times n$ matrix.  Because the monomials are far from
    orthogonal, $G$ is far from the identity:
    """)
    return


@app.cell
def _(A):
    G = A.T @ A
    print(np.round(G, 4))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Least-squares fitting

    Given a quasimatrix $A$ and a target function $f$, we can solve the
    continuous least-squares problem

    $$\min_{\mathbf{c}} \| A\mathbf{c} - f \|_2$$

    with `A.solve(f)`.  This is the continuous analogue of the matrix
    backslash operator.
    """)
    return


@app.cell
def _(A, x):
    f = np.exp(x) * np.sin(6 * x)
    c = A.solve(f)
    print("Least-squares coefficients c:")
    print(c)
    return c, f


@app.cell
def _(A, c, f):
    ffit = A @ c
    _fig, _ax = plt.subplots()
    f.plot(ax=_ax, label="f", linewidth=2)
    ffit.plot(ax=_ax, label="$\\tilde{f}$", linestyle="--", color="r")
    _ax.legend()
    _ax.grid(True)
    _ax.set_title(f"Degree-5 least-squares fit  (error = {(f - ffit).norm(2):.4f})")
    _fig
    return (ffit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The convenience wrapper `polyfit(f, n)` does the same thing in one call:
    """)
    return


@app.cell
def _(f, ffit, polyfit):
    ffit2 = polyfit(f, 5)
    print(f"‖ffit − polyfit(f, 5)‖ = {(ffit - ffit2).norm(2):.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Beyond polynomials: a hat-function basis

    Quasimatrix columns need not be polynomials.  Here we construct 11
    piecewise-linear hat functions centred at equally spaced points on
    $[-1, 1]$ and use them as a finite-element-style basis for
    least-squares fitting.  All columns share the same breakpoints so
    their domains match in arithmetic.
    """)
    return


@app.cell
def _(Quasimatrix, chebfun, f):
    _bkpts = [round(-1 + _k * 0.2, 10) for _k in range(11)]
    hat_cols = []
    for _j in range(11):
        _xj = round(-1 + _j * 0.2, 10)
        hat_cols.append(chebfun(lambda t, __xj=_xj: np.maximum(0, 1 - 5 * np.abs(t - __xj)), _bkpts))
    A2 = Quasimatrix(hat_cols)

    _fig, _axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=(1, 2))
    A2.plot(ax=_axes[0])
    _axes[0].set_title("Hat-function basis")
    _axes[0].grid(True)
    _axes[0].tick_params(labelbottom=False)

    c2 = A2.solve(f)
    ffit_hat = A2 @ c2
    f.plot(ax=_axes[1], label="f", linewidth=2)
    ffit_hat.plot(ax=_axes[1], label="$\\tilde{f}$", linestyle="--", color="r")
    _axes[1].legend()
    _axes[1].grid(True)
    _axes[1].set_title(f"Hat-function fit  (error = {(f - ffit_hat).norm(2):.4f})")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## QR factorisation

    The reduced QR factorisation $A = QR$ produces an $\infty \times n$
    quasimatrix $Q$ whose columns are orthonormal in $L^2$ and an
    $n \times n$ upper-triangular factor $R$.

    When the input columns are the monomials $1, x, \ldots, x^n$,
    Gram–Schmidt orthogonalisation recovers the **Legendre polynomials**
    (up to $L^2$ normalisation).
    """)
    return


@app.cell
def _(A):
    Q, R = A.qr()

    _fig, _ax = plt.subplots()
    Q.plot(ax=_ax)
    _ax.grid(True)
    _ax.set_title("Columns of Q  (L²-normalised Legendre polynomials)")
    _fig
    return Q, R


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The orthonormality relation $Q^\top Q = I$ holds to machine precision:
    """)
    return


@app.cell
def _(Q):
    QTQ = Q.T @ Q
    print("Q^T Q:")
    print(np.round(QTQ, 10))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we rescale each column so that $P_k(1) = 1$, we recover the
    standard Legendre polynomials.  The inverse of the rescaled $R$
    then holds the monomial expansion coefficients — for instance
    $P_3(x) = \tfrac{5}{2}x^3 - \tfrac{3}{2}x$.
    """)
    return


@app.cell
def _(Q, R):
    R_leg = R.copy()
    Q_leg_cols = [col.copy() for col in Q.columns]
    for _j in range(len(Q_leg_cols)):
        _v = float(Q_leg_cols[_j](1.0))
        R_leg[_j, :] *= _v
        Q_leg_cols[_j] = (1.0 / _v) * Q_leg_cols[_j]

    _fig, _ax = plt.subplots()
    from chebpy.quasimatrix import Quasimatrix as _QM

    _QM(Q_leg_cols).plot(ax=_ax)
    _ax.grid(True)
    _ax.set_title("Legendre polynomials P₀, P₁, …, P₅  (normalised so P(1) = 1)")

    print("inv(R) — monomial coefficients of Legendre polynomials:")
    print(np.round(np.linalg.inv(R_leg), 4))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SVD, norms, and condition numbers

    The singular value decomposition $A = U \Sigma V^\top$ generalises
    naturally to quasimatrices.  The singular values tell us about the
    "geometry" of the column space: how much the quasimatrix stretches
    or compresses different directions.
    """)
    return


@app.cell
def _(A):
    _U, S, V = A.svd()
    print("Singular values:")
    print(S)
    print(f"\n‖A‖₂    = {A.norm(2):.15f}")
    print(f"cond(A) = {A.cond():.15f}")
    return (V,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first and last right singular vectors $v_1$ and $v_n$ tell us
    which coefficient vectors produce the largest and smallest functions
    in $L^2$.  Multiplying $A$ by these vectors gives the
    **extreme singular functions**:
    """)
    return


@app.cell
def _(A, V):
    _v1 = V[:, 0]
    _vn = V[:, -1]

    _fig, _ax = plt.subplots()
    (A @ _v1).plot(ax=_ax, label="Av₁  (largest)")
    (A @ _vn).plot(ax=_ax, label="Avₙ  (smallest)")
    _ax.legend()
    _ax.grid(True)
    _ax.set_title("Extreme singular functions")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Several matrix norms extend directly to the quasimatrix setting:

    | Norm | Definition | Value |
    |---|---|---|
    | 1-norm | $\max_j \lVert A_j \rVert_1$ | maximum column $L^1$ norm |
    | $\infty$-norm | $\max_x \sum_j \lvert A_j(x)\rvert$ | maximum absolute row sum |
    | Frobenius | $\bigl(\sum_k \sigma_k^2\bigr)^{1/2}$ | Hilbert–Schmidt norm |
    """)
    return


@app.cell
def _(A):
    print(f"‖A‖₁   = {A.norm(1)}")
    print(f"‖A‖_∞  = {A.norm(np.inf)}")
    print(f"‖A‖_F  = {A.norm('fro'):.15f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Rank, null space, and pseudoinverse

    Three functions that live on the same interval can still be linearly
    *dependent*.  The Pythagorean identity $\sin^2 x + \cos^2 x = 1$
    means the quasimatrix $B = [1,\; \sin^2 x,\; \cos^2 x]$ has rank 2
    and a one-dimensional null space.
    """)
    return


@app.cell
def _(Quasimatrix, x):
    B = Quasimatrix([1, np.sin(x) ** 2, np.cos(x) ** 2])
    print(f"rank(B) = {B.rank()}")
    print(f"\nnull(B) =\n{B.null()}")
    return (B,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The null vector $[-1/\sqrt{3},\; 1/\sqrt{3},\; 1/\sqrt{3}]$
    encodes exactly the identity $\sin^2 x + \cos^2 x - 1 = 0$.
    """)
    return


@app.cell
def _(B):
    O = B.orth()
    print(f"orth(B) shape: {O.shape}")
    print(f"Orthonormality check (O^T O):\n{np.round(O.T @ O, 10)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The pseudoinverse maps a function $f$ to the least-squares
    coefficients, so `pinv(A) @ f` and `A.solve(f)` agree:
    """)
    return


@app.cell
def _(A, f):
    P = A.pinv()
    c_solve = A.solve(f)
    c_pinv = P @ f
    print(f"‖A.solve(f) − pinv(A) @ f‖ = {np.linalg.norm(c_solve - c_pinv):.2e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why orthogonal bases matter

    The monomial basis $\{1, x, \ldots, x^n\}$ becomes spectacularly
    ill-conditioned as $n$ grows — the condition number grows
    exponentially.  This is precisely why orthogonal polynomial bases
    (Legendre, Chebyshev) are preferred in practice.
    """)
    return


@app.cell
def _(Quasimatrix, x):
    _conds = []
    _ns = range(1, 16)
    for _n in _ns:
        _cols = [x**_k for _k in range(_n + 1)]
        from chebpy import chebfun as _cf

        _cols[0] = _cf(1.0)
        _An = Quasimatrix(_cols)
        _conds.append(_An.cond())

    _fig, _ax = plt.subplots()
    _ax.semilogy(list(_ns), _conds, "o-")
    _ax.set_xlabel("n")
    _ax.set_ylabel("cond(A)")
    _ax.set_title("Condition number of the monomial quasimatrix")
    _ax.grid(True)
    _fig
    return


if __name__ == "__main__":
    app.run()
