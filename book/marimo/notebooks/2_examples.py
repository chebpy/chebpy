"""Marimo ChebyshevPolynomial Demonstration.

This notebook demonstrates the properties and methods of the ChebyshevPolynomial class
from the chebpy library. Each cell showcases a different aspect of working with
Chebyshev polynomials.
"""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun"]
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

    import chebpy.chebyshev as cheb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Basic API Examples

    This notebook demonstrates core properties and methods of `ChebyshevPolynomial`
    using a simple constant polynomial.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Create a constant Chebyshev polynomial with value 5.

    `from_constant` creates a degree-0 polynomial representing a constant function.
    """)
    return


@app.cell
def _():
    f = cheb.from_constant(c=5)
    f
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Check whether the polynomial is constant.
    """)
    return


@app.cell
def _(f):
    f.isconst
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Check whether the polynomial has complex coefficients.
    """)
    return


@app.cell
def _(f):
    f.iscomplex
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Check whether the polynomial is empty (has no coefficients).
    """)
    return


@app.cell
def _(f):
    f.isempty
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the size (number of coefficients).
    """)
    return


@app.cell
def _(f):
    f.size
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the symbol used for the independent variable (typically `x`).
    """)
    return


@app.cell
def _(f):
    f.symbol
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get function values at Chebyshev points.
    """)
    return


@app.cell
def _(f):
    f.values
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the vertical scale (maximum absolute function value).
    """)
    return


@app.cell
def _(f):
    f.vscale
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compute the indefinite integral (antiderivative).
    """)
    return


@app.cell
def _(f):
    f.cumsum()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compute the definite integral over the domain.
    """)
    return


@app.cell
def _(f):
    f.sum()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the polynomial degree.
    """)
    return


@app.cell
def _(f):
    f.degree()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compute the derivative.
    """)
    return


@app.cell
def _(f):
    f.deriv()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the polynomial domain.
    """)
    return


@app.cell
def _(f):
    f.domain
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Plot the polynomial over its domain.
    """)
    return


@app.cell
def _(f):
    f.plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Find the roots of the polynomial.
    """)
    return


@app.cell
def _(f):
    f.roots()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Prolong to a higher-degree representation without changing the function.
    """)
    return


@app.cell
def _(f):
    f.prolong(n=4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get the name of the basis used by this polynomial.
    """)
    return


@app.cell
def _(f):
    f.basis_name
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Get a specific basis polynomial; here, degree 1 gives $T_1(x)=x$.
    """)
    return


@app.cell
def _(f):
    f.basis(deg=1)
    return


if __name__ == "__main__":
    app.run()
