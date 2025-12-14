"""Marimo ChebyshevPolynomial Demonstration."""

# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "chebpy",
# ]
#
# [tool.uv.sources]
# chebpy = { path = "../..", editable=True }
# 
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    import marimo as mo  # noqa: F401


@app.cell
def _():
    mo.md(
        """
    # ChebyshevPolynomial Demonstration

    This notebook demonstrates the usage of the `ChebyshevPolynomial` class from the `chebpy` library.

    The `ChebyshevPolynomial` class provides an immutable representation of Chebyshev polynomials
    and various factory functions to construct them.
    """
    )
    return


if __name__ == "__main__":
    app.run()
