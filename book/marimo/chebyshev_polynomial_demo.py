"""Marimo ChebyshevPolynomial Demonstration."""

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    pass


@app.cell
def _(mo):
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
