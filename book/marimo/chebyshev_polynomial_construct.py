"""Marimo ChebyshevPolynomial Demonstration.

This notebook demonstrates the properties and methods of the ChebyshevPolynomial class
from the chebpy library. Each cell showcases a different aspect of working with
Chebyshev polynomials.
"""

# /// script
# dependencies = [
#     "marimo==0.18.4",
# ]
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    # Import the chebyshev module from chebpy
    import chebpy.chebyshev as cheb


@app.cell
def _():
    # Create a constant Chebyshev polynomial with value 5
    # from_constant creates a degree 0 polynomial representing a constant function
    f = cheb.from_constant(c=5)
    f
    return (f,)


@app.cell
def _(f):
    # Check if the polynomial is a constant function
    # Returns True for our example since we created a constant polynomial
    f.isconst
    return


@app.cell
def _(f):
    # Check if the polynomial has complex coefficients
    # Returns False for our example since we used a real constant
    f.iscomplex
    return


@app.cell
def _(f):
    # Check if the polynomial is empty (has no coefficients)
    # Returns False for our example since we have a non-empty polynomial
    f.isempty
    return


@app.cell
def _(f):
    # Get the size (number of coefficients) of the polynomial
    # For a constant polynomial, this is 1
    f.size
    return


@app.cell
def _(f):
    # Get the symbol used for the independent variable in the polynomial
    # By default, this is 'x'
    f.symbol
    return


@app.cell
def _(f):
    # Get the values of the polynomial at Chebyshev points
    # These are the function values at specific points in the domain
    f.values
    return


@app.cell
def _(f):
    # Get the vertical scale of the polynomial
    # This is the maximum absolute value of the function values
    f.vscale
    return


@app.cell
def _(f):
    # Calculate the indefinite integral (antiderivative) of the polynomial
    # For a constant polynomial f(x) = 5, the result is f(x) = 5x + C
    f.cumsum()
    return


@app.cell
def _(f):
    # Calculate the definite integral of the polynomial over its domain
    # For a constant polynomial f(x) = 5 on domain [-1, 1], this is 5 * (1 - (-1)) = 10
    f.sum()
    return


@app.cell
def _(f):
    # Get the degree of the polynomial
    # For a constant polynomial, the degree is 0
    f.degree()
    return


@app.cell
def _(f):
    # Calculate the derivative of the polynomial
    # For a constant polynomial, the derivative is zero
    f.deriv()
    return


@app.cell
def _(f):
    # Get the domain of the polynomial
    # By default, Chebyshev polynomials are defined on [-1, 1]
    f.domain
    return


@app.cell
def _(f):
    # Plot the polynomial over its domain
    # For a constant polynomial, this is a horizontal line
    f.plot()
    return


@app.cell
def _(f):
    # Find the roots of the polynomial
    # A constant polynomial (unless it's zero) has no roots
    f.roots()
    return


@app.cell
def _(f):
    # Extend the polynomial to a higher degree representation
    # This doesn't change the function but increases the internal representation size
    f.prolong(n=4)
    return


@app.cell
def _(f):
    # Get the name of the basis used for the polynomial
    # This should be 'Chebyshev' for our polynomial
    f.basis_name
    return


@app.cell
def _(f):
    # Get a specific basis polynomial of the given degree
    # This returns the Chebyshev polynomial T_1(x) = x
    f.basis(deg=1)
    return


@app.cell
def _():
    # Empty cell for additional experimentation
    return


if __name__ == "__main__":
    app.run()
