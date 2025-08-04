"""Tests for the plotting functionality of ChebyshevPolynomial."""

import pytest
from matplotlib import pyplot as plt

from chebpy.core.chebyshev import from_coefficients
from tests.generic.plotting import test_plot  # noqa: F401


@pytest.fixture
def plotting_fixtures():
    """Fixture providing various ChebyshevPolynomial objects for plotting tests."""
    # Create a simple polynomial: 1 + 2x + 3x^2
    simple_poly = from_coefficients([1, 2, 3])

    # Create a polynomial with custom domain: 1 + 2x + 3x^2 on [0, 2]
    custom_domain_poly = from_coefficients([1, 2, 3], domain=(0, 2))

    # Create a complex polynomial: (1+1j) + (2+2j)x + (3+3j)x^2
    complex_poly = from_coefficients([1 + 1j, 2 + 2j, 3 + 3j])

    return {
        "simple": simple_poly,
        "custom_domain": custom_domain_poly,
        "complex": complex_poly,
    }


@pytest.fixture
def constfun():
    """Fixture providing a constant ChebyshevPolynomial for generic tests."""
    return from_coefficients([42.0])


@pytest.fixture
def complexfun():
    """Fixture providing a complex ChebyshevPolynomial for generic tests."""
    # Create a simple complex polynomial: x
    return from_coefficients([0, 1])


def test_plot_with_options(plotting_fixtures):
    """Test plotting a ChebyshevPolynomial with custom options."""
    fig, ax = plt.subplots()
    poly = plotting_fixtures["simple"]
    poly.plot(ax=ax, color="r", linestyle="--", linewidth=2)
    # No assertion needed; test passes if no exception is raised


def test_plot_custom_domain(plotting_fixtures):
    """Test plotting a ChebyshevPolynomial with a custom domain."""
    fig, ax = plt.subplots()
    poly = plotting_fixtures["custom_domain"]
    poly.plot(ax=ax)
    # No assertion needed; test passes if no exception is raised


def test_plot_complex_polynomial(plotting_fixtures):
    """Test plotting a complex ChebyshevPolynomial."""
    fig, ax = plt.subplots()
    poly = plotting_fixtures["complex"]
    poly.plot(ax=ax)
    # No assertion needed; test passes if no exception is raised
