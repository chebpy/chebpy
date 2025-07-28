"""Unit-tests for Chebfun ufunc operations.

This module contains tests for the ufunc operations of Chebfun,
including absolute, trigonometric, exponential, and logarithmic functions.
"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from .conftest import (
    eps, ufuncEmptyCaseTester, ufuncTester,
    uf1, uf2, uf3
)


@pytest.fixture
def ufuncs_fixtures():
    """Create fixtures for testing ufunc operations.

    This fixture creates an empty Chebfun object and an array of test points
    for evaluating functions.

    Returns:
        dict: Dictionary containing:
            emptyfun: Empty Chebfun object
            yy: Array of test points in [-1, 1]
    """
    emptyfun = Chebfun.initempty()
    yy = np.linspace(-1, 1, 2000)
    return {"emptyfun": emptyfun, "yy": yy}


def test_abs_absolute_alias(ufuncs_fixtures):
    """Test that abs and absolute are aliases.

    This test verifies that the abs and absolute methods of Chebfun
    are aliases for the same function.

    Args:
        ufuncs_fixtures: Fixture providing test objects.
    """
    emptyfun = ufuncs_fixtures["emptyfun"]
    assert Chebfun.abs == Chebfun.absolute


# Define the ufuncs to test
ufuncs = (
    np.absolute,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)


# Test empty cases for all ufuncs
@pytest.mark.parametrize("ufunc", ufuncs)
def test_empty_case(ufunc, ufuncs_fixtures):
    """Test ufunc operations on empty Chebfun objects.

    This test verifies that applying a ufunc to an empty Chebfun object
    results in an empty Chebfun object.

    Args:
        ufunc: The ufunc to test
        ufuncs_fixtures: Fixture providing test objects.
    """
    emptyfun = ufuncs_fixtures["emptyfun"]
    test_func = ufuncEmptyCaseTester(ufunc)
    test_func(emptyfun)


# Define the ufunc test parameters
ufunc_test_params = [
    (
        np.absolute,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.arccos,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arccosh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arcsin,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arcsinh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arctan,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arctanh,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.exp2,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.log,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log10,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log1p,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf1, (-3, 3)], eps),
            ([uf2, (-3, 3)], eps),
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
]


# Generate test functions for ufuncs
@pytest.mark.parametrize("ufunc,test_cases", ufunc_test_params)
def test_ufuncs(ufunc, test_cases, ufuncs_fixtures):
    """Test ufunc operations on Chebfun objects.

    This test verifies that applying a ufunc to a Chebfun object
    produces the expected result within a specified tolerance.

    Args:
        ufunc: The ufunc to test
        test_cases: List of test cases, each containing a function, interval, and tolerance
        ufuncs_fixtures: Fixture providing test objects.
    """
    yy = ufuncs_fixtures["yy"]
    for ([f, intvl], tol) in test_cases:
        from chebpy.core.utilities import Interval
        interval = Interval(*intvl)
        test_func = ufuncTester(ufunc, f, interval, tol)
        test_func(yy)