"""Unit-tests for Chebfun ufunc operations.

This module contains tests for the ufunc operations of Chebfun,
including absolute, trigonometric, exponential, and logarithmic functions.
"""

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun

from ..utilities import eps


def test_abs_absolute_alias():
    """Test that abs and absolute are aliases.

    This test verifies that the abs and absolute methods of Chebfun
    are aliases for the same function.
    """
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
def test_empty_case(ufunc, emptyfun):
    """Test ufunc operations on empty Chebfun objects.

    This test verifies that applying a ufunc to an empty Chebfun object
    results in an empty Chebfun object.

    Args:
        ufunc: The ufunc to test
        emptyfun: Fixture providing an empty Chebfun object
    """
    assert getattr(emptyfun, ufunc.__name__)().isempty

@pytest.fixture()
def ufunc_parameter(uf1, uf2, uf3):
    """List of ufunc test parameters."""
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
    return ufunc_test_params


# Generate test functions for ufuncs
def test_ufuncs(ufunc_parameter):
    """Test ufunc operations on Chebfun objects.

    This test verifies that applying a ufunc to a Chebfun object
    produces the expected result within a specified tolerance.

    Args:
        ufunc_parameter: List of tuples, each containing a ufunc and a list of test cases
            where each test case contains a function, interval, and tolerance
    """
    yy = np.linspace(-1, 1, 2000)

    for ufunc, test_cases in ufunc_parameter:
        for [f, intvl], tol in test_cases:
            from chebpy.core.utilities import Interval

            interval = Interval(*intvl)

            a, b = interval
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

            def gg(x):
                return ufunc(f(x))

            gg_result = getattr(ff, ufunc.__name__)()

            xx = interval(yy)
            vscl = gg_result.vscale
            lscl = sum([fun.size for fun in gg_result])
            assert np.max(gg(xx) - gg_result(xx)) <= vscl * lscl * tol
