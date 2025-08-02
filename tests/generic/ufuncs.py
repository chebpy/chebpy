"""Generic test functions for universal functions (ufuncs).

This module contains test functions for universal functions (ufuncs) that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech). These tests
focus on operations with empty function objects and various mathematical functions.
"""

import numpy as np
import pytest


# Define utility functions for testing
def uf1(x):
    """Identity function."""
    return x


def uf2(x):
    """Sine function with offset."""
    return np.sin(x - 0.5)


def uf3(x):
    """Sine function with scaling and offset."""
    return np.sin(25 * x - 1)


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
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)


def ufunc_parameter():
    """Generate test parameters for ufunc tests.

    This function returns a list of tuples, each containing:
    1. A NumPy ufunc to test
    2. A test function to apply the ufunc to
    3. A domain interval for testing

    Returns:
        list: List of (ufunc, test_function, domain) tuples for testing
    """
    # Test parameters for ufunc tests
    ufunc_test_params = [
        (np.absolute, uf1, (-3, -0.5)),
        (np.arccos, uf1, (-0.8, 0.8)),
        (np.arccosh, uf1, (2, 3)),
        (np.arcsin, uf1, (-0.8, 0.8)),
        (np.arcsinh, uf1, (2, 3)),
        (np.arctan, uf1, (-0.8, 0.8)),
        (np.arctanh, uf1, (-0.8, 0.8)),
        (np.cos, uf1, (-3, 3)),
        (np.cosh, uf1, (-3, 3)),
        (np.exp, uf1, (-3, 3)),
        (np.exp2, uf1, (-3, 3)),
        (np.expm1, uf1, (-3, 3)),
        (np.log, uf1, (2, 3)),
        (np.log2, uf1, (2, 3)),
        (np.log10, uf1, (2, 3)),
        (np.log1p, uf1, (-0.8, 0.8)),
        (np.sinh, uf1, (-3, 3)),
        (np.sin, uf1, (-3, 3)),
        (np.tan, uf1, (-0.8, 0.8)),
        (np.tanh, uf1, (-3, 3)),
        (np.sqrt, uf1, (2, 3)),
        (np.cos, uf2, (-3, 3)),
        (np.cosh, uf2, (-3, 3)),
        (np.exp, uf2, (-3, 3)),
        (np.expm1, uf2, (-3, 3)),
        (np.sinh, uf2, (-3, 3)),
        (np.sin, uf2, (-3, 3)),
        (np.tan, uf2, (-0.8, 0.8)),
        (np.tanh, uf2, (-3, 3)),
        (np.cos, uf3, (-3, 3)),
        (np.cosh, uf3, (-3, 3)),
        (np.exp, uf3, (-3, 3)),
        (np.expm1, uf3, (-3, 3)),
        (np.sinh, uf3, (-3, 3)),
        (np.sin, uf3, (-3, 3)),
        (np.tan, uf3, (-0.8, 0.8)),
        (np.tanh, uf3, (-3, 3)),
    ]
    return ufunc_test_params


# Test empty cases for all ufuncs
@pytest.mark.parametrize("ufunc", ufuncs)
def test_emptycase(ufunc, emptyfun):
    """Test that applying ufuncs to empty Bndfun objects returns empty Bndfun objects."""
    assert getattr(emptyfun, ufunc.__name__)().isempty
