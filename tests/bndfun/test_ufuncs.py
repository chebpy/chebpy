"""Unit-tests for Bndfun numpy ufunc overloads."""

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval

from ..utilities import eps, sin


# Define utility functions for testing
def uf1(x):
    """Identity function."""
    return x


def uf2(x):
    """Sine function with offset."""
    return sin(x - 0.5)


def uf3(x):
    """Sine function with scaling and offset."""
    return sin(25 * x - 1)


# List of ufuncs to test
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


# Test empty cases for all ufuncs
@pytest.mark.parametrize("ufunc", ufuncs)
def test_emptycase(ufunc, emptyfun):
    """Test that applying ufuncs to empty Bndfun objects returns empty Bndfun objects."""
    assert getattr(emptyfun, ufunc.__name__)().isempty


# Test parameters for ufunc tests
ufunc_test_params = [
    (np.absolute, uf1, (-3, -0.5), eps),
    (np.arccos, uf1, (-0.8, 0.8), eps),
    (np.arccosh, uf1, (2, 3), eps),
    (np.arcsin, uf1, (-0.8, 0.8), eps),
    (np.arcsinh, uf1, (2, 3), eps),
    (np.arctan, uf1, (-0.8, 0.8), eps),
    (np.arctanh, uf1, (-0.8, 0.8), eps),
    (np.cos, uf1, (-3, 3), eps),
    (np.cosh, uf1, (-3, 3), eps),
    (np.exp, uf1, (-3, 3), eps),
    (np.exp2, uf1, (-3, 3), eps),
    (np.expm1, uf1, (-3, 3), eps),
    (np.log, uf1, (2, 3), eps),
    (np.log2, uf1, (2, 3), eps),
    (np.log10, uf1, (2, 3), eps),
    (np.log1p, uf1, (-0.8, 0.8), eps),
    (np.sinh, uf1, (-3, 3), eps),
    (np.sin, uf1, (-3, 3), eps),
    (np.tan, uf1, (-0.8, 0.8), eps),
    (np.tanh, uf1, (-3, 3), eps),
    (np.sqrt, uf1, (2, 3), eps),
    (np.cos, uf2, (-3, 3), eps),
    (np.cosh, uf2, (-3, 3), eps),
    (np.exp, uf2, (-3, 3), eps),
    (np.expm1, uf2, (-3, 3), eps),
    (np.sinh, uf2, (-3, 3), eps),
    (np.sin, uf2, (-3, 3), eps),
    (np.tan, uf2, (-0.8, 0.8), eps),
    (np.tanh, uf2, (-3, 3), eps),
    (np.cos, uf3, (-3, 3), eps),
    (np.cosh, uf3, (-3, 3), eps),
    (np.exp, uf3, (-3, 3), eps),
    (np.expm1, uf3, (-3, 3), eps),
    (np.sinh, uf3, (-3, 3), eps),
    (np.sin, uf3, (-3, 3), eps),
    (np.tan, uf3, (-0.8, 0.8), eps),
    (np.tanh, uf3, (-3, 3), eps),
]


@pytest.mark.parametrize("ufunc, f, interval, tol", ufunc_test_params)
def test_ufunc(ufunc, f, interval, tol):
    """Test applying ufuncs to Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(f, subinterval)

    def gg(x):
        return ufunc(f(x))

    gg_result = getattr(ff, ufunc.__name__)()

    xx = subinterval(yy)
    vscl = gg_result.vscale
    lscl = gg_result.size
    assert np.max(gg(xx) - gg_result(xx)) <= vscl * lscl * tol
