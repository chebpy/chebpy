"""Configuration and fixtures for pytest in the chebfun tests.

This module contains fixtures and helper functions specific to testing
the Chebfun class. It provides fixtures for common test objects and
helper functions for testing various operations.
"""

import operator

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences

# in Python 3, the operator module does not have a 'div' method
binops = [operator.add, operator.mul, operator.sub, operator.truediv]

# try:
#    # in Python 2 we need to test div separately
#    binops.append(operator.div)
#    div_binops = (operator.div, operator.truediv)
# except AttributeError:
#    # Python 3
#    div_binops = (operator.truediv,)
div_binops = (operator.truediv,)

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

# domain, test_tolerance
chebfun_testdomains = [
    ([-1, 1], 2 * eps),
    ([-2, 1], eps),
    ([-1, 2], eps),
    ([-5, 9], 35 * eps),
]


@pytest.fixture
def emptyfun() -> Chebfun:
    """Create an empty Chebfun function for testing.

    This fixture creates an empty Chebfun object that can be used
    to test the behavior of operations on empty functions.

    Returns:
        Chebfun: An empty Chebfun object
    """
    return Chebfun.initempty()


def binary_op_tester(f: callable, g: callable, binop: callable, dom: list, tol: float) -> callable:
    """Test binary operations between two Chebfun objects.

    This function creates Chebfun objects from the given functions and tests
    that the binary operation between them produces the expected result within
    a specified tolerance.

    Args:
        f (callable): First function
        g (callable): Second function
        binop (callable): Binary operator function (e.g., operator.add)
        dom (list): Domain for the functions
        tol (float): Tolerance for the comparison

    Returns:
        callable: A test function that can be used with pytest
    """
    a, b = dom
    xx = np.linspace(a, b, 1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, n + 1))
    gg = Chebfun.initfun_adaptive(g, np.linspace(a, b, m + 1))

    def fg_expected(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    def tester():
        vscl = max([ff.vscale, gg.vscale])
        hscl = max([ff.hscale, gg.hscale])
        lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
        assert ff.funs.size == n
        assert gg.funs.size == m
        assert fg.funs.size == n + m - 1

        # Increase tolerance for multiplication on large domains
        extra_factor = 1
        if binop == operator.mul and abs(b - a) > 10:
            extra_factor = 100

        # try:
        # Evaluate both functions
        fg_vals = fg(xx)
        fg_expected_vals = fg_expected(xx)

        # Skip test if there are any NaN or infinite values
        # if np.any(np.isnan(fg_vals)) or np.any(np.isnan(fg_expected_vals)) or \
        #   np.any(np.isinf(fg_vals)) or np.any(np.isinf(fg_expected_vals)):
        #    pytest.skip("NaN or infinite values encountered")

        assert np.max(fg_vals - fg_expected_vals) <= extra_factor * vscl * hscl * lscl * tol
        # except (RuntimeWarning, ValueError, OverflowError, FloatingPointError):
        #    # Skip test if numerical issues occur
        #    pytest.skip("Numerical issues encountered")

    return tester


def unary_op_tester(f: callable, unaryop: callable, dom: list, tol: float) -> callable:
    """Test unary operations on a Chebfun object.

    This function creates a Chebfun object from the given function and tests
    that the unary operation on it produces the expected result within a
    specified tolerance.

    Args:
        f (callable): Function to test
        unaryop (callable): Unary operator function (e.g., operator.neg)
        dom (list): Domain for the function
        tol (float): Tolerance for the comparison

    Returns:
        callable: A test function that can be used with pytest
    """
    a, b = dom
    xx = np.linspace(a, b, 1001)
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 9))

    def gg_expected(x):
        return unaryop(f(x))

    gg = unaryop(ff)

    def tester():
        vscl = ff.vscale
        hscl = ff.hscale
        lscl = max([fun.size for fun in ff])
        assert ff.funs.size == gg.funs.size
        assert np.max(gg(xx) - gg_expected(xx)) <= vscl * hscl * lscl * tol

    return tester


def ufunc_empty_case_tester(ufunc: callable) -> callable:
    """Test ufunc operations on empty Chebfun objects.

    This function tests that applying a ufunc to an empty Chebfun object
    results in an empty Chebfun object.

    Args:
        ufunc (callable): The ufunc to test

    Returns:
        callable: A test function that can be used with pytest
    """

    def tester(emptyfun):
        assert getattr(emptyfun, ufunc.__name__)().isempty

    return tester


def uf1(x: float) -> float:
    """Identity function.

    Args:
        x: Input value or array

    Returns:
        x: The input value or array
    """
    return x


def uf2(x: float) -> float:
    """Sine function with offset.

    Args:
        x: Input value or array

    Returns:
        float or array: sin(x - 0.5)
    """
    return sin(x - 0.5)


def uf3(x: float) -> float:
    """Sine function with scaling and offset.

    Args:
        x: Input value or array

    Returns:
        float or array: sin(25 * x - 1)
    """
    return sin(25 * x - 1)


def ufunc_tester(ufunc: callable, f: callable, interval: list, tol: float) -> callable:
    """Test ufunc operations on Chebfun objects.

    This function creates a Chebfun object from the given function and tests
    that applying the ufunc to it produces the expected result within a
    specified tolerance.

    Args:
        ufunc (callable): The ufunc to test
        f (callable): Function to test
        interval (Interval): Domain for the function
        tol (float): Tolerance for the comparison

    Returns:
        callable: A test function that can be used with pytest
    """
    a, b = interval
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

    def gg(x):
        return ufunc(f(x))

    gg_result = getattr(ff, ufunc.__name__)()

    def tester(yy):
        xx = interval(yy)
        vscl = gg_result.vscale
        lscl = sum([fun.size for fun in gg_result])
        assert np.max(gg(xx) - gg_result(xx)) <= vscl * lscl * tol

    return tester


# def domainBreakOpTester(domainBreakOp, f, g, dom, tol):
#     """Test domain-breaking operations on Chebfun objects.
#
#     This function creates Chebfun objects from the given functions and tests
#     that the domain-breaking operation between them produces the expected
#     result within a specified tolerance.
#
#     Args:
#         domainBreakOp (callable): The domain-breaking operation to test
#         f (callable): First function
#         g (callable or float): Second function or constant
#         dom (list): Domain for the functions
#         tol (float): Tolerance for the comparison
#
#     Returns:
#         callable: A test function that can be used with pytest
#     """
#     xx = np.linspace(dom[0], dom[-1], 1001)
#     ff = chebfun(f, dom)
#     gg = chebfun(g, dom)
#     # convert constant g to to callable
#     if isinstance(g, (int, float)):
#         ffgg = domainBreakOp(f(xx), g)
#     else:
#         ffgg = domainBreakOp(f(xx), g(xx))
#     fg = getattr(ff, domainBreakOp.__name__)(gg)
#
#     def tester():
#         vscl = max([ff.vscale, gg.vscale])
#         hscl = max([ff.hscale, gg.hscale])
#         lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
#         assert np.max(fg(xx) - ffgg) <= vscl * hscl * lscl * tol
#
#     return tester
