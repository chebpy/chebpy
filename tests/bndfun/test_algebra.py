"""Unit-tests for Bndfun algebraic operations."""

import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval

from ..generic.algebra import *
from ..utilities import eps


# Helper function for binary operator tests
def binary_op_test(f, g, subinterval, binop, yy):
    """Test a binary operation between two Bndfun objects."""
    ff = Bndfun.initfun_adaptive(f, subinterval)
    gg = Bndfun.initfun_adaptive(g, subinterval)

    def fg_expected(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale])
    lscl = max([ff.size, gg.size])
    xx = subinterval(yy)
    assert np.max(np.abs(fg(xx) - fg_expected(xx))) <= 6 * vscl * lscl * eps


# Test binary operations between Bndfun objects
subintervals = [Interval(-0.5, 0.9), Interval(-1.2, 1.3), Interval(-2.2, -1.9), Interval(0.4, 1.3)]


@pytest.mark.parametrize("subinterval", subintervals)
@pytest.mark.parametrize("f, g", [(np.exp, np.sin), (np.exp, lambda x: 2 - x), (lambda x: 2 - x, np.exp)])
def test_binary_operations(f, g, subinterval, binop=operator.pow):
    """Test binary operations between Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    binary_op_test(f, g, subinterval, binop, yy)


@pytest.mark.parametrize("subinterval", subintervals)
@pytest.mark.parametrize("f, g", [(np.exp, np.sin), (np.exp, lambda x: 2 - x), (lambda x: 2 - x, np.exp)])
def test_pow_operations(f, g, subinterval, binop=operator.pow):
    """Test power operations between Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    binary_op_test(f, g, subinterval, binop, yy)


# Helper function for unary operator tests
def unary_op_test(unaryop, f, subinterval, yy):
    """Test a unary operation on a Bndfun object."""
    test_fun = Bndfun.initfun_adaptive(f, subinterval)
    xx = subinterval(yy)


    assert np.max(np.abs((unaryop(test_fun))(xx) - unaryop(test_fun(xx)))) <= 4e1 * eps


@pytest.mark.parametrize("unaryop", [operator.pos, operator.neg])
def test_unary_operations(unaryop, testfunctions):
    """Test unary operations on Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    for f, _, _ in testfunctions:
        unary_op_test(unaryop, f, subinterval, yy)
