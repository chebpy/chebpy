"""Unit-tests for Chebtech2 algebraic operations.

This module contains tests for the algebraic operations of Chebtech2,
including addition, subtraction, multiplication, division, and powers.
"""
import pytest

from chebpy.core.chebtech import Chebtech2

from ..generic.algebra import *


@pytest.mark.parametrize("binop", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_binary_operations(binop, testfunctions):
    """Test binary operations between two Chebtech objects.

    This test verifies that binary operations (addition, subtraction,
    multiplication, division) between two Chebtech2 objects produce the
    expected results within a tolerance that scales with the size and
    scale of the functions.

    Args:
        binop: Binary operator function (e.g., operator.add)
        testfunctions: List of test functions, each represented as a tuple:
                      (fun, funlen, has_roots).
    """

    def binary_op_tester(f: callable, g: callable, nf: int, ng: int):
        """Perform binary operation between two Chebtech2 functions."""
        ff = Chebtech2.initfun_fixedlen(f, nf)
        gg = Chebtech2.initfun_fixedlen(g, ng)
        xx = np.linspace(-1, 1, 1000)

        def fg_expected(x):
            return binop(f(x), g(x))

        fg = binop(ff, gg)

        vscl = max(ff.vscale, gg.vscale, fg.vscale)
        lscl = max(ff.size, gg.size, fg.size)
        tol = 5e1 * eps * lscl * vscl

        return fg(xx), fg_expected(xx), tol

    for f, nf, _ in testfunctions:
        for g, ng, has_roots_g in testfunctions:
            if binop is operator.truediv and (g.__name__ == "zerofun" or has_roots_g):
                continue

            actual, expected, tol = binary_op_tester(f, g, nf, ng)
            absdiff = np.max(np.abs(actual - expected))
            assert absdiff <= tol, (
                f"{binop.__name__} failed for {f.__name__} and {g.__name__}: "
                f"max diff = {absdiff:.2e}, tol = {tol:.2e}"
            )


@pytest.mark.parametrize("unaryop", [operator.pos, operator.neg])
def test_unary_operations(unaryop, testfunctions):
    """Test unary operations on a Chebtech object.

    This test verifies that unary operations (positive, negative) on
    Chebtech2 objects produce the expected results within a tolerance
    that scales with the size and scale of the function.

    Args:
        unaryop: Unary operator function (e.g., operator.neg)
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length, and _ is an unused parameter.
    """
    for f, nf, _ in testfunctions:
        ff = Chebtech2.initfun_fixedlen(f, nf)
        xx = np.linspace(-1, 1, 1000)
        gg = unaryop(ff)

        absdiff = np.max(np.abs(unaryop(f(xx)) - gg(xx)))

        vscl = max([ff.vscale, gg.vscale])
        lscl = max([ff.size, gg.size])
        tol = 5e1 * eps * lscl * vscl

        assert absdiff <= tol
