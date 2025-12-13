"""Unit-tests for Chebfun algebraic operations.

This module contains tests for the algebraic operations of Chebfun,
including addition, subtraction, multiplication, division, and powers.
"""

import numpy as np

from chebpy import chebfun
from chebpy.chebfun import Chebfun

from ..generic.algebra import *  # noqa: F401, F403


class TestChebfunAlgebraEdgeCases:
    """Additional edge case tests for Chebfun algebraic operations."""

    def test_rpow_with_scalar(self):
        """Test reverse power operation (scalar ** chebfun)."""
        f = chebfun(lambda x: x, [-1, 1])
        result = 2**f
        expected = chebfun(lambda x: 2**x, [-1, 1])
        xx = np.linspace(-1, 1, 100)
        assert np.allclose(result(xx), expected(xx), atol=1e-10)

    def test_rpow_with_chebfun(self):
        """Test that rpow works when both arguments are chebfuns."""
        base = chebfun(lambda x: 2 + 0 * x, [-1, 1])  # Constant 2
        exponent = chebfun(lambda x: x, [-1, 1])
        result = base**exponent  # Should compute using __pow__
        xx = np.linspace(-1, 1, 50)
        expected = 2**xx
        assert np.allclose(result(xx), expected, atol=1e-10)

    def test_domain_setter(self):
        """Test domain property setter."""
        f = chebfun(lambda x: x**2, [-2, 2])
        # Set domain to restrict
        f.domain = [-1, 1]
        assert np.allclose(f.support, [-1, 1])
        # Verify restriction worked
        xx = np.linspace(-1, 1, 50)
        assert np.allclose(f(xx), xx**2, atol=1e-10)

    def test_binop_with_empty_second_arg(self):
        """Test binary operation when second argument is empty."""
        f = chebfun(lambda x: x, [-1, 1])
        empty = Chebfun.initempty()
        result = f + empty
        assert result.isempty

    def test_binop_with_scalar_no_simplify(self):
        """Test that binary ops with scalars don't simplify."""
        f = chebfun(lambda x: x**2, [-1, 0, 1])
        # Add scalar - should not simplify
        result = f + 5
        # Verify it still has multiple pieces
        assert len(result.funs) == len(f.funs)
