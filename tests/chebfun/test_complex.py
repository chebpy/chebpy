"""Unit-tests for Chebtech complex function operations.

This module contains tests for complex function operations in Chebtech,
including roots, rho ellipse construction, calculus, and real/imag methods.
"""

import numpy as np

from ..generic.complex import (  # noqa: F401
    test_calculus,
    test_complexfun_properties,
    test_real_imag,
    test_rho_ellipse_construction,
    test_roots,
)


class TestChebfunComplexEdgeCases:
    """Additional edge case tests for Chebfun complex operations."""

    def test_real_of_complex_chebfun(self):
        """Test real() method on complex chebfun."""
        from chebpy import chebfun

        f = chebfun(lambda x: x + 1j * x**2, [-1, 1])
        f_real = f.real()
        xx = np.linspace(-1, 1, 50)
        assert np.allclose(f_real(xx), xx, atol=1e-10)
        assert not f_real.iscomplex

    def test_real_of_real_chebfun(self):
        """Test real() method on real chebfun (should return self)."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        f_real = f.real()
        assert f_real is f  # Should return the same object

    def test_imag_of_complex_chebfun(self):
        """Test imag() method on complex chebfun."""
        from chebpy import chebfun

        f = chebfun(lambda x: x + 1j * x**2, [-1, 1])
        f_imag = f.imag()
        xx = np.linspace(-1, 1, 50)
        assert np.allclose(f_imag(xx), xx**2, atol=1e-10)

    def test_imag_of_real_chebfun(self):
        """Test imag() method on real chebfun (should return zero)."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        f_imag = f.imag()
        xx = np.linspace(-1, 1, 50)
        assert np.allclose(f_imag(xx), 0, atol=1e-14)
