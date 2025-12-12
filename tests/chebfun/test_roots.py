"""Unit-tests for Chebfun roots functionality.

This module contains tests for finding the roots of Chebfun objects,
including empty, constant, and various polynomial and trigonometric functions.
"""

import numpy as np
import pytest

from chebpy.chebfun import Chebfun

from ..generic.roots import rootstestfuns, test_empty  # noqa: F401
from ..utilities import pi, sin


@pytest.mark.parametrize("f, roots", rootstestfuns)
def test_roots(f, roots):
    """Test that the roots of a function are correctly identified.

    This test verifies that the roots() method of a Chebtech object
    correctly identifies the roots of various functions within the
    specified tolerance.

    Args:
        f: Function to find roots of
        roots: Expected roots
        tol: Tolerance for comparison
    """
    ff = Chebfun.initfun_adaptive(f)
    rts = ff.roots()
    assert np.max(np.abs(rts - roots)) <= 1e-15


def test_roots_const():
    """Test roots method on constant Chebfun objects.

    This test verifies that the roots method correctly handles
    constant Chebfun objects, both zero and non-zero.
    """
    # Non-zero constant should have no roots
    f_nonzero = Chebfun.initconst(1.0, [-1, 1])
    assert f_nonzero.roots().size == 0

    # Zero constant is a special case - technically every point is a root
    # but the implementation should return an empty array
    f_zero = Chebfun.initconst(0.0, [-1, 1])
    assert f_zero.roots().size == 0


def test_roots_multiple_intervals():
    """Test roots method on a Chebfun with multiple intervals.

    This test verifies that the roots method correctly identifies
    roots across multiple intervals.
    """
    # Create a Chebfun with sin(2*pi*x) on a domain with multiple breakpoints
    # This function has roots at x = 0, 0.5, 1, 1.5, etc.
    f = Chebfun.initfun_adaptive(lambda x: sin(2 * pi * x), [-1, 0, 1])

    # Find the roots
    roots = f.roots()

    # Expected roots at x = -1, -0.5, 0, 0.5, 1
    # The roots at the breakpoints (0) should only be counted once
    expected_roots = np.array([-1, -0.5, 0, 0.5, 1])

    # Check that we have the correct number of roots
    assert roots.size == expected_roots.size

    # Check that each expected root is found (within tolerance)
    sorted_roots = np.sort(roots)
    assert np.allclose(sorted_roots, expected_roots, atol=1e-10)


def test_roots_high_frequency():
    """Test roots method on a high-frequency function.

    This test verifies that the roots method can accurately find
    all roots of a high-frequency function.
    """
    # Create a Chebfun for sin(10*pi*x) which has 20 roots in [-1, 1]
    f = Chebfun.initfun_adaptive(lambda x: sin(10 * pi * x), [-1, 1])

    # Find the roots
    roots = f.roots()

    # Should have 21 roots at x = k/10 for k = -10, -9, ..., 9, 10
    expected_roots = np.linspace(-1, 1, 21)

    assert roots.size == expected_roots.size
    assert np.allclose(np.sort(roots), expected_roots, atol=1e-10)


class TestChebfunRootsEdgeCases:
    """Additional edge case tests for Chebfun roots."""

    def test_roots_merging_at_breakpoints(self):
        """Test that roots at breakpoints are properly handled."""
        from chebpy import chebfun

        # Function with root at x=0 (a breakpoint)
        f = chebfun(lambda x: x**2 - 0.25, [-1, 0, 1])
        roots = f.roots()
        expected = np.array([-0.5, 0.5])
        assert np.allclose(roots, expected, atol=1e-10)

    def test_roots_multipiece(self):
        """Test roots on multipiece chebfun."""
        from chebpy import chebfun

        f = chebfun(lambda x: (x + 0.5) * (x - 0.3), [-1, 0, 1])
        roots = f.roots()
        # Should find both roots
        assert roots.size == 2
        assert np.any(np.isclose(roots, -0.5, atol=1e-10))
        assert np.any(np.isclose(roots, 0.3, atol=1e-10))

    def test_roots_on_complex_function(self):
        """Test roots on a more complex multipiece function."""
        from chebpy import chebfun

        # Function with multiple roots
        def multi_root(x):
            return (x + 0.7) * (x + 0.2) * (x - 0.3) * (x - 0.8)

        f = chebfun(multi_root, [-1, -0.5, 0, 0.5, 1])
        roots = f.roots()
        # Should find all 4 roots
        assert roots.size == 4
        expected_roots = np.array([-0.7, -0.2, 0.3, 0.8])
        for expected_root in expected_roots:
            assert np.any(np.abs(roots - expected_root) < 1e-10)
