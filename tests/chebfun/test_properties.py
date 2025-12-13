"""Unit-tests for Chebfun properties.

This module contains tests for the various properties of Chebfun objects,
including breakpoints, domain, hscale, isempty, isconst, support, and vscale.
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.chebfun import Chebfun
from chebpy.utilities import Domain


@pytest.fixture
def properties_fixtures():
    """Create Chebfun objects for testing properties.

    This fixture creates several Chebfun objects with different characteristics
    for testing various properties.

    Returns:
        dict: Dictionary containing:
            f0: Empty Chebfun
            f1: Constant Chebfun
            f2: Identity Chebfun on [-1, 1]
            f3: Chebfun on a piecewise domain
    """
    f0 = Chebfun.initempty()
    f1 = Chebfun.initconst(1.0)
    f2 = Chebfun.initidentity([-1, 1])
    f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-2, 0, 1, 3])
    return {"f0": f0, "f1": f1, "f2": f2, "f3": f3}


def test_breakpoints(properties_fixtures):
    """Test the breakpoints property of Chebfun objects.

    This test verifies that the breakpoints property correctly returns
    the domain breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = properties_fixtures["f3"]
    assert np.all(f3.breakpoints == np.array([-2, 0, 1, 3]))


def test_domain(properties_fixtures):
    """Test the domain property of Chebfun objects.

    This test verifies that the domain property correctly returns
    a Domain object with the expected breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert f0.domain == Domain([])
    assert f1.domain == Domain([-1, 1])
    assert f2.domain == Domain([-1, 1])
    assert f3.domain == Domain([-2, 0, 1, 3])


def test_hscale(properties_fixtures):
    """Test the hscale property of Chebfun objects.

    This test verifies that the hscale property correctly returns
    the horizontal scale (maximum domain width) for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f3 = properties_fixtures["f3"]
    assert f0.hscale == 0.0
    assert f1.hscale == 1.0
    assert f3.hscale == 3.0


def test_isempty(properties_fixtures):
    """Test the isempty property of Chebfun objects.

    This test verifies that the isempty property correctly identifies
    empty Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    assert f0.isempty
    assert not f1.isempty


def test_isconst(properties_fixtures):
    """Test the isconst property of Chebfun objects.

    This test verifies that the isconst property correctly identifies
    constant Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert not f0.isconst
    assert f1.isconst
    assert not f2.isconst
    assert not f3.isconst


def test_support(properties_fixtures):
    """Test the support property of Chebfun objects.

    This test verifies that the support property correctly returns
    the first and last breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert f0.support.size == 0
    assert np.all(f1.support == np.array([-1, 1]))
    assert np.all(f2.support == np.array([-1, 1]))
    assert np.all(f3.support == np.array([-2, 3]))


def test_vscale(properties_fixtures):
    """Test the vscale property of Chebfun objects.

    This test verifies that the vscale property correctly returns
    the vertical scale (maximum absolute value) for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = properties_fixtures["f1"]
    assert f1.vscale == 1.0


class TestChebfunPropertiesEdgeCases:
    """Additional edge case tests for Chebfun properties."""

    def test_repr_multipiece_shows_total_length(self):
        """Test that __repr__ shows total length for multipiece chebfuns."""
        f = chebfun(lambda x: np.abs(x), [-1, 0, 1])
        repr_str = repr(f)
        assert "total length" in repr_str
        # Check that total length is sum of pieces
        total = sum(fun.size for fun in f)
        assert f"total length = {total}" in repr_str

    def test_breakpoints_property(self):
        """Test breakpoints property."""
        f = chebfun(lambda x: x, [-1, -0.5, 0, 0.5, 1])
        breakpoints = f.breakpoints
        expected = np.array([-1, -0.5, 0, 0.5, 1])
        assert np.allclose(breakpoints, expected, atol=1e-14)

    def test_copy_method(self):
        """Test copy creates independent chebfun."""
        f = chebfun(lambda x: x**2, [-1, 1])
        f_copy = f.copy()
        # Modify original shouldn't affect copy
        xx = np.linspace(-1, 1, 20)
        assert np.allclose(f(xx), f_copy(xx), atol=1e-14)
        # Verify it's a different object
        assert f is not f_copy

    def test_len_single_piece(self):
        """Test __len__ returns total size for single-piece chebfun."""
        f = chebfun(lambda x: np.sin(x), [-1, 1])
        # len should match the size of the single fun
        assert len(f) == f.funs[0].size
        # For smooth sin(x), should be relatively small
        assert len(f) < 30

    def test_len_multi_piece(self):
        """Test __len__ returns sum of sizes for multi-piece chebfun."""
        # Piecewise function with breakpoint at 0
        f = chebfun(lambda x: np.abs(x), [-1, 0, 1])
        # len should be sum of all pieces
        expected_len = sum(fun.size for fun in f.funs)
        assert len(f) == expected_len
        # Should have 2 pieces
        assert len(f.funs) == 2

    def test_len_constant(self):
        """Test __len__ for constant chebfun."""
        f = chebfun(3.14, [-1, 1])
        # Constant should have minimal length
        assert len(f) == 1

    def test_len_empty(self):
        """Test __len__ for empty chebfun."""
        f = Chebfun.initempty()
        # Empty chebfun has no funs
        assert len(f) == 0

    def test_len_behavior(self):
        """Test that len(f) returns the expected value."""
        # Create several test functions
        f1 = chebfun(lambda x: np.exp(x), [0, 1])
        f2 = chebfun(lambda x: x**2, [-1, 1])
        f3 = chebfun(lambda x: np.sin(10 * x), [0, 2 * np.pi])

        # Length should be positive integer
        assert isinstance(len(f1), int)
        assert len(f1) > 0
        assert isinstance(len(f2), int)
        assert len(f2) > 0
        assert isinstance(len(f3), int)
        assert len(f3) > 0

        # More oscillatory function should need more points
        assert len(f3) > len(f1)
