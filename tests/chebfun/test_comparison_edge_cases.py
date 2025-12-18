"""Tests for chebfun comparison operations."""

import numpy as np

from chebpy.api import chebfun
from chebpy.chebfun import Chebfun


class TestChebfunEdgeCases:
    """Test chebfun edge cases."""

    def test_eq_different_types(self):
        """Test equality comparison with non-Chebfun objects."""
        # Create a simple Chebfun
        f = chebfun(lambda x: x**2)

        # Compare with non-Chebfun objects
        assert (f == "not a chebfun") is False
        assert (f == 5) is False
        assert (f == [1, 2, 3]) is False

    def test_eq_empty_chebfuns(self):
        """Test equality comparison with empty Chebfun objects."""
        # Create empty Chebfuns
        f1 = Chebfun.initempty()
        f2 = Chebfun.initempty()

        # They should be equal
        assert f1 == f2

    def test_cumsum_multiple_funs(self):
        """Test cumsum with multiple functions to ensure continuity."""
        # Create a piecewise Chebfun with multiple pieces
        x = np.linspace(-1, 1, 5)
        f = chebfun(lambda t: np.abs(t), domain=x)

        # Compute the cumulative sum
        F = f.cumsum()  # noqa: N806

        # Check that F is continuous by evaluating at breakpoints
        for i in range(1, len(x) - 1):
            left_val = F(x[i] - 1e-10)
            right_val = F(x[i] + 1e-10)
            assert np.abs(left_val - right_val) < 1e-9

        # Check that F'(x) ≈ f(x)
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(F.diff()(xx) - f(xx))) < 1e-10

    def test_maximum_minimum_different_supports(self):
        """Test maximum and minimum with functions having different supports."""
        # Create two Chebfuns with different supports
        f = chebfun(lambda x: x**2, domain=[-1, 1])
        g = chebfun(lambda x: 1 - x**2, domain=[0, 2])

        # Test maximum
        h_max = f.maximum(g)

        # The result should be defined on [0, 1] (the intersection of domains)
        assert h_max.support[0] == 0
        assert h_max.support[1] == 1

        # Test minimum
        h_min = f.minimum(g)

        # The result should be defined on [0, 1] (the intersection of domains)
        assert h_min.support[0] == 0
        assert h_min.support[1] == 1

    def test_maximum_minimum_no_intersection(self):
        """Test maximum and minimum with functions having no intersection."""
        # Create two Chebfuns with non-overlapping supports
        f = chebfun(lambda x: x**2, domain=[-2, -1])
        g = chebfun(lambda x: 1 - x**2, domain=[1, 2])

        # Test maximum - should return empty
        h_max = f.maximum(g)
        assert h_max.isempty

        # Test minimum - should return empty
        h_min = f.minimum(g)
        assert h_min.isempty

    def test_maximum_minimum_empty_switch(self):
        """Test maximum and minimum that result in an empty switch."""
        # Create a special case where the switch would be empty
        # This is a bit tricky to construct, but we can try with functions
        # that are exactly equal at all points
        f = chebfun(lambda x: x**2, domain=[-1, 1])
        g = chebfun(lambda x: x**2, domain=[-1, 1])

        # The difference f-g has no roots, but the algorithm should handle this
        h_max = f.maximum(g)
        assert not h_max.isempty

        h_min = f.minimum(g)
        assert not h_min.isempty


class TestChebfunEqualityEdgeCases:
    """Test equality comparison edge cases."""

    def test_eq_with_non_chebfun(self):
        """Test equality comparison with non-Chebfun types."""
        f = chebfun(lambda x: x, [-1, 1])
        assert not (f == 5)
        assert not (f == "string")
        assert not (f == [1, 2, 3])

    def test_eq_different_vscales(self):
        """Test equality with different vertical scales."""
        f1 = chebfun(lambda x: x, [-1, 1])
        f2 = chebfun(lambda x: x + 1e-14, [-1, 1])  # Very close but different
        # Should be equal within tolerance
        assert f1 == f2

    def test_eq_different_types(self):
        """Test equality comparison with non-Chebfun objects."""
        # Create a simple Chebfun
        f = chebfun(lambda x: x**2)

        # Compare with non-Chebfun objects
        assert (f == "not a chebfun") is False
        assert (f == 5) is False
        assert (f == [1, 2, 3]) is False

    def test_eq_empty_chebfuns(self):
        """Test equality comparison with empty Chebfun objects."""
        # Create empty Chebfuns
        f1 = Chebfun.initempty()
        f2 = Chebfun.initempty()

        # They should be equal
        assert f1 == f2
