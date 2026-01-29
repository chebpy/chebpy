"""Additional tests to improve coverage for chebfun.py."""

import numpy as np

from chebpy.api import chebfun
from chebpy.chebfun import Chebfun


class TestAdditionalCoverage:
    """Additional tests to improve coverage for chebfun.py."""

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

    def test_absolute_method(self):
        """Test the absolute value method (__abs__) for Chebfun."""
        # Create a Chebfun that has negative values
        # Use a smooth function to avoid non-convergence warning for |x| near 0
        f = chebfun(lambda x: x**2 - 0.5, domain=[-1, 1])

        # Test __abs__
        abs_f = abs(f)

        # Check that it's a Chebfun
        assert isinstance(abs_f, Chebfun)

        # Check that it gives the correct values at sample points away from zeros
        xx = np.array([-1, -0.5, 0, 0.5, 1])
        expected = np.abs(xx**2 - 0.5)
        actual = abs_f(xx)
        assert np.max(np.abs(actual - expected)) < 1e-6

    def test_imag_complex_chebfun(self):
        """Test the imag() method on a complex Chebfun."""
        # Create a complex Chebfun
        f = chebfun(lambda x: np.exp(1j * np.pi * x), domain=[-1, 1])

        # Verify it's complex
        assert f.iscomplex

        # Get the imaginary part
        imag_f = f.imag()

        # Check that it's a Chebfun
        assert isinstance(imag_f, Chebfun)

        # Check that it gives the correct values
        xx = np.linspace(-1, 1, 100)
        expected = np.imag(np.exp(1j * np.pi * xx))
        actual = imag_f(xx)
        assert np.max(np.abs(actual - expected)) < 1e-10

    def test_diff_type_error(self):
        """Test that diff() raises TypeError for non-integer n."""
        import pytest

        f = chebfun(lambda x: x**2)

        # Should raise TypeError for float
        with pytest.raises(TypeError):
            f.diff(1.5)

        # Should raise TypeError for string
        with pytest.raises(TypeError):
            f.diff("1")
