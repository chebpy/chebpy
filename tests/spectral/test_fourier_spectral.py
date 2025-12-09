"""Tests for Fourier spectral methods in spectral.py."""
import numpy as np
import pytest

from src.chebpy.spectral import fourier_diff_matrix, fourier_points_scaled
from src.chebpy.utilities import Interval


class TestFourierPoints:
    """Test Fourier collocation point generation."""

    def test_fourier_points_count(self):
        """Test correct number of points returned."""
        interval = Interval(0, 2*np.pi)
        for n in [4, 8, 16, 32]:
            pts = fourier_points_scaled(n, interval)
            assert len(pts) == n

    def test_fourier_points_spacing(self):
        """Test equally-spaced points."""
        n = 8
        interval = Interval(0, 2*np.pi)
        pts = fourier_points_scaled(n, interval)

        # Check spacing
        spacing = np.diff(pts)
        assert np.allclose(spacing, 2*np.pi / n)

        # First point at left endpoint
        assert np.allclose(pts[0], 0.0)

        # Last point should NOT be at right endpoint (periodic)
        assert not np.allclose(pts[-1], 2*np.pi)
        assert np.allclose(pts[-1], 2*np.pi - 2*np.pi/n)

    def test_fourier_points_arbitrary_interval(self):
        """Test on arbitrary interval."""
        n = 10
        interval = Interval(-1, 3)
        pts = fourier_points_scaled(n, interval)

        assert len(pts) == n
        assert np.allclose(pts[0], -1.0)
        h = 4 / n  # (3 - (-1)) / n
        assert np.allclose(np.diff(pts), h)


class TestFourierDiffMatrix:
    """Test Fourier differentiation matrices."""

    def test_diff_matrix_shape(self):
        """Test matrix has correct shape."""
        n = 16
        interval = Interval(0, 2*np.pi)

        D = fourier_diff_matrix(n, interval, order=1)
        assert D.shape == (n, n)

        D2 = fourier_diff_matrix(n, interval, order=2)
        assert D2.shape == (n, n)

    def test_diff_sin_to_cos(self):
        """Test D @ sin(x) = cos(x)."""
        n = 32
        interval = Interval(0, 2*np.pi)

        # Get points and matrix
        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        # Test on sin(x)
        f_vals = np.sin(x)
        df_vals = D @ f_vals
        expected = np.cos(x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-12, f"Error: {error}"

    def test_diff_cos_to_neg_sin(self):
        """Test D @ cos(x) = -sin(x)."""
        n = 32
        interval = Interval(0, 2*np.pi)

        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        f_vals = np.cos(x)
        df_vals = D @ f_vals
        expected = -np.sin(x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-12, f"Error: {error}"

    def test_diff_sin_5x(self):
        """Test D @ sin(5x) = 5*cos(5x)."""
        n = 64
        interval = Interval(0, 2*np.pi)

        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        f_vals = np.sin(5*x)
        df_vals = D @ f_vals
        expected = 5 * np.cos(5*x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10, f"Error: {error}"

    def test_second_derivative_sin(self):
        """Test D2 @ sin(x) = -sin(x)."""
        n = 32
        interval = Interval(0, 2*np.pi)

        x = fourier_points_scaled(n, interval)
        D2 = fourier_diff_matrix(n, interval, order=2)

        f_vals = np.sin(x)
        d2f_vals = D2 @ f_vals
        expected = -np.sin(x)

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-11, f"Error: {error}"

    def test_higher_order_derivative(self):
        """Test 4th derivative: D^4 @ sin(x) = sin(x)."""
        n = 64
        interval = Interval(0, 2*np.pi)

        x = fourier_points_scaled(n, interval)
        D4 = fourier_diff_matrix(n, interval, order=4)

        f_vals = np.sin(x)
        d4f_vals = D4 @ f_vals
        expected = np.sin(x)

        error = np.max(np.abs(d4f_vals - expected))
        assert error < 1e-9, f"Error: {error}"

    def test_arbitrary_interval_scaling(self):
        """Test derivative on arbitrary interval."""
        n = 32
        interval = Interval(0, 1)  # [0, 1] instead of [0, 2pi]

        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        # Test f(x) = sin(2*pi*x) on [0, 1]
        # df/dx = 2*pi*cos(2*pi*x)
        f_vals = np.sin(2*np.pi * x)
        df_vals = D @ f_vals
        expected = 2*np.pi * np.cos(2*np.pi * x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-11, f"Error: {error}"

    def test_edge_cases(self):
        """Test edge cases."""
        interval = Interval(0, 2*np.pi)

        # n=1 case
        D = fourier_diff_matrix(1, interval, order=1)
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

        # order=0 (identity)
        D0 = fourier_diff_matrix(8, interval, order=0)
        assert np.allclose(D0, np.eye(8))

    def test_odd_vs_even_n(self):
        """Test both odd and even n work correctly."""
        interval = Interval(0, 2*np.pi)

        for n in [15, 16, 31, 32]:  # Test odd and even
            x = fourier_points_scaled(n, interval)
            D = fourier_diff_matrix(n, interval, order=1)

            f_vals = np.sin(x)
            df_vals = D @ f_vals
            expected = np.cos(x)

            error = np.max(np.abs(df_vals - expected))
            assert error < 1e-11, f"Failed for n={n}, error={error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
