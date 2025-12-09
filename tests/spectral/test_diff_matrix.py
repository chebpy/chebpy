"""Unit tests for differentiation matrices."""

import numpy as np

from chebpy import chebfun
from chebpy.spectral import cheb_points_scaled, diff_matrix
from chebpy.utilities import Interval


class TestChebPointsScaled:
    """Tests for scaled Chebyshev points."""

    def test_standard_domain(self):
        """Test Chebyshev points on [-1, 1]."""
        interval = Interval(-1, 1)
        pts = cheb_points_scaled(4, interval)
        assert len(pts) == 5
        assert np.isclose(pts[0], -1.0)  # Chebyshev points go from -1 to 1
        assert np.isclose(pts[-1], 1.0)

    def test_scaled_domain(self):
        """Test Chebyshev points on [0, 1]."""
        interval = Interval(0, 1)
        pts = cheb_points_scaled(4, interval)
        assert len(pts) == 5
        assert np.isclose(pts[0], 0.0)
        assert np.isclose(pts[-1], 1.0)
        assert np.all((pts >= 0) & (pts <= 1))

    def test_arbitrary_domain(self):
        """Test Chebyshev points on arbitrary interval."""
        interval = Interval(-5, 5)
        pts = cheb_points_scaled(10, interval)
        assert len(pts) == 11
        assert np.isclose(pts[0], -5.0, atol=1e-14)
        assert np.isclose(pts[-1], 5.0, atol=1e-14)


class TestDiffMatrix:
    """Tests for differentiation matrices."""

    def test_first_derivative_polynomial(self):
        """Test first derivative of polynomial."""
        # Test D * [x^2 values] ≈ [2x values]
        n = 8
        domain = Interval(-1, 1)
        D = diff_matrix(n, domain, order=1)

        x = cheb_points_scaled(n, domain)
        f_vals = x**2
        df_vals = D @ f_vals

        expected = 2 * x
        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-12, f"Error in derivative: {error}"

    def test_first_derivative_trig(self):
        """Test first derivative of trigonometric function."""
        # Test D * [sin(x) values] ≈ [cos(x) values]
        n = 16
        domain = Interval(-np.pi, np.pi)
        D = diff_matrix(n, domain, order=1)

        x = cheb_points_scaled(n, domain)
        f_vals = np.sin(x)
        df_vals = D @ f_vals

        expected = np.cos(x)
        error = np.max(np.abs(df_vals - expected))
        assert error < 3e-10, f"Error in derivative: {error}"  # Relax tolerance slightly

    def test_second_derivative(self):
        """Test second derivative."""
        # Test D^2 * [x^3 values] ≈ [6x values]
        n = 12
        domain = Interval(-1, 1)
        D2 = diff_matrix(n, domain, order=2)

        x = cheb_points_scaled(n, domain)
        f_vals = x**3
        d2f_vals = D2 @ f_vals

        expected = 6 * x
        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-10, f"Error in second derivative: {error}"

    def test_second_derivative_exponential(self):
        """Test second derivative of exponential."""
        # Test D^2 * [exp(x) values] ≈ [exp(x) values]
        n = 20
        domain = Interval(-1, 1)
        D2 = diff_matrix(n, domain, order=2)

        x = cheb_points_scaled(n, domain)
        f_vals = np.exp(x)
        d2f_vals = D2 @ f_vals

        expected = np.exp(x)
        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-9, f"Error in second derivative: {error}"

    def test_third_derivative(self):
        """Test third derivative."""
        # Test D^3 * [x^4 values] ≈ [24x values]
        n = 16
        domain = Interval(-1, 1)
        D3 = diff_matrix(n, domain, order=3)

        x = cheb_points_scaled(n, domain)
        f_vals = x**4
        d3f_vals = D3 @ f_vals

        expected = 24 * x
        error = np.max(np.abs(d3f_vals - expected))
        assert error < 1e-8, f"Error in third derivative: {error}"

    def test_scaled_domain(self):
        """Test differentiation on scaled domain [0, 2]."""
        n = 12
        domain = Interval(0, 2)
        D = diff_matrix(n, domain, order=1)

        x = cheb_points_scaled(n, domain)
        f_vals = x**2
        df_vals = D @ f_vals

        expected = 2 * x
        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-11, f"Error on scaled domain: {error}"

    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        n = 10
        domain = Interval(-1, 1)
        D = diff_matrix(n, domain)
        assert D.shape == (n + 1, n + 1)

    def test_sparse_format(self):
        """Test that matrix is returned in sparse format."""
        from scipy import sparse

        domain = Interval(-1, 1)
        D = diff_matrix(10, domain)
        assert sparse.issparse(D)
        assert D.format == 'csr'


class TestDiffMatrixIntegration:
    """Integration tests with chebfun."""

    def test_chebfun_derivative(self):
        """Test that diff_matrix agrees with chebfun.diff()."""
        # Create a chebfun
        f = chebfun(lambda x: np.sin(3 * x), [-1, 1])

        # Compute derivative using chebfun
        df_chebfun = f.diff()

        # Compute derivative using diff_matrix
        n = len(f.funs[0].coeffs) - 1
        domain = Interval(-1, 1)
        D = diff_matrix(n, domain)
        x = cheb_points_scaled(n, domain)
        f_vals = f(x)
        df_vals = D @ f_vals

        # Compare
        df_expected = df_chebfun(x)
        error = np.max(np.abs(df_vals - df_expected))
        assert error < 1e-10, f"Mismatch with chebfun.diff(): {error}"


class TestDriscollHaleDiffMatrix:
    """Tests for Driscoll-Hale rectangular differentiation matrices."""

    def test_shape_second_derivative(self):
        """D2 should map (n+1) values to (n-1) values."""
        from chebpy.spectral import diff_matrix_driscoll_hale

        n = 10
        D2 = diff_matrix_driscoll_hale(n, [-1, 1], order=2)
        assert D2.shape == (n - 1, n + 1)  # (9, 11)

    def test_shape_first_derivative(self):
        """D1 should map (n+1) values to (n) values."""
        from chebpy.spectral import diff_matrix_driscoll_hale

        n = 10
        D1 = diff_matrix_driscoll_hale(n, [-1, 1], order=1)
        assert D1.shape == (n, n + 1)  # (10, 11)

    def test_accuracy_polynomial(self):
        """Test accuracy on a polynomial f(x) = x^3, f'' = 6x."""
        from chebpy.spectral import diff_matrix_driscoll_hale, cheb_points_scaled
        from chebpy.utilities import Interval

        n = 10
        interval = [-1, 1]
        D2 = diff_matrix_driscoll_hale(n, interval, order=2)

        x_input = cheb_points_scaled(n, Interval(*interval))
        x_output = cheb_points_scaled(n - 2, Interval(*interval))

        f_vals = x_input**3
        f_deriv2_expected = 6 * x_output
        f_deriv2_computed = D2 @ f_vals

        np.testing.assert_allclose(f_deriv2_computed, f_deriv2_expected, atol=1e-10)

    def test_scaled_interval(self):
        """Test on non-standard interval [0, 2]."""
        from chebpy.spectral import diff_matrix_driscoll_hale, cheb_points_scaled
        from chebpy.utilities import Interval

        n = 12
        interval = [0, 2]
        D1 = diff_matrix_driscoll_hale(n, interval, order=1)

        x_input = cheb_points_scaled(n, Interval(*interval))
        x_output = cheb_points_scaled(n - 1, Interval(*interval))

        # f(x) = x^2, f'(x) = 2x
        f_vals = x_input**2
        f_deriv_expected = 2 * x_output
        f_deriv_computed = D1 @ f_vals

        np.testing.assert_allclose(f_deriv_computed, f_deriv_expected, atol=1e-10)
