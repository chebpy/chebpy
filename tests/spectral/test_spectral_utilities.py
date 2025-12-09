"""Comprehensive tests for spectral.py utility functions to achieve >95% coverage."""

import warnings

import numpy as np
import pytest
from scipy import sparse

from chebpy import chebfun
from chebpy.spectral import (
    _barydiff_matrix,
    barycentric_matrix,
    cheb_points_scaled,
    diff_matrix,
    diff_matrix_rectangular,
    fourier_diff_matrix,
    fourier_points_scaled,
    identity_matrix,
    mult_matrix,
    projection_matrix_rectangular,
)
from chebpy.utilities import Interval


class TestChebPointsScaledEdgeCases:
    """Test edge cases for cheb_points_scaled."""

    def test_single_point(self):
        """Test n=0 returns 1 point."""
        interval = Interval(-1, 1)
        pts = cheb_points_scaled(0, interval)
        assert len(pts) == 1
        assert np.isclose(pts[0], 0.0)

    def test_two_points(self):
        """Test n=1 returns 2 points."""
        interval = Interval(-1, 1)
        pts = cheb_points_scaled(1, interval)
        assert len(pts) == 2
        assert np.isclose(pts[0], -1.0)
        assert np.isclose(pts[1], 1.0)


class TestDiffMatrixEdgeCases:
    """Test edge cases and all code paths in diff_matrix."""

    def test_n_zero_edge_case(self):
        """Test n=0 returns 1x1 matrix."""
        D = diff_matrix(0, [-1, 1])
        assert D.shape == (1, 1)
        assert sparse.issparse(D)

    def test_n_one_case(self):
        """Test n=1 case (2x2 matrix)."""
        D = diff_matrix(1, [-1, 1], order=1)
        assert D.shape == (2, 2)
        # For 2 points, derivative should map correctly
        x = cheb_points_scaled(1, Interval(-1, 1))
        f_vals = x**1  # Linear function
        df_vals = D @ f_vals
        # Derivative of x is 1 everywhere
        assert np.allclose(df_vals, 1.0, atol=1e-10)

    def test_high_order_warning(self):
        """Test warning for high-order derivatives."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            D = diff_matrix(8, [-1, 1], order=7)
            assert len(w) == 1
            assert "order 7" in str(w[0].message)
            assert "numerically unstable" in str(w[0].message)

    def test_very_high_order_warning(self):
        """Test warning for very high order."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            D = diff_matrix(10, [-1, 1], order=10)
            assert len(w) == 1
            assert "order 10" in str(w[0].message)

    def test_first_order_code_path(self):
        """Test order=1 uses standard formula (not barycentric)."""
        # This tests lines 168-176 (standard formula for first order)
        n = 8
        D1 = diff_matrix(n, [-1, 1], order=1)

        # Verify it works correctly
        x = cheb_points_scaled(n, Interval(-1, 1))
        f_vals = x**3
        df_vals = D1 @ f_vals
        expected = 3 * x**2

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10

    def test_second_order_uses_barycentric(self):
        """Test order=2 uses barycentric formula."""
        # This tests the order >= 2 branch (line 162)
        n = 10
        D2 = diff_matrix(n, [-1, 1], order=2)

        x = cheb_points_scaled(n, Interval(-1, 1))
        f_vals = x**4
        d2f_vals = D2 @ f_vals
        expected = 12 * x**2

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-9

    def test_fourth_order_derivative(self):
        """Test 4th order derivative."""
        n = 16
        D4 = diff_matrix(n, [-1, 1], order=4)

        x = cheb_points_scaled(n, Interval(-1, 1))
        f_vals = x**5
        d4f_vals = D4 @ f_vals
        expected = 120 * x

        error = np.max(np.abs(d4f_vals - expected))
        assert error < 1e-7

    def test_fifth_order_derivative(self):
        """Test 5th order derivative computes without error."""
        # High-order derivatives are numerically unstable, just verify they run
        n = 20
        D5 = diff_matrix(n, [-1, 1], order=5)

        # Verify shape and sparsity
        assert D5.shape == (n + 1, n + 1)
        assert sparse.issparse(D5)

        # Verify it can be applied
        x = cheb_points_scaled(n, Interval(-1, 1))
        f_vals = x**2
        d5f_vals = D5 @ f_vals
        assert d5f_vals.shape == (n + 1,)

    def test_sixth_order_derivative(self):
        """Test 6th order derivative (no warning at order=6)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            D6 = diff_matrix(20, [-1, 1], order=6)
            # Should not warn at order=6
            assert len(w) == 0

        # Verify shape and it can be applied
        assert D6.shape == (21, 21)
        x = cheb_points_scaled(20, Interval(-1, 1))
        f_vals = x**3
        d6f_vals = D6 @ f_vals
        assert d6f_vals.shape == (21,)


class TestDiffMatrixRectangular:
    """Test rectangular differentiation matrices."""

    def test_square_case_equals_standard(self):
        """Test m=n reduces to standard diff_matrix."""
        n = 8
        D_rect = diff_matrix_rectangular(n, n, [-1, 1], order=1)
        D_std = diff_matrix(n, [-1, 1], order=1)

        # Should have same shape
        assert D_rect.shape == D_std.shape
        assert D_rect.shape == (n + 1, n + 1)

    def test_overdetermined_shape(self):
        """Test rectangular matrix has correct shape."""
        n = 8
        m = 16
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=1)

        assert D_rect.shape == (m + 1, n + 1)
        assert sparse.issparse(D_rect)

    def test_rectangular_first_derivative(self):
        """Test rectangular matrix for first derivative."""
        n = 8
        m = 16
        interval = Interval(-1, 1)
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=1)

        # Get n+1 collocation points for input
        x_in = cheb_points_scaled(n, interval)
        f_vals = x_in**3

        # Apply rectangular differentiation
        df_vals = D_rect @ f_vals

        # Get m+1 collocation points for output
        x_out = cheb_points_scaled(m, interval)
        expected = 3 * x_out**2

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-9

    def test_rectangular_second_derivative(self):
        """Test rectangular matrix for second derivative."""
        n = 16
        m = 32
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=2)

        assert D_rect.shape == (m + 1, n + 1)

        interval = Interval(-1, 1)
        x_in = cheb_points_scaled(n, interval)
        f_vals = np.sin(x_in)

        d2f_vals = D_rect @ f_vals
        x_out = cheb_points_scaled(m, interval)
        expected = -np.sin(x_out)

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-7  # Relaxed tolerance for rectangular 2nd order

    def test_rectangular_third_derivative(self):
        """Test rectangular matrix for third derivative."""
        n = 12
        m = 24
        interval = Interval(-1, 1)
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=3)

        x_in = cheb_points_scaled(n, interval)
        f_vals = x_in**5

        d3f_vals = D_rect @ f_vals
        x_out = cheb_points_scaled(m, interval)
        expected = 60 * x_out**2

        error = np.max(np.abs(d3f_vals - expected))
        assert error < 1e-7

    def test_rectangular_error_m_less_than_n(self):
        """Test error when m < n."""
        with pytest.raises(ValueError, match="m >= n"):
            diff_matrix_rectangular(10, 5, [-1, 1])

    def test_rectangular_with_interval_object(self):
        """Test with Interval object."""
        n = 6
        m = 12
        interval = Interval(0, 2)
        D_rect = diff_matrix_rectangular(n, m, interval, order=1)

        x_in = cheb_points_scaled(n, interval)
        f_vals = x_in**2

        df_vals = D_rect @ f_vals
        x_out = cheb_points_scaled(m, interval)
        expected = 2 * x_out

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10


class TestMultMatrix:
    """Test multiplication matrices."""

    def test_mult_matrix_shape(self):
        """Test multiplication matrix has correct shape."""
        f = chebfun(lambda x: x**2, [-1, 1])
        n = 8
        M = mult_matrix(f, n, interval=Interval(-1, 1))

        assert M.shape == (n + 1, n + 1)
        assert sparse.issparse(M)

    def test_mult_matrix_diagonal(self):
        """Test multiplication matrix is diagonal."""
        f = chebfun(lambda x: x**2, [-1, 1])
        n = 8
        M = mult_matrix(f, n, interval=Interval(-1, 1))

        # Check it's diagonal
        M_dense = M.toarray()
        off_diagonal = M_dense - np.diag(np.diag(M_dense))
        assert np.allclose(off_diagonal, 0)

    def test_mult_matrix_values(self):
        """Test multiplication matrix has correct diagonal values."""
        f = chebfun(lambda x: 2 * x + 1, [-1, 1])
        n = 8
        interval = Interval(-1, 1)
        M = mult_matrix(f, n, interval=interval)

        x = cheb_points_scaled(n, interval)
        expected_diag = 2 * x + 1

        M_dense = M.toarray()
        actual_diag = np.diag(M_dense)

        error = np.max(np.abs(actual_diag - expected_diag))
        assert error < 1e-13

    def test_mult_matrix_explicit_interval_param(self):
        """Test mult_matrix with explicit interval parameter."""
        # This is already well-tested, but ensures we have explicit interval path
        f = chebfun(lambda x: x**2 + 1, [-1, 1])
        n = 8
        interval = Interval(-1, 1)

        # Explicitly provide interval
        M = mult_matrix(f, n, interval=interval)

        x = cheb_points_scaled(n, interval)
        expected_diag = x**2 + 1

        M_dense = M.toarray()
        actual_diag = np.diag(M_dense)

        error = np.max(np.abs(actual_diag - expected_diag))
        assert error < 1e-13

    def test_mult_matrix_action(self):
        """Test multiplication matrix action M @ v."""
        f = chebfun(lambda x: x, [-1, 1])
        n = 8
        interval = Interval(-1, 1)
        M = mult_matrix(f, n, interval=interval)

        # M @ ones should give x values
        ones = np.ones(n + 1)
        result = M @ ones

        x = cheb_points_scaled(n, interval)
        error = np.max(np.abs(result - x))
        assert error < 1e-13


class TestIdentityMatrix:
    """Test identity matrix construction."""

    def test_identity_shape(self):
        """Test identity matrix has correct shape."""
        for n in [0, 1, 4, 8, 16]:
            I = identity_matrix(n)
            assert I.shape == (n + 1, n + 1)

    def test_identity_is_sparse(self):
        """Test identity matrix is sparse."""
        I = identity_matrix(10)
        assert sparse.issparse(I)
        assert I.format == "csr"

    def test_identity_values(self):
        """Test identity matrix has ones on diagonal."""
        n = 8
        I = identity_matrix(n)
        I_dense = I.toarray()

        expected = np.eye(n + 1)
        assert np.allclose(I_dense, expected)

    def test_identity_action(self):
        """Test identity matrix action I @ v = v."""
        n = 10
        I = identity_matrix(n)

        v = np.random.randn(n + 1)
        result = I @ v

        assert np.allclose(result, v)


class TestBarycentricMatrix:
    """Test barycentric interpolation matrices."""

    def test_barycentric_shape(self):
        """Test barycentric matrix has correct shape."""
        x_eval = np.array([0.0, 0.5, 1.0])
        n = 8
        E = barycentric_matrix(x_eval, n, Interval(-1, 1))

        assert E.shape == (3, n + 1)
        assert sparse.issparse(E)

    def test_barycentric_scalar_input(self):
        """Test barycentric matrix with scalar input."""
        x_eval = 0.5
        n = 8
        E = barycentric_matrix(x_eval, n, Interval(-1, 1))

        assert E.shape == (1, n + 1)

    def test_barycentric_interpolation(self):
        """Test barycentric interpolation accuracy."""
        n = 16
        interval = Interval(-1, 1)
        x_cheb = cheb_points_scaled(n, interval)

        # Function values at Chebyshev points
        f_vals = np.sin(x_cheb)

        # Evaluate at arbitrary points
        x_eval = np.linspace(-0.9, 0.9, 20)
        E = barycentric_matrix(x_eval, n, interval)

        # Interpolate
        f_interp = E @ f_vals

        # Check against true values
        expected = np.sin(x_eval)
        error = np.max(np.abs(f_interp - expected))
        assert error < 1e-12

    def test_barycentric_at_nodes(self):
        """Test barycentric interpolation at collocation nodes."""
        n = 8
        interval = Interval(-1, 1)
        x_cheb = cheb_points_scaled(n, interval)

        # Evaluate at the nodes themselves
        E = barycentric_matrix(x_cheb, n, interval)

        # Should be identity-like
        f_vals = np.random.randn(n + 1)
        result = E @ f_vals

        error = np.max(np.abs(result - f_vals))
        assert error < 1e-13

    def test_barycentric_endpoint(self):
        """Test barycentric at endpoints."""
        n = 10
        interval = Interval(-1, 1)
        x_cheb = cheb_points_scaled(n, interval)

        # Evaluate at left endpoint
        E_left = barycentric_matrix(np.array([-1.0]), n, interval)
        f_vals = np.random.randn(n + 1)
        result = (E_left @ f_vals)[0]

        # Should equal first value (Chebyshev points include endpoints)
        assert np.isclose(result, f_vals[0], atol=1e-13)


class TestProjectionMatrixRectangular:
    """Test projection matrices for rectangular collocation."""

    def test_projection_shape(self):
        """Test projection matrix has correct shape."""
        n = 8
        m = 16
        PS = projection_matrix_rectangular(n, m, Interval(-1, 1))

        assert PS.shape == (n + 1, m + 1)
        assert sparse.issparse(PS)

    def test_projection_error_m_less_than_n(self):
        """Test error when m < n."""
        with pytest.raises(ValueError, match="m >= n"):
            projection_matrix_rectangular(10, 5, Interval(-1, 1))

    def test_projection_interpolation(self):
        """Test projection matrix interpolates correctly."""
        n = 8
        m = 16
        interval = Interval(-1, 1)
        PS = projection_matrix_rectangular(n, m, interval)

        # Values at m+1 collocation points
        x_colloc = cheb_points_scaled(m, interval)
        f_vals = np.sin(x_colloc)

        # Project to n+1 coefficient points
        f_coeff = PS @ f_vals

        # Check values at coefficient points
        x_coeff = cheb_points_scaled(n, interval)
        expected = np.sin(x_coeff)

        error = np.max(np.abs(f_coeff - expected))
        assert error < 1e-12

    def test_projection_compose_with_diff(self):
        """Test PS @ D_rect gives something reasonable."""
        n = 6
        m = 12
        interval = Interval(-1, 1)

        PS = projection_matrix_rectangular(n, m, interval)
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=1)

        # Compose: PS @ D_rect should be (n+1, n+1)
        A = PS @ D_rect
        assert A.shape == (n + 1, n + 1)

    def test_projection_non_matching_nodes(self):
        """Test projection matrix when n and m differ (no exact node matches)."""
        # Use n and m that don't share nodes (not m=2*n)
        n = 7
        m = 11  # Not a multiple of n
        interval = Interval(-1, 1)

        PS = projection_matrix_rectangular(n, m, interval)

        # Get both grids
        x_coeff = cheb_points_scaled(n, interval)
        x_colloc = cheb_points_scaled(m, interval)

        # Most points should not match (triggers barycentric interpolation lines 459-462)
        # Test interpolation works
        f_vals_colloc = np.sin(x_colloc)
        f_vals_coeff = PS @ f_vals_colloc

        expected = np.sin(x_coeff)
        error = np.max(np.abs(f_vals_coeff - expected))
        assert error < 1e-12


class TestBaryDiffMatrix:
    """Test internal _barydiff_matrix function."""

    def test_barydiff_empty_array(self):
        """Test empty array case."""
        x = np.array([])
        w = np.array([])
        D = _barydiff_matrix(x, w, order=1)

        assert D.shape == (0,)
        assert len(D) == 0

    def test_barydiff_single_point(self):
        """Test single point case."""
        x = np.array([0.0])
        w = np.array([1.0])
        D = _barydiff_matrix(x, w, order=1)

        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_barydiff_order_zero(self):
        """Test order=0 returns identity."""
        x = np.linspace(-1, 1, 5)
        w = np.ones(5)
        D = _barydiff_matrix(x, w, order=0)

        assert np.allclose(D, np.eye(5))

    def test_barydiff_without_angles(self):
        """Test without angle argument (t=None path)."""
        from chebpy.algorithms import barywts2, chebpts2

        n = 8
        x = chebpts2(n + 1)
        w = barywts2(n + 1)

        # Call without t argument
        D = _barydiff_matrix(x, w, order=1, t=None)

        # Test it works
        f_vals = x**2
        df_vals = D @ f_vals
        expected = 2 * x

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10

    def test_barydiff_order_1(self):
        """Test order=1 code path."""
        from chebpy.algorithms import barywts2, chebpts2

        n = 10
        x = chebpts2(n + 1)
        w = barywts2(n + 1)
        t = np.arccos(x)

        D = _barydiff_matrix(x, w, order=1, t=t)

        # Verify shape
        assert D.shape == (n + 1, n + 1)

        # Test on polynomial
        f_vals = x**3
        df_vals = D @ f_vals
        expected = 3 * x**2

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10

    def test_barydiff_order_2(self):
        """Test order=2 code path."""
        from chebpy.algorithms import barywts2, chebpts2

        n = 12
        x = chebpts2(n + 1)
        w = barywts2(n + 1)
        t = np.arccos(x)

        D = _barydiff_matrix(x, w, order=2, t=t)

        # Test on polynomial
        f_vals = x**4
        d2f_vals = D @ f_vals
        expected = 12 * x**2

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-9

    def test_barydiff_order_3_and_higher(self):
        """Test order >= 3 code path (loop)."""
        from chebpy.algorithms import barywts2, chebpts2

        n = 14
        x = chebpts2(n + 1)
        w = barywts2(n + 1)
        t = np.arccos(x)

        # Test order 3
        D3 = _barydiff_matrix(x, w, order=3, t=t)
        f_vals = x**5
        d3f_vals = D3 @ f_vals
        expected = 60 * x**2
        error = np.max(np.abs(d3f_vals - expected))
        assert error < 1e-8

        # Test order 4
        D4 = _barydiff_matrix(x, w, order=4, t=t)
        f_vals = x**6
        d4f_vals = D4 @ f_vals
        expected = 360 * x**2
        error = np.max(np.abs(d4f_vals - expected))
        assert error < 1e-7

    def test_barydiff_symmetry_even_n(self):
        """Test symmetry forcing for even N."""
        from chebpy.algorithms import barywts2, chebpts2

        # Even N to trigger symmetry code (lines 311-316)
        n = 10  # N = 11 (odd), try n=9 for N=10 (even)
        x = chebpts2(n)
        w = barywts2(n)
        t = np.arccos(x)

        D = _barydiff_matrix(x, w, order=1, t=t)

        # Check diagonal symmetry
        diag = np.diag(D)
        N = len(diag)
        half_N = N // 2

        # Bottom half should be negative of top half
        for k in range(half_N):
            idx = N - 1 - k
            assert np.isclose(diag[idx], -diag[k], atol=1e-10)


class TestFourierDiffMatrixEdgeCases:
    """Test edge cases for Fourier differentiation."""

    def test_fourier_n_zero(self):
        """Test n=0 case."""
        D = fourier_diff_matrix(0, Interval(0, 2 * np.pi))
        assert D.shape == (0,)

    def test_fourier_n_one(self):
        """Test n=1 case."""
        D = fourier_diff_matrix(1, Interval(0, 2 * np.pi))
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_fourier_order_zero(self):
        """Test order=0 returns identity."""
        n = 16
        D = fourier_diff_matrix(n, Interval(0, 2 * np.pi), order=0)
        assert np.allclose(D, np.eye(n))

    def test_fourier_first_order_odd(self):
        """Test first-order derivative with odd n."""
        n = 15  # Odd
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        # Test on sin(x)
        f_vals = np.sin(x)
        df_vals = D @ f_vals
        expected = np.cos(x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-11

    def test_fourier_first_order_even(self):
        """Test first-order derivative with even n."""
        n = 16  # Even
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)
        D = fourier_diff_matrix(n, interval, order=1)

        # Test on sin(2x)
        f_vals = np.sin(2 * x)
        df_vals = D @ f_vals
        expected = 2 * np.cos(2 * x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-11

    def test_fourier_second_order_odd(self):
        """Test second-order derivative with odd n."""
        n = 17  # Odd
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)
        D2 = fourier_diff_matrix(n, interval, order=2)

        # Test on sin(x)
        f_vals = np.sin(x)
        d2f_vals = D2 @ f_vals
        expected = -np.sin(x)

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-10

    def test_fourier_second_order_even(self):
        """Test second-order derivative with even n."""
        n = 18  # Even
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)
        D2 = fourier_diff_matrix(n, interval, order=2)

        # Test on cos(3x)
        f_vals = np.cos(3 * x)
        d2f_vals = D2 @ f_vals
        expected = -9 * np.cos(3 * x)

        error = np.max(np.abs(d2f_vals - expected))
        assert error < 1e-9

    def test_fourier_higher_order_fft_odd(self):
        """Test higher-order derivatives use FFT path (odd n)."""
        n = 31  # Odd
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)

        # Test 3rd derivative
        D3 = fourier_diff_matrix(n, interval, order=3)
        f_vals = np.sin(x)
        d3f_vals = D3 @ f_vals
        expected = -np.cos(x)

        error = np.max(np.abs(d3f_vals - expected))
        assert error < 1e-10

    def test_fourier_higher_order_fft_even(self):
        """Test higher-order derivatives use FFT path (even n)."""
        n = 32  # Even
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)

        # Test 4th derivative
        D4 = fourier_diff_matrix(n, interval, order=4)
        f_vals = np.sin(2 * x)
        d4f_vals = D4 @ f_vals
        expected = 16 * np.sin(2 * x)

        error = np.max(np.abs(d4f_vals - expected))
        assert error < 1e-9

    def test_fourier_fifth_order(self):
        """Test 5th order derivative."""
        n = 64
        interval = Interval(0, 2 * np.pi)
        x = fourier_points_scaled(n, interval)

        D5 = fourier_diff_matrix(n, interval, order=5)
        f_vals = np.sin(x)
        d5f_vals = D5 @ f_vals
        # 5th derivative of sin: sin -> cos -> -sin -> -cos -> sin -> cos
        expected = np.cos(x)

        error = np.max(np.abs(d5f_vals - expected))
        assert error < 1e-8  # Relaxed tolerance for 5th order FFT

    def test_fourier_arbitrary_interval(self):
        """Test Fourier on arbitrary interval [0, 1]."""
        n = 32
        interval = Interval(0, 1)
        x = fourier_points_scaled(n, interval)

        D = fourier_diff_matrix(n, interval, order=1)

        # Function: sin(2*pi*x)
        # Derivative: 2*pi*cos(2*pi*x)
        f_vals = np.sin(2 * np.pi * x)
        df_vals = D @ f_vals
        expected = 2 * np.pi * np.cos(2 * np.pi * x)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-10


class TestIntegrationWithChebfun:
    """Integration tests combining multiple functions."""

    def test_diff_then_mult(self):
        """Test D @ (M @ v) for solving linear operators."""
        n = 12
        interval = Interval(-1, 1)

        # Create multiplication matrix for x
        f = chebfun(lambda x: x, [-1, 1])
        M = mult_matrix(f, n, interval=interval)

        # Create differentiation matrix
        D = diff_matrix(n, [-1, 1], order=1)

        # Test (D @ M) @ v
        x = cheb_points_scaled(n, interval)
        v = np.sin(x)

        result = D @ (M @ v)

        # d/dx(x * sin(x)) = sin(x) + x*cos(x)
        expected = np.sin(x) + x * np.cos(x)

        error = np.max(np.abs(result - expected))
        assert error < 1e-10

    def test_rectangular_workflow(self):
        """Test complete rectangular workflow: D_rect, then project."""
        n = 16
        m = 32
        interval = Interval(-1, 1)

        # Rectangular differentiation: (m+1, n+1)
        D_rect = diff_matrix_rectangular(n, m, [-1, 1], order=1)

        # Input values at n+1 points
        x_in = cheb_points_scaled(n, interval)
        f_vals = np.exp(x_in)

        # Differentiate (gives m+1 values)
        df_vals = D_rect @ f_vals

        # Expected at m+1 points
        x_out = cheb_points_scaled(m, interval)
        expected = np.exp(x_out)

        error = np.max(np.abs(df_vals - expected))
        assert error < 1e-8  # Relaxed for rectangular workflow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
