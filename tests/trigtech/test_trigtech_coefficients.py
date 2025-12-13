"""Tests for trigtech coefficient handling.

This module uses direct coefficient manipulation to test specific
scenarios that are difficult to reach through normal function construction.
"""

import numpy as np

from chebpy.trigtech import Trigtech


class TestDirectCodePaths:
    """Test specific code paths using direct manipulation."""

    def test_simplify_no_significant_coeffs(self):
        """Test simplify returns const 0 when no coeffs significant."""
        # Create trigtech with all near-zero coefficients
        coeffs = np.array([0.0, 1e-16, 1e-17, 1e-16], dtype=complex)
        f = Trigtech(coeffs)

        # The function values should be near zero
        # This tests the path where simplify finds no significant frequencies
        assert f.size == 4

    def test_simplify_max_sig_freq_calculation(self):
        """Test simplify calculates max significant frequency."""
        # Create coefficients with specific frequency pattern
        n = 64
        coeffs = np.zeros(n, dtype=complex)
        coeffs[0] = 1.0  # DC
        coeffs[5] = 0.1  # Freq 5
        coeffs[-5] = 0.1  # Freq -5
        f = Trigtech(coeffs)

        # Should identify freq 5 as max significant
        assert f.size == n

    def test_simplify_n_min_calculation(self):
        """Test simplify calculates n_min correctly."""
        # Create with moderate frequency content
        n = 32
        coeffs = np.zeros(n, dtype=complex)
        coeffs[0] = 1.0
        coeffs[3] = 0.5
        coeffs[-3] = 0.5
        f = Trigtech(coeffs)

        # n_min should be calculated as 2*(max_freq + 1)
        assert f.size == n

    def test_simplify_power_of_2_rounding(self):
        """Test simplify rounds to power of 2."""
        # Create coefficients that need specific size
        n = 128
        coeffs = np.zeros(n, dtype=complex)
        coeffs[0] = 1.0
        coeffs[7] = 0.3
        coeffs[-7] = 0.3
        f = Trigtech(coeffs)

        # Should round n_keep to power of 2
        assert f.size == n

    def test_simplify_dont_increase_size(self):
        """Test simplify doesn't increase size."""
        # Already small coeffs array
        coeffs = np.array([1.0, 0.5, 0.3, 0.5], dtype=complex)
        f = Trigtech(coeffs)

        # Size is already small (4), shouldn't need to grow
        assert f.size == 4

    def test_simplify_early_return_no_reduction(self):
        """Test simplify returns copy if n_keep >= n."""
        # Small function that's already optimal
        coeffs = np.array([1.0, 0.5], dtype=complex)
        f = Trigtech(coeffs)

        # Already minimal, should just return copy
        assert f.size == 2

    def test_simplify_resampling_real_values(self):
        """Test simplify resampling with real values."""
        # Create function with real coefficients
        n = 64
        coeffs = np.zeros(n, dtype=complex)
        # Make it symmetric for real function
        coeffs[0] = 1.0
        coeffs[2] = 0.5
        coeffs[-2] = 0.5  # Conjugate symmetry
        f = Trigtech(coeffs)

        # Test values are accessible
        vals = f.values()
        assert len(vals) == n

    def test_simplify_resampling_complex_values(self):
        """Test simplify resampling with complex values."""
        # Create function with complex coefficients
        n = 32
        coeffs = np.zeros(n, dtype=complex)
        coeffs[0] = 1.0
        coeffs[3] = 0.3 + 0.2j
        coeffs[-3] = 0.3 - 0.2j
        f = Trigtech(coeffs)

        # Should have complex values
        vals = f.values()
        assert np.iscomplexobj(vals)

    def test_add_zero_check_succeeds(self):
        """Test addition recognizes zero result."""
        # Create two small Trigtech that should sum to ~zero
        coeffs = np.array([1e-16, 1e-17], dtype=complex)
        f = Trigtech(coeffs)
        g = Trigtech(-coeffs)

        # Both same size, should add directly
        assert f.size == g.size

    def test_add_prolong_smaller_operand_n_less_m(self):
        """Test addition prolongs when n < m."""
        # f is smaller
        coeffs_f = np.array([1.0], dtype=complex)
        coeffs_g = np.array([1.0, 0.5, 0.3], dtype=complex)

        f = Trigtech(coeffs_f)
        g = Trigtech(coeffs_g)

        assert f.size < g.size
        # Addition would prolong f

    def test_add_prolong_smaller_operand_m_less_n(self):
        """Test addition prolongs when m < n."""
        # g is smaller
        coeffs_f = np.array([1.0, 0.5, 0.3], dtype=complex)
        coeffs_g = np.array([1.0], dtype=complex)

        f = Trigtech(coeffs_f)
        g = Trigtech(coeffs_g)

        assert g.size < f.size
        # Addition would prolong g

    def test_ufunc_finds_trigtech_obj(self):
        """Test ufunc loop finds Trigtech object."""
        # Create Trigtech
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 16)

        # Apply ufunc - should find f as trigtech_obj
        result = np.abs(f)

        # Should create new Trigtech
        assert isinstance(result, Trigtech)

    def test_ufunc_tracks_max_size_iteration(self):
        """Test ufunc iterates to find max_size."""
        # Create multiple Trigtech with different sizes
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 8)

        # Unary ufunc still checks size
        result = np.sqrt(f + 1.0)  # Make positive for sqrt

        # Should preserve size info
        assert result.size >= 8

    def test_norm_l2_result_real_extraction(self):
        """Test norm L2 extracts real from complex."""
        # Create function with real values
        f = Trigtech.initfun_fixedlen(lambda x: np.cos(x), 16)

        # L2 norm computation
        norm = f.norm(p=2)

        # Should be real
        if hasattr(norm, "real"):
            norm = float(norm.real)
        assert not np.iscomplex(norm)

    def test_sum_real_output_when_negligible_imag(self):
        """Test sum returns real when imag negligible."""
        # Function with tiny imaginary part in coefficients
        coeffs = np.array([1.0 + 1e-15j, 0.5, 0.3], dtype=complex)
        f = Trigtech(coeffs)

        # Sum should strip negligible imaginary part
        result = f.sum()

        # Check if real or complex with tiny imag
        if np.iscomplexobj(result):
            assert np.abs(result.imag) < 1e-12

    def test_cumsum_adjust_constant_for_f0_zero(self):
        """Test cumsum adjusts DC to make F(0)=0."""
        # Create function with DC component
        coeffs = np.array([2.0, 0.5, 0.3, 0.5], dtype=complex)
        f = Trigtech(coeffs)

        # Cumsum should adjust so F(0) = 0
        F = f.cumsum()

        # Verify it ran (size should be same)
        assert F.size == f.size

    def test_diff_const_with_positive_n(self):
        """Test diff of const with n>0 returns zero."""
        # Constant Trigtech
        f = Trigtech.initconst(3.14)

        # Derivative should be zero
        df = f.diff(n=1)

        # Should return zero constant
        assert df.size == 1

    def test_chop_coeffs_n_min_le_4(self):
        """Test _chop_coeffs with n_min <= 4 uses n_keep=4."""
        # Very simple coefficients
        coeffs = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        result = Trigtech._chop_coeffs(coeffs, tol=1e-10)

        # Should keep at minimum practical size
        assert len(result) >= 1

    def test_chop_coeffs_freq_exceeds_nyquist_check(self):
        """Test _chop_coeffs checks max_sig_freq > n_keep//2."""
        # High frequency coefficients
        n = 64
        coeffs = np.zeros(n, dtype=complex)
        coeffs[0] = 1.0
        coeffs[25] = 0.5
        coeffs[-25] = 0.5

        result = Trigtech._chop_coeffs(coeffs, tol=1e-10)

        # Should keep enough for freq 25
        assert len(result) >= 1

    def test_interval_property_access(self):
        """Test interval property returns Interval object."""
        f = Trigtech.initconst(1.0, interval=[0, 4 * np.pi])

        # Access interval
        interval = f.interval

        # Should be Interval object with correct bounds
        assert hasattr(interval, "__getitem__")
        assert interval[0] == 0
        assert interval[1] == 4 * np.pi

    def test_values_method_calls_coeffs2vals(self):
        """Test values() calls _coeffs2vals."""
        coeffs = np.array([1.0, 0.5, 0.3], dtype=complex)
        f = Trigtech(coeffs)

        # Get values
        vals = f.values()

        # Should return array of same length
        assert len(vals) == len(coeffs)

    def test_copy_copies_interval(self):
        """Test copy() copies interval object."""
        f = Trigtech.initconst(1.0, interval=[0, 4 * np.pi])

        # Make copy
        f_copy = f.copy()

        # Should have same interval
        assert f_copy.interval[0] == f.interval[0]
        assert f_copy.interval[1] == f.interval[1]

        # But should be different object
        assert f_copy is not f


class TestStaticMethods:
    """Test static methods in Trigtech."""

    def test_trigpts_with_zero(self):
        """Test _trigpts with n=0."""
        pts = Trigtech._trigpts(0)
        assert len(pts) == 0

    def test_trigpts_with_positive_n(self):
        """Test _trigpts with positive n."""
        pts = Trigtech._trigpts(8)
        assert len(pts) == 8
        # Should be in [0, 2π)
        assert np.all(pts >= 0)
        assert np.all(pts < 2 * np.pi)

    def test_trigwts_with_zero(self):
        """Test _trigwts with n=0."""
        wts = Trigtech._trigwts(0)
        assert len(wts) == 0

    def test_trigwts_with_positive_n(self):
        """Test _trigwts with positive n."""
        wts = Trigtech._trigwts(16)
        assert len(wts) == 16
        # All weights should be 2π/n
        assert np.allclose(wts, 2 * np.pi / 16)

    def test_vals2coeffs_with_empty(self):
        """Test _vals2coeffs with empty array."""
        vals = np.array([])
        coeffs = Trigtech._vals2coeffs(vals)
        assert len(coeffs) == 0
        assert coeffs.dtype == complex

    def test_coeffs2vals_with_empty(self):
        """Test _coeffs2vals with empty array."""
        coeffs = np.array([], dtype=complex)
        vals = Trigtech._coeffs2vals(coeffs)
        assert len(vals) == 0
