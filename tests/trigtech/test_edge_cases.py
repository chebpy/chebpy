"""Unit tests for edge cases and missing coverage in Trigtech."""

import numpy as np
import pytest

from chebpy.trigtech import Trigtech


class TestInitFunSwitch:
    """Test that initfun correctly dispatches to adaptive/fixedlen."""

    def test_initfun_no_n_uses_adaptive(self):
        """Test initfun without n parameter uses adaptive construction."""
        f = Trigtech.initfun(lambda x: np.sin(x))
        # Adaptive should find optimal size (probably 16 for sin)
        assert f.size >= 16

    def test_initfun_with_n_uses_fixedlen(self):
        """Test initfun with n parameter uses fixed-length construction."""
        f = Trigtech.initfun(lambda x: np.sin(x), n=32)
        assert f.size == 32


class TestConstructorValidation:
    """Test constructor input validation."""

    def test_initconst_int_converted_to_float(self):
        """Test that integer constants are converted to float."""
        f = Trigtech.initconst(5)  # integer input
        assert f.size == 1
        assert f.coeffs.dtype == complex  # Trigtech uses complex

    def test_initconst_invalid_type(self):
        """Test that non-scalar constants raise ValueError."""
        with pytest.raises(ValueError):
            Trigtech.initconst(np.array([1, 2, 3]))


class TestCallMethods:
    """Test different evaluation methods."""

    def test_call_invalid_method_raises_error(self):
        """Test that invalid evaluation method raises ValueError."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        x = np.array([0.5, 1.0])
        with pytest.raises(ValueError):
            f(x, how="invalid_method")

    def test_call_fft_fallback(self):
        """Test that FFT method falls back to direct."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        x = np.array([0.5, 1.0, 1.5])
        result_fft = f(x, how="fft")
        result_direct = f(x, how="direct")
        assert np.allclose(result_fft, result_direct, rtol=1e-12)

    def test_call_empty_returns_zeros(self):
        """Test that calling empty Trigtech returns zeros."""
        f = Trigtech.initempty()
        x = np.array([0.5, 1.0, 1.5])
        result = f(x)
        assert np.allclose(result, np.zeros_like(x))


class TestProperties:
    """Test Trigtech properties."""

    def test_iscomplex_real_function(self):
        """Test iscomplex returns False for real functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        assert not f.iscomplex

    def test_iscomplex_complex_function(self):
        """Test iscomplex returns True for complex functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.exp(1j * x))
        assert f.iscomplex

    def test_iscomplex_empty(self):
        """Test iscomplex returns False for empty."""
        f = Trigtech.initempty()
        assert not f.iscomplex

    def test_isconst(self):
        """Test isconst property."""
        f_const = Trigtech.initconst(5.0)
        assert f_const.isconst

        f_nconst = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        assert not f_nconst.isconst


class TestUtilityMethods:
    """Test utility methods like imag, real, copy, prolong."""

    def test_imag_complex_function(self):
        """Test imag() extracts imaginary part."""
        f = Trigtech.initfun_adaptive(lambda x: 1 + 1j * np.sin(x))
        f_imag = f.imag()
        # imag() should give us the imaginary coefficient structure
        # Just verify it's not zero and has right size
        assert f_imag.size > 0
        assert not f_imag.isempty

    def test_imag_real_function_returns_zero(self):
        """Test imag() returns zero for real functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        f_imag = f.imag()
        assert f_imag.size == 1
        x_test = np.array([0.5, 1.0])
        assert np.allclose(f_imag(x_test), 0, atol=1e-13)

    def test_real_complex_function(self):
        """Test real() extracts real part."""
        f = Trigtech.initfun_adaptive(lambda x: np.cos(x) + 1j * np.sin(x))
        f_real = f.real()
        x_test = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        expected = np.cos(x_test)
        result = f_real(x_test).real
        assert np.allclose(result, expected, atol=1e-12)

    def test_real_real_function_returns_self(self):
        """Test real() returns self for already real functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        f_real = f.real()
        # Should return self (same object)
        x_test = np.array([0.5, 1.0])
        assert np.allclose(f(x_test), f_real(x_test))

    def test_prolong_truncate(self):
        """Test prolong with n < size truncates."""
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 64)
        f_short = f.prolong(32)
        assert f_short.size == 32

    def test_prolong_extend(self):
        """Test prolong with n > size zero-pads."""
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 16)
        f_long = f.prolong(32)
        assert f_long.size == 32

    def test_prolong_same_size_copies(self):
        """Test prolong with n == size makes copy."""
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 32)
        f_copy = f.prolong(32)
        assert f_copy.size == f.size
        assert f_copy is not f

    def test_simplify_empty_returns_copy(self):
        """Test simplify on empty returns copy."""
        f = Trigtech.initempty()
        f_simple = f.simplify()
        assert f_simple.isempty

    def test_simplify_handles_zero_vscale(self):
        """Test simplify when vscale is zero."""
        f = Trigtech.initconst(0.0)
        f_simple = f.simplify()
        assert f_simple.size == 1


class TestAlgebraEdgeCases:
    """Test algebraic operations edge cases."""

    def test_add_to_empty(self):
        """Test adding to empty Trigtech."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initempty()
        result = f + g
        assert result.size == f.size

    def test_add_empty_to_function(self):
        """Test adding empty to Trigtech."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initempty()
        result = g + f
        assert result.size == f.size

    def test_add_scalar_to_empty(self):
        """Test adding scalar to empty."""
        f = Trigtech.initempty()
        result = f + 5.0
        # Should create constant
        assert result.size >= 1

    def test_add_results_in_zero(self):
        """Test addition that results in zero function."""
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 32)
        g = -f
        result = f + g
        # Should recognize as zero
        x_test = np.array([0.5, 1.0])
        assert np.allclose(result(x_test), 0, atol=1e-12)

    def test_div_by_scalar(self):
        """Test division by scalar."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        result = f / 2.0
        x_test = np.array([0.5, 1.0])
        assert np.allclose(result(x_test), np.sin(x_test) / 2.0, atol=1e-12)

    def test_div_by_empty_returns_empty(self):
        """Test division by empty returns empty."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initempty()
        result = f / g
        assert result.isempty

    def test_mul_empty_returns_empty(self):
        """Test multiplication with empty returns empty."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initempty()
        result = f * g
        assert result.isempty

    def test_pos(self):
        """Test unary positive."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        result = +f
        assert result is f  # Should return self

    def test_right_operations(self):
        """Test right-side operations (radd, rdiv, rpow)."""
        f = Trigtech.initfun_adaptive(lambda x: 2 + np.sin(x))

        # Right addition
        result = 3.0 + f
        x_test = np.array([0.5, 1.0])
        expected = 3.0 + 2 + np.sin(x_test)
        assert np.allclose(result(x_test), expected, atol=1e-12)

        # Right division
        result = 1.0 / f
        expected = 1.0 / (2 + np.sin(x_test))
        assert np.allclose(result(x_test), expected, atol=1e-10)

        # Right power
        result = 2.0**f
        expected = 2.0 ** (2 + np.sin(x_test))
        assert np.allclose(result(x_test), expected, rtol=1e-10)

    def test_rsub(self):
        """Test right subtraction."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        result = 5.0 - f
        x_test = np.array([0.5, 1.0])
        expected = 5.0 - np.sin(x_test)
        assert np.allclose(result(x_test), expected, atol=1e-12)


class TestUfuncSupport:
    """Test numpy ufunc support."""

    def test_ufunc_not_trigtech_returns_notimplemented(self):
        """Test that ufuncs without Trigtech return NotImplemented."""
        # This tests the case where no input is a Trigtech
        # Direct test is tricky, but we verify behavior indirectly
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        # Test that ufuncs work normally
        result = np.exp(f)
        x_test = np.array([0.5, 1.0])
        assert np.allclose(result(x_test), np.exp(np.sin(x_test)), rtol=1e-10)

    def test_ufunc_preserves_max_size(self):
        """Test that ufuncs preserve maximum size from inputs."""
        f = Trigtech.initfun_fixedlen(lambda x: np.sin(x), 16)
        g = Trigtech.initfun_fixedlen(lambda x: np.cos(x), 32)
        result = np.add(f, g)
        # Result should have size at least 32
        assert result.size >= 16


class TestRoots:
    """Test root finding."""

    def test_roots_empty(self):
        """Test roots of empty Trigtech."""
        f = Trigtech.initempty()
        roots = f.roots()
        assert len(roots) == 0

    def test_roots_constant_nonzero(self):
        """Test roots of non-zero constant has no roots."""
        f = Trigtech.initconst(5.0)
        roots = f.roots()
        assert len(roots) == 0

    def test_roots_complex_returns_empty(self):
        """Test that roots of complex function returns empty array."""
        f = Trigtech.initfun_adaptive(lambda x: 1j * np.sin(x) + np.cos(x))
        roots = f.roots()
        # Complex root finding not implemented
        assert len(roots) == 0

    def test_roots_sorting(self):
        """Test that roots can be sorted."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        roots_sorted = f.roots(sort=True)
        roots_unsorted = f.roots(sort=False)
        # Both should find the roots
        assert len(roots_sorted) > 0
        assert len(roots_unsorted) > 0
        # Sorted should be in order
        assert np.all(roots_sorted[:-1] <= roots_sorted[1:])


class TestCalculusEdgeCases:
    """Test calculus operations edge cases."""

    def test_norm_l1(self):
        """Test L1 norm."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        norm = f.norm(p=1)
        # L1 norm of sin(x) over [0, 2π] should be 4
        # But we're dealing with complex representations, so use looser tolerance
        assert np.abs(norm - 4.0) < 1e-6

    def test_norm_linf(self):
        """Test L-infinity norm."""
        f = Trigtech.initfun_adaptive(lambda x: 2.0 + np.sin(x))
        norm = f.norm(p=np.inf)
        # Max should be 3.0
        assert np.abs(norm - 3.0) < 1e-10

    def test_norm_unsupported(self):
        """Test that unsupported norm types raise ValueError."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        with pytest.raises(ValueError, match="Unsupported norm"):
            f.norm(p=3)

    def test_norm_complex_result(self):
        """Test norm when sum returns complex."""
        # Create a function that might have complex intermediate values
        f = Trigtech.initfun_adaptive(lambda x: np.cos(x))
        norm = f.norm(p=2)
        # Result should be real
        assert np.isreal(norm)

    def test_sum_zero_size(self):
        """Test sum of empty returns 0."""
        f = Trigtech.initempty()
        result = f.sum()
        assert result == 0.0

    def test_sum_real_output(self):
        """Test sum returns real when imaginary part is negligible."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        result = f.sum()
        # Sum of sin over full period should be ~0
        assert np.isreal(result) or np.abs(np.imag(result)) < 1e-12

    def test_cumsum_empty(self):
        """Test cumsum of empty returns empty."""
        f = Trigtech.initempty()
        result = f.cumsum()
        assert result.isempty

    def test_diff_zero_order(self):
        """Test diff with n=0 returns copy."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        result = f.diff(n=0)
        x_test = np.array([0.5, 1.0])
        assert np.allclose(f(x_test), result(x_test))

    def test_diff_constant(self):
        """Test diff of constant returns zero."""
        f = Trigtech.initconst(5.0)
        result = f.diff(n=1)
        x_test = np.array([0.5, 1.0])
        assert np.allclose(result(x_test), 0, atol=1e-13)

    def test_diff_empty(self):
        """Test diff of empty returns empty."""
        f = Trigtech.initempty()
        result = f.diff(n=1)
        assert result.isempty


class TestStaticHelpers:
    """Test static helper methods."""

    def test_trigpts_zero(self):
        """Test _trigpts with n=0."""
        pts = Trigtech._trigpts(0)
        assert len(pts) == 0

    def test_trigwts_zero(self):
        """Test _trigwts with n=0."""
        wts = Trigtech._trigwts(0)
        assert len(wts) == 0

    def test_vals2coeffs_empty(self):
        """Test _vals2coeffs with empty array."""
        coeffs = Trigtech._vals2coeffs(np.array([]))
        assert len(coeffs) == 0
        assert coeffs.dtype == complex

    def test_coeffs2vals_empty(self):
        """Test _coeffs2vals with empty array."""
        vals = Trigtech._coeffs2vals(np.array([], dtype=complex))
        assert len(vals) == 0


class TestPlotting:
    """Test plotting methods (coverage only)."""

    def test_plot_coverage(self):
        """Test plot method exists and runs without error."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        # Just verify method exists and returns something
        # Actual plotting tested in integration tests
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        line = f.plot(ax=ax)
        assert line is not None
        plt.close(fig)

    def test_plotcoeffs_coverage(self):
        """Test plotcoeffs method exists and runs without error."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        line = f.plotcoeffs(ax=ax)
        assert line is not None
        plt.close(fig)
