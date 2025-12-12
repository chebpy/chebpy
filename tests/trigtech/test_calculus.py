"""Unit-tests for calculus operations on Trigtech objects.

This module tests differentiation, integration, and related calculus operations
on Trigtech objects representing periodic functions.
"""

import numpy as np

from chebpy.trigtech import Trigtech


class TestDifferentiation:
    """Tests for differentiation of periodic functions."""

    def test_diff_sin(self):
        """Test differentiation of sin(kx) gives k*cos(kx)."""
        k = 3
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        df = f.diff()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = k * np.cos(k * x_test)
        error = np.max(np.abs(df(x_test) - expected))
        assert error < 1e-12, f"Differentiation error: {error}"

    def test_diff_cos(self):
        """Test differentiation of cos(kx) gives -k*sin(kx)."""
        k = 2
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))
        df = f.diff()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = -k * np.sin(k * x_test)
        error = np.max(np.abs(df(x_test) - expected))
        assert error < 1e-12, f"Differentiation error: {error}"

    def test_diff_sum(self):
        """Test differentiation of sum of periodic functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x) + np.cos(3 * x))
        df = f.diff()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = 2 * np.cos(2 * x_test) - 3 * np.sin(3 * x_test)
        error = np.max(np.abs(df(x_test) - expected))
        assert error < 1e-11, f"Differentiation error: {error}"

    def test_diff_constant(self):
        """Test differentiation of constant gives zero."""
        c = 2.71828
        f = Trigtech.initconst(c)
        df = f.diff()

        # Derivative of constant should be very small (essentially zero)
        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        vals = df(x_test)
        assert np.max(np.abs(vals)) < 1e-12

    def test_diff_order_2(self):
        """Test second derivative: d²/dx² sin(kx) = -k²sin(kx)."""
        k = 2
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        d2f = f.diff(2)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = -(k**2) * np.sin(k * x_test)
        error = np.max(np.abs(d2f(x_test) - expected))
        assert error < 1e-11, f"Second derivative error: {error}"

    def test_diff_order_3(self):
        """Test third derivative of cos(kx)."""
        k = 3
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))
        d3f = f.diff(3)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        # d/dx cos(kx) = -k*sin(kx)
        # d²/dx² cos(kx) = -k²*cos(kx)
        # d³/dx³ cos(kx) = k³*sin(kx)
        expected = k**3 * np.sin(k * x_test)
        error = np.max(np.abs(d3f(x_test) - expected))
        assert error < 1e-10, f"Third derivative error: {error}"

    def test_diff_preserves_periodicity(self):
        """Test that differentiation preserves periodicity."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x) + np.cos(3 * x))
        df = f.diff()

        val_0 = df(np.array([0.0]))[0]
        val_2pi = df(np.array([2 * np.pi]))[0]
        assert np.abs(val_0 - val_2pi) < 1e-12, "Derivative should be periodic"

    def test_diff_chain_rule(self):
        """Test differentiation of composed functions."""
        # f(x) = sin(2x), g(x) = exp(f(x))
        # g'(x) = 2*cos(2x)*exp(sin(2x))
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x))
        g = np.exp(f)
        dg = g.diff()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = 2 * np.cos(2 * x_test) * np.exp(np.sin(2 * x_test))
        error = np.max(np.abs(dg(x_test) - expected))
        assert error < 1e-10, f"Chain rule differentiation error: {error}"


class TestIntegration:
    """Tests for integration of periodic functions."""

    def test_cumsum_sin(self):
        """Test indefinite integral of sin(kx) gives -cos(kx)/k + C."""
        k = 2
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        F = f.cumsum()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        # Integral with F(0) = 0
        expected = (-np.cos(k * x_test) + np.cos(0)) / k
        error = np.max(np.abs(F(x_test) - expected))
        assert error < 1e-11, f"Indefinite integral error: {error}"

    def test_cumsum_cos(self):
        """Test indefinite integral of cos(kx) gives sin(kx)/k + C."""
        k = 3
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))
        F = f.cumsum()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        # Integral with F(0) = 0
        expected = np.sin(k * x_test) / k
        error = np.max(np.abs(F(x_test) - expected))
        assert error < 1e-11, f"Indefinite integral error: {error}"

    def test_sum_periodic(self):
        """Test definite integral over full period [0, 2π].

        For periodic functions, the integral of sin(kx) and cos(kx) over
        a full period should be zero (for k ≠ 0).
        """
        # sin(2x) integrated from 0 to 2π should be 0
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x))
        integral = f.sum()
        assert np.abs(integral) < 1e-12, f"Integral of sin(2x) should be 0, got {integral}"

        # cos(3x) integrated from 0 to 2π should be 0
        g = Trigtech.initfun_adaptive(lambda x: np.cos(3 * x))
        integral = g.sum()
        assert np.abs(integral) < 1e-12, f"Integral of cos(3x) should be 0, got {integral}"

    def test_sum_constant(self):
        """Test definite integral of constant over [0, 2π]."""
        c = 1.5
        f = Trigtech.initconst(c)
        integral = f.sum()
        expected = c * 2 * np.pi
        assert np.abs(integral - expected) < 1e-12

    def test_sum_mean(self):
        """Test that integral equals mean times period."""
        f = Trigtech.initfun_adaptive(lambda x: 2 + np.sin(x) + np.cos(2 * x))

        integral = f.sum()
        mean = f.mean()

        # Integral = mean * (2π)
        assert np.abs(integral - mean * 2 * np.pi) < 1e-11

    def test_diff_cumsum_inverse(self):
        """Test that diff and cumsum are (approximately) inverse operations."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(3 * x) + np.cos(2 * x))

        # Integrate then differentiate
        F = f.cumsum()
        f_back = F.diff()

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        error = np.max(np.abs(f(x_test) - f_back(x_test)))
        assert error < 1e-10, f"diff(cumsum(f)) != f, error = {error}"

    def test_fundamental_theorem(self):
        """Test fundamental theorem of calculus: ∫_a^b f'(x)dx = f(b) - f(a)."""
        # Start with F(x) = sin(2x)
        F = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x))
        f = F.diff()  # f = 2*cos(2x)

        # Integrate f over [0, 2π]
        integral = f.sum()

        # Should equal F(2π) - F(0) = sin(4π) - sin(0) = 0
        F_2pi = F(np.array([2 * np.pi]))[0]
        F_0 = F(np.array([0.0]))[0]
        expected = F_2pi - F_0

        assert np.abs(integral - expected) < 1e-11


class TestOrthogonality:
    """Tests for orthogonality properties of Fourier basis."""

    def test_sine_cosine_orthogonal(self):
        """Test that sin(kx) and cos(mx) are orthogonal over [0, 2π]."""
        k, m = 2, 3
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(m * x))

        # Inner product: ∫ sin(kx)*cos(mx) dx over [0, 2π]
        product = f * g
        inner_product = product.sum()

        assert np.abs(inner_product) < 1e-11, f"<sin({k}x), cos({m}x)> should be 0"

    def test_sine_orthogonal_different_k(self):
        """Test that sin(kx) and sin(mx) are orthogonal for k ≠ m."""
        k, m = 2, 3
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        g = Trigtech.initfun_adaptive(lambda x: np.sin(m * x))

        product = f * g
        inner_product = product.sum()

        assert np.abs(inner_product) < 1e-11, f"<sin({k}x), sin({m}x)> should be 0"

    def test_cosine_orthogonal_different_k(self):
        """Test that cos(kx) and cos(mx) are orthogonal for k ≠ m."""
        k, m = 1, 4
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(m * x))

        product = f * g
        inner_product = product.sum()

        assert np.abs(inner_product) < 1e-11, f"<cos({k}x), cos({m}x)> should be 0"


class TestNorms:
    """Tests for norms of periodic functions."""

    def test_norm_sin(self):
        """Test L2 norm of sin(kx) over [0, 2π]."""
        k = 2
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))

        # ||sin(kx)||₂² = ∫₀^{2π} sin²(kx) dx = π
        norm = f.norm()
        expected = np.sqrt(np.pi)

        assert np.abs(norm - expected) < 1e-11, f"Norm error: got {norm}, expected {expected}"

    def test_norm_cos(self):
        """Test L2 norm of cos(kx) over [0, 2π]."""
        k = 3
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))

        # ||cos(kx)||₂² = ∫₀^{2π} cos²(kx) dx = π
        norm = f.norm()
        expected = np.sqrt(np.pi)

        assert np.abs(norm - expected) < 1e-11, f"Norm error: got {norm}, expected {expected}"

    def test_norm_constant(self):
        """Test L2 norm of constant function."""
        c = 2.0
        f = Trigtech.initconst(c)

        # ||c||₂² = ∫₀^{2π} c² dx = c² * 2π
        norm = f.norm()
        expected = c * np.sqrt(2 * np.pi)

        assert np.abs(norm - expected) < 1e-11


class TestMean:
    """Tests for mean values of periodic functions."""

    def test_mean_sin(self):
        """Test that mean of sin(kx) over [0, 2π] is zero."""
        k = 3
        f = Trigtech.initfun_adaptive(lambda x: np.sin(k * x))
        mean = f.mean()
        assert np.abs(mean) < 1e-12, f"Mean of sin({k}x) should be 0, got {mean}"

    def test_mean_cos(self):
        """Test that mean of cos(kx) over [0, 2π] is zero (for k > 0)."""
        k = 2
        f = Trigtech.initfun_adaptive(lambda x: np.cos(k * x))
        mean = f.mean()
        assert np.abs(mean) < 1e-12, f"Mean of cos({k}x) should be 0, got {mean}"

    def test_mean_constant(self):
        """Test mean of constant function."""
        c = 3.14
        f = Trigtech.initconst(c)
        mean = f.mean()
        assert np.abs(mean - c) < 1e-12

    def test_mean_offset_trig(self):
        """Test mean of constant + trig function."""
        c = 2.5
        f = Trigtech.initfun_adaptive(lambda x: c + np.sin(2 * x) + np.cos(3 * x))
        mean = f.mean()
        # Mean should be c (trig terms average to zero)
        assert np.abs(mean - c) < 1e-11
