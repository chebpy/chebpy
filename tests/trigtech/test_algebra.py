"""Unit-tests for algebraic operations on Trigtech objects.

This module tests addition, subtraction, multiplication, division, and other
algebraic operations on Trigtech objects representing periodic functions.
"""

import numpy as np
import pytest

from chebpy.trigtech import Trigtech

# Ensure reproducibility
rng = np.random.default_rng(0)


class TestArithmetic:
    """Tests for basic arithmetic operations."""

    def test_addition(self):
        """Test addition of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(2 * x))
        h = f + g

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sin(x_test) + np.cos(2 * x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_subtraction(self):
        """Test subtraction of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(x))
        h = f - g

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sin(2 * x_test) - np.cos(x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_multiplication(self):
        """Test multiplication of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(x))
        h = f * g

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sin(x_test) * np.cos(x_test)
        # Product should equal sin(2x)/2
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_division(self):
        """Test division of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x) + 2)
        g = Trigtech.initfun_adaptive(lambda x: np.cos(x) + 3)
        h = f / g

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = (np.sin(x_test) + 2) / (np.cos(x_test) + 3)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_scalar_multiplication(self):
        """Test multiplication by a scalar."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(3 * x))
        c = 2.5
        h = c * f

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = c * np.sin(3 * x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_scalar_addition(self):
        """Test addition of a scalar."""
        f = Trigtech.initfun_adaptive(lambda x: np.cos(2 * x))
        c = 1.5
        h = f + c

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.cos(2 * x_test) + c
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_negation(self):
        """Test negation of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x) + np.cos(2 * x))
        h = -f

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = -(np.sin(x_test) + np.cos(2 * x_test))
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_power(self):
        """Test exponentiation of Trigtech objects."""
        f = Trigtech.initfun_adaptive(lambda x: 1 + 0.5 * np.cos(x))
        h = f**2

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = (1 + 0.5 * np.cos(x_test)) ** 2
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11


class TestTranscendental:
    """Tests for transcendental functions."""

    def test_exp(self):
        """Test exponential of Trigtech."""
        f = Trigtech.initfun_adaptive(lambda x: 0.1 * np.sin(x))
        h = np.exp(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.exp(0.1 * np.sin(x_test))
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_sin(self):
        """Test sine of Trigtech.

        Note: Creating a Trigtech from x (non-periodic) will generate a warning,
        but the final result sin(x) is periodic and should be accurate.
        """
        # Expect warning since x is not periodic on [0, 2π]
        with pytest.warns(UserWarning, match="did not converge"):
            f = Trigtech.initfun_adaptive(lambda x: x)
        h = np.sin(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sin(x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_cos(self):
        """Test cosine of Trigtech.

        Note: Creating a Trigtech from 2*x (non-periodic) will generate a warning,
        but the final result cos(2*x) is periodic and should be accurate.
        """
        # Expect warning since 2*x is not periodic on [0, 2π]
        with pytest.warns(UserWarning, match="did not converge"):
            f = Trigtech.initfun_adaptive(lambda x: 2 * x)
        h = np.cos(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.cos(2 * x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_sqrt(self):
        """Test square root of Trigtech."""
        f = Trigtech.initfun_adaptive(lambda x: 2 + np.cos(x))
        h = np.sqrt(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sqrt(2 + np.cos(x_test))
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_log(self):
        """Test natural logarithm of Trigtech."""
        f = Trigtech.initfun_adaptive(lambda x: 2 + np.sin(x))
        h = np.log(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.log(2 + np.sin(x_test))
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11


class TestComposition:
    """Tests for function composition."""

    def test_compose_with_trig(self):
        """Test composition with trigonometric functions."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        # Compose: sin(sin(x))
        h = np.sin(f)

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.sin(np.sin(x_test))
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11

    def test_compose_polynomial_like(self):
        """Test composition that creates higher harmonics."""
        f = Trigtech.initfun_adaptive(lambda x: np.cos(x))
        # (cos(x))^3 = (3cos(x) + cos(3x))/4
        h = f**3

        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = np.cos(x_test) ** 3
        assert np.max(np.abs(h(x_test) - expected)) < 1e-11


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_operations(self):
        """Test arithmetic with empty Trigtech."""
        f = Trigtech.initempty()
        g = Trigtech.initfun_adaptive(lambda x: np.sin(x))

        # Empty + non-empty should give non-empty
        h = f + g
        assert not h.isempty

    def test_const_operations(self):
        """Test operations involving constant Trigtech."""
        c = Trigtech.initconst(3.0)
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))

        h = c + f
        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        expected = 3.0 + np.sin(x_test)
        assert np.max(np.abs(h(x_test) - expected)) < 1e-12

    def test_zero_function(self):
        """Test operations with zero function."""
        zero = Trigtech.initconst(0.0)
        f = Trigtech.initfun_adaptive(lambda x: np.cos(2 * x))

        h = zero + f
        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        assert np.max(np.abs(h(x_test) - np.cos(2 * x_test))) < 1e-12


class TestPropertiesPreserved:
    """Tests that algebraic operations preserve periodicity."""

    def test_addition_preserves_periodicity(self):
        """Verify that sum of periodic functions is periodic."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(2 * x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(3 * x))
        h = f + g

        assert np.abs(h(np.array([0.0]))[0] - h(np.array([2 * np.pi]))[0]) < 1e-12

    def test_product_preserves_periodicity(self):
        """Verify that product of periodic functions is periodic."""
        f = Trigtech.initfun_adaptive(lambda x: np.sin(x))
        g = Trigtech.initfun_adaptive(lambda x: np.cos(2 * x))
        h = f * g

        assert np.abs(h(np.array([0.0]))[0] - h(np.array([2 * np.pi]))[0]) < 1e-12

    def test_composition_preserves_periodicity(self):
        """Verify that composition preserves periodicity."""
        f = Trigtech.initfun_adaptive(lambda x: 0.5 * np.sin(x))
        h = np.exp(f)

        assert np.abs(h(np.array([0.0]))[0] - h(np.array([2 * np.pi]))[0]) < 1e-12
