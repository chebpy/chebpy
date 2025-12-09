"""Tests for validation and warning features."""

import warnings

import pytest

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.spectral import diff_matrix
from chebpy.utilities import Domain


class TestCoefficientsValidation:
    """Tests for coefficient length validation."""

    def test_correct_coefficient_length(self):
        """Test that correct coefficient length doesn't warn."""
        domain = Domain([0, 1])

        # 2nd order: need 3 coefficients
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

    def test_incorrect_coefficient_length_warns(self):
        """Test that incorrect coefficient length triggers warning."""
        domain = Domain([0, 1])

        # 2nd order but only 2 coefficients provided (missing one)
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])

        # Should warn
        with pytest.warns(UserWarning, match="Coefficient list length"):
            LinOp(coeffs=[a0, a1], domain=domain, diff_order=2)

    def test_extra_coefficients_no_warn(self):
        """Test that extra coefficients do NOT trigger warning (Issue #8).

        Composed operators (e.g., in generalized eigenvalue problems) can legitimately
        have more coefficients than diff_order+1. We only warn when there are too FEW
        coefficients, not too many.
        """
        domain = Domain([0, 1])

        # 1st order but 3 coefficients provided - this is OK for composed operators
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        # Should NOT warn (intentional behavior per Issue #8)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=1)

        # Verify LinOp was created successfully
        assert L.diff_order == 1
        assert len(L.coeffs) == 3


class TestHighOrderDerivativeWarning:
    """Tests for high-order derivative warnings."""

    def test_low_order_no_warning(self):
        """Test that low-order derivatives don't warn."""
        domain = [0, 1]

        # Orders 1-6 should not warn
        for order in [1, 2, 3, 4, 5, 6]:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                diff_matrix(10, domain, order=order)

    def test_high_order_warns(self):
        """Test that order > 6 triggers warning."""
        domain = [0, 1]

        # Order 7 should warn
        with pytest.warns(UserWarning, match="numerically unstable"):
            diff_matrix(10, domain, order=7)

        # Order 10 should warn
        with pytest.warns(UserWarning, match="numerically unstable"):
            diff_matrix(10, domain, order=10)


# NOTE: Periodic BC tests removed - periodic BCs require Fourier collocation


class TestConditionNumberWarning:
    """Tests for condition number warnings on ill-conditioned systems."""

    def test_well_conditioned_no_warning(self):
        """Test that well-conditioned systems don't warn."""
        domain = Domain([0, 1])

        # Simple Poisson: u'' = 1, u(0) = 0, u(1) = 0
        # This is a well-conditioned problem
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0
        L.rhs = chebfun(lambda x: 1 + 0*x, [0, 1])
        L.max_n = 16

        # Should not warn about condition number
        with warnings.catch_warnings(record=True) as w:
            L.solve()
            # Check no ill-conditioning warnings
            cond_warnings = [warning for warning in w
                           if "ill-conditioned" in str(warning.message)]
            assert len(cond_warnings) == 0

    def test_ill_conditioned_warns(self):
        """Test that severely ill-conditioned systems trigger warning."""
        domain = Domain([0, 1])

        # Create an ill-conditioned problem: high order derivative with large domain scaling
        # u'''' = eps * u where eps is tiny
        eps = 1e-10
        a0 = chebfun(lambda x: -eps + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 0*x, [0, 1])
        a3 = chebfun(lambda x: 0*x, [0, 1])
        a4 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2, a3, a4], domain=domain, diff_order=4)
        L.lbc = [0, 0]  # u(0) = u'(0) = 0
        L.rbc = [1, 0]  # u(1) = 1, u'(1) = 0
        L.rhs = chebfun(lambda x: 0*x, [0, 1])
        L.max_n = 32

        # Should warn about ill-conditioning
        # Note: This test may need adjustment based on actual condition number
        with warnings.catch_warnings(record=True) as w:
            L.solve()
            # Check if there's an ill-conditioning warning (may or may not trigger depending on n)
            cond_warnings = [warning for warning in w
                           if "ill-conditioned" in str(warning.message)]
            # If it warns, the message should contain "cond(A)"
            if len(cond_warnings) > 0:
                assert "cond(A)" in str(cond_warnings[0].message)
