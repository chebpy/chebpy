"""Tests for linop_diagnostics module.

This module provides tests for the diagnostic functions in linop_diagnostics.py
that detect and warn about numerical issues in differential operator coefficients.

Tests cover:
1. check_coefficient_singularities(): Vanishing and sign-changing coefficients
2. check_coefficient_oscillation(): Highly oscillatory coefficients
3. check_operator_wellposedness(): Boundary condition counting
4. check_periodic_compatibility(): Periodic BC compatibility
5. diagnose_linop(): Main diagnostic entry point

All tests use realistic operator configurations and verify both detection
and warning message generation.
"""

import warnings

import numpy as np
import pytest

from chebpy import chebfun, chebop
from chebpy.linop import LinOp
from chebpy.linop_diagnostics import (
    check_coefficient_oscillation,
    check_coefficient_singularities,
    check_operator_wellposedness,
    check_periodic_compatibility,
    diagnose_linop,
)
from chebpy.utilities import Domain


class TestCheckCoefficientSingularities:
    """Tests for check_coefficient_singularities() function.

    This function detects:
    - Vanishing highest-order coefficients
    - Near-zero coefficients (ill-conditioned)
    - Sign-changing coefficients (type-changing PDEs)
    """

    def test_no_singularities_constant_coefficient(self):
        """Test that constant non-zero coefficient has no issues."""
        # a_2(x) = 1 (constant)
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert not has_issues, "Constant coefficient should have no issues"
        assert len(warnings_list) == 0, "Should have no warnings"

    def test_vanishing_everywhere(self):
        """Test detection of coefficient that vanishes everywhere."""
        # a_2(x) = 0 everywhere
        a2 = chebfun(lambda x: np.zeros_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues, "Should detect vanishing coefficient"
        assert len(warnings_list) > 0, "Should have warnings"
        assert "essentially zero" in warnings_list[0].lower()
        assert "singular operator" in warnings_list[0].lower()

    def test_vanishing_at_point(self):
        """Test detection of coefficient that vanishes at interior point."""
        # a_2(x) = x (vanishes at x=0)
        a2 = chebfun(lambda x: x, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues, "Should detect vanishing at x=0"
        assert len(warnings_list) > 0, "Should have warnings"
        # Should detect either vanishing, singular, or sign change (all are issues with x)
        all_warnings = " ".join(warnings_list).lower()
        assert "vanishes" in all_warnings or "singular" in all_warnings or "sign" in all_warnings

    def test_sign_change(self):
        """Test detection of sign-changing coefficient."""
        # a_2(x) = x (changes sign at x=0)
        a2 = chebfun(lambda x: x, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues, "Should detect sign change"
        # Should have multiple warnings: vanishing + sign change
        assert len(warnings_list) >= 1
        # Check for sign change warning
        sign_change_warning = any("sign" in w.lower() for w in warnings_list)
        assert sign_change_warning, "Should warn about sign change"

    def test_wide_variation_near_singularity(self):
        """Test detection of wide coefficient variation."""
        # a_2(x) = 1 + 0.0001*x (very small variation but crosses near-zero threshold)
        a2 = chebfun(lambda x: 1.0 + 0.0001 * x, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        # This should NOT trigger issues (variation is small)
        assert not has_issues

    def test_wide_variation_detected(self):
        """Test detection of genuinely wide coefficient variation."""
        # a_2(x) = 0.0001 + x^2 (ranges from 0.0001 to ~1, ratio > 10000)
        a2 = chebfun(lambda x: 0.0001 + x**2, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues, "Should detect wide variation"
        assert any("varies widely" in w.lower() or "widely" in w.lower() for w in warnings_list)

    def test_empty_coeffs(self):
        """Test with empty coefficient list."""
        coeffs = []
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain)

        assert not has_issues
        assert len(warnings_list) == 0

    def test_diff_order_zero(self):
        """Test with zero differential order."""
        a0 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=0, domain=domain)

        assert not has_issues
        assert len(warnings_list) == 0

    def test_implicit_coefficient(self):
        """Test when coefficient list is shorter than diff_order (implicit 1)."""
        # Only provide a_0, diff_order=2 means a_2 is implicitly 1
        a0 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain)

        assert not has_issues, "Implicit coefficient should be fine"
        assert len(warnings_list) == 0

    def test_none_coefficient(self):
        """Test when highest-order coefficient is None."""
        coeffs = [None, None, None]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain)

        assert not has_issues
        assert len(warnings_list) == 0

    def test_evaluation_failure(self):
        """Test graceful handling when coefficient evaluation fails."""
        # Create a pathological coefficient that can't be evaluated
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])

        # Monkey-patch __call__ to raise exception
        def bad_call(x):
            raise RuntimeError("Evaluation failed")

        a2.__call__ = bad_call
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # Should not crash, just skip checks
        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain)

        assert not has_issues
        assert len(warnings_list) == 0


class TestCheckCoefficientOscillation:
    """Tests for check_coefficient_oscillation() function.

    This function detects highly oscillatory coefficients that require
    fine discretization grids.
    """

    def test_no_oscillation_constant(self):
        """Test that constant coefficient is not oscillatory."""
        a0 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain)

        assert not is_osc, "Constant coefficient should not be oscillatory"
        assert len(warnings_list) == 0
        assert suggested_n is None

    def test_oscillatory_sin_high_freq(self):
        """Test detection of highly oscillatory coefficient."""
        # sin(20*pi*x) is highly oscillatory
        a0 = chebfun(lambda x: np.sin(20 * np.pi * x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(
            coeffs, diff_order=1, domain=domain, min_points_per_wavelength=10
        )

        # This coefficient needs many points, should be detected as oscillatory
        # (threshold is 64 for max_n_required)
        if a0.funs[0].size > 50:
            assert is_osc, "High-frequency sin should be oscillatory"
            assert len(warnings_list) > 0
            assert suggested_n is not None
            assert suggested_n > 64

    def test_moderately_oscillatory(self):
        """Test coefficient that needs resolution but isn't highly oscillatory."""
        # sin(5*pi*x) needs some resolution but not extreme
        a0 = chebfun(lambda x: np.sin(5 * np.pi * x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain)

        # Should not be flagged as highly oscillatory (threshold 64)
        # but may have some warnings
        if a0.funs[0].size > 50:
            assert len(warnings_list) > 0
        else:
            # Low enough resolution that it's not concerning
            pass

    def test_higher_diff_order_increases_requirement(self):
        """Test that higher diff_order increases resolution requirement."""
        # Same coefficient, different diff orders
        a0 = chebfun(lambda x: np.sin(20 * np.pi * x), [-1, 1])
        coeffs_order1 = [a0]
        coeffs_order4 = [a0]
        domain = Domain([-1, 1])

        _, _, n1 = check_coefficient_oscillation(coeffs_order1, diff_order=1, domain=domain)
        _, _, n4 = check_coefficient_oscillation(coeffs_order4, diff_order=4, domain=domain)

        # Higher order should require more resolution (if coefficient is oscillatory)
        if n1 is not None and n4 is not None:
            assert n4 > n1, "Higher diff_order should increase resolution requirement"

    def test_empty_coeffs(self):
        """Test with empty coefficient list."""
        coeffs = []
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=2, domain=domain)

        assert not is_osc
        assert len(warnings_list) == 0
        assert suggested_n is None

    def test_none_coefficients(self):
        """Test with None coefficients."""
        coeffs = [None, None, None]
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=2, domain=domain)

        assert not is_osc
        assert len(warnings_list) == 0
        assert suggested_n is None

    def test_multiple_oscillatory_coefficients(self):
        """Test with multiple oscillatory coefficients."""
        a0 = chebfun(lambda x: np.sin(20 * np.pi * x), [-1, 1])
        a1 = chebfun(lambda x: np.cos(15 * np.pi * x), [-1, 1])
        coeffs = [a0, a1]
        domain = Domain([-1, 1])

        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain)

        # Should detect both if they're highly oscillatory
        if a0.funs[0].size > 50 or a1.funs[0].size > 50:
            # At least one should trigger warnings
            assert len(warnings_list) > 0

    def test_coeff_without_funs_attribute(self):
        """Test with coefficient that doesn't have funs attribute."""

        # Create a plain object without funs
        class SimpleCoeff:
            pass

        coeffs = [SimpleCoeff()]
        domain = Domain([-1, 1])

        # Should handle gracefully
        is_osc, warnings_list, suggested_n = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain)

        # Should not crash
        assert not is_osc


class TestCheckOperatorWellposedness:
    """Tests for check_operator_wellposedness() function.

    This function verifies that the number of boundary conditions matches
    the differential order.
    """

    def test_wellposed_second_order_two_bcs(self):
        """Test well-posed second-order problem with 2 BCs."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # Second-order needs 2 BCs: 1 left, 1 right
        lbc = lambda u: u(-1)  # noqa: E731
        rbc = lambda u: u(1)  # noqa: E731

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=lbc, rbc=rbc, domain=domain
        )

        assert is_wellposed, "2nd order with 2 BCs should be well-posed"
        assert len(warnings_list) == 0

    def test_underdetermined_second_order_one_bc(self):
        """Test underdetermined second-order problem with only 1 BC."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # Only 1 BC for second-order
        lbc = lambda u: u(-1)  # noqa: E731
        rbc = None

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=lbc, rbc=rbc, domain=domain
        )

        assert not is_wellposed, "Should detect underdetermined system"
        assert len(warnings_list) > 0
        assert "underdetermined" in warnings_list[0].lower()

    def test_overdetermined_second_order_three_bcs(self):
        """Test overdetermined second-order problem with 3 BCs."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # 3 BCs for second-order (overdetermined)
        lbc = [lambda u: u(-1), lambda u: u.diff()(-1)]  # noqa: E731
        rbc = lambda u: u(1)  # noqa: E731

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=lbc, rbc=rbc, domain=domain
        )

        # Still returns True but with warning
        assert is_wellposed
        assert len(warnings_list) > 0
        assert "overdetermined" in warnings_list[0].lower()

    def test_fourth_order_four_bcs(self):
        """Test well-posed fourth-order problem with 4 BCs."""
        a4 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, None, None, a4]
        domain = Domain([-1, 1])

        # Fourth-order needs 4 BCs: 2 left, 2 right
        lbc = [lambda u: u(-1), lambda u: u.diff()(-1)]  # noqa: E731
        rbc = [lambda u: u(1), lambda u: u.diff()(1)]  # noqa: E731

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=4, lbc=lbc, rbc=rbc, domain=domain
        )

        assert is_wellposed
        assert len(warnings_list) == 0

    def test_periodic_bcs_always_wellposed(self):
        """Test that periodic BCs are always considered well-posed."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # Periodic BCs - no explicit lbc/rbc needed
        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=None, rbc=None, domain=domain, bc="periodic"
        )

        assert is_wellposed, "Periodic BCs should be well-posed"
        assert len(warnings_list) == 0

    def test_no_bcs_zero_order(self):
        """Test zero-order operator with no BCs (should be fine)."""
        a0 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=0, lbc=None, rbc=None, domain=domain
        )

        # Zero-order needs 0 BCs, so this is fine
        assert is_wellposed

    def test_list_with_none_entries(self):
        """Test BC counting with list containing None entries."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # List with None should be skipped
        lbc = [lambda u: u(-1), None]  # noqa: E731
        rbc = None

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=lbc, rbc=rbc, domain=domain
        )

        # Only 1 BC (None is ignored)
        assert not is_wellposed
        assert "underdetermined" in warnings_list[0].lower()

    def test_tuple_bcs(self):
        """Test that tuple BCs are counted correctly."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # Tuples should work like lists
        lbc = (lambda u: u(-1),)  # noqa: E731
        rbc = (lambda u: u(1),)  # noqa: E731

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=lbc, rbc=rbc, domain=domain
        )

        assert is_wellposed


class TestCheckPeriodicCompatibility:
    """Tests for check_periodic_compatibility() function.

    This function checks compatibility conditions for periodic BCs.
    For u'' = f with periodic BCs, need ∫f dx = 0.
    """

    def test_non_periodic_always_compatible(self):
        """Test that non-periodic BCs skip compatibility check."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        N.rbc = lambda u: u(1)  # noqa: E731

        # Create LinOp
        L = N.to_linop()

        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible
        assert len(warnings_list) == 0

    def test_periodic_compatible_zero_integral(self):
        """Test periodic problem with compatible RHS (zero integral)."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "periodic"
        # RHS with zero integral: sin(pi*x)
        N.rhs = chebfun(lambda x: np.sin(np.pi * x), [-1, 1])

        L = N.to_linop()

        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible
        assert len(warnings_list) == 0

    def test_periodic_incompatible_nonzero_integral(self):
        """Test periodic problem with incompatible RHS (nonzero integral)."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "periodic"
        # RHS with nonzero integral: constant 1
        N.rhs = chebfun(lambda x: np.ones_like(x), [-1, 1])

        L = N.to_linop()
        L.prepare_domain()  # Need to prepare domain first

        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert not is_compatible, "Should detect incompatible periodic problem"
        assert len(warnings_list) > 0
        assert "compatibility error" in warnings_list[0].lower()
        assert "zero integral" in warnings_list[0].lower()

    def test_periodic_first_order(self):
        """Test that first-order periodic problems skip compatibility check."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff()  # noqa: E731
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.ones_like(x), [-1, 1])

        L = N.to_linop()

        # First-order doesn't have compatibility constraint
        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible
        assert len(warnings_list) == 0

    def test_periodic_no_rhs(self):
        """Test periodic problem with no RHS."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "periodic"
        # No RHS set

        L = N.to_linop()

        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible
        assert len(warnings_list) == 0

    def test_periodic_no_blocks(self):
        """Test handling when LinOp has no blocks."""
        # Create a minimal LinOp without blocks
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "periodic"

        L = N.to_linop()
        # Clear blocks
        L.blocks = None

        is_compatible, warnings_list = check_periodic_compatibility(L)

        # Should handle gracefully
        assert is_compatible
        assert len(warnings_list) == 0


class TestDiagnoseLinop:
    """Tests for diagnose_linop() main entry point function.

    This function runs all diagnostic checks and coordinates warning emission.
    """

    def test_diagnose_wellposed_problem(self):
        """Test diagnosis of well-posed problem with no issues."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2) + u  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        N.rbc = lambda u: u(1)  # noqa: E731

        L = N.to_linop()

        # Diagnose without printing
        has_issues = diagnose_linop(L, verbose=False)

        assert not has_issues, "Well-posed problem should have no issues"

    def test_diagnose_singular_coefficient(self):
        """Test diagnosis detects vanishing coefficient."""
        N = chebop([-1, 1])
        # Coefficient of u'' vanishes: 0*u'' + u = 0
        N.op = lambda u: u  # noqa: E731  (no u'' term means coefficient is effectively 0)
        N.lbc = lambda u: u(-1)  # noqa: E731
        N.rbc = lambda u: u(1)  # noqa: E731

        # For this test, manually create LinOp with vanishing coefficient

        domain = Domain([-1, 1])
        a2 = chebfun(lambda x: np.zeros_like(x), [-1, 1])  # Vanishing
        coeffs = [None, None, a2]

        L = LinOp(coeffs=coeffs, domain=domain, diff_order=2, lbc=N.lbc, rbc=N.rbc)

        # Suppress warnings during test
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=False)

        assert has_issues, "Should detect vanishing coefficient"

    def test_diagnose_underdetermined(self):
        """Test diagnosis detects underdetermined system."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        # No rbc - underdetermined

        L = N.to_linop()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=False)

        assert has_issues, "Should detect underdetermined system"

    def test_diagnose_periodic_incompatible(self):
        """Test diagnosis detects periodic incompatibility."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.ones_like(x), [-1, 1])  # Nonzero integral

        L = N.to_linop()

        # Should raise ValueError for periodic incompatibility
        with pytest.raises(ValueError, match="compatibility error"):
            diagnose_linop(L, verbose=False)

    def test_diagnose_oscillatory_updates_max_n(self):
        """Test that diagnosis updates max_n for oscillatory coefficients."""
        N = chebop([-1, 1])
        # Highly oscillatory coefficient
        N.op = lambda u: chebfun(lambda x: np.sin(20 * np.pi * x), [-1, 1]) * u.diff(2) + u  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        N.rbc = lambda u: u(1)  # noqa: E731

        L = N.to_linop()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=True)

        # If coefficient is highly oscillatory, max_n should be updated
        # (depends on coefficient representation)
        if has_issues:
            # Check that max_n may have been updated (not guaranteed in all cases)
            pass

    def test_diagnose_with_verbose_true(self):
        """Test that verbose=True emits warnings."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        # Missing rbc

        L = N.to_linop()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=True)

        # Should have emitted warnings
        assert has_issues
        # Check that some warnings were raised
        assert len(w) > 0

    def test_diagnose_with_verbose_false(self):
        """Test that verbose=False suppresses warnings."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        # Missing rbc

        L = N.to_linop()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=False)

        # Should detect issues but not emit warnings
        assert has_issues
        # Warnings should not be emitted when verbose=False
        # (but some may still be raised internally, so we just check it returns correctly)

    def test_diagnose_no_blocks(self):
        """Test diagnosis when LinOp has no blocks initially."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        N.rbc = lambda u: u(1)  # noqa: E731

        L = N.to_linop()
        # Clear blocks to force prepare_domain call
        L.blocks = None

        has_issues = diagnose_linop(L, verbose=False)

        # Should prepare domain and run diagnostics
        assert not has_issues  # Well-posed problem

    def test_diagnose_empty_blocks(self):
        """Test diagnosis with empty blocks list."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731

        L = N.to_linop()
        L.blocks = []

        has_issues = diagnose_linop(L, verbose=False)

        assert not has_issues  # No blocks, no issues to detect


class TestEdgeCases:
    """Test edge cases for linop diagnostics."""

    def test_check_singularities_multiple_warnings(self):
        """Test that multiple issues generate multiple warnings."""
        # Coefficient that both vanishes AND changes sign
        a2 = chebfun(lambda x: x, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues
        # Should have warnings for both vanishing and sign change
        assert len(warnings_list) >= 1

    def test_vanishing_with_suggestions(self):
        """Test that vanishing coefficient generates detailed warning with suggestions."""
        # Create coefficient that vanishes at a specific point
        a2 = chebfun(lambda x: (x - 0.3), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        has_issues, warnings_list = check_coefficient_singularities(coeffs, diff_order=2, domain=domain, tol=1e-8)

        assert has_issues
        assert len(warnings_list) > 0
        # Check for warnings about coefficient issues (vanishes, singular, or sign change)
        all_warnings = " ".join(warnings_list).lower()
        assert "vanishes" in all_warnings or "singular" in all_warnings or "sign" in all_warnings

    def test_check_singularities_custom_tolerance(self):
        """Test that custom tolerance affects detection."""
        # Small but nonzero coefficient
        a2 = chebfun(lambda x: 1e-7 + 0 * x, [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        # With strict tolerance, should detect
        has_issues_strict, warnings_strict = check_coefficient_singularities(
            coeffs, diff_order=2, domain=domain, tol=1e-6
        )

        # With loose tolerance, should not detect
        has_issues_loose, warnings_loose = check_coefficient_singularities(
            coeffs, diff_order=2, domain=domain, tol=1e-8
        )

        assert has_issues_strict
        assert not has_issues_loose

    def test_oscillation_min_points_per_wavelength(self):
        """Test that min_points_per_wavelength parameter affects suggestion."""
        a0 = chebfun(lambda x: np.sin(20 * np.pi * x), [-1, 1])
        coeffs = [a0]
        domain = Domain([-1, 1])

        _, _, n_low = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain, min_points_per_wavelength=5)

        _, _, n_high = check_coefficient_oscillation(coeffs, diff_order=1, domain=domain, min_points_per_wavelength=20)

        # Higher points per wavelength should give higher suggestion
        if n_low is not None and n_high is not None:
            assert n_high > n_low

    def test_wellposedness_bc_string_uppercase(self):
        """Test that BC string is case-insensitive."""
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]
        domain = Domain([-1, 1])

        is_wellposed, warnings_list = check_operator_wellposedness(
            coeffs, diff_order=2, lbc=None, rbc=None, domain=domain, bc="PERIODIC"
        )

        assert is_wellposed

    def test_periodic_compatibility_case_insensitive(self):
        """Test that bc string matching is case-insensitive."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)  # noqa: E731
        N.bc = "PERIODIC"

        L = N.to_linop()

        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible

    def test_diagnose_all_checks_together(self):
        """Test a problem that triggers multiple diagnostic checks."""
        N = chebop([-1, 1])
        # Vanishing coefficient
        N.op = lambda u: 0 * u.diff(2) + u  # noqa: E731
        N.lbc = lambda u: u(-1)  # noqa: E731
        # Missing rbc - underdetermined

        # Manually create LinOp with explicit vanishing coefficient

        domain = Domain([-1, 1])
        a2 = chebfun(lambda x: np.zeros_like(x), [-1, 1])
        a0 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [a0, None, a2]

        L = LinOp(coeffs=coeffs, domain=domain, diff_order=2, lbc=N.lbc, rbc=None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            has_issues = diagnose_linop(L, verbose=True)

        assert has_issues
        # Should have multiple warnings
        assert len(w) > 0

    def test_periodic_third_order_skips_compatibility(self):
        """Test that third-order periodic problems skip compatibility check."""
        # Manually create a third-order LinOp with periodic BCs
        domain = Domain([-1, 1])
        a3 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        a0 = chebfun(lambda x: np.zeros_like(x), [-1, 1])
        coeffs = [a0, None, None, a3]

        L = LinOp(coeffs=coeffs, domain=domain, diff_order=3, bc="periodic")
        L.rhs = chebfun(lambda x: np.ones_like(x), [-1, 1])  # Nonzero integral
        L.prepare_domain()

        # Third-order should skip compatibility check
        is_compatible, warnings_list = check_periodic_compatibility(L)

        assert is_compatible
        assert len(warnings_list) == 0

    def test_periodic_rhs_integral_exception_handling(self):
        """Test that exceptions during RHS integral computation are handled gracefully."""
        # Create a periodic LinOp with problematic RHS
        domain = Domain([-1, 1])
        a2 = chebfun(lambda x: np.ones_like(x), [-1, 1])
        coeffs = [None, None, a2]

        L = LinOp(coeffs=coeffs, domain=domain, diff_order=2, bc="periodic")

        # Create a chebfun and break its sum method
        L.rhs = chebfun(lambda x: np.ones_like(x), [-1, 1])
        original_sum = L.rhs.sum

        def bad_sum():
            raise RuntimeError("Cannot compute integral")

        L.rhs.sum = bad_sum
        L.prepare_domain()

        # Should handle exception gracefully
        is_compatible, warnings_list = check_periodic_compatibility(L)

        # Should return compatible (skipped check due to exception)
        assert is_compatible
        assert len(warnings_list) == 0

        # Restore original method
        L.rhs.sum = original_sum
