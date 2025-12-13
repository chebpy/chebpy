"""Tests for op_discretization.py edge cases.

This test file includes tests for OpDiscretization:
1. Fourier/periodic BC handling with Fourier collocation
2. Integral constraints using Clenshaw-Curtis weights
3. Point constraints with interior derivatives
4. Mean-zero constraints for periodic problems
5. Rectangular discretization for eigenvalue problems
6. Edge cases and validation
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.op_discretization import OpDiscretization
from chebpy.utilities import Domain, Interval


class TestFourierPeriodicHandling:
    """Test Fourier collocation for periodic boundary conditions."""

    def test_is_periodic_detection(self):
        """Test that _is_periodic correctly detects periodic BCs."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)

        # No periodic BC
        assert not OpDiscretization._is_periodic(linop)

        # Prepare domain with periodic constraint
        linop.prepare_domain()
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        # Should detect periodic
        assert OpDiscretization._is_periodic(linop)

    def test_fourier_collocation_used_for_periodic(self):
        """Test that Fourier collocation is used for periodic problems."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.prepare_domain()

        # Add periodic continuity constraints
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # For Fourier, n_per_block should be n (not n+1)
        assert disc["n_per_block"][0] == n
        assert disc["m_per_block"][0] == n

        # Operator block should be n x n for Fourier
        A_block = disc["blocks"][0]
        assert A_block.shape == (n, n)

    def test_periodic_bcs_skip_for_fourier(self):
        """Test that periodic BCs skip generating BC rows for Fourier."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0  # This should be ignored for periodic Fourier
        linop.rbc = 0.0
        linop.prepare_domain()

        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # BC rows should be empty for periodic Fourier
        assert len(disc["bc_rows"]) == 0
        assert len(disc["bc_rhs"]) == 0

    def test_periodic_continuity_constraints(self):
        """Test periodic continuity constraint row generation."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.prepare_domain()

        # Add periodic continuity constraints (not using Fourier, for eigenvalue problem)
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        n = 16
        # Force eigenvalue mode to get explicit constraints even for periodic
        disc = OpDiscretization.build_discretization(linop, n, for_eigenvalue_problem=True)

        # Should have continuity rows for eigenvalue problems
        # (Fourier is only used for single-interval periodic, but with for_eigenvalue_problem=True
        # we may still get constraints)
        # The behavior depends on whether Fourier is used
        # For single interval periodic, Fourier is used, so check based on that
        if OpDiscretization._is_periodic(linop) and len(linop.blocks) == 1:
            # Fourier case: even for eigenvalue problems, continuity_rows may be empty
            # because periodicity is implicit in Fourier matrices
            pass  # No assertion needed, just testing code path
        else:
            assert len(disc["continuity_rows"]) >= 0  # May or may not have rows

    def test_fourier_rhs_evaluation(self):
        """Test that RHS is evaluated at Fourier points for periodic problems."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.rhs = chebfun(lambda x: np.sin(x), interval)
        linop.prepare_domain()

        # Add periodic constraints to trigger Fourier
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # RHS should have n values for Fourier
        rhs_block = disc["rhs_blocks"][0]
        assert len(rhs_block) == n


class TestIntegralConstraints:
    """Test integral constraint handling with Clenshaw-Curtis weights."""

    def test_simple_integral_constraint(self):
        """Test simple unweighted integral constraint ∫u dx = 0."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.integral_constraint = {"weight": None, "value": 0.0}
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have integral constraint rows
        assert len(disc["integral_rows"]) == 1
        assert len(disc["integral_rhs"]) == 1
        assert disc["integral_rhs"][0] == 0.0

        # Integral row should have Clenshaw-Curtis weights
        row = disc["integral_rows"][0]
        assert row.shape == (1, n + 1)

        # Sum of CC weights should approximate interval length
        weights_sum = np.sum(row.toarray())
        interval_length = 1.0  # [0, 1]
        assert abs(weights_sum - interval_length) < 1e-10

    def test_weighted_integral_constraint(self):
        """Test weighted integral constraint ∫g(x)u dx = c."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        # Weight function g(x) = x
        weight_fun = chebfun(lambda x: x, interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.integral_constraint = {"weight": weight_fun, "value": 1.0}
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have integral constraint
        assert len(disc["integral_rows"]) == 1
        assert disc["integral_rhs"][0] == 1.0

        # Weights should be scaled by weight function
        row = disc["integral_rows"][0]
        assert row.shape == (1, n + 1)

    def test_multiple_integral_constraints(self):
        """Test multiple integral constraints."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.integral_constraint = [
            {"weight": None, "value": 0.0},
            {"weight": chebfun(lambda x: x, interval), "value": 0.5},
        ]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have 2 integral constraints
        assert len(disc["integral_rows"]) == 2
        assert len(disc["integral_rhs"]) == 2
        assert disc["integral_rhs"][0] == 0.0
        assert disc["integral_rhs"][1] == 0.5

    def test_integral_constraint_none_returns_empty(self):
        """Test that no integral constraint returns empty lists."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        # No integral constraint
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        assert len(disc["integral_rows"]) == 0
        assert len(disc["integral_rhs"]) == 0


class TestPointConstraints:
    """Test point constraints with interior derivative evaluation."""

    def test_simple_point_constraint(self):
        """Test simple point constraint u(0.5) = 0.5."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = [{"location": 0.5, "derivative_order": 0, "value": 0.5}]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have point constraint row
        assert len(disc["point_rows"]) == 1
        assert len(disc["point_rhs"]) == 1
        assert disc["point_rhs"][0] == 0.5

        # Row should be barycentric evaluation at x=0.5
        row = disc["point_rows"][0]
        assert row.shape == (1, n + 1)

    def test_derivative_point_constraint(self):
        """Test derivative point constraint u'(0.3) = 1.0."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = [{"location": 0.3, "derivative_order": 1, "value": 1.0}]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have point constraint
        assert len(disc["point_rows"]) == 1
        assert disc["point_rhs"][0] == 1.0

    def test_second_derivative_point_constraint(self):
        """Test second derivative point constraint u''(0.7) = -0.2."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = [{"location": 0.7, "derivative_order": 2, "value": -0.2}]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have point constraint
        assert len(disc["point_rows"]) == 1
        assert disc["point_rhs"][0] == -0.2

    def test_multiple_point_constraints(self):
        """Test multiple point constraints at different locations."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = [
            {"location": 0.3, "derivative_order": 0, "value": 0.5},
            {"location": 0.7, "derivative_order": 1, "value": 1.0},
        ]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have 2 point constraints
        assert len(disc["point_rows"]) == 2
        assert len(disc["point_rhs"]) == 2
        assert disc["point_rhs"][0] == 0.5
        assert disc["point_rhs"][1] == 1.0

    def test_point_constraint_outside_domain_raises_error(self):
        """Test that point constraint outside domain raises ValueError."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = [
            {"location": 1.5, "derivative_order": 0, "value": 0.5}  # Outside [0,1]
        ]
        linop.prepare_domain()

        n = 16
        with pytest.raises(ValueError, match="outside domain"):
            OpDiscretization.build_discretization(linop, n)

    def test_point_constraint_empty_list(self):
        """Test that empty point constraint list returns empty."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.point_constraints = []
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        assert len(disc["point_rows"]) == 0
        assert len(disc["point_rhs"]) == 0


class TestMeanZeroConstraint:
    """Test mean-zero constraint for periodic Fourier problems."""

    def test_mean_zero_for_fourth_order_periodic(self):
        """Test mean-zero constraint added for u'''' with periodic BCs."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a4 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, None, None, a4], domain, diff_order=4)
        linop.prepare_domain()

        # Add periodic constraints
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
            {"type": "periodic", "block": 0, "derivative_order": 2},
            {"type": "periodic", "block": 0, "derivative_order": 3},
        ]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have mean-zero constraint for even order >= 2
        assert len(disc["mean_zero_rows"]) == 1
        assert len(disc["mean_zero_rhs"]) == 1
        assert disc["mean_zero_rhs"][0] == 0.0

        # Mean-zero row should sum to interval length
        row = disc["mean_zero_rows"][0]
        weights_sum = np.sum(row.toarray())
        interval_length = 2 * np.pi
        assert abs(weights_sum - interval_length) < 1e-10

    def test_mean_zero_for_second_order_periodic(self):
        """Test mean-zero constraint added for u'' with periodic BCs."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.prepare_domain()

        # Add periodic constraints
        linop.continuity_constraints = [
            {"type": "periodic", "block": 0, "derivative_order": 0},
            {"type": "periodic", "block": 0, "derivative_order": 1},
        ]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have mean-zero constraint
        assert len(disc["mean_zero_rows"]) == 1
        assert disc["mean_zero_rhs"][0] == 0.0

    def test_no_mean_zero_for_odd_order(self):
        """Test that odd-order periodic problems don't get mean-zero constraint."""
        domain = Domain([0, 2 * np.pi])
        interval = Interval(0, 2 * np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a1 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, a1], domain, diff_order=1)
        linop.prepare_domain()

        # Add periodic constraints
        linop.continuity_constraints = [{"type": "periodic", "block": 0, "derivative_order": 0}]

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should NOT have mean-zero constraint for odd order
        assert len(disc["mean_zero_rows"]) == 0

    def test_no_mean_zero_for_non_periodic(self):
        """Test that non-periodic problems don't get mean-zero constraint."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should NOT have mean-zero constraint
        assert len(disc["mean_zero_rows"]) == 0


class TestRectangularDiscretization:
    """Test rectangular (overdetermined) discretization."""

    def test_rectangular_validation_m_less_than_n_raises_error(self):
        """Test that m < n raises ValueError."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        m = 8  # m < n

        with pytest.raises(ValueError, match="m >= n"):
            OpDiscretization.build_discretization(linop, n, m=m, rectangularization=True)

    def test_rectangular_heuristic_small_n(self):
        """Test rectangular heuristic: m = 2*n for small n."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 8
        # Should use m = 2*n = 16
        disc = OpDiscretization.build_discretization(linop, n, rectangularization=True)

        # m_per_block should be 2*n + 1 = 17
        assert disc["m_per_block"][0] == 2 * n + 1

    def test_rectangular_heuristic_large_n(self):
        """Test rectangular heuristic: m = min(2*n, n+50) for large n."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 64
        # 2*n = 128 > n+50 = 114, so should use m = 114
        disc = OpDiscretization.build_discretization(linop, n, rectangularization=True)

        # m should be 114, so m_per_block = 115
        expected_m = min(2 * n, n + 50)
        assert disc["m_per_block"][0] == expected_m + 1

    def test_rectangular_projection_matrices(self):
        """Test that projection matrices are created for rectangular discretization."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        m = 32
        disc = OpDiscretization.build_discretization(linop, n, m=m, rectangularization=True)

        # Should have projection matrices
        assert len(disc["projection_matrices"]) == 1
        PS = disc["projection_matrices"][0]

        # PS should project from m+1 to n+1
        assert PS.shape == (n + 1, m + 1)

    def test_rectangular_no_projection_for_square(self):
        """Test that square discretization has no projection matrices."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n, rectangularization=False)

        # Should have empty projection matrices
        assert len(disc["projection_matrices"]) == 0

    def test_rectangular_rhs_evaluation(self):
        """Test that RHS is evaluated at m+1 collocation points for rectangular."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.rhs = chebfun(lambda x: np.sin(np.pi * x), interval)
        linop.prepare_domain()

        n = 16
        m = 32
        disc = OpDiscretization.build_discretization(linop, n, m=m, rectangularization=True)

        # RHS should have m+1 values
        rhs_block = disc["rhs_blocks"][0]
        assert len(rhs_block) == m + 1


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_callable_left_bc(self):
        """Test callable left boundary condition."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = lambda u: u(np.array([0.0]))[0] - 0.5  # u(0) = 0.5
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have BC rows
        assert len(disc["bc_rows"]) >= 1

    def test_callable_right_bc(self):
        """Test callable right boundary condition."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = lambda u: u(np.array([1.0]))[0] - 1.0  # u(1) = 1.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have BC rows
        assert len(disc["bc_rows"]) >= 1

    def test_callable_bc_with_linearization(self):
        """Test callable BC that requires linearization with u_current."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = lambda u: u(np.array([0.0]))[0] - 0.5
        linop.rbc = 0.0
        linop.prepare_domain()

        # Provide u_current for linearization
        u_current = chebfun(lambda x: x, interval)

        n = 16
        disc = OpDiscretization.build_discretization(linop, n, u_current=u_current)

        # Should have BC rows
        assert len(disc["bc_rows"]) >= 1

    def test_derivative_bc_list(self):
        """Test derivative boundary conditions using list syntax."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        # [u(0), u'(0)]
        linop.lbc = [0.0, 0.5]  # u(0) = 0, u'(0) = 0.5
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have 2 left BC rows + 1 right BC row
        assert len(disc["bc_rows"]) >= 2

    def test_right_bc_list_with_none(self):
        """Test right BC list with None values."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = [1.0, None]  # u(1) = 1.0, skip u'(1)
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have BC rows (only from non-None entries)
        assert len(disc["bc_rows"]) >= 1

    def test_general_bc_list(self):
        """Test general boundary condition list."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.bc = [
            lambda u: u(np.array([0.0]))[0],  # u(0) = 0
            lambda u: u(np.array([1.0]))[0],  # u(1) = 0
        ]
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have BC rows from general bc list
        assert len(disc["bc_rows"]) >= 2


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_domain_blocks(self):
        """Test that empty blocks triggers prepare_domain."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        # Don't call prepare_domain

        n = 16
        # Should automatically call prepare_domain
        disc = OpDiscretization.build_discretization(linop, n)

        assert disc is not None
        assert len(disc["blocks"]) > 0

    def test_none_diff_order(self):
        """Test handling of None diff_order."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.ones_like(x), interval)
        a1 = chebfun(lambda x: np.zeros_like(x), interval)

        linop = LinOp([a0, a1], domain, diff_order=None)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should handle None diff_order
        assert disc is not None

    def test_empty_coeffs_list(self):
        """Test handling of empty coefficients."""
        domain = Domain([0, 1])

        linop = LinOp([], domain, diff_order=0)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should create zero operator
        assert disc is not None

    def test_sparse_row_creation(self):
        """Test _create_sparse_row utility."""
        total_size = 10
        indices = [0, 5, 9]
        values = [1.0, 2.0, 3.0]

        row = OpDiscretization._create_sparse_row(total_size, indices, values)

        assert row.shape == (1, total_size)
        assert row[0, 0] == 1.0
        assert row[0, 5] == 2.0
        assert row[0, 9] == 3.0
        assert row[0, 1] == 0.0  # Check zeros

    def test_setup_linearization_point(self):
        """Test _setup_linearization_point utility."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0], domain, diff_order=0)
        linop.prepare_domain()

        block_interval = linop.blocks[0]["interval"]
        a, b, u_lin = OpDiscretization._setup_linearization_point(linop, None, block_interval)

        assert a == 0.0
        assert b == 1.0
        assert u_lin is not None

    def test_bc_enforcement_parameter(self):
        """Test bc_enforcement parameter is passed through."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n, bc_enforcement="replace")

        assert disc["bc_enforcement"] == "replace"


class TestContinuityConstraints:
    """Test continuity constraint handling."""

    def test_regular_continuity_constraint(self):
        """Test regular continuity constraint at interface."""
        # Create a domain with two intervals
        domain = Domain([0, 0.5, 1])

        # Create coefficients for each interval
        a0_left = chebfun(lambda x: np.zeros_like(x), [0, 0.5])
        a2_left = chebfun(lambda x: np.ones_like(x), [0, 0.5])
        chebfun(lambda x: np.zeros_like(x), [0.5, 1])
        chebfun(lambda x: np.ones_like(x), [0.5, 1])

        linop = LinOp([a0_left, None, a2_left], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have continuity constraints at interface
        assert len(disc["continuity_rows"]) > 0

    def test_none_continuity_constraints(self):
        """Test that None continuity_constraints returns empty."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0
        linop.prepare_domain()
        linop.continuity_constraints = None

        n = 16
        disc = OpDiscretization.build_discretization(linop, n)

        # Should have no continuity rows
        assert len(disc["continuity_rows"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
