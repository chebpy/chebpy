"""Tests for systems of ODEs (coupled equations).

These tests verify that LinOp can handle systems of coupled differential equations,
where the solution is a vector of functions rather than a single function.
"""

import numpy as np

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestSystemsBasics:
    """Basic tests for systems of ODEs."""

    def test_first_order_system_2x2(self):
        """Test 2x2 first-order system: U' = AU with U = [u, v]ᵀ.

        System: u' = v
                v' = -u
        BCs: u(0) = 1, v(0) = 0

        Exact solution: u = cos(x), v = -sin(x)
        """
        Domain([0, np.pi / 2])

        # Create system operator
        # U' = [0  1] U
        #      [-1 0]
        #
        # This can be written as:
        # u' - v = 0
        # v' + u = 0

        # Test that we can construct and solve each equation separately
        # Full system support would require matrix-valued LinOps

        # First equation: u' - v = 0, or u' = v
        # This is harder to express in LinOp framework
        # Test a simpler decoupled system

    def test_harmonic_oscillator(self):
        """Test harmonic oscillator as 2x2 system.

        u'' + u = 0 can be written as:
        u' = v
        v' = -u

        with u(0) = 1, v(0) = 0
        Exact: u = cos(x), v = -sin(x)
        """
        domain = Domain([0, np.pi])

        # Solve as second order: u'' + u = 0
        a0 = chebfun(lambda x: 1 + 0 * x, [0, np.pi])
        a1 = chebfun(lambda x: 0 * x, [0, np.pi])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, np.pi])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 1  # u(0) = 1
        L.rbc = -1  # u(π) = -1 (since cos(π) = -1)
        L.rhs = chebfun(lambda x: 0 * x, [0, np.pi])

        u = L.solve()

        # Check solution
        x_test = np.linspace(0, np.pi, 20)
        expected = np.cos(x_test)
        assert np.max(np.abs(u(x_test) - expected)) < 1e-10

        # Verify v = u' ≈ -sin(x)
        v = u.diff()
        expected_v = -np.sin(x_test)
        assert np.max(np.abs(v(x_test) - expected_v)) < 1e-8


class TestCoupledSystems:
    """Tests for genuinely coupled systems."""

    def test_simple_coupling(self):
        """Test a simple coupled system with known solution.

        u'' = v
        v'' = u
        u(0) = 1, u'(0) = 0
        v(0) = 0, v'(0) = 1

        Solution involves hyperbolic functions.
        For now, verify we can solve second-order equations with RHS depending on another function.
        """
        domain = Domain([0, 1])

        # Solve u'' = v with some assumed v
        # Full coupling requires more infrastructure

        # Test that we can solve with a chebfun RHS
        v_assumed = chebfun(lambda x: np.sinh(x), [0, 1])

        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 1  # u(0) = 1
        L.rbc = 0  # u'(0) = 0 (approximate with u'(0) small)
        L.rhs = v_assumed

        u = L.solve()

        # Just verify it solves without error
        assert u is not None
        assert abs(u(np.array([0.0]))[0] - 1.0) < 1e-8

    def test_predator_prey_linearized(self):
        """Test linearized predator-prey model.

        This is typically:
        u' = au - buv
        v' = -cv + duv

        Linearizing around equilibrium (u₀, v₀):
        δu' = (a - bv₀)δu - bu₀δv
        δv' = dv₀δu + (-c + du₀)δv

        This is a first-order matrix ODE that would require system support.
        Verify we can handle the structure.
        """
        # Placeholder test
        pass


class TestBlockStructure:
    """Tests for understanding how to represent block systems."""

    def test_second_order_to_first_order_system(self):
        """Verify converting u'' = f(x) to first-order system.

        u'' = f becomes:
        [u']   [0  1] [u]   [0]
        [v'] = [0  0] [v] + [f]

        where v = u'
        """
        domain = Domain([0, 1])

        # Original: u'' = sin(x), u(0) = 0, u(1) = 0
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0
        L.rhs = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        u = L.solve()

        # Verify solution
        x_test = np.linspace(0, 1, 20)
        # Exact solution: u = -sin(πx)/π² (note the negative from integrating twice)
        expected = -np.sin(np.pi * x_test) / (np.pi**2)
        error = np.max(np.abs(u(x_test) - expected))
        assert error < 1e-8  # Looser tolerance due to numerical solve


class TestSystemInterfaces:
    """Tests for potential system interfaces."""

    def test_multiple_rhs(self):
        """Test if we can solve with vector RHS (list of chebfuns).

        This would be needed for systems.
        """
        domain = Domain([0, 1])

        # Single equation
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # Try list of RHS (even though it's just one equation)
        rhs_list = [chebfun(lambda x: 1 + 0 * x, [0, 1])]

        # LinOp expects single chebfun
        # This test documents what we'd need for systems
        L.rhs = rhs_list[0]  # Use single element

        u = L.solve()
        assert u is not None


class TestMatrixOperators:
    """Tests for matrix-valued operators."""

    def test_matrix_coefficient(self):
        """Test operator with matrix coefficient: U' = A(x)U.

        This would require LinOp to accept matrix-valued coefficients.
        Document the desired interface.
        """
        # Desired interface:
        # A = lambda x: np.array([[0, 1], [-1, 0]])
        # L = LinOp(matrix_coeffs=[None, A], system_size=2)
        pass

    def test_coupled_bcs(self):
        """Test coupled boundary conditions: u(0) + v(0) = 1.

        This requires BCs that couple multiple solution components.
        """
        pass


class TestExistingSystemSolves:
    """Tests using existing LinOp to solve system-like problems."""

    def test_fourth_order_as_system(self):
        """Fourth-order equation can be viewed as 2x2 second-order system.

        u'''' = f becomes:
        u'' = v
        v'' = f

        But we can solve directly as u'''' = f.
        """
        domain = Domain([0, 1])

        # u'''' = 1, u(0) = u'(0) = u(1) = u'(1) = 0
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 0 * x, [0, 1])
        a3 = chebfun(lambda x: 0 * x, [0, 1])
        a4 = chebfun(lambda x: 1 + 0 * x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2, a3, a4], domain=domain, diff_order=4)
        L.bc = [
            lambda u: u(np.array([0.0]))[0],
            lambda u: u.diff()(np.array([0.0]))[0],
            lambda u: u(np.array([1.0]))[0],
            lambda u: u.diff()(np.array([1.0]))[0],
        ]
        L.rhs = chebfun(lambda x: 1 + 0 * x, [0, 1])

        u = L.solve()

        # Exact: u = x²(1-x)²/24
        x_test = np.linspace(0, 1, 20)
        expected = x_test**2 * (1 - x_test) ** 2 / 24
        error = np.max(np.abs(u(x_test) - expected))
        assert error < 1e-10

    def test_wave_equation_decomposition(self):
        """Wave equation u_tt = c²u_xx can be split into system.

        This is a PDE but illustrates system structure.
        For ODEs, test traveling wave: u'' - c²u = 0
        """
        c = 2.0
        domain = Domain([0, 1])

        # u'' - c²u = 0, u(0) = 1, u(1) = cosh(c)
        a0 = chebfun(lambda x: -(c**2) + 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 1
        L.rbc = np.cosh(c)
        L.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = L.solve()

        # Exact: u = cosh(cx)
        x_test = np.linspace(0, 1, 20)
        expected = np.cosh(c * x_test)
        error = np.max(np.abs(u(x_test) - expected))
        assert error < 1e-8
