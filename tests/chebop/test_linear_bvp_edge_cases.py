"""Tests for linear BVP solving with Chebop.

This test suite ensures chebop works correctly for linear problems.
Tests cover:
- Various orders (0th through 4th derivatives)
- Dirichlet, Neumann, and mixed boundary conditions
- Different domains
- Variable coefficients
- Operator combinations
"""

import numpy as np

from chebpy import chebfun, chebop


class TestBasicLinearBVPs:
    """Test basic linear boundary value problems."""

    def test_poisson_dirichlet(self):
        """Test u'' = f with Dirichlet BCs: u(0)=u(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 0
        N.rhs = chebfun(lambda x: -(np.pi**2) * np.sin(np.pi * x), [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

    def test_poisson_nonzero_dirichlet(self):
        """Test u'' = f with non-zero Dirichlet BCs."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 1
        N.rbc = 2
        # Solution: u'' = -2, u(0)=1, u(1)=2
        # General: u = -x² + C1*x + C2
        # u(0) = C2 = 1, u(1) = -1 + C1 + 1 = C1 = 2
        # Therefore: u = -x² + 2x + 1
        N.rhs = chebfun(lambda x: -2 + 0 * x, [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: -(x**2) + 2 * x + 1, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

        # Check BCs
        assert abs(u(np.array([0.0]))[0] - 1) < 1e-12
        assert abs(u(np.array([1.0]))[0] - 2) < 1e-12

    def test_helmholtz(self):
        """Test u'' - u = f with Dirichlet BCs."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) - u
        N.lbc = 0
        N.rbc = 0
        # For u = sin(πx): u'' - u = -π²sin(πx) - sin(πx)
        N.rhs = chebfun(lambda x: (-(np.pi**2) - 1) * np.sin(np.pi * x), [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-11

    def test_first_order_ode(self):
        """Test u' + u = f."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff() + u
        N.lbc = 1
        # Solution: u = e^(-x), u' + u = -e^(-x) + e^(-x) = 0
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.exp(-x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-9


class TestNeumannBoundaryConditions:
    """Test Neumann and mixed boundary conditions."""

    def test_neumann_left(self):
        """Test u'' = f with u'(0)=0, u(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = [None, 0]  # u'(0) = 0
        N.rbc = 0  # u(1) = 0
        N.rhs = chebfun(lambda x: 1 + 0 * x, [0, 1])

        u = N.solve()
        # Exact: u'' = 1, u'(0) = 0, u(1) = 0
        # u' = x + C1, u'(0) = C1 = 0
        # u = x²/2 + C2, u(1) = 1/2 + C2 = 0, C2 = -1/2
        # u = x²/2 - 1/2
        u_exact = chebfun(lambda x: x**2 / 2 - 1 / 2, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

        # Check BCs
        u_prime = u.diff()
        assert abs(u_prime(np.array([0.0]))[0]) < 1e-11
        assert abs(u(np.array([1.0]))[0]) < 1e-12

    def test_neumann_right(self):
        """Test u'' = f with u(0)=0, u'(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0  # u(0) = 0
        N.rbc = [None, 0]  # u'(1) = 0
        N.rhs = chebfun(lambda x: 1 + 0 * x, [0, 1])

        u = N.solve()
        # Exact: u'' = 1, u(0) = 0, u'(1) = 0
        # u' = x + C1, u'(1) = 1 + C1 = 0, C1 = -1
        # u = x²/2 - x + C2, u(0) = C2 = 0
        # u = x²/2 - x
        u_exact = chebfun(lambda x: x**2 / 2 - x, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

        # Check BCs
        u_prime = u.diff()
        assert abs(u(np.array([0.0]))[0]) < 1e-12
        assert abs(u_prime(np.array([1.0]))[0]) < 1e-11

    def test_neumann_both(self):
        """Test u'' = f with u'(0)=0, u'(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = [None, 0]  # u'(0) = 0
        N.rbc = [None, 0]  # u'(1) = 0
        # Use forcing that's compatible with Neumann BCs
        # u = cos(πx), u' = -πsin(πx), u'(0) = u'(1) = 0
        # u'' = -π²cos(πx)
        N.rhs = chebfun(lambda x: -(np.pi**2) * np.cos(np.pi * x), [0, 1])

        u = N.solve()
        # Note: solution is determined up to a constant
        # Check that u - u_exact is constant
        chebfun(lambda x: np.cos(np.pi * x), [0, 1])

        # Check derivative BCs
        u_prime = u.diff()
        assert abs(u_prime(np.array([0.0]))[0]) < 1e-10
        assert abs(u_prime(np.array([1.0]))[0]) < 1e-10

        # Check that solution satisfies the ODE
        residual = u.diff(2) - N.rhs
        x_test = np.linspace(0.1, 0.9, 50)
        res_norm = np.max(np.abs(residual(x_test)))
        assert res_norm < 1e-10

    def test_mixed_nonzero_neumann(self):
        """Test u'' = f with u'(0)=1, u(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = [None, 1]  # u'(0) = 1
        N.rbc = 0  # u(1) = 0
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()
        # Exact: u'' = 0, u'(0) = 1, u(1) = 0
        # u' = C1, u'(0) = C1 = 1
        # u = x + C2, u(1) = 1 + C2 = 0, C2 = -1
        # u = x - 1
        u_exact = chebfun(lambda x: x - 1, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

        # Check BCs
        u_prime = u.diff()
        assert abs(u_prime(np.array([0.0]))[0] - 1) < 1e-11
        assert abs(u(np.array([1.0]))[0]) < 1e-12


class TestHigherOrderOperators:
    """Test higher-order differential operators."""

    def test_third_order(self):
        """Test u''' = 0 with u(0)=0, u'(0)=1, u(1)=0."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(3)
        N.lbc = [0, 1]  # u(0) = 0, u'(0) = 1
        N.rbc = 0  # u(1) = 0
        # Solution: u''' = 0 means u = C1*x²/2 + C2*x + C3
        # u(0) = C3 = 0, u'(0) = C2 = 1, u(1) = C1/2 + 1 = 0 → C1 = -2
        # Therefore: u = -x² + x
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: -(x**2) + x, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

    def test_fourth_order(self):
        """Test u'''' = f (biharmonic) with clamped boundary conditions."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(4)
        N.lbc = [0, 0]  # u(0) = u'(0) = 0
        N.rbc = [0, 0]  # u(1) = u'(1) = 0
        # Solution: u = x²(1-x)²
        # u' = 2x(1-x)² - 2x²(1-x) = 2x(1-x)(1-2x)
        # u'' = 2(1-x)(1-2x) + 2x(-2x) + 2x(1-x)(-2) = 2(1-6x+6x²)
        # u''' = -12 + 24x
        # u'''' = 24
        N.rhs = chebfun(lambda x: 24 + 0 * x, [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: x**2 * (1 - x) ** 2, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-9


class TestVariableCoefficients:
    """Test operators with variable coefficients."""

    def test_variable_coefficient_second_order(self):
        """Test -u'' + xu = f (simpler variable coefficient)."""
        N = chebop([0, 1])
        # Use identity function for x
        x_fun = chebfun(lambda x: x, [0, 1])
        # -u'' + xu = f
        N.op = lambda u: -u.diff(2) + x_fun * u
        N.lbc = 0
        N.rbc = 0
        # Use u = sin(πx): u' = πcos(πx), u'' = -π²sin(πx)
        # -u'' + xu = π²sin(πx) + xsin(πx)
        N.rhs = chebfun(lambda x: (np.pi**2 + x) * np.sin(np.pi * x), [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-10

    def test_sturm_liouville(self):
        """Test -u'' - u' + xu = f (variable coefficient Sturm-Liouville)."""
        N = chebop([0, 1])
        # Variable coefficient operator
        x_fun = chebfun(lambda x: x, [0, 1])
        N.op = lambda u: -u.diff(2) - u.diff() + x_fun * u
        N.lbc = 0
        N.rbc = 0
        # Use u = sin(πx): u' = πcos(πx), u'' = -π²sin(πx)
        # -u'' - u' + xu = π²sin(πx) - πcos(πx) + xsin(πx)
        N.rhs = chebfun(
            lambda x: np.pi**2 * np.sin(np.pi * x) - np.pi * np.cos(np.pi * x) + x * np.sin(np.pi * x), [0, 1]
        )

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-9


class TestDifferentDomains:
    """Test BVPs on various domains."""

    def test_domain_minus_one_to_one(self):
        """Test on [-1, 1]."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 0
        # u = 1 - x², u'' = -2
        N.rhs = chebfun(lambda x: -2 + 0 * x, [-1, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: 1 - x**2, [-1, 1])

        x_test = np.linspace(-1, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12

    def test_domain_large(self):
        """Test on [0, 10]."""
        N = chebop([0, 10])
        N.op = lambda u: u.diff(2) + u
        N.lbc = 0
        N.rbc = 0
        # u = sin(x), u'' + u = -sin(x) + sin(x) = 0
        # But u(0) = u(10) = 0 requires sin(10) = 0, which is false
        # Use u = sin(πx/10)
        N.rhs = chebfun(lambda x: (-((np.pi / 10) ** 2) + 1) * np.sin(np.pi * x / 10), [0, 10])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x / 10), [0, 10])

        x_test = np.linspace(0, 10, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-10

    def test_domain_asymmetric(self):
        """Test on [-2, 3]."""
        N = chebop([-2, 3])
        N.op = lambda u: u.diff(2)
        N.lbc = 1
        N.rbc = 2
        # u = 1 + (x+2)/5 = 1 + x/5 + 2/5, u'' = 0
        N.rhs = chebfun(lambda x: 0 * x, [-2, 3])

        u = N.solve()
        u_exact = chebfun(lambda x: 1 + (x + 2) / 5, [-2, 3])

        x_test = np.linspace(-2, 3, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-12


class TestOperatorCombinations:
    """Test combinations of operators."""

    def test_sum_of_operators(self):
        """Test (D² + I)u = f."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u
        N.lbc = 0
        N.rbc = 0
        # u = sin(πx), u'' + u = -π²sin(πx) + sin(πx)
        N.rhs = chebfun(lambda x: (1 - np.pi**2) * np.sin(np.pi * x), [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-11

    def test_scaled_operator(self):
        """Test αD²u = f."""
        alpha = 2.5
        N = chebop([0, 1])
        N.op = lambda u: alpha * u.diff(2)
        N.lbc = 0
        N.rbc = 0
        # u = sin(πx), αu'' = -απ²sin(πx)
        N.rhs = chebfun(lambda x: -alpha * np.pi**2 * np.sin(np.pi * x), [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-11


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_constant_solution(self):
        """Test problem with constant solution."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 5
        N.rbc = 5
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()
        u_exact = chebfun(lambda x: 5 + 0 * x, [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 2e-12

    def test_periodic_forcing(self):
        """Test with periodic forcing term (non-resonant)."""
        N = chebop([0, np.pi])
        N.op = lambda u: u.diff(2) + 4 * u
        N.lbc = 0
        N.rbc = 0
        # Use sin(3x) forcing to avoid resonance with sin(2x) eigenfunction
        # For u = Asin(3x): u'' + 4u = -9Asin(3x) + 4Asin(3x) = -5Asin(3x)
        # So if RHS = sin(3x), then A = -1/5
        N.rhs = chebfun(lambda x: np.sin(3 * x), [0, np.pi])

        u = N.solve()
        u_exact = chebfun(lambda x: -np.sin(3 * x) / 5, [0, np.pi])

        x_test = np.linspace(0, np.pi, 100)
        error = np.max(np.abs((u - u_exact)(x_test)))
        assert error < 1e-10
