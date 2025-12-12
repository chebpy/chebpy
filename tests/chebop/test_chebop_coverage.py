"""Tests for chebop.py coverage - targeting untested code paths.

All tests use the chebop API function (lowercase) from chebpy.
"""

import numpy as np
import pytest

from chebpy import chebfun, chebop


class TestChebopRepr:
    """Test Chebop string representations."""

    def test_repr_basic(self):
        """Test __repr__ method."""
        N = chebop([0, 1])
        r = repr(N)
        assert "Chebop" in r
        assert "domain" in r

    def test_repr_with_operator(self):
        """Test __repr__ with operator defined."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        r = repr(N)
        assert "defined" in r

    def test_repr_with_bcs(self):
        """Test __repr__ with BCs."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 1
        r = repr(N)
        assert "lbc=True" in r
        assert "rbc=True" in r


class TestIVPDetection:
    """Test IVP vs BVP detection."""

    def test_standard_bvp_not_ivp(self):
        """Test that standard BVP is not detected as IVP."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 0
        assert not N._is_ivp()

    def test_callable_bc_not_ivp(self):
        """Test that callable BCs are not IVP."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = lambda u: u
        N.rbc = 0
        assert not N._is_ivp()

    def test_left_ivp(self):
        """Test left IVP detection."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff()
        N.lbc = [1.0]  # Initial value only at left
        assert N._is_ivp()

    def test_right_ivp(self):
        """Test right (final value) IVP detection."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff()
        N.rbc = [1.0]  # Final value only at right
        assert N._is_ivp()


class TestChebopConstructorVariants:
    """Test various chebop constructor patterns via the API."""

    def test_constructor_op_domain(self):
        """Test chebop(op, domain) style."""
        op = lambda u: u.diff(2)
        N = chebop(op, [0, 1])
        assert N.op is not None
        a, b = N.domain.support
        assert a == 0 and b == 1

    def test_constructor_domain_op_kwarg(self):
        """Test chebop(domain, op=func) style."""
        N = chebop([0, 1], op=lambda u: u.diff(2))
        assert N.op is not None
        a, b = N.domain.support
        assert a == 0 and b == 1

    def test_constructor_two_scalars(self):
        """Test chebop(a, b) style for domain."""
        N = chebop(0, 1)
        a, b = N.domain.support
        assert a == 0 and b == 1


class TestLinearOperatorMethods:
    """Test linear operator analysis methods."""

    def test_diff_order_detection(self):
        """Test differential order detection."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u
        N.lbc = 0
        N.rbc = 0
        N.analyze_operator()
        assert N._diff_order == 2

    def test_linearity_detection_linear(self):
        """Test linear operator is detected as linear."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u
        N.lbc = 0
        N.rbc = 0
        N.analyze_operator()
        assert N._is_linear

    def test_linearity_detection_nonlinear(self):
        """Test nonlinear operator is detected as nonlinear."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u**2
        N.lbc = 0
        N.rbc = 0
        N.analyze_operator()
        assert not N._is_linear


class TestSystemDetection:
    """Test system vs scalar operator detection."""

    def test_scalar_operator(self):
        """Test scalar operator detection."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u
        N.lbc = 0
        N.rbc = 0
        N.analyze_operator()
        assert not N._is_system

    def test_system_operator(self):
        """Test system operator detection."""
        N = chebop([0, 1])
        N.op = lambda u, v: [u.diff() - v, v.diff() + u]
        N.lbc = [0, 0]
        N.rbc = [0, 0]
        N.analyze_operator()
        assert N._is_system
        assert N._num_equations == 2
        assert N._num_variables == 2


class TestSolveMethods:
    """Test solve method variants using chebop API."""

    def test_solve_simple_bvp(self):
        """Test solving simple BVP."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 1
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # u'' = 0, u(0) = 0, u(1) = 1 => u = x
        x_test = np.linspace(0, 1, 50)
        error = np.max(np.abs(u(x_test) - x_test))
        assert error < 1e-10

    def test_solve_with_nonzero_rhs(self):
        """Test solving BVP with non-zero RHS."""
        N = chebop([0, np.pi])
        N.op = lambda u: u.diff(2)
        N.lbc = 0
        N.rbc = 0
        N.rhs = chebfun(lambda x: -np.sin(x), [0, np.pi])

        u = N.solve()

        # u'' = -sin(x), u(0) = u(π) = 0 => u = sin(x)
        x_test = np.linspace(0.1, np.pi - 0.1, 50)
        u_exact = np.sin(x_test)
        error = np.max(np.abs(u(x_test) - u_exact))
        assert error < 1e-10


class TestEigenvalueProblems:
    """Test eigenvalue methods using chebop API."""

    def test_eigs_simple(self):
        """Test simple eigenvalue problem."""
        N = chebop([0, np.pi])
        N.op = lambda u: -u.diff(2)
        N.lbc = 0
        N.rbc = 0

        evals, efuns = N.eigs(k=3)

        # -u'' = λu, u(0) = u(π) = 0 => λ_n = n^2
        expected = np.array([1.0, 4.0, 9.0])
        for i in range(3):
            rel_err = abs(evals[i] - expected[i]) / expected[i]
            assert rel_err < 1e-10, f"Eigenvalue {i}: {evals[i]} vs {expected[i]}"


class TestVariableCoefficients:
    """Test operators with variable coefficients."""

    def test_variable_coeff_simple(self):
        """Test operator with variable coefficient using chebfun."""
        N = chebop([0, 1])
        x = chebfun(lambda x: x, [0, 1])
        N.op = lambda u: u.diff(2) + x * u
        N.lbc = 0
        N.rbc = 0
        N.rhs = chebfun(lambda t: np.sin(np.pi * t) * (np.pi**2 + t), [0, 1])

        u = N.solve()

        # Check BCs
        assert abs(u(0.0)) < 1e-10
        assert abs(u(1.0)) < 1e-10


class TestPeriodicBVPs:
    """Test periodic boundary conditions."""

    def test_periodic_second_order(self):
        """Test second order ODE with periodic BCs."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        # RHS with zero mean for well-posedness
        N.rhs = chebfun(lambda x: -4 * np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Check periodicity
        assert abs(u(0.0) - u(2 * np.pi)) < 1e-10


class TestNeumannBCs:
    """Test Neumann boundary conditions."""

    def test_neumann_left(self):
        """Test Neumann BC at left endpoint."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = [None, 0]  # u'(0) = 0
        N.rbc = 0  # u(1) = 0
        N.rhs = chebfun(lambda x: 1 + 0 * x, [0, 1])

        u = N.solve()

        # Check Neumann BC
        uprime = u.diff()
        assert abs(uprime(0.0)) < 1e-10


class TestIVPSolving:
    """Test initial value problem solving."""

    def test_ivp_first_order(self):
        """Test first order IVP: u' = u, u(0) = 1 => u = e^x."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff() - u
        N.lbc = 1.0  # u(0) = 1 (scalar BC)
        # No rbc - this makes it an IVP with solution u = e^x
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # Check initial condition
        assert abs(u(0.0) - 1.0) < 1e-8

    def test_ivp_second_order(self):
        """Test second order IVP: u'' = -u, u(0)=0, u'(0)=1 => u = sin(x)."""
        N = chebop([0, np.pi])
        N.op = lambda u: u.diff(2) + u
        N.lbc = [0.0, 1.0]  # u(0)=0, u'(0)=1

        u = N.solve()

        # Check solution: u = sin(x)
        x_test = np.linspace(0.1, np.pi - 0.1, 20)
        error = np.max(np.abs(u(x_test) - np.sin(x_test)))
        assert error < 1e-6


class TestBCConditionTypes:
    """Test various boundary condition types."""

    def test_callable_lbc_neumann(self):
        """Test callable left boundary condition (Neumann)."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = lambda u: u.diff()  # u'(0) = 0
        N.rbc = 1  # u(1) = 1
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # Check Neumann BC at left
        uprime = u.diff()
        assert abs(uprime(0.0)) < 1e-8
        # Check Dirichlet at right
        assert abs(u(1.0) - 1.0) < 1e-8

    def test_callable_rbc_neumann(self):
        """Test callable right boundary condition (Neumann)."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 1  # u(0) = 1
        N.rbc = lambda u: u.diff()  # u'(1) = 0
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # u'' = 0, u(0) = 1, u'(1) = 0 => u = 1 (constant)
        assert abs(u(0.0) - 1.0) < 1e-8
        uprime = u.diff()
        assert abs(uprime(1.0)) < 1e-8

    def test_neumann_rbc(self):
        """Test Neumann BC at right endpoint."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.lbc = 1  # u(0) = 1
        N.rbc = [None, 0]  # u'(1) = 0
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # Check BCs
        assert abs(u(0.0) - 1.0) < 1e-10
        uprime = u.diff()
        assert abs(uprime(1.0)) < 1e-10


class TestChainedDerivatives:
    """Test operators with chained derivatives."""

    def test_chained_diff(self):
        """Test operator using chained diff calls."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff().diff()  # Same as u.diff(2)
        N.lbc = 0
        N.rbc = 1
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # u'' = 0, u(0) = 0, u(1) = 1 => u = x
        x_test = np.linspace(0, 1, 20)
        error = np.max(np.abs(u(x_test) - x_test))
        assert error < 1e-10


class TestHigherOrderODEs:
    """Test higher order differential equations."""

    def test_fourth_order_beam(self):
        """Test fourth order ODE (beam equation)."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(4)
        # Clamped-clamped BCs
        N.lbc = [0, 0]  # u(0) = 0, u'(0) = 0
        N.rbc = [0, 0]  # u(1) = 0, u'(1) = 0
        N.rhs = chebfun(lambda x: 1 + 0 * x, [0, 1])

        u = N.solve()

        # Check BCs
        assert abs(u(0.0)) < 1e-10
        assert abs(u(1.0)) < 1e-10
        uprime = u.diff()
        assert abs(uprime(0.0)) < 1e-10
        assert abs(uprime(1.0)) < 1e-10


class TestFromDict:
    """Test from_dict class method."""

    def test_from_dict_basic(self):
        """Test creating chebop from specification dict."""
        from chebpy.chebop import Chebop

        spec = {
            "domain": [0, 1],
            "op": lambda u: u.diff(2),
            "lbc": 0,
            "rbc": 1,
        }
        N = Chebop.from_dict(spec)
        assert N.op is not None
        a, b = N.domain.support
        assert a == 0 and b == 1


class TestMaxnorm:
    """Test maxnorm helper."""

    def test_maxnorm_none(self):
        """Test maxnorm with None returns inf."""
        from chebpy.chebop import Chebop

        result = Chebop._maxnorm(None)
        assert result == np.inf


class TestNormalizeBC:
    """Test BC normalization."""

    def test_normalize_none(self):
        """Test normalize_bc with None."""
        from chebpy.chebop import Chebop

        count, vals = Chebop._normalize_bc(None)
        assert count == 0
        assert vals == []

    def test_normalize_scalar(self):
        """Test normalize_bc with scalar."""
        from chebpy.chebop import Chebop

        count, vals = Chebop._normalize_bc(1.5)
        assert count == 1
        assert vals == [1.5]

    def test_normalize_list(self):
        """Test normalize_bc with list."""
        from chebpy.chebop import Chebop

        count, vals = Chebop._normalize_bc([1, 2, 3])
        assert count == 3
        assert vals == [1, 2, 3]


class TestOperatorWithX:
    """Test operators that use x variable."""

    def test_op_with_x_arg(self):
        """Test operator with explicit x argument."""
        N = chebop([0, 1])
        N.op = lambda x, u: u.diff(2) + x * u  # Variable coefficient
        N.lbc = 0
        N.rbc = 0
        N.rhs = chebfun(lambda t: np.sin(np.pi * t), [0, 1])

        N.analyze_operator()
        assert N._diff_order == 2


class TestThirdOrderODE:
    """Test third order differential equations."""

    def test_third_order_bvp(self):
        """Test third order BVP."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(3)
        N.lbc = [0, 0]  # u(0) = 0, u'(0) = 0
        N.rbc = [1]  # u(1) = 1
        N.rhs = chebfun(lambda x: 0 * x, [0, 1])

        u = N.solve()

        # Check BCs
        assert abs(u(0.0)) < 1e-10
        assert abs(u(1.0) - 1.0) < 1e-10


class TestDomainKwarg:
    """Test domain passed as keyword argument."""

    def test_domain_kwarg(self):
        """Test chebop with domain as kwarg."""
        N = chebop(domain=[0, 2])
        a, b = N.domain.support
        assert a == 0 and b == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
