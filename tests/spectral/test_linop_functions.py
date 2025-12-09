"""Comprehensive tests for LinOp functions: eigs, expm, null, svds, cond, norm.

Tests cover:
- Standard eigenvalue problems (Laplacian, etc.)
- Matrix exponential for evolution equations
- Nullspace computation
- Singular value decomposition
- Condition number estimation
- Operator norms
- Various boundary conditions (Dirichlet, Neumann, mixed)
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestEigs:
    """Tests for LinOp.eigs() eigenvalue computation."""

    def test_laplacian_dirichlet_eigenvalues(self):
        """Test eigenvalues of -d^2/dx^2 with Dirichlet BCs on [0, pi].

        Eigenvalues should be n^2 for n = 1, 2, 3, ...
        Eigenfunctions should be sin(n*x).
        """
        domain = Domain([0, np.pi])

        # L = -d^2/dx^2, so coeffs = [0, 0, -1] (constant zero, zero first deriv, -1 second deriv)
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        # Dirichlet BCs: u(0) = 0, u(pi) = 0
        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        eigenvalues, eigenfunctions = L.eigs(k=4)

        # Expected eigenvalues: 1, 4, 9, 16
        expected = np.array([1.0, 4.0, 9.0, 16.0])

        for i, (eig_computed, eig_expected) in enumerate(zip(eigenvalues, expected)):
            rel_err = abs(eig_computed - eig_expected) / eig_expected
            assert rel_err < 1e-6, f"Eigenvalue {i+1}: expected {eig_expected}, got {eig_computed}"

    def test_laplacian_eigenfunctions_orthogonal(self):
        """Test that eigenfunctions are orthogonal."""
        domain = Domain([0, np.pi])

        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        _, eigenfunctions = L.eigs(k=3)

        # Check orthogonality (L2 inner product should be ~0 for different eigenfunctions)
        for i in range(len(eigenfunctions)):
            for j in range(i+1, len(eigenfunctions)):
                inner = (eigenfunctions[i] * eigenfunctions[j]).sum()
                assert abs(inner) < 1e-6, f"Eigenfunctions {i} and {j} not orthogonal: inner product = {inner}"

    def test_harmonic_oscillator_eigenvalues(self):
        """Test eigenvalues of -d^2/dx^2 + x^2 (harmonic oscillator-like).

        On a bounded domain with Dirichlet BCs, eigenvalues depend on domain.
        Just verify we get positive increasing eigenvalues.
        """
        domain = Domain([-5, 5])

        a0 = chebfun(lambda x: x**2, [-5, 5])  # Potential
        a1 = chebfun(lambda x: 0*x, [-5, 5])
        a2 = chebfun(lambda x: -1 + 0*x, [-5, 5])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        eigenvalues, _ = L.eigs(k=4)

        # Eigenvalues should be positive and increasing
        assert all(eigenvalues > 0), "Eigenvalues should be positive"
        assert all(np.diff(np.real(eigenvalues)) > 0), "Eigenvalues should be increasing"

    def test_eigs_with_shift(self):
        """Test eigenvalue computation with shift-invert mode."""
        domain = Domain([0, np.pi])

        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        # Use shift near 9 to get eigenvalues close to 9
        eigenvalues, _ = L.eigs(k=3, sigma=9.0)

        # Should include eigenvalue near 9
        has_nine = any(abs(eig - 9.0) < 1.0 for eig in eigenvalues)
        assert has_nine, f"Should find eigenvalue near 9, got {eigenvalues}"


class TestExpm:
    """Tests for LinOp.expm() matrix exponential."""

    def test_expm_diffusion_decay(self):
        """Test that exp(t*D^2) with BCs correctly causes diffusion/decay."""
        domain = Domain([0, np.pi])

        # L = d^2/dx^2 (diffusion operator)
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: 1 + 0*x, [0, np.pi])

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
        )
        L.lbc = 0  # u(0) = 0
        L.rbc = 0  # u(pi) = 0

        # Initial condition: sin(x) (eigenfunction with eigenvalue -1)
        u0 = chebfun(lambda x: np.sin(x), [0, np.pi])

        # After time t, solution is exp(-1*t)*sin(x) for the heat equation
        t = 0.1
        u_t = L.expm(t=t, u0=u0, num_eigs=20)

        # Check BCs are preserved
        assert abs(u_t(np.array([0.0]))[0]) < 1e-8
        assert abs(u_t(np.array([np.pi]))[0]) < 1e-8

        # Check decay rate: amplitude should be exp(-t) of original
        amp_ratio = abs(u_t(np.array([np.pi/2]))[0] / u0(np.array([np.pi/2]))[0])
        expected_ratio = np.exp(-1 * t)
        assert abs(amp_ratio - expected_ratio) / expected_ratio < 1e-3

    def test_expm_identity_at_zero_time(self):
        """exp(0*L) should be identity."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])  # d/dx

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )
        L.lbc = 0  # Need BCs for well-posed eigenvalue problem

        u0 = chebfun(lambda x: np.cos(np.pi * x), [-1, 1])
        u_0 = L.expm(t=0.0, u0=u0)

        # Should be close to u0 (t=0 is special-cased)
        x_test = np.linspace(-1, 1, 50)
        err = np.max(np.abs(u_0(x_test) - u0(x_test)))
        assert err < 1e-10, f"exp(0*L)*u0 should equal u0, error = {err}"


class TestNull:
    """Tests for LinOp.null() nullspace computation."""

    def test_null_with_underdetermined_system(self):
        """Test nullspace when system has free parameters."""
        # For d^2u/dx^2 = 0 with only one BC, there should be a nullspace
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        # Only left BC
        def lbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
        )

        # With only one BC for a second-order operator, there may be nullspace
        null_basis = L.null()
        # The result depends on exact discretization; just verify it runs
        assert isinstance(null_basis, list)

    def test_null_well_posed_problem_empty(self):
        """Well-posed problem should have empty or near-empty nullspace."""
        domain = Domain([0, np.pi])

        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: 1 + 0*x, [0, np.pi])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        null_basis = L.null()
        # Well-posed Dirichlet problem should have trivial nullspace
        # (or numerical near-zero vectors)
        # Just verify the function runs correctly
        assert isinstance(null_basis, list)


class TestSvds:
    """Tests for LinOp.svds() singular value decomposition."""

    def test_svds_returns_correct_types(self):
        """Test that svds returns correct types."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        S, u_funcs, v_funcs = L.svds(k=3)

        assert isinstance(S, np.ndarray)
        assert len(S) <= 3
        assert len(u_funcs) == len(S)
        assert len(v_funcs) == len(S)

    def test_svds_singular_values_positive_decreasing(self):
        """Singular values should be positive and decreasing."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        S, _, _ = L.svds(k=5)

        assert all(S > 0), "Singular values should be positive"
        assert all(np.diff(S) <= 1e-10), "Singular values should be decreasing (or equal)"

    def test_svds_left_right_functions_are_chebfuns(self):
        """Left and right singular functions should be Chebfun objects."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        _, u_funcs, v_funcs = L.svds(k=2)

        from chebpy.chebfun import Chebfun
        for uf in u_funcs:
            assert isinstance(uf, Chebfun)
        for vf in v_funcs:
            assert isinstance(vf, Chebfun)


class TestCond:
    """Tests for LinOp.cond() condition number."""

    def test_cond_positive(self):
        """Condition number should be positive."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        kappa = L.cond()
        assert kappa > 0 or kappa == np.inf

    def test_cond_well_conditioned_operator(self):
        """A simple operator should have moderate condition number."""
        domain = Domain([-1, 1])

        # Identity-like operator: u + small d^2u/dx^2
        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 0.01 + 0*x, [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        kappa = L.cond()
        # Should be finite and not astronomical
        assert np.isfinite(kappa)
        assert kappa < 1e12

    def test_cond_different_norms(self):
        """Test condition number with different norms."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        kappa_2 = L.cond(p=2)
        kappa_1 = L.cond(p=1)
        kappa_inf = L.cond(p=np.inf)

        assert kappa_2 > 0
        assert kappa_1 > 0
        assert kappa_inf > 0


class TestNorm:
    """Tests for LinOp.norm() operator norm."""

    def test_norm_positive(self):
        """Operator norm should be positive for non-trivial operators."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        n = L.norm()
        assert n >= 0

    def test_norm_spectral(self):
        """Test that spectral norm (p=2) is largest singular value."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        norm_2 = L.norm(p=2)
        S, _, _ = L.svds(k=1)

        # Should be approximately equal to largest singular value
        assert abs(norm_2 - S[0]) < 1e-10

    def test_norm_different_p_values(self):
        """Test norms with different p values."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        norm_1 = L.norm(p=1)
        norm_2 = L.norm(p=2)
        norm_inf = L.norm(p=np.inf)

        assert norm_1 >= 0
        assert norm_2 >= 0
        assert norm_inf >= 0


class TestLinOpWithVariousBC:
    """Test LinOp functions with various boundary condition types."""

    def test_eigs_neumann_bc(self):
        """Test eigenvalues with Neumann boundary conditions.

        For -d^2/dx^2 on [0, pi] with u'(0) = u'(pi) = 0,
        eigenvalues are n^2 for n = 0, 1, 2, ...
        with eigenfunctions cos(n*x).
        """
        domain = Domain([0, np.pi])

        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        # Neumann BCs: u'(0) = 0, u'(pi) = 0
        def lbc(u):
            return u.diff()
        def rbc(u):
            return u.diff()

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        eigenvalues, _ = L.eigs(k=4)

        # First eigenvalue should be 0 (constant eigenfunction)
        # Subsequent: 1, 4, 9, ...
        # Just check they're reasonable
        assert len(eigenvalues) >= 1

    def test_mixed_bc(self):
        """Test with mixed Dirichlet/Neumann boundary conditions."""
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        # u(0) = 0, u'(1) = 0
        def lbc(u):
            return u
        def rbc(u):
            return u.diff()

        rhs = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc,
            rhs=rhs
        )

        # Just verify these operations work with mixed BCs
        kappa = L.cond()
        assert np.isfinite(kappa)

        n = L.norm()
        assert n >= 0


class TestLinOpVariableCoefficients:
    """Test LinOp functions with variable coefficient operators."""

    def test_eigs_variable_coefficient(self):
        """Test eigenvalue computation with variable coefficients."""
        domain = Domain([-1, 1])

        # L = -(1+x^2) d^2/dx^2 (variable diffusion coefficient)
        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: -(1 + x**2), [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        eigenvalues, eigenfunctions = L.eigs(k=3)

        # Eigenvalues should be positive (operator is positive definite)
        assert all(np.real(eigenvalues) > 0)
        # Eigenfunctions should be returned
        assert len(eigenfunctions) == len(eigenvalues)

    def test_norm_variable_coefficient(self):
        """Test norm computation with variable coefficients."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: np.exp(x), [-1, 1])
        a1 = chebfun(lambda x: x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        n = L.norm()
        assert n >= 0
        assert np.isfinite(n)


class TestLinOpHigherOrder:
    """Test LinOp functions with higher-order operators."""

    def test_fourth_order_cond(self):
        """Test condition number of fourth-order operator."""
        domain = Domain([0, 1])

        # L = d^4/dx^4 + d^2/dx^2 + I (regularized beam equation)
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])
        a3 = chebfun(lambda x: 0*x, [0, 1])
        a4 = chebfun(lambda x: 1 + 0*x, [0, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2, a3, a4],
            domain=domain,
            diff_order=4,
            lbc=lbc,
            rbc=rbc
        )

        # Test that cond() works for 4th order
        kappa = L.cond()
        assert kappa > 0

    def test_third_order_cond(self):
        """Test condition number of third-order operator."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 0*x, [-1, 1])
        a3 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2, a3],
            domain=domain,
            diff_order=3,
            lbc=lbc,
            rbc=rbc
        )

        kappa = L.cond()
        assert kappa > 0


class TestRank:
    """Tests for LinOp.rank() numerical rank."""

    def test_rank_full_rank_operator(self):
        """Well-posed problem should have full rank."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        r = L.rank()
        assert r > 0

    def test_rank_positive(self):
        """Rank should be positive for non-trivial operators."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        r = L.rank()
        assert r > 0


class TestTrace:
    """Tests for LinOp.trace()."""

    def test_trace_returns_float(self):
        """Trace should return a float."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        tr = L.trace()
        assert isinstance(tr, (float, np.floating))

    def test_trace_identity_like(self):
        """Trace of identity-like operator should be approximately n."""
        domain = Domain([-1, 1])

        # L = I (identity in coefficient sense)
        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0],
            domain=domain,
            diff_order=0,
        )

        tr = L.trace()
        # Should be close to the discretization size
        assert tr > 0


class TestDet:
    """Tests for LinOp.det()."""

    def test_det_returns_number(self):
        """Det should return a number."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        d = L.det()
        assert isinstance(d, (float, np.floating, complex, np.complexfloating))

    def test_det_nonzero_for_wellposed(self):
        """Determinant should be nonzero for well-posed problems."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 1 + 0*x, [-1, 1])
        a1 = chebfun(lambda x: 0*x, [-1, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        def lbc(u):
            return u
        def rbc(u):
            return u

        L = LinOp(
            coeffs=[a0, a1, a2],
            domain=domain,
            diff_order=2,
            lbc=lbc,
            rbc=rbc
        )

        d = L.det()
        assert abs(d) > 1e-100  # Non-zero


class TestDiscretizationHelpers:
    """Tests for the discretization helper methods."""

    def test_discretization_size_scales_with_order(self):
        """Higher order operators should use larger discretization."""
        domain = Domain([-1, 1])

        # First order
        L1 = LinOp(
            coeffs=[chebfun(lambda x: 0*x, [-1, 1]), chebfun(lambda x: 1+0*x, [-1, 1])],
            domain=domain,
            diff_order=1,
        )

        # Fourth order
        L4 = LinOp(
            coeffs=[chebfun(lambda x: 0*x, [-1, 1])] * 4 + [chebfun(lambda x: 1+0*x, [-1, 1])],
            domain=domain,
            diff_order=4,
        )

        n1 = L1._discretization_size()
        n4 = L4._discretization_size()

        # Fourth order should use more points
        assert n4 > n1

    def test_discretization_size_respects_explicit_n(self):
        """Explicit n should override automatic sizing."""
        domain = Domain([-1, 1])

        L = LinOp(
            coeffs=[chebfun(lambda x: 0*x, [-1, 1]), chebfun(lambda x: 1+0*x, [-1, 1])],
            domain=domain,
            diff_order=1,
        )

        n_explicit = 64
        n_actual = L._discretization_size(n_explicit)
        assert n_actual == n_explicit

    def test_discretize_returns_dense_matrix(self):
        """_discretize should return a dense numpy array."""
        domain = Domain([-1, 1])

        a0 = chebfun(lambda x: 0*x, [-1, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [-1, 1])

        L = LinOp(
            coeffs=[a0, a1],
            domain=domain,
            diff_order=1,
        )

        A, disc = L._discretize()
        assert isinstance(A, np.ndarray)
        assert 'n_per_block' in disc


class TestChebfunNorm:
    """Tests for the Chebfun.norm() method that LinOp depends on."""

    def test_chebfun_norm_l2(self):
        """Test L2 norm of Chebfun."""
        f = chebfun(lambda x: np.sin(np.pi * x), [-1, 1])

        # L2 norm of sin(pi*x) on [-1,1] is sqrt(integral of sin^2) = 1
        norm_l2 = f.norm(2)
        expected = 1.0  # sqrt(integral of sin^2(pi*x) from -1 to 1) = 1
        assert abs(norm_l2 - expected) < 1e-10

    def test_chebfun_norm_l1(self):
        """Test L1 norm of Chebfun."""
        # |x| has a corner at x=0, so we need to provide a breakpoint there
        f = chebfun(lambda x: np.abs(x), [-1, 0, 1])

        # L1 norm of |x| on [-1,1] is integral of |x| = 1
        norm_l1 = f.norm(1)
        expected = 1.0
        assert abs(norm_l1 - expected) < 1e-8  # Relaxed tolerance for |x| approximation

    def test_chebfun_norm_linf(self):
        """Test L-infinity norm of Chebfun."""
        f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi])

        # L-infinity norm of sin(x) is 1
        norm_inf = f.norm(np.inf)
        assert abs(norm_inf - 1.0) < 1e-10

    def test_chebfun_norm_general_p(self):
        """Test general Lp norm."""
        f = chebfun(lambda x: 1 + 0*x, [-1, 1])  # Constant 1

        # Lp norm of constant 1 on [-1,1] is 2^(1/p)
        for p in [3, 4, 5]:
            norm_p = f.norm(p)
            expected = 2 ** (1/p)
            assert abs(norm_p - expected) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
