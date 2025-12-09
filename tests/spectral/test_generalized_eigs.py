"""Tests for generalized eigenvalue problems L[u] = λ * M[u].

These tests verify that the generalized eigenvalue solver correctly handles
weighted eigenvalue problems with mass matrices.
"""

import numpy as np

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestGeneralizedEigenvalues:
    """Tests for generalized eigenvalue problems with mass matrices."""

    def test_identity_mass_matrix(self):
        """Test that M = I gives same result as standard eigenvalue problem.

        For L[u] = λ * I[u] = λ * u, should match standard problem.
        """
        domain = Domain([0, np.pi])

        # L = -d^2/dx^2
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # M = I (identity operator)
        m0 = chebfun(lambda x: 1 + 0*x, [0, np.pi])
        M = LinOp(coeffs=[m0], domain=domain, diff_order=0)
        M.lbc = 0
        M.rbc = 0

        # Compute eigenvalues both ways
        evals_standard, _ = L.eigs(k=4)
        evals_generalized, _ = L.eigs(k=4, mass_matrix=M)

        # Should be identical
        evals_standard = np.sort(evals_standard)
        evals_generalized = np.sort(evals_generalized)

        for i in range(4):
            rel_err = abs(evals_standard[i] - evals_generalized[i]) / abs(evals_standard[i])
            assert rel_err < 1e-6, f"Eigenvalue {i}: {evals_standard[i]} vs {evals_generalized[i]}"

    def test_constant_weight_function(self):
        """Test -u'' = λ * c * u with constant weight c.

        Eigenvalues should be (n*π)^2 / c.
        """
        domain = Domain([0, 1])

        # L = -d^2/dx^2
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # M[u] = 2*u (constant weight c=2)
        m0 = chebfun(lambda x: 2 + 0*x, [0, 1])
        M = LinOp(coeffs=[m0], domain=domain, diff_order=0)
        M.lbc = 0
        M.rbc = 0

        # Compute eigenvalues
        evals, _ = L.eigs(k=4, mass_matrix=M)
        evals = np.sort(evals)

        # Expected: (n*π)^2 / 2 for n = 1, 2, 3, 4
        expected = np.array([(n*np.pi)**2 / 2 for n in range(1, 5)])

        for i in range(4):
            rel_err = abs(evals[i] - expected[i]) / expected[i]
            assert rel_err < 1e-3, f"Eigenvalue {i}: {evals[i]} vs {expected[i]}"

    def test_variable_weight_function(self):
        """Test -u'' = λ * x * u (Sturm-Liouville with variable weight).

        This is a weighted eigenvalue problem on [0, 1] with weight x.
        """
        domain = Domain([0, 1])

        # L = -d^2/dx^2
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # M[u] = x * u (variable weight)
        m0 = chebfun(lambda x: x, [0, 1])
        M = LinOp(coeffs=[m0], domain=domain, diff_order=0)
        M.lbc = 0
        M.rbc = 0

        # Compute eigenvalues
        evals, efuns = L.eigs(k=3, mass_matrix=M)

        # Check that we got positive eigenvalues
        assert len(evals) == 3
        assert all(evals > 0), f"Expected positive eigenvalues, got {evals}"

        # Check that eigenfunctions satisfy BCs
        for i, ef in enumerate(efuns):
            bc_left = abs(ef(np.array([0.0]))[0])
            bc_right = abs(ef(np.array([1.0]))[0])
            assert bc_left < 1e-10, f"Eigenfunction {i}: u(0) = {bc_left}"
            assert bc_right < 1e-10, f"Eigenfunction {i}: u(1) = {bc_right}"

        # Check orthogonality with respect to weight
        # <u, v>_weight = ∫ u(x) * v(x) * x dx
        for i in range(3):
            for j in range(i+1, 3):
                # Compute weighted inner product
                product = (efuns[i] * efuns[j] * m0).sum()
                assert abs(product) < 1e-10, f"Eigenfunctions {i} and {j} not orthogonal: product={product}"

    def test_beam_vibration_problem(self):
        """Test u'''' = λ * u'' (beam with rotational inertia).

        This is a generalized eigenvalue problem modeling beam vibration
        where the mass matrix includes rotational inertia effects.

        For simply supported beam: u(0) = u''(0) = u(1) = u''(1) = 0

        NOTE: This problem involves a mass matrix with derivatives (M = d²/dx²),
        which makes it more numerically challenging than scalar mass matrices.
        The key is that BCs apply to the main operator L, and M is projected
        into the same BC-satisfying subspace.
        """
        domain = Domain([0, 1])

        # L = d^4/dx^4
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 0*x, [0, 1])
        a3 = chebfun(lambda x: 0*x, [0, 1])
        a4 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2, a3, a4], domain=domain, diff_order=4)

        # Simply supported BCs: u(0) = u''(0) = u(1) = u''(1) = 0
        # These BCs are attached to L (the main operator)
        L.bc = [
            lambda u: u(np.array([0.0]))[0],           # u(0) = 0
            lambda u: u.diff(2)(np.array([0.0]))[0],   # u''(0) = 0
            lambda u: u(np.array([1.0]))[0],           # u(1) = 0
            lambda u: u.diff(2)(np.array([1.0]))[0],   # u''(1) = 0
        ]

        # M = d^2/dx^2 (rotational inertia)
        # M does NOT get BCs - it will be projected into L's BC-satisfying subspace
        m0 = chebfun(lambda x: 0*x, [0, 1])
        m1 = chebfun(lambda x: 0*x, [0, 1])
        m2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        M = LinOp(coeffs=[m0, m1, m2], domain=domain, diff_order=2)
        # No BCs on M - per Chebfun convention

        # Compute eigenvalues
        evals, efuns = L.eigs(k=3, mass_matrix=M)

        # Mathematical note: For u'''' = λu'' with u=sin(nπx):
        # u'' = -(nπ)² sin(nπx), u'''' = (nπ)⁴ sin(nπx)
        # So (nπ)⁴ sin(nπx) = λ(-(nπ)² sin(nπx))
        # Therefore λ = -(nπ)²
        # Eigenvalues are NEGATIVE for this problem.

        assert len(evals) == 3
        # Expected: -(nπ)² for n = 1, 2, 3
        expected = -np.array([(n*np.pi)**2 for n in range(1, 4)])

        # Sort eigenvalues by magnitude for comparison
        idx_sorted = np.argsort(np.abs(evals))
        evals_sorted = evals[idx_sorted]

        for i in range(3):
            rel_err = abs(evals_sorted[i] - expected[i]) / abs(expected[i])
            assert rel_err < 0.05, f"Eigenvalue {i}: {evals_sorted[i]} vs {expected[i]}, rel_err={rel_err}"

        # Check eigenfunctions satisfy BCs
        for i, ef in enumerate(efuns):
            bc_u0 = abs(ef(np.array([0.0]))[0])
            bc_u1 = abs(ef(np.array([1.0]))[0])
            bc_u2_0 = abs(ef.diff(2)(np.array([0.0]))[0])
            bc_u2_1 = abs(ef.diff(2)(np.array([1.0]))[0])

            assert bc_u0 < 1e-10, f"Eigenfunction {i}: u(0) = {bc_u0}"
            assert bc_u1 < 1e-10, f"Eigenfunction {i}: u(1) = {bc_u1}"
            # Note: Second derivative BCs have looser tolerance due to numerical differentiation
            # Using 3e-8 to account for improved but still approximate derivative evaluation at boundaries
            # With barycentric formula: typical errors are 1-2e-8, allow 3e-8 for safety
            assert bc_u2_0 < 3e-8, f"Eigenfunction {i}: u''(0) = {bc_u2_0}"
            assert bc_u2_1 < 3e-8, f"Eigenfunction {i}: u''(1) = {bc_u2_1}"

    def test_mixed_bcs_generalized(self):
        """Test generalized problem with mixed (Neumann-Dirichlet) BCs."""
        domain = Domain([0, np.pi])

        # L = -d^2/dx^2
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = lambda u: u.diff()(np.array([0.0]))[0]  # Neumann
        L.rbc = 0  # Dirichlet

        # M[u] = u
        m0 = chebfun(lambda x: 1 + 0*x, [0, np.pi])
        M = LinOp(coeffs=[m0], domain=domain, diff_order=0)
        M.lbc = lambda u: u.diff()(np.array([0.0]))[0]
        M.rbc = 0

        # Compute eigenvalues
        evals, efuns = L.eigs(k=3, mass_matrix=M)

        # Expected: ((n + 1/2))^2 for n = 0, 1, 2
        expected = np.array([(n + 0.5)**2 for n in range(3)])

        evals = np.sort(evals)
        for i in range(3):
            rel_err = abs(evals[i] - expected[i]) / expected[i]
            assert rel_err < 1e-2, f"Eigenvalue {i}: {evals[i]} vs {expected[i]}"


class TestGeneralizedEigenvaluesRegression:
    """Regression tests to ensure generalized eigenvalues work correctly."""

    def test_no_mass_matrix_unchanged(self):
        """Verify that not providing mass_matrix gives same results as before."""
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # Should work without mass_matrix parameter
        evals, efuns = L.eigs(k=3)

        assert len(evals) == 3
        assert len(efuns) == 3

        # Expected: (n*π)^2 for n = 1, 2, 3
        expected = np.array([(n*np.pi)**2 for n in range(1, 4)])
        evals = np.sort(evals)

        for i in range(3):
            rel_err = abs(evals[i] - expected[i]) / expected[i]
            assert rel_err < 1e-3
