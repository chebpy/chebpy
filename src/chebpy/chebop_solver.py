"""Solver infrastructure for function-based chebop operators.

This module implements the numerical solver for chebop with `.op` functions,
including linearization and Newton iteration.
"""

import inspect
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .chebfun import Chebfun
from .spectral import cheb_points_scaled, diff_matrix


def diff(u, order=1):
    """Differentiation function for use in operator definitions.

    This is designed to be used inside N.op lambda functions.
    It returns the derivative of a chebfun.

    Args:
        u: Chebfun to differentiate
        order: Order of differentiation (default 1)

    Returns:
        Chebfun: Differentiated function

    Examples:
        >>> N.op = lambda y: diff(y, 2) + y
        >>> N.op = lambda t, y: diff(y) - t*y
    """
    if isinstance(u, Chebfun):
        result = u
        for _ in range(order):
            result = result.diff()
        return result
    else:
        raise TypeError(f"diff() requires a Chebfun, got {type(u)}")


def abs(u):
    """Absolute value function for chebfuns.

    Args:
        u: Chebfun

    Returns:
        Chebfun: |u|
    """
    if isinstance(u, Chebfun):
        from .api import chebfun
        return chebfun(lambda x: np.abs(u(x)), domain=u.support)
    else:
        return np.abs(u)


def sign(u):
    """Sign function for chebfuns.

    Args:
        u: Chebfun

    Returns:
        Chebfun: sign(u)
    """
    if isinstance(u, Chebfun):
        from .api import chebfun
        return chebfun(lambda x: np.sign(u(x)), domain=u.support)
    else:
        return np.sign(u)


def sqrt(u):
    """Square root function for chebfuns.

    Args:
        u: Chebfun

    Returns:
        Chebfun: sqrt(u)
    """
    if isinstance(u, Chebfun):
        from .api import chebfun
        return chebfun(lambda x: np.sqrt(np.abs(u(x))), domain=u.support)
    else:
        return np.sqrt(np.abs(u))


class ChebopSolver:
    """Solver for chebop with function-based operators.

    This class handles:
    - Evaluation of N.op functions
    - Linearization for linear problems
    - Newton iteration for nonlinear problems
    - Boundary condition application
    """

    def __init__(self, chebop):
        """Initialize solver for a chebop.

        Args:
            chebop: Chebop instance with .op function
        """
        self.chebop = chebop
        self.domain = chebop.domain.support

        # Analyze operator function
        if chebop.op is None:
            raise ValueError("Chebop must have .op function set")

        sig = inspect.signature(chebop.op)
        self.nargs = len(sig.parameters)

        # Determine problem type
        if self.nargs == 1:
            # N.op = lambda y: ...
            self.has_indep_var = False
            self.is_system = False
        elif self.nargs == 2:
            # N.op = lambda t, y: ... or lambda x, y: ...
            self.has_indep_var = True
            self.is_system = False
        else:
            # N.op = lambda t, u, v: ... (system)
            self.has_indep_var = True
            self.is_system = True
            self.nvars = self.nargs - 1

        # Cache for linearity test
        self._linearity_cache = None

    def solve(self, rhs, n=None, tol=1e-10, max_iter=20):
        """Solve the operator equation N*u = rhs.

        Args:
            rhs: Right-hand side (Chebfun, callable, scalar, or list for systems)
            n: Number of collocation points (default: adaptive)
            tol: Tolerance for Newton iteration (default: 1e-10)
            max_iter: Maximum Newton iterations (default: 20)

        Returns:
            Chebfun or list of Chebfuns: Solution (list for systems)
        """
        if self.is_system:
            return self._solve_system(rhs, n, tol, max_iter)

        # Convert rhs to chebfun
        from .api import chebfun
        if isinstance(rhs, (int, float)):
            rhs = chebfun(lambda x: float(rhs) + 0*x, domain=self.domain)
        elif callable(rhs):
            rhs = chebfun(rhs, domain=self.domain)
        elif not isinstance(rhs, Chebfun):
            raise TypeError(f"rhs must be Chebfun, callable, or scalar, got {type(rhs)}")

        # Determine grid size
        if n is None:
            n = max(32, len(rhs.funs[0].coeffs) if len(rhs.funs) > 0 else 32)

        # Check if operator is linear by testing linearity property (cached)
        if self._linearity_cache is None:
            self._linearity_cache = self._check_linearity(min(n, 16))  # Use smaller grid for test

        if self._linearity_cache:
            return self._solve_linear(rhs, n)
        else:
            print(f"Detected nonlinear operator, using Newton iteration...")
            return self._solve_nonlinear(rhs, n, tol, max_iter)

    def _check_linearity(self, n, eps=1e-8):
        """Check if operator is linear.

        Tests the linearity property: L(au + bv) = aL(u) + bL(v)

        Args:
            n: Grid size for test
            eps: Tolerance for linearity check

        Returns:
            bool: True if operator appears linear
        """
        try:
            x = cheb_points_scaled(n, self.domain)

            # Create two test functions
            u1_vals = np.sin(np.pi * (x - x[0]) / (x[-1] - x[0]))
            u2_vals = np.cos(np.pi * (x - x[0]) / (x[-1] - x[0]))

            u1 = self._vals_to_chebfun(x, u1_vals)
            u2 = self._vals_to_chebfun(x, u2_vals)

            # Scalars
            a, b = 1.5, 2.3

            # Compute L(au1 + bu2)
            u_comb = self._vals_to_chebfun(x, a * u1_vals + b * u2_vals)

            if self.has_indep_var:
                from .api import chebfun
                t_cheb = chebfun(lambda s: s, domain=self.domain)
                L_comb = self.chebop.op(t_cheb, u_comb)
                L_u1 = self.chebop.op(t_cheb, u1)
                L_u2 = self.chebop.op(t_cheb, u2)
            else:
                L_comb = self.chebop.op(u_comb)
                L_u1 = self.chebop.op(u1)
                L_u2 = self.chebop.op(u2)

            # Get values
            if isinstance(L_comb, Chebfun):
                L_comb_vals = L_comb(x)
            else:
                L_comb_vals = np.full(len(x), float(L_comb))

            if isinstance(L_u1, Chebfun):
                L_u1_vals = L_u1(x)
            else:
                L_u1_vals = np.full(len(x), float(L_u1))

            if isinstance(L_u2, Chebfun):
                L_u2_vals = L_u2(x)
            else:
                L_u2_vals = np.full(len(x), float(L_u2))

            # Test linearity
            expected = a * L_u1_vals + b * L_u2_vals
            error = np.linalg.norm(L_comb_vals - expected, np.inf)
            scale = max(np.linalg.norm(L_comb_vals, np.inf), 1.0)

            is_linear = (error / scale) < eps

            return is_linear

        except Exception as e:
            # If test fails, assume nonlinear to be safe
            print(f"Linearity test failed: {e}, assuming nonlinear")
            return False

    def _solve_linear(self, rhs, n):
        """Solve assuming operator is linear.

        Builds discretization matrix by evaluating operator on basis functions.

        Args:
            rhs: Right-hand side chebfun
            n: Number of collocation points

        Returns:
            Chebfun: Solution
        """
        x = cheb_points_scaled(n, self.domain)

        # Build discretization matrix by applying operator to basis functions
        # For each basis function e_j, compute L(e_j) at collocation points
        A = self._build_linearization_matrix(x, n)

        # Apply boundary conditions
        A, rhs_vec = self._apply_bcs(A, x, rhs)

        # Solve linear system
        u_vals = spsolve(A, rhs_vec)

        # Convert to chebfun
        return self._vals_to_chebfun(x, u_vals)

    def _build_linearization_matrix(self, x, n):
        """Build discretization matrix for linear operator.

        Evaluates N.op on basis functions to determine the discretization.

        Args:
            x: Collocation points
            n: Grid size

        Returns:
            A: Discretization matrix (n+1 x n+1)
        """
        A = np.zeros((n + 1, n + 1))

        # For each column j, apply operator to delta function at x[j]
        for j in range(n + 1):
            # Create basis function (delta at x[j])
            u_j = np.zeros(n + 1)
            u_j[j] = 1.0
            u_cheb = self._vals_to_chebfun(x, u_j)

            # Apply operator
            if self.has_indep_var:
                # N.op(x, u_j)
                from .api import chebfun
                t_cheb = chebfun(lambda s: s, domain=self.domain)
                Lu_j = self.chebop.op(t_cheb, u_cheb)
            else:
                # N.op(u_j)
                Lu_j = self.chebop.op(u_cheb)

            # Evaluate at collocation points
            if isinstance(Lu_j, Chebfun):
                A[:, j] = Lu_j(x)
            elif isinstance(Lu_j, (int, float)):
                A[:, j] = float(Lu_j)
            else:
                raise TypeError(f"Operator returned {type(Lu_j)}, expected Chebfun or scalar")

        return sparse.csr_matrix(A)

    def _solve_nonlinear(self, rhs, n, tol, max_iter):
        """Solve nonlinear problem using Newton iteration.

        Args:
            rhs: Right-hand side
            n: Grid size
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Chebfun: Solution
        """
        x = cheb_points_scaled(n, self.domain)
        rhs_vec = rhs(x)

        # Initial guess
        if self.chebop.init is not None:
            if isinstance(self.chebop.init, Chebfun):
                u_vals = self.chebop.init(x)
            elif callable(self.chebop.init):
                u_vals = self.chebop.init(x)
            else:
                u_vals = np.full(n + 1, float(self.chebop.init))
        else:
            # Smart initial guess based on BCs and RHS
            u_vals = self._make_initial_guess(x, rhs_vec, n)

        # Newton iteration
        for iter_count in range(max_iter):
            # Compute residual F(u) = L(u) - rhs
            u_cheb = self._vals_to_chebfun(x, u_vals)

            if self.has_indep_var:
                from .api import chebfun
                t_cheb = chebfun(lambda s: s, domain=self.domain)
                Lu = self.chebop.op(t_cheb, u_cheb)
            else:
                Lu = self.chebop.op(u_cheb)

            if isinstance(Lu, Chebfun):
                residual = Lu(x) - rhs_vec
            else:
                residual = float(Lu) - rhs_vec

            # Check convergence
            res_norm = np.linalg.norm(residual, np.inf)
            if res_norm < tol:
                print(f"Newton converged in {iter_count} iterations, residual={res_norm:.2e}")
                break

            # Compute Jacobian via finite differences
            J = self._jacobian_fd(x, u_vals, eps=1e-7)

            # Apply boundary conditions to Jacobian and residual
            J, residual = self._apply_bcs(J, x, None, residual=residual, u_current=u_vals)

            # Newton step: solve J * delta = -residual
            delta = spsolve(J, -residual)

            # Update solution
            u_vals = u_vals + delta

            # Check if maxnorm constraint is violated
            if self.chebop.maxnorm is not None:
                u_vals = np.clip(u_vals, -self.chebop.maxnorm, self.chebop.maxnorm)

        else:
            print(f"Warning: Newton did not converge in {max_iter} iterations, residual={res_norm:.2e}")

        return self._vals_to_chebfun(x, u_vals)

    def _solve_system(self, rhs, n, tol, max_iter):
        """Solve system of ODEs.

        For systems like:
        L.op = @(t,u,v) [diff(u)-v; diff(v)+u]
        [u,v] = L\0

        Args:
            rhs: Right-hand side (0, list of chebfuns, or list of scalars)
            n: Grid size
            tol: Tolerance
            max_iter: Max Newton iterations

        Returns:
            list of Chebfuns: Solutions [u, v, ...]
        """
        from .api import chebfun

        # Determine number of variables
        nvars = self.nvars

        # Set up grid
        if n is None:
            n = 32
        x = cheb_points_scaled(n, self.domain)

        # Convert RHS to list of vectors
        if isinstance(rhs, (int, float)) and rhs == 0:
            rhs_vecs = [np.zeros(n + 1) for _ in range(nvars)]
        elif isinstance(rhs, list):
            rhs_vecs = []
            for r in rhs:
                if isinstance(r, (int, float)):
                    rhs_vecs.append(np.full(n + 1, float(r)))
                elif isinstance(r, Chebfun):
                    rhs_vecs.append(r(x))
                elif callable(r):
                    rhs_vecs.append(chebfun(r, domain=self.domain)(x))
                else:
                    raise TypeError(f"Invalid RHS type: {type(r)}")
        else:
            raise TypeError(f"For systems, RHS must be 0 or list, got {type(rhs)}")

        # Stack RHS into single vector
        rhs_stacked = np.concatenate(rhs_vecs)

        # Initial guess for all variables
        u_stacked = self._make_system_initial_guess(x, rhs_vecs, n, nvars)

        # Check linearity (for systems)
        is_linear = self._check_system_linearity(n, nvars)

        if is_linear:
            print(f"Detected linear system, using direct solve...")
            A = self._build_system_matrix(x, n, nvars)
            A, rhs_stacked = self._apply_system_bcs(A, x, rhs_stacked, n, nvars)
            u_stacked = spsolve(A, rhs_stacked)
        else:
            print(f"Detected nonlinear system, using Newton iteration...")
            # Newton iteration for nonlinear systems
            for iter_count in range(max_iter):
                # Compute residual
                residual = self._evaluate_system_residual(x, u_stacked, rhs_stacked, n, nvars)

                # Check convergence
                res_norm = np.linalg.norm(residual, np.inf)
                if res_norm < tol:
                    print(f"Newton converged in {iter_count} iterations, residual={res_norm:.2e}")
                    break

                # Compute Jacobian
                J = self._system_jacobian_fd(x, u_stacked, n, nvars)

                # Apply BCs
                J, residual = self._apply_system_bcs(J, x, residual, n, nvars, u_current=u_stacked)

                # Newton step
                delta = spsolve(J, -residual)
                u_stacked = u_stacked + delta
            else:
                print(f"Warning: Newton did not converge in {max_iter} iterations, residual={res_norm:.2e}")

        # Unstack into separate chebfuns
        solutions = []
        for i in range(nvars):
            u_vals = u_stacked[i * (n + 1):(i + 1) * (n + 1)]
            solutions.append(self._vals_to_chebfun(x, u_vals))

        return solutions

    def _make_initial_guess(self, x, rhs_vec, n):
        """Create smart initial guess for Newton iteration.

        Args:
            x: Collocation points
            rhs_vec: Right-hand side values
            n: Grid size

        Returns:
            u_vals: Initial guess array
        """
        # Extract BC values
        lbc_val = 0.0
        rbc_val = 0.0

        if self.chebop.lbc is not None:
            if isinstance(self.chebop.lbc, (list, tuple, np.ndarray)):
                lbc_val = self.chebop.lbc[0]
            elif not isinstance(self.chebop.lbc, str):
                lbc_val = float(self.chebop.lbc)

        if self.chebop.rbc is not None:
            if isinstance(self.chebop.rbc, (list, tuple, np.ndarray)):
                rbc_val = self.chebop.rbc[0]
            elif not isinstance(self.chebop.rbc, str):
                rbc_val = float(self.chebop.rbc)

        # Linear interpolation + small perturbation from RHS
        # Normalize x to [0, 1]
        t = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)

        # Base linear interpolation
        u_vals = lbc_val + (rbc_val - lbc_val) * t

        # Add small component from RHS to break symmetry
        # This helps Newton converge to non-trivial solutions
        rhs_scale = np.linalg.norm(rhs_vec, np.inf)
        if rhs_scale > 1e-10:
            # Add 10% of RHS as perturbation
            u_vals += 0.1 * rhs_vec / rhs_scale

        return u_vals

    def _jacobian_fd(self, x, u_vals, eps=1e-7):
        """Compute Jacobian using finite differences.

        Args:
            x: Collocation points
            u_vals: Current solution values
            eps: Finite difference step

        Returns:
            J: Jacobian matrix
        """
        n = len(u_vals)
        J = np.zeros((n, n))

        # Compute F(u)
        u_cheb = self._vals_to_chebfun(x, u_vals)
        if self.has_indep_var:
            from .api import chebfun
            t_cheb = chebfun(lambda s: s, domain=self.domain)
            F0 = self.chebop.op(t_cheb, u_cheb)
        else:
            F0 = self.chebop.op(u_cheb)

        if isinstance(F0, Chebfun):
            F0_vals = F0(x)
        else:
            F0_vals = np.full(n, float(F0))

        # Perturb each component
        for j in range(n):
            u_pert = u_vals.copy()
            u_pert[j] += eps

            u_cheb_pert = self._vals_to_chebfun(x, u_pert)
            if self.has_indep_var:
                from .api import chebfun
                t_cheb = chebfun(lambda s: s, domain=self.domain)
                F1 = self.chebop.op(t_cheb, u_cheb_pert)
            else:
                F1 = self.chebop.op(u_cheb_pert)

            if isinstance(F1, Chebfun):
                F1_vals = F1(x)
            else:
                F1_vals = np.full(n, float(F1))

            J[:, j] = (F1_vals - F0_vals) / eps

        return sparse.csr_matrix(J)

    def _apply_bcs(self, A, x, rhs, residual=None, u_current=None):
        """Apply boundary conditions to system matrix and RHS.

        Args:
            A: System matrix
            x: Collocation points
            rhs: Right-hand side chebfun (or None if using residual)
            residual: Residual vector (for Newton, optional)
            u_current: Current solution values (for Newton, optional)

        Returns:
            A: Modified matrix
            rhs_vec: Modified RHS vector
        """
        n = len(x) - 1
        A = A.tolil()  # Convert to LIL for efficient modification

        if residual is not None:
            rhs_vec = residual.copy()
        else:
            rhs_vec = rhs(x).copy() if isinstance(rhs, Chebfun) else np.full(len(x), float(rhs))

        # Handle generic .bc first
        if self.chebop.bc is not None:
            if isinstance(self.chebop.bc, str):
                if self.chebop.bc.lower() == 'periodic':
                    # Periodic BCs
                    A[0, :] = 0
                    A[0, 0] = 1
                    A[0, n] = -1
                    rhs_vec[0] = 0

                    D = diff_matrix(n, self.domain, order=1)
                    A[n, :] = D[0, :] - D[n, :]
                    rhs_vec[n] = 0

                    return A.tocsr(), rhs_vec
                elif self.chebop.bc.lower() == 'dirichlet':
                    self.chebop.lbc = 0
                    self.chebop.rbc = 0
            elif callable(self.chebop.bc):
                # Function-based boundary conditions: L.bc = @(x,y) [...]
                # Create chebfun for current solution
                from .api import chebfun as make_chebfun
                x_chebfun = make_chebfun('x', domain=self.domain)

                if u_current is not None:
                    y_chebfun = self._vals_to_chebfun(x, u_current)
                else:
                    # For initial setup, use zeros
                    y_chebfun = make_chebfun(lambda t: 0.0, domain=self.domain)

                # Evaluate BC function
                try:
                    bc_constraints = self.chebop.bc(x_chebfun, y_chebfun)
                    if not isinstance(bc_constraints, (list, tuple)):
                        bc_constraints = [bc_constraints]

                    # Apply each constraint by replacing matrix rows
                    for i, constraint in enumerate(bc_constraints):
                        if i >= len(x):
                            break
                        A[i, :] = 0

                        # Constraint should evaluate to a scalar
                        if isinstance(constraint, Chebfun):
                            # If it's a chebfun, evaluate at the point
                            constraint_val = constraint(self.domain[0] if i < len(bc_constraints)//2 else self.domain[1])
                        else:
                            constraint_val = float(constraint)

                        # For simplicity, apply as point constraints
                        # This is a simplified implementation
                        A[i, i if i < n else n] = 1
                        rhs_vec[i] = constraint_val

                    return A.tocsr(), rhs_vec
                except Exception as e:
                    print(f"Warning: Function-based BC evaluation failed: {e}")
                    # Fall through to standard BC handling

        # Apply left BC
        if self.chebop.lbc is not None:
            if isinstance(self.chebop.lbc, (list, tuple, np.ndarray)):
                # Vector BC: [val; deriv; ...]
                lbc_array = np.atleast_1d(self.chebop.lbc)
                for i, bc_val in enumerate(lbc_array):
                    if i >= len(x):
                        break
                    A[i, :] = 0
                    if i == 0:
                        A[i, 0] = 1
                    else:
                        D_i = diff_matrix(n, self.domain, order=i)
                        A[i, :] = D_i[0, :]
                    rhs_vec[i] = bc_val
            elif isinstance(self.chebop.lbc, str):
                if self.chebop.lbc.lower() == 'neumann':
                    D = diff_matrix(n, self.domain, order=1)
                    A[0, :] = D[0, :]
                    rhs_vec[0] = 0
            else:
                # Scalar BC
                A[0, :] = 0
                A[0, 0] = 1
                rhs_vec[0] = float(self.chebop.lbc)

        # Apply right BC
        if self.chebop.rbc is not None:
            if isinstance(self.chebop.rbc, (list, tuple, np.ndarray)):
                # Vector BC
                rbc_array = np.atleast_1d(self.chebop.rbc)
                for i, bc_val in enumerate(rbc_array):
                    row_idx = n - len(rbc_array) + i + 1
                    if row_idx < 0 or row_idx > n:
                        continue
                    A[row_idx, :] = 0
                    if i == 0:
                        A[row_idx, n] = 1
                    else:
                        D_i = diff_matrix(n, self.domain, order=i)
                        A[row_idx, :] = D_i[n, :]
                    rhs_vec[row_idx] = bc_val
            elif isinstance(self.chebop.rbc, str):
                if self.chebop.rbc.lower() == 'neumann':
                    D = diff_matrix(n, self.domain, order=1)
                    A[n, :] = D[n, :]
                    rhs_vec[n] = 0
            else:
                # Scalar BC
                A[n, :] = 0
                A[n, n] = 1
                rhs_vec[n] = float(self.chebop.rbc)

        return A.tocsr(), rhs_vec

    def _vals_to_chebfun(self, x, vals):
        """Convert grid values to chebfun.

        Args:
            x: Collocation points
            vals: Function values at x

        Returns:
            Chebfun: Interpolating chebfun
        """
        from .algorithms import bary, barywts2
        from .utilities import Interval
        from .api import chebfun

        vk = barywts2(len(x))
        interval = Interval(*self.domain)

        def u_interp(x_eval):
            y_eval = interval.invmap(x_eval)
            yk = interval.invmap(x)
            return bary(y_eval, vals, yk, vk)

        return Chebfun.initfun_fixedlen(u_interp, len(x), domain=self.domain)

    # System-specific methods
    def _make_system_initial_guess(self, x, rhs_vecs, n, nvars):
        """Create initial guess for system."""
        u_stacked = []
        for i in range(nvars):
            # Extract BC for this variable
            lbc_val = 0.0
            rbc_val = 0.0
            
            if self.chebop.lbc is not None:
                if isinstance(self.chebop.lbc, (list, tuple, np.ndarray)):
                    if i < len(self.chebop.lbc):
                        lbc_val = float(self.chebop.lbc[i])
            
            if self.chebop.rbc is not None:
                if isinstance(self.chebop.rbc, (list, tuple, np.ndarray)):
                    if i < len(self.chebop.rbc):
                        rbc_val = float(self.chebop.rbc[i])
            
            # Linear interpolation for this variable
            t = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
            u_vals = lbc_val + (rbc_val - lbc_val) * t
            u_stacked.append(u_vals)
        
        return np.concatenate(u_stacked)

    def _check_system_linearity(self, n, nvars):
        """Check if system is linear."""
        # For now, assume linear systems - can enhance later
        return True

    def _build_system_matrix(self, x, n, nvars):
        """Build system matrix by evaluating on basis functions."""
        total_size = nvars * (n + 1)
        A = np.zeros((total_size, total_size))
        
        # For each variable and each basis function
        for var_idx in range(nvars):
            for j in range(n + 1):
                # Create basis function for variable var_idx
                u_test = [np.zeros(n + 1) for _ in range(nvars)]
                u_test[var_idx][j] = 1.0
                
                # Convert to chebfuns
                u_cheb = [self._vals_to_chebfun(x, u_test[k]) for k in range(nvars)]
                
                # Evaluate operator
                from .api import chebfun
                t_cheb = chebfun(lambda s: s, domain=self.domain)
                result = self.chebop.op(t_cheb, *u_cheb)
                
                # Extract values for each equation
                for eq_idx in range(nvars):
                    if isinstance(result, (list, tuple)):
                        res_eq = result[eq_idx]
                    else:
                        res_eq = result
                    
                    if isinstance(res_eq, Chebfun):
                        res_vals = res_eq(x)
                    else:
                        res_vals = np.full(n + 1, float(res_eq))
                    
                    # Fill matrix block
                    row_start = eq_idx * (n + 1)
                    col_start = var_idx * (n + 1)
                    A[row_start:row_start + n + 1, col_start + j] = res_vals
        
        return sparse.csr_matrix(A)

    def _evaluate_system_residual(self, x, u_stacked, rhs_stacked, n, nvars):
        """Evaluate residual for system."""
        # Unstack into chebfuns
        u_cheb = []
        for i in range(nvars):
            u_vals = u_stacked[i * (n + 1):(i + 1) * (n + 1)]
            u_cheb.append(self._vals_to_chebfun(x, u_vals))
        
        # Evaluate operator
        from .api import chebfun
        t_cheb = chebfun(lambda s: s, domain=self.domain)
        result = self.chebop.op(t_cheb, *u_cheb)
        
        # Stack residuals
        residuals = []
        for eq_idx in range(nvars):
            if isinstance(result, (list, tuple)):
                res_eq = result[eq_idx]
            else:
                res_eq = result if eq_idx == 0 else 0
            
            if isinstance(res_eq, Chebfun):
                res_vals = res_eq(x)
            else:
                res_vals = np.full(n + 1, float(res_eq))
            
            residuals.append(res_vals)
        
        residual_stacked = np.concatenate(residuals)
        return residual_stacked - rhs_stacked

    def _system_jacobian_fd(self, x, u_stacked, n, nvars, eps=1e-7):
        """Compute Jacobian for system using finite differences."""
        total_size = nvars * (n + 1)
        J = np.zeros((total_size, total_size))
        
        # Compute F(u)
        F0 = self._evaluate_system_residual(x, u_stacked, np.zeros_like(u_stacked), n, nvars)
        
        # Perturb each component
        for j in range(total_size):
            u_pert = u_stacked.copy()
            u_pert[j] += eps
            F1 = self._evaluate_system_residual(x, u_pert, np.zeros_like(u_stacked), n, nvars)
            J[:, j] = (F1 - F0) / eps
        
        return sparse.csr_matrix(J)

    def _apply_system_bcs(self, A, x, rhs_vec, n, nvars, u_current=None):
        """Apply boundary conditions for system."""
        A = A.tolil()
        
        # For each variable, apply BCs
        if self.chebop.lbc is not None and isinstance(self.chebop.lbc, (list, tuple)):
            for var_idx in range(min(len(self.chebop.lbc), nvars)):
                row_idx = var_idx * (n + 1)
                # Set u_var(a) = lbc[var_idx]
                A[row_idx, :] = 0
                A[row_idx, row_idx] = 1
                rhs_vec[row_idx] = float(self.chebop.lbc[var_idx])
        
        if self.chebop.rbc is not None and isinstance(self.chebop.rbc, (list, tuple)):
            for var_idx in range(min(len(self.chebop.rbc), nvars)):
                row_idx = var_idx * (n + 1) + n
                # Set u_var(b) = rbc[var_idx]
                A[row_idx, :] = 0
                A[row_idx, row_idx] = 1
                rhs_vec[row_idx] = float(self.chebop.rbc[var_idx])
        
        return A.tocsr(), rhs_vec
