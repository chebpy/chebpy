"""Differential operators with boundary conditions.

This module provides the Chebop class for representing and solving
differential equations using spectral collocation methods.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .chebfun import Chebfun
from .spectral import cheb_points_scaled, diff_matrix, identity_matrix, mult_matrix
from .utilities import Domain


class Chebop:
    """Differential operator with boundary conditions.

    Represents linear differential operators that can be applied to chebfuns
    and solved via spectral collocation. Supports boundary conditions and
    operator algebra.

    Attributes:
        domain (Domain): Domain on which operator is defined
        difforder (int): Order of differential operator
        lbc: Left boundary condition(s)
        rbc: Right boundary condition(s)
    """

    def __init__(self, *args, **kwargs):
        """Initialize a chebop on the given domain.

        Usage:
            N = Chebop([a, b])                    # domain only
            N = Chebop(a, b)                      # separate endpoints
            N = Chebop(@(y) diff(y,2)+y, [a, b])  # with operator function
            N = Chebop(@(y) diff(y,2)+y, [a, b], lbc, rbc)  # with BCs

        Args:
            *args: Variable arguments (see usage above)
            **kwargs: Optional keyword arguments
        """
        # Parse arguments to match MATLAB chebop constructor
        op_func = None
        domain = None
        lbc_arg = None
        rbc_arg = None

        if len(args) == 1:
            # Chebop([a, b]) or Chebop(domain)
            domain = args[0]
        elif len(args) == 2:
            if callable(args[0]):
                # Chebop(@(y) ..., [a, b])
                op_func = args[0]
                domain = args[1]
            else:
                # Chebop(a, b)
                domain = [args[0], args[1]]
        elif len(args) == 3:
            # Chebop(@(y) ..., [a, b], lbc) or Chebop(a, b, lbc)
            if callable(args[0]):
                op_func = args[0]
                domain = args[1]
                lbc_arg = args[2]
            else:
                domain = [args[0], args[1]]
                lbc_arg = args[2]
        elif len(args) == 4:
            # Chebop(@(y) ..., [a, b], lbc, rbc) or Chebop(a, b, lbc, rbc)
            if callable(args[0]):
                op_func = args[0]
                domain = args[1]
                lbc_arg = args[2]
                rbc_arg = args[3]
            else:
                domain = [args[0], args[1]]
                lbc_arg = args[2]
                rbc_arg = args[3]

        self.domain = Domain(domain) if not isinstance(domain, Domain) else domain
        self._matrix_func = None
        self._forward_func = None
        self.difforder = 0
        self.lbc = lbc_arg
        self.rbc = rbc_arg
        self.bc = None  # Generic boundary conditions
        self._scale_factor = 1.0
        self.op = op_func  # Operator function (e.g., @(y) diff(y,2)+y)
        self.init = None  # Initial guess for nonlinear problems
        self.maxnorm = None  # Maximum norm for solution

    @classmethod
    def identity(cls, domain):
        """Create identity operator I.

        Args:
            domain (array-like): Domain for the operator.

        Returns:
            Chebop: Identity operator on given domain.
        """
        op = cls(domain)
        op._matrix_func = lambda n: identity_matrix(n)
        op._forward_func = lambda f: f
        op.difforder = 0
        return op

    @classmethod
    def diff(cls, domain, order=1):
        """Create differentiation operator D^k.

        Args:
            domain (array-like): Domain for the operator.
            order (int, optional): Order of differentiation. Defaults to 1.

        Returns:
            Chebop: Differentiation operator on given domain.
        """
        op = cls(domain)
        op._matrix_func = lambda n: diff_matrix(n, op.domain.support, order=order)
        # Create forward function that applies diff() k times
        def apply_diff_k_times(f, k=order):
            result = f
            for _ in range(k):
                result = result.diff()
            return result
        op._forward_func = apply_diff_k_times
        op.difforder = order
        return op

    @classmethod
    def diag(cls, chebfun):
        """Create multiplication operator by a chebfun.

        Args:
            chebfun: Chebfun to multiply by.

        Returns:
            Chebop: Multiplication operator.
        """
        op = cls(chebfun.support)
        op._matrix_func = lambda n: mult_matrix(chebfun, n)
        op._forward_func = lambda f: chebfun * f
        op.difforder = 0
        return op

    def __call__(self, f):
        """Apply operator to a chebfun (forward mode).

        Args:
            f (Chebfun): Function to apply operator to.

        Returns:
            Chebfun: Result of applying operator.
        """
        if self._forward_func is None:
            raise NotImplementedError("Forward operation not defined for this operator")
        return self._forward_func(f)

    def __add__(self, other):
        """Add two operators or add a scalar.

        Args:
            other: Another Chebop or a scalar.

        Returns:
            Chebop: Sum of operators.
        """
        result = Chebop(self.domain)

        if np.isscalar(other):
            # L + c -> (L + cI)
            result._matrix_func = lambda n: self._matrix_func(n) + other * identity_matrix(n)
            if self._forward_func:
                result._forward_func = lambda f: self._forward_func(f) + other
            result.difforder = self.difforder
        else:
            # L1 + L2
            result._matrix_func = lambda n: self._matrix_func(n) + other._matrix_func(n)
            if self._forward_func and other._forward_func:
                result._forward_func = lambda f: self._forward_func(f) + other._forward_func(f)
            result.difforder = max(self.difforder, other.difforder)

        return result

    def __mul__(self, other):
        """Multiply operator by scalar or compose with another operator.

        Args:
            other: Scalar or another Chebop.

        Returns:
            Chebop: Scaled or composed operator.
        """
        result = Chebop(self.domain)

        if np.isscalar(other):
            # c * L
            result._matrix_func = lambda n: other * self._matrix_func(n)
            if self._forward_func:
                result._forward_func = lambda f: other * self._forward_func(f)
            result.difforder = self.difforder
            result._scale_factor = other
        else:
            # L1 * L2 (composition: (L1*L2)*f = L1(L2(f)))
            result._matrix_func = lambda n: self._matrix_func(n) @ other._matrix_func(n)
            if self._forward_func and other._forward_func:
                result._forward_func = lambda f: self._forward_func(other._forward_func(f))
            result.difforder = self.difforder + other.difforder

        return result

    __rmul__ = __mul__

    def __sub__(self, other):
        """Subtract operators or subtract a scalar."""
        return self + (-1) * other

    def __pow__(self, n):
        """Raise operator to integer power.

        Args:
            n (int): Power to raise operator to.

        Returns:
            Chebop: L^n
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("Power must be a positive integer")

        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def __truediv__(self, other):
        r"""Backslash operator: L \ f solves L*u = f.

        This is an alias for solve() to match MATLAB syntax.
        In MATLAB: u = L \ f
        In Python: u = L / f (since we can't overload backslash)

        Args:
            other: Right-hand side (Chebfun or callable)

        Returns:
            Chebfun: Solution to L*u = other
        """
        return self.solve(other)

    def solve(self, f, n=None):
        """Solve Lu = f with boundary conditions.

        Args:
            f (Chebfun or callable): Right-hand side.
            n (int, optional): Number of collocation points. If None, adaptive.

        Returns:
            Chebfun: Solution to Lu = f.
        """
        # Check if operator function is set
        if self.op is not None:
            # Use function-based solver
            from .chebop_solver import ChebopSolver
            solver = ChebopSolver(self)
            return solver.solve(f, n=n)

        # Original algebraic operator path
        # Convert f to chebfun if needed
        if callable(f):
            f = Chebfun.initfun_adaptive(f, domain=self.domain.support)

        # Determine number of points
        if n is None:
            # Start with a reasonable guess based on f
            n = max(32, len(f.funs[0].coeffs) if len(f.funs) > 0 else 32)

        # Get collocation points
        x = cheb_points_scaled(n, self.domain.support)

        # Build system matrix
        A = self._matrix_func(n)

        # Apply boundary conditions
        A = self._apply_boundary_conditions(A, n)

        # Sample right-hand side
        rhs = f(x)

        # Adjust RHS for boundary conditions
        rhs = self._adjust_rhs(rhs, n)

        # Solve system
        u_vals = spsolve(A, rhs)

        # Convert to chebfun using initfun_fixedlen
        from .utilities import Interval

        # Create interpolating function
        def u_interp(x_eval):
            # Barycentric interpolation
            from .algorithms import bary, barywts2
            xk = x
            fk = u_vals
            vk = barywts2(len(xk))
            # Map x_eval to [-1, 1] for barycentric formula
            interval = Interval(*self.domain.support)
            y_eval = interval.invmap(x_eval)
            # Scale points to [-1, 1]
            yk = interval.invmap(xk)
            return bary(y_eval, fk, yk, vk)

        result = Chebfun.initfun_fixedlen(u_interp, n + 1, domain=self.domain.support)
        return result

    def _apply_boundary_conditions(self, A, n):
        """Apply boundary conditions to system matrix.

        Args:
            A: System matrix (will be modified in-place).
            n: Number of interior points.

        Returns:
            Modified matrix with BCs applied.
        """
        # Handle generic .bc property first
        if self.bc is not None:
            if isinstance(self.bc, str):
                if self.bc.lower() == 'dirichlet':
                    # Apply Dirichlet BCs at both ends: u(a) = u(b) = 0
                    self.lbc = 0
                    self.rbc = 0
                elif self.bc.lower() == 'periodic':
                    # Periodic BCs: u(a) = u(b), u'(a) = u'(b)
                    # Replace first and last rows with periodicity conditions
                    A = A.tolil()

                    # u(a) = u(b): u[0] - u[n] = 0
                    A[0, :] = 0
                    A[0, 0] = 1
                    A[0, n] = -1

                    # u'(a) = u'(b): D[0,:]*u - D[n,:]*u = 0
                    D = diff_matrix(n, self.domain.support, order=1)
                    A[n, :] = D[0, :] - D[n, :]

                    return A.tocsr()

        A = A.tolil()  # Convert to LIL format for efficient row modification

        # Apply left BCs
        if self.lbc is not None:
            if isinstance(self.lbc, str):
                if self.lbc.lower() == 'neumann':
                    # Neumann: u'(a) = 0
                    D = diff_matrix(n, self.domain.support, order=1)
                    A[0, :] = D[0, :]
                elif self.lbc.lower() == 'dirichlet':
                    # Dirichlet: u(a) = 0
                    A[0, :] = 0
                    A[0, 0] = 1
            elif np.isscalar(self.lbc):
                # Dirichlet: u(a) = lbc
                A[0, :] = 0
                A[0, 0] = 1
            elif isinstance(self.lbc, (list, tuple, np.ndarray)):
                # Vector BC: [u(a); u'(a); u''(a); ...]
                # Each element specifies a BC for a derivative order
                lbc_array = np.atleast_1d(self.lbc)
                for i, bc_val in enumerate(lbc_array):
                    if i >= n + 1:
                        break  # Can't have more BCs than grid points
                    if i == 0:
                        # u(a) = bc_val
                        A[i, :] = 0
                        A[i, 0] = 1
                    else:
                        # u^(i)(a) = bc_val (ith derivative)
                        D_i = diff_matrix(n, self.domain.support, order=i)
                        A[i, :] = D_i[0, :]

        # Apply right BCs
        if self.rbc is not None:
            if isinstance(self.rbc, str):
                if self.rbc.lower() == 'neumann':
                    # Neumann: u'(b) = 0
                    D = diff_matrix(n, self.domain.support, order=1)
                    A[n, :] = D[n, :]
                elif self.rbc.lower() == 'dirichlet':
                    # Dirichlet: u(b) = 0
                    A[n, :] = 0
                    A[n, n] = 1
            elif np.isscalar(self.rbc):
                # Dirichlet: u(b) = rbc
                A[n, :] = 0
                A[n, n] = 1
            elif isinstance(self.rbc, (list, tuple, np.ndarray)):
                # Vector BC: [u(b); u'(b); u''(b); ...]
                rbc_array = np.atleast_1d(self.rbc)
                for i, bc_val in enumerate(rbc_array):
                    if n - i < 0:
                        break  # Can't have more BCs than grid points
                    if i == 0:
                        # u(b) = bc_val
                        A[n - i, :] = 0
                        A[n - i, n] = 1
                    else:
                        # u^(i)(b) = bc_val (ith derivative)
                        D_i = diff_matrix(n, self.domain.support, order=i)
                        A[n - i, :] = D_i[n, :]

        return A.tocsr()

    def _adjust_rhs(self, rhs, n):
        """Adjust RHS vector for boundary conditions.

        Args:
            rhs: Right-hand side vector.
            n: Number of interior points.

        Returns:
            Adjusted RHS vector.
        """
        rhs = rhs.copy()

        # Handle periodic BCs first
        if self.bc is not None and isinstance(self.bc, str) and self.bc.lower() == 'periodic':
            # For periodic BCs, boundary rows should be 0
            rhs[0] = 0
            rhs[n] = 0
            return rhs

        # Adjust for left BC
        if self.lbc is not None:
            if isinstance(self.lbc, str):
                if self.lbc.lower() in ['neumann', 'dirichlet']:
                    # Homogeneous BC
                    rhs[0] = 0
            elif np.isscalar(self.lbc):
                # Dirichlet BC: set RHS to BC value
                rhs[0] = self.lbc
            elif isinstance(self.lbc, (list, tuple, np.ndarray)):
                # Vector BC: set RHS for each BC row
                lbc_array = np.atleast_1d(self.lbc)
                for i, bc_val in enumerate(lbc_array):
                    if i >= n + 1:
                        break
                    rhs[i] = bc_val

        # Adjust for right BC
        if self.rbc is not None:
            if isinstance(self.rbc, str):
                if self.rbc.lower() in ['neumann', 'dirichlet']:
                    # Homogeneous BC
                    rhs[n] = 0
            elif np.isscalar(self.rbc):
                # Dirichlet BC: set RHS to BC value
                rhs[n] = self.rbc
            elif isinstance(self.rbc, (list, tuple, np.ndarray)):
                # Vector BC: set RHS for each BC row
                rbc_array = np.atleast_1d(self.rbc)
                for i, bc_val in enumerate(rbc_array):
                    if n - i < 0:
                        break
                    rhs[n - i] = bc_val

        return rhs

    def eigs(self, k=6, sigma=None, n=None):
        """Compute eigenvalues and eigenfunctions.

        Solves the eigenvalue problem: L*u = λ*u

        Args:
            k (int, optional): Number of eigenvalues to compute. Default is 6.
            sigma (float or str, optional): Target for eigenvalues.
                - float: Find eigenvalues closest to sigma
                - 'SM': Smallest magnitude
                - 'LM': Largest magnitude
                - 'SR': Smallest real part
                - 'LR': Largest real part
                Default is 'SM' (smallest magnitude).
            n (int, optional): Number of collocation points.

        Returns:
            tuple: (vals, vecs) where vals are eigenvalues and vecs are eigenvectors

        Example:
            >>> L = Chebop.diff([0, np.pi], order=2)
            >>> L.lbc = 0; L.rbc = 0
            >>> vals, vecs = L.eigs(10)  # 10 smallest eigenvalues
        """
        import scipy.linalg

        # Determine number of points
        if n is None:
            n = max(128, 6 * k)  # Use more points for better accuracy

        # Build system matrix (without BCs initially)
        A = self._matrix_func(n).toarray()

        # For Dirichlet BCs, we solve interior problem
        # Remove boundary rows and columns
        has_dirichlet = False
        if self.lbc is not None and (np.isscalar(self.lbc) or
                                     (isinstance(self.lbc, str) and self.lbc.lower() == 'dirichlet')):
            has_dirichlet = True
        if self.rbc is not None and (np.isscalar(self.rbc) or
                                     (isinstance(self.rbc, str) and self.rbc.lower() == 'dirichlet')):
            has_dirichlet = True

        if has_dirichlet:
            # Solve interior eigenvalue problem (remove boundary points)
            A_interior = A[1:-1, 1:-1]
            vals_full, vecs_interior = scipy.linalg.eig(A_interior)
        else:
            # Apply BCs normally
            A_bc = self._apply_boundary_conditions(sparse.csr_matrix(A), n).toarray()
            vals_full, vecs_full = scipy.linalg.eig(A_bc)
            vecs_interior = vecs_full

        # Set default sigma
        if sigma is None:
            sigma = 'SM'

        # Sort eigenvalues based on sigma
        if isinstance(sigma, str):
            if sigma == 'SM':
                idx = np.argsort(np.abs(vals_full))
            elif sigma == 'LM':
                idx = np.argsort(np.abs(vals_full))[::-1]
            elif sigma == 'SR':
                idx = np.argsort(np.real(vals_full))
            elif sigma == 'LR':
                idx = np.argsort(np.real(vals_full))[::-1]
            else:
                idx = np.argsort(np.abs(vals_full))
        else:
            idx = np.argsort(np.abs(vals_full - sigma))

        # Filter out potential spurious eigenvalues (too large or NaN)
        valid_idx = []
        for i in idx:
            if not np.isnan(vals_full[i]) and not np.isinf(vals_full[i]):
                if len(valid_idx) == 0 or not np.isclose(vals_full[i], vals_full[valid_idx[-1]], rtol=1e-6):
                    valid_idx.append(i)
                    if len(valid_idx) >= k:
                        break

        # Return k unique eigenvalues/eigenvectors
        vals = vals_full[valid_idx[:k]]
        vecs = vecs_interior[:, valid_idx[:k]]

        return vals, vecs

    def expm(self, t, u0, n=None):
        """Compute operator exponential: exp(t*L)*u0.

        Solves the evolution equation: du/dt = L*u with initial condition u(0) = u0

        Args:
            t (float or array): Time value(s) at which to evaluate solution
            u0 (Chebfun or callable): Initial condition
            n (int, optional): Number of collocation points

        Returns:
            Chebfun: Solution u(t) = exp(t*L)*u0

        Example:
            >>> # Heat equation: u_t = u_xx on [-1, 1]
            >>> L = Chebop.diff([-1, 1], order=2)
            >>> L.lbc = 0; L.rbc = 0
            >>> u0 = chebfun(lambda x: np.exp(-20*x**2), [-1, 1])
            >>> u = L.expm(0.1, u0)  # Solution at t=0.1
        """
        from scipy.linalg import expm as matrix_expm

        # Convert u0 to chebfun if needed
        if callable(u0):
            u0 = Chebfun.initfun_adaptive(u0, domain=self.domain.support)

        # Determine number of points
        if n is None:
            n = max(64, len(u0.funs[0].coeffs) if len(u0.funs) > 0 else 64)

        # Get collocation points
        x = cheb_points_scaled(n, self.domain.support)

        # Build system matrix
        A = self._matrix_func(n)

        # Apply boundary conditions
        A = self._apply_boundary_conditions(A, n)

        # Sample initial condition
        u0_vals = u0(x)

        # Compute matrix exponential
        if np.isscalar(t):
            # Single time value
            expA = matrix_expm(t * A.toarray())
            u_vals = expA @ u0_vals
        else:
            # Multiple time values - return array of solutions
            raise NotImplementedError("Multiple time values not yet supported")

        # Convert to chebfun
        from .utilities import Interval
        from .algorithms import bary, barywts2

        def u_interp(x_eval):
            xk = x
            fk = u_vals
            vk = barywts2(len(xk))
            interval = Interval(*self.domain.support)
            y_eval = interval.invmap(x_eval)
            yk = interval.invmap(xk)
            return bary(y_eval, fk, yk, vk)

        result = Chebfun.initfun_fixedlen(u_interp, n + 1, domain=self.domain.support)
        return result

    def matrix(self, n=None):
        """Access the discretization matrix.

        Args:
            n (int, optional): Number of collocation points. If None, uses default.

        Returns:
            sparse matrix: Discretized operator matrix

        Example:
            >>> L = Chebop.diff([0, 1], order=2)
            >>> A = L.matrix(32)  # 33x33 differentiation matrix
        """
        if n is None:
            n = 64

        A = self._matrix_func(n)
        return A
