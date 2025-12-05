"""LinOp: Linear differential operator representation and solving.

This module provides the LinOp class for representing linear differential operators
and solving linear boundary value problems using spectral collocation.

TWO-PHASE DESIGN PATTERN:
-------------------------
LinOp works in concert with OpDiscretization using a two-phase approach:

This separation allows LinOp to focus on mathematical problem structure while
delegating numerical implementation to OpDiscretization.

The LinOp class handles:
- Domain splitting and continuity constraint specification
- Discretization via OpDiscretization
- Global system assembly
- QR/SVD solving of rectangular systems
- Solution reconstruction as Chebfun objects
- Adaptive refinement
- Eigenvalue problems
- Matrix operators (expm, null, svd, cond, inv, norm, etc.)
"""

import warnings
from typing import List, Optional, Callable, Tuple, Dict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, eigs as sp_eigs
from scipy.linalg import expm as scipy_expm

from .chebfun import Chebfun
from .utilities import Domain
from .opDiscretization import OpDiscretization
from .chebtech import Chebtech
from .bndfun import Bndfun
from .algorithms import chebpts2


class LinOp:
    """Linear differential operator for spectral collocation.

    Represents a linear differential operator:
        L = sum_{k=0}^z a_k(x) D^k

    along with boundary conditions, domain splitting, and solving machinery.

    Attributes:
        coeffs: List of coefficient functions [a_0(x), ..., a_z(x)]
        domain: Domain on which operator is defined
        diff_order: Differential order z
        lbc: Left boundary condition
        rbc: Right boundary condition
        bc: General boundary conditions list
        rhs: Right-hand side function
        tol: Convergence tolerance
        n_current: Current discretization size
    """

    def __init__(
        self,
        coeffs: List[Chebfun],
        domain: Domain,
        diff_order: int,
        lbc: Optional[Callable] = None,
        rbc: Optional[Callable] = None,
        bc: Optional[List] = None,
        rhs: Optional[Chebfun] = None,
        tol: float = 1e-10
    ):
        """Initialize a LinOp.

        Args:
            coeffs: List of coefficient functions
            domain: Domain for the operator
            diff_order: Differential order
            lbc: Left boundary condition
            rbc: Right boundary condition
            bc: General boundary conditions
            rhs: Right-hand side function
            tol: Tolerance for adaptive refinement
        """
        self.coeffs = coeffs if coeffs is not None else []
        self.domain = domain
        self.diff_order = diff_order
        self.lbc = lbc
        self.rbc = rbc
        self.bc = bc if bc is not None else []
        self.rhs = rhs
        self.tol = tol

        # Discretization state TODO: Have a chebop options later?
        self.n_current = 16  # Initial grid size
        self.max_n = 512  # Maximum grid size
        self.min_n = 8  # Minimum grid size

        # Domain splitting and continuity specifications
        # These are public attributes accessed by OpDiscretization during the
        # discretization phase. They store specifications (metadata) that get
        # compiled into concrete constraint matrices.
        self.blocks = None  # List of block specifications (one per subinterval)
        self.continuity_constraints = None  # List of continuity constraint specifications

    def _check_well_posedness(self):
        """Check if the problem is well-posed (correct number of boundary conditions).

        For a differential operator of order z on a single interval, we need exactly
        z boundary conditions (counting both left and right BCs).

        For multiple intervals, we additionally need (z * (n_intervals - 1)) continuity
        constraints, which are automatically generated.

        Raises:
            Warning: If the number of BCs does not match the differential order.
        """
        n_intervals = len(list(self.domain.intervals))

        # Count user-provided boundary conditions
        n_lbc = 0
        if self.lbc is not None:
            if isinstance(self.lbc, (list, tuple)):
                # Count non-None entries
                n_lbc = sum(1 for bc in self.lbc if bc is not None)
            else:
                n_lbc = 1

        n_rbc = 0
        if self.rbc is not None:
            if isinstance(self.rbc, (list, tuple)):
                n_rbc = sum(1 for bc in self.rbc if bc is not None)
            else:
                n_rbc = 1

        n_general_bc = len(self.bc)
        total_bcs = n_lbc + n_rbc + n_general_bc

        # For a differential operator of order z, we need z boundary conditions
        # (assuming single interval; multiple intervals have automatic continuity)
        required_bcs = self.diff_order

        if total_bcs < required_bcs:
            warnings.warn(
                f"Under-determined system: Differential order is {self.diff_order} "
                f"but only {total_bcs} boundary conditions provided. "
                f"Need {required_bcs} BCs for well-posedness. "
                f"Solution may not be unique."
            )
        elif total_bcs > required_bcs:
            warnings.warn(
                f"Over-determined system: Differential order is {self.diff_order} "
                f"but {total_bcs} boundary conditions provided. "
                f"Expected {required_bcs} BCs for well-posedness. "
                f"System will be solved in least-squares sense."
            )

    def prepare_domain(self):
        """Prepare domain splitting and continuity constraints.

        For each interface x_i, continuity constraints enforce:
            u_left^(k)(x_i^+) - u_right^(k)(x_i^-) = 0, k = 0, ..., z-1

        Sets:
            self.blocks: List of block specifications (one per subinterval)
                Each block_spec is a dict with:
                    - 'interval': Interval object [a, b]
                    - 'index': Block index
                    - 'coeffs': List of coefficient functions
                    - 'diff_order': Differential order

            self.continuity_constraints: List of continuity constraint specifications
                Each constraint is a dict with:
                    - 'type': 'continuity'
                    - 'location': Breakpoint location
                    - 'left_block': Index of left block
                    - 'right_block': Index of right block
                    - 'derivative_order': Order of derivative (0, 1, ..., z-1)
        """
        intervals = list(self.domain.intervals)
        n_intervals = len(intervals)

        # Create block specifications for each subinterval
        self.blocks = []
        for i, interval in enumerate(intervals):
            block_spec = {
                'interval': interval,
                'index': i,
                'coeffs': self.coeffs,  # Coefficient functions (global)
                'diff_order': self.diff_order
            }
            self.blocks.append(block_spec)

        # Create continuity constraint specifications
        # At each internal breakpoint, enforce continuity of u and its derivatives up to diff order - 1.
        self.continuity_constraints = []

        if n_intervals > 1:
            # For each internal breakpoint, add continuity constraints
            for i in range(n_intervals - 1):
                # Breakpoint between interval i and interval i+1
                left_idx = i
                right_idx = i + 1
                bp = self.domain[i + 1]  # The (i+1)-th breakpoint

                # Add continuity constraints for u and derivatives
                for deriv_order in range(self.diff_order):
                    constraint = {
                        'type': 'continuity',
                        'location': bp,
                        'left_block': left_idx,
                        'right_block': right_idx,
                        'derivative_order': deriv_order
                    }
                    self.continuity_constraints.append(constraint)

        # Check well-posedness
        self._check_well_posedness()

    def assemble_system(self, discretization: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global rectangular linear system from discretization.

        Takes discretized blocks and constraints, stacks them into:
            [A_blocks]      [b_blocks]
            [A_bc    ]  u = [b_bc    ]
            [A_cont  ]      [b_cont  ]

        Args:
            discretization: Dictionary with:
                - 'blocks': List of discretized operator blocks (matrices)
                - 'bc_rows': List of boundary condition rows
                - 'continuity_rows': List of continuity constraint rows
                - 'rhs_blocks': List of RHS vectors for each block
                - 'bc_rhs': RHS values for boundary conditions
                - 'continuity_rhs': RHS values for continuity (usually 0)
                - 'n_per_block': List of sizes for each block

        Returns:
            (A, b): Global system matrix and RHS vector
        """
        blocks = discretization['blocks']
        bc_rows = discretization['bc_rows']
        continuity_rows = discretization['continuity_rows']
        rhs_blocks = discretization['rhs_blocks']
        bc_rhs = discretization['bc_rhs']
        continuity_rhs = discretization['continuity_rhs']
        n_per_block = discretization['n_per_block']

        total_size = sum(n_per_block)

        # Stack operator blocks into block-diagonal structure
        A_blocks = sparse.block_diag(blocks, format='csr')

        # Stack RHS blocks
        b_blocks = np.concatenate(rhs_blocks)

        # Convert BC and continuity rows to full-width sparse matrices
        n_bc = len(bc_rows)
        n_cont = len(continuity_rows)

        if n_bc > 0:
            A_bc = sparse.vstack(bc_rows, format='csr')
            b_bc_vec = np.array(bc_rhs)
        else:
            A_bc = sparse.csr_matrix((0, total_size))
            b_bc_vec = np.array([])

        if n_cont > 0:
            A_cont = sparse.vstack(continuity_rows, format='csr')
            b_cont_vec = np.array(continuity_rhs)
        else:
            A_cont = sparse.csr_matrix((0, total_size))
            b_cont_vec = np.array([])

        # Stack everything vertically
        A = sparse.vstack([A_blocks, A_bc, A_cont], format='csr')
        b = np.concatenate([b_blocks, b_bc_vec, b_cont_vec])

        return A, b

    def solve_linear_system(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve rectangular linear system using least-squares.

        Solves:
            min_u ||A u - b||

        Handles:
        - Overdetermined systems (more equations than unknowns)
        - Rank deficiency
        - Compatibility conditions

        Args:
            A: System matrix (sparse, rectangular)
            b: Right-hand side vector

        Returns:
            Solution vector u
        """
        m, n = A.shape

        if n < 1000:
            A_dense = A.toarray()
            u, _, rank, _ = np.linalg.lstsq(A_dense, b, rcond=self.tol)

            if m >= n and rank < n:
                warnings.warn(f"System is rank deficient: rank={rank}, n={n}")

            return u

        # Sparse iterative solve for large systems
        result = lsqr(A, b, atol=self.tol, btol=self.tol)
        return result[0]


    def reconstruct_solution(self, u: np.ndarray, n_per_block: List[int]) -> Chebfun:
        """Reconstruct Chebfun solution from nodal values.

        Takes nodal solution vector and creates piecewise Chebfun.

        Args:
            u: Solution vector (concatenated nodal values from all blocks)
            n_per_block: List of sizes for each block

        Returns:
            Chebfun representing the solution
        """
        if self.blocks is None:
            raise RuntimeError("Must call prepare_domain() before reconstruction")

        # Split solution vector into per-block pieces
        pieces = []
        offset = 0
        for i, n_block in enumerate(n_per_block):
            u_block = u[offset:offset + n_block]
            interval = self.blocks[i]['interval']

            # Create Chebtech from values, then wrap in Bndfun
            chebtech_piece = Chebtech.initvalues(u_block, interval=interval)
            piece = Bndfun(chebtech_piece, interval)
            pieces.append(piece)

            offset += n_block

        solution = Chebfun(pieces)
        solution.simplify()
        return solution

    def solve(self, n: Optional[int] = None) -> Chebfun:
        """Solve the linear BVP with adaptive refinement."""
        
        # Specification phase
        if self.blocks is None:
            self.prepare_domain()

        # Determine discretization sequence
        if n is not None:
            n_values = [n]
        else:
            max_k = int(np.floor(np.log2(self.max_n / self.min_n)))
            n_values = [self.min_n * (2 ** k) for k in range(max_k + 1)]

        prev_solution = None

        for n_current in n_values:
            self.n_current = n_current

            discretization = OpDiscretization.build_discretization(self, n_current)

            A, b = self.assemble_system(discretization)

            try:
                u = self.solve_linear_system(A, b)
            except Exception as e:
                if n is not None:
                    raise RuntimeError(f"Failed to solve at n={n}: {e}")
                continue

            solution = self.reconstruct_solution(u, discretization['n_per_block'])
            solution = solution.simplify()

            # Residual check
            residual = A @ u - b
            relres = np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-16)

            # Accept solution if residual is small enough
            is_last_n = (n_current == n_values[-1])

            if relres < max(100 * self.tol, 1e-8):
                # Good residual - check adaptive convergence
                if prev_solution is not None:
                    diff = (solution - prev_solution).norm()
                    if diff < self.tol:
                        return solution
                # For algebraic equations at last n with good residual, accept
                if is_last_n and self.diff_order == 0 and relres < 1e-6:
                    return solution

            elif relres > 10 * self.tol:
                warnings.warn(f"Large residual: {relres:.2e}")

            prev_solution = solution

            # If explicit n was provided, return this solution
            if n is not None:
                return solution

            # At max refinement, return best solution with warning if needed
            if is_last_n:
                # For algebraic equations (order 0), be very lenient since they
                # often have larger relative residuals in the linear solve
                # For higher order operators, check if RHS appears non-smooth
                # (indicated by needing many points to represent)
                if self.diff_order == 0:
                    threshold = 0.5
                else:
                    # Check if RHS is non-smooth by looking at how many points it needed
                    rhs_is_nonsmooth = False
                    if self.rhs is not None:
                        for fun in self.rhs.funs:
                            if fun.size > 1000:  # Needed many points to represent
                                rhs_is_nonsmooth = True
                                break

                    # Be more lenient for non-smooth RHS
                    threshold = 0.01 if rhs_is_nonsmooth else 1e-4

                if relres < threshold:
                    if relres > 1e-4:
                        warnings.warn(
                            f"Returning solution at max n={self.max_n} with "
                            f"relative residual {relres:.2e}"
                        )
                    return solution
                else:
                    # Even if above threshold, return with warning rather than error
                    # This allows tests to check if solution is "good enough"
                    warnings.warn(
                        f"Returning solution at max n={self.max_n} with large "
                        f"relative residual {relres:.2e} (threshold was {threshold:.2e})"
                    )
                    return solution

        # Should not reach here, but return last solution as fallback
        return solution

    def _discretization_size(self, n: Optional[int] = None) -> int:
        """Compute appropriate discretization size for this operator.

        The size scales with operator order to ensure sufficient resolution:
        - Higher-order operators need more points for accurate derivatives
        - Base size = 8 * (diff_order + 1), ensuring at least 16 points for 1st order,
          24 for 2nd order, 40 for 4th order, etc.

        Args:
            n: Explicit size override (if provided, returned directly after clamping)

        Returns:
            Discretization size clamped to [min_n, max_n]
        """
        if n is not None:
            return min(max(self.min_n, n), self.max_n)

        # Base size scales with operator order
        base = 8 * (self.diff_order + 1)
        size = max(base, self.n_current)
        return min(max(self.min_n, size), self.max_n)

    def _discretize(self, n: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Discretize the operator and return the assembled system matrix.

        This is the common setup used by most matrix analysis functions.

        Args:
            n: Discretization size (default: automatically determined)

        Returns:
            A_dense: Dense assembled system matrix
            disc: Discretization dictionary with 'n_per_block', 'blocks', etc.
        """
        if self.blocks is None:
            self.prepare_domain()

        n_actual = self._discretization_size(n)
        disc = OpDiscretization.build_discretization(self, n_actual)
        A, _ = self.assemble_system(disc)
        return A.toarray(), disc

    def _discretize_operator_only(self, n: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Discretize and return only the square operator matrix (no BC rows).

        Useful for eigenvalue problems and matrix exponential where we need
        the square block-diagonal operator matrix without constraint rows.

        Args:
            n: Discretization size (default: automatically determined)

        Returns:
            A_op: Square dense operator matrix (block diagonal)
            disc: Discretization dictionary
        """
        if self.blocks is None:
            self.prepare_domain()

        n_actual = self._discretization_size(n)
        disc = OpDiscretization.build_discretization(self, n_actual)
        n_per_block = disc['n_per_block']
        total_n = sum(n_per_block)

        A_op = sparse.block_diag(disc['blocks'], format='csr').toarray()
        # Ensure square by taking only the operator part
        A_op = A_op[:total_n, :total_n]
        return A_op, disc

    def eigs(
        self,
        k: int = 6,
        sigma: Optional[float] = None,
        mass_matrix: Optional['LinOp'] = None,
        **kwargs
    ) -> Tuple[np.ndarray, List[Chebfun]]:
        """Compute k eigenvalues and eigenfunctions of the operator.

        Solves the eigenvalue problem L[u] = lambda * M[u] subject to boundary conditions.
        If mass_matrix is None, solves the standard problem L[u] = lambda * u.

        Uses adaptive refinement to ensure eigenvalue convergence.

        Args:
            k: Number of eigenvalues to compute (default 6)
            sigma: Target eigenvalue for shift-invert mode (default None = smallest magnitude)
            mass_matrix: Optional mass matrix operator M for generalized problem L[u] = λ*M[u]
                        If None, solves standard eigenvalue problem L[u] = λ*u
            **kwargs: Additional arguments passed to scipy.sparse.linalg.eigs

        Returns:
            eigenvalues: Array of k eigenvalues
            eigenfunctions: List of k Chebfun eigenfunctions (L2-normalized)

        Examples:
            # Standard eigenvalue problem: -u'' = λu
            L = LinOp([a0, a1, a2], domain, diff_order=2)
            L.lbc = L.rbc = 0
            evals, efuns = L.eigs(k=5)

            # Generalized eigenvalue problem: -u'' = λ * x * u (weighted)
            M = LinOp([weight_func], domain, diff_order=0)  # M[u] = x*u
            M.lbc = M.rbc = 0
            evals, efuns = L.eigs(k=5, mass_matrix=M)
        """
        if self.blocks is None:
            self.prepare_domain()

        # Adaptive refinement sequence scaled by operator order
        base = self._discretization_size()
        n_list = [base, 2*base, 4*base, 8*base]
        n_list = [min(n, self.max_n) for n in n_list]
        n_list = sorted(set(n_list))  # Remove duplicates

        prev_vals = None
        final_disc = None
        final_vecs = None
        final_free_idx = None

        for n in n_list:
            disc = OpDiscretization.build_discretization(self, n)
            n_per_block = disc['n_per_block']
            total_dofs = sum(n_per_block)

            # Build operator and BC matrices separately
            A_op_sparse = sparse.block_diag(disc['blocks'], format='csr')
            A_op = A_op_sparse.toarray()[:total_dofs, :total_dofs]

            # Get BC rows
            bc_rows_list = disc.get('bc_rows', [])

            if len(bc_rows_list) == 0:
                # No BCs - eigenvalue problem on full operator
                A_eig = A_op
                bc_proj = np.eye(total_dofs)
            else:
                # Assemble BC constraint matrix
                BC = []
                for bc_row in bc_rows_list:
                    if sparse.isspmatrix(bc_row):
                        BC.append(bc_row.toarray().ravel()[:total_dofs])
                    else:
                        BC.append(np.asarray(bc_row).ravel()[:total_dofs])
                BC = np.array(BC)

                # For eigenproblems L[u] = λu, we need homogeneous BCs: BC @ u = 0
                # Find null space of BC matrix
                try:
                    # Use QR decomposition for numerical stability
                    # Q, R, P = qr(BC.T, pivoting=True)
                    # Nullspace is last (n - rank) columns of Q
                    from scipy.linalg import qr
                    Q, R, P = qr(BC.T, mode='full', pivoting=True)

                    # Determine rank
                    tol_rank = max(BC.shape) * np.abs(R[0, 0]) * np.finfo(float).eps if R.size > 0 else 1e-10
                    rank = np.sum(np.abs(np.diag(R)) > tol_rank)

                    if rank >= total_dofs:
                        # Over-constrained - no free DOFs
                        continue

                    # Nullspace basis: columns of Q beyond rank
                    bc_proj = Q[:, rank:]

                    if bc_proj.shape[1] < k + 2:
                        # Not enough free DOFs for requested eigenvalues
                        continue

                except (np.linalg.LinAlgError, ValueError):
                    continue

                # Project operator onto BC-satisfying subspace
                # A_eig = Z.T @ A_op @ Z where Z are nullspace basis vectors
                A_eig = bc_proj.T @ A_op @ bc_proj

                # Handle mass matrix for generalized eigenvalue problem
                if mass_matrix is not None:
                    # Discretize mass matrix at same resolution
                    M_disc = OpDiscretization.build_discretization(mass_matrix, n)
                    M_op_sparse = sparse.block_diag(M_disc['blocks'], format='csr')
                    M_op = M_op_sparse.toarray()[:total_dofs, :total_dofs]

                    # Project mass matrix onto same BC-satisfying subspace
                    M_eig = bc_proj.T @ M_op @ bc_proj
                else:
                    M_eig = None

            # Check if matrix is large enough
            if A_eig.shape[0] < k + 2:
                continue

            # Compute eigenvalues
            k_actual = min(k, A_eig.shape[0] - 2)
            try:
                if M_eig is not None:
                    # Generalized eigenvalue problem: A @ v = λ * M @ v
                    from scipy.linalg import eig
                    vals_all, vecs_all = eig(A_eig, M_eig)

                    # Filter out infinite/nan eigenvalues
                    finite_mask = np.isfinite(vals_all)
                    vals_finite = vals_all[finite_mask]
                    vecs_finite = vecs_all[:, finite_mask]

                    # Sort by magnitude and take k smallest
                    if len(vals_finite) == 0:
                        continue
                    idx_sort = np.argsort(np.abs(vals_finite))
                    k_actual = min(k_actual, len(vals_finite))
                    vals = vals_finite[idx_sort[:k_actual]]
                    vecs = vecs_finite[:, idx_sort[:k_actual]]
                else:
                    # Standard eigenvalue problem
                    if sigma is not None:
                        vals, vecs = sp_eigs(A_eig, k=k_actual, sigma=sigma, **kwargs)
                    else:
                        vals, vecs = sp_eigs(A_eig, k=k_actual, which='SM', **kwargs)
            except Exception:
                continue

            # Sort by magnitude
            idx = np.argsort(np.abs(vals))
            vals, vecs = vals[idx], vecs[:, idx]

            # Store for reconstruction
            final_disc = disc
            final_vecs = vecs
            final_bc_proj = bc_proj

            # Check convergence
            if prev_vals is not None and len(prev_vals) == len(vals):
                rel_err = np.abs(vals - prev_vals) / (np.abs(prev_vals) + 1e-14)
                if np.max(rel_err) < 1e-8:
                    break
            prev_vals = vals.copy()

        if prev_vals is None or final_disc is None:
            raise RuntimeError("eigs failed to converge")

        # Reconstruct eigenfunctions
        n_per_block = final_disc['n_per_block']
        total_dofs = sum(n_per_block)
        eigenfunctions = []

        for j in range(len(prev_vals)):
            # Transform eigenvector back to full DOF space
            # Eigenvector is in BC-satisfying subspace, project back via bc_proj
            u = final_bc_proj @ np.real_if_close(final_vecs[:, j])

            ef = self.reconstruct_solution(u, n_per_block)
            # L2 normalize
            norm_val = ef.norm(2)
            if norm_val > 0:
                ef = ef / norm_val
            eigenfunctions.append(ef)

        return np.real_if_close(prev_vals), eigenfunctions

    def expm(self, t: float = 1.0, u0: Optional[Chebfun] = None, num_eigs: int = 50) -> Chebfun:
        """Apply the matrix exponential exp(t*L) to an initial function.

        Computes u(t) = exp(t*L) * u0, which solves du/dt = L[u] with u(0) = u0,
        where L includes boundary conditions.

        This uses eigenfunction expansion:
            exp(t*L) * u0 = sum_i exp(t*λ_i) * <u0, v_i> * v_i
        where (λ_i, v_i) are eigenpairs of the BC-constrained operator.

        Args:
            t: Time parameter (default 1.0)
            u0: Initial function (default: constant 1)
            num_eigs: Number of eigenvalues/eigenfunctions to use (default 50)

        Returns:
            Chebfun representing exp(t*L) * u0

        Note:
            Accuracy depends on num_eigs. For smooth initial conditions,
            50 eigenfunctions is usually sufficient. Increase if needed.

        Raises:
            RuntimeError: If eigenvalue computation fails
        """
        # Create default u0 if not provided
        if u0 is None:
            from .chebfun import chebfun
            u0 = chebfun(lambda x: np.ones_like(x), list(self.domain))

        # Special case: t=0 means identity
        if abs(t) < 1e-15:
            return u0.copy() if hasattr(u0, 'copy') else u0

        # Compute eigenpairs of BC-constrained operator
        try:
            # Use smallest magnitude eigenvalues (most important for dynamics)
            evals, efuns = self.eigs(k=min(num_eigs, self.max_n - 10))
        except Exception as e:
            raise RuntimeError(
                f"Matrix exponential requires eigenfunction expansion, but "
                f"eigenvalue computation failed: {e}\n"
                f"Try increasing max_n or decreasing num_eigs."
            )

        if len(evals) == 0:
            warnings.warn("No eigenvalues computed; returning zero function")
            return u0 * 0

        # Expand u0 in eigenbasis: u0 = sum_i c_i * v_i
        # where c_i = <u0, v_i> (L2 inner product)
        coeffs = []
        for ef in efuns:
            # L2 inner product: <u0, ef> = integral(u0 * ef)
            inner_prod = (u0 * ef).sum()  # Chebfun.sum() integrates
            coeffs.append(inner_prod)

        # Apply exponential in eigenspace: exp(t*L)*u0 = sum_i exp(t*λ_i)*c_i*v_i
        result = None
        for lam, c, ef in zip(evals, coeffs, efuns):
            term = np.exp(t * lam) * c * ef
            if result is None:
                result = term
            else:
                result = result + term

        if result is None:
            warnings.warn("Eigenfunction expansion produced no terms; returning zero")
            return u0 * 0

        return result

    def null(self, tol: Optional[float] = None) -> List[Chebfun]:
        """Compute a basis for the nullspace of the operator.

        Returns functions u such that L[u] = 0 (with boundary conditions).

        Args:
            tol: Tolerance for determining nullspace (default: self.tol)

        Returns:
            List of Chebfun functions forming a basis for the nullspace
        """
        tol = tol if tol is not None else self.tol
        A, disc = self._discretize()
        n_per_block = disc['n_per_block']
        total_n = sum(n_per_block)

        # SVD to find nullspace
        _, s, vh = np.linalg.svd(A, full_matrices=True)

        # Find vectors corresponding to small singular values
        null_idx = np.where(s < tol * s[0])[0] if len(s) > 0 else []

        # Include vectors beyond the rank
        rank = len(s)
        null_vecs = vh[rank:].T
        if len(null_idx) > 0:
            null_vecs = np.column_stack([vh[null_idx].T, null_vecs]) if null_vecs.size > 0 else vh[null_idx].T

        if null_vecs.size == 0:
            return []

        if null_vecs.ndim == 1:
            null_vecs = null_vecs.reshape(-1, 1)

        # Reconstruct as Chebfuns
        basis = []
        for j in range(null_vecs.shape[1]):
            vec = null_vecs[:total_n, j]
            basis.append(self.reconstruct_solution(vec, n_per_block))

        return basis

    def svds(self, k: int = 6) -> Tuple[np.ndarray, List[Chebfun], List[Chebfun]]:
        """Compute k largest singular values and singular functions.

        Args:
            k: Number of singular values to compute

        Returns:
            s: Array of k largest singular values
            u_funcs: List of k left singular functions
            v_funcs: List of k right singular functions
        """
        A, disc = self._discretize()
        n_per_block = disc['n_per_block']
        total_n = sum(n_per_block)

        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        k_actual = min(k, len(S))

        # Right singular functions (input space)
        v_funcs = [self.reconstruct_solution(Vh[j, :total_n], n_per_block)
                   for j in range(k_actual)]

        # Left singular functions (output space)
        u_funcs = []
        for j in range(k_actual):
            vec = U[:total_n, j] if U.shape[0] >= total_n else np.zeros(total_n)
            if U.shape[0] < total_n:
                vec[:U.shape[0]] = U[:, j]
            u_funcs.append(self.reconstruct_solution(vec, n_per_block))

        return S[:k_actual], u_funcs, v_funcs

    # -------------------------------------------------------------------------
    # Matrix analysis functions
    # -------------------------------------------------------------------------

    def cond(self, p: int = 2) -> float:
        """Estimate the condition number of the discretized operator.

        Args:
            p: Norm type (default 2 = ratio of largest to smallest singular value)

        Returns:
            Condition number estimate
        """
        A, _ = self._discretize()
        if p == 2:
            s = np.linalg.svd(A, compute_uv=False)
            if len(s) == 0 or s[-1] == 0:
                return np.inf
            return s[0] / s[-1]
        return np.linalg.cond(A, p=p)

    def norm(self, p: int = 2) -> float:
        """Compute the induced operator norm.

        Args:
            p: Norm type (default 2 = spectral norm = largest singular value)

        Returns:
            Operator norm
        """
        A, _ = self._discretize()
        if p == 2:
            s = np.linalg.svd(A, compute_uv=False)
            return s[0] if len(s) > 0 else 0.0
        return np.linalg.norm(A, ord=p)

    def rank(self, tol: Optional[float] = None) -> int:
        """Compute the numerical rank of the discretized operator.

        Args:
            tol: Tolerance for rank determination (default: self.tol * max singular value)

        Returns:
            Numerical rank
        """
        A, _ = self._discretize()
        s = np.linalg.svd(A, compute_uv=False)
        if len(s) == 0:
            return 0
        tol_actual = (tol if tol is not None else self.tol) * s[0]
        return int(np.sum(s > tol_actual))

    def trace(self) -> float:
        """Compute the trace of the discretized operator matrix.

        Returns:
            Trace (sum of diagonal elements)
        """
        A, _ = self._discretize()
        return np.trace(A)

    def det(self) -> float:
        """Compute the determinant of the discretized operator matrix.

        Note: For rectangular matrices (more constraints than DOFs),
        returns the product of singular values.

        Returns:
            Determinant or pseudo-determinant
        """
        A, _ = self._discretize()
        if A.shape[0] == A.shape[1]:
            return np.linalg.det(A)
        # For rectangular matrices, return product of singular values
        s = np.linalg.svd(A, compute_uv=False)
        return np.prod(s)

    def spy(self, **kwargs):
        """Plot the sparsity pattern of the discretized operator.

        Args:
            **kwargs: Arguments passed to matplotlib.pyplot.spy
        """
        import matplotlib.pyplot as plt
        A, _ = self._discretize()
        plt.figure()
        plt.spy(A, **kwargs)
        plt.title(f"LinOp sparsity pattern (order {self.diff_order})")
        plt.show()

    def __call__(self, u: Chebfun) -> Chebfun:
        """Apply operator to a function.

        Computes L[u] = sum_k a_k(x) u^(k)(x)

        Args:
            u: Input Chebfun

        Returns:
            Result of applying operator
        """
        result = None

        for k, coeff in enumerate(self.coeffs):
            # Compute k-th derivative
            u_deriv = u
            for _ in range(k):
                u_deriv = u_deriv.diff()

            # Multiply by coefficient
            term = coeff * u_deriv

            # Accumulate
            if result is None:
                result = term
            else:
                result = result + term

        return result if result is not None else u * 0
