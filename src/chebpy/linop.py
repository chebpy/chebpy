"""LinOp: Linear differential operator representation and solving.

This module provides the LinOp class for representing linear differential operators
and solving linear boundary value problems using spectral collocation.

The LinOp class handles:
- Domain splitting and continuity constraint specification
- Discretization via op_discretization
- Global system assembly
- QR/SVD solving of rectangular systems
- Solution reconstruction as Chebfun objects
- Adaptive refinement
- Eigenvalue problems
- Matrix operators (expm, null, svd, cond, inv, norm, etc.)
"""

import logging
import traceback
import warnings
from collections.abc import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, eig, lu_factor, lu_solve, qr
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.special import comb

from .algorithms import standard_chop
from .bndfun import Bndfun
from .chebfun import Chebfun
from .chebtech import Chebtech
from .linop_diagnostics import diagnose_linop
from .op_discretization import OpDiscretization
from .order_detection_ast import OrderTracerAST
from .sparse_utils import sparse_to_dense
from .spectral import cheb_points_scaled, ultraspherical_solve
from .trigtech import Trigtech
from .utilities import Domain, Interval

# Initialize logger
logger = logging.getLogger(__name__)


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
        coeffs: list[Chebfun],
        domain: Domain,
        diff_order: int,
        lbc: Callable | None = None,
        rbc: Callable | None = None,
        bc: list | None = None,
        rhs: Chebfun | None = None,
        tol: float = 1e-10,
        point_constraints: list[dict] | None = None,
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
            point_constraints: List of interior point constraints. Each constraint
                is a dict with keys:
                - 'location': x-coordinate where constraint applies
                - 'derivative_order': order of derivative (0 for value)
                - 'value': value the derivative should equal
        """
        self.coeffs = coeffs if coeffs is not None else []
        self.domain = domain
        self.diff_order = diff_order
        self.lbc = lbc
        self.rbc = rbc
        self.bc = bc if bc is not None else []
        self.rhs = rhs
        self.tol = tol
        self.point_constraints = point_constraints if point_constraints is not None else []

        # Validate coefficients length matches diff_order
        # Note: For composed operators (e.g., in generalized eigenvalue problems),
        # the coefficient list may be longer than diff_order+1. We only warn if
        # the list is shorter (which could indicate missing terms).
        # None entries in coeffs represent zero coefficients (e.g., [a0, None, a2] for u'' + u with no u' term).
        if diff_order is not None and coeffs is not None:
            expected_len = diff_order + 1
            # Count total list length (including None entries for zero coefficients)
            actual_len = len(coeffs)
            if actual_len < expected_len and actual_len != 0:
                warnings.warn(
                    f"Coefficient list length ({actual_len}) is less than diff_order+1 ({expected_len}). "
                    f"This may cause incorrect results. Expected at least {expected_len} coefficients "
                    f"for order-{diff_order} operator. Use None for zero coefficients "
                    f"(e.g., [a0, None, a2] for a2*u'' + a0*u)."
                )

        # Performance optimization flags
        # Set by Newton iteration to avoid expensive diagnostics on each linearization
        self._disable_diagnostics = False

        # Integral constraints
        # Can be a single constraint dict or a list of constraint dicts
        # Each dict has {'weight': chebfun or None, 'value': float}
        # None weight means ∫u dx, otherwise ∫weight*u dx
        self.integral_constraint = None

        # Discretization state TODO: Have a Chebop options later?
        self.n_current = 16  # Initial grid size

        # Cap max_n based on differential order to avoid catastrophic ill-conditioning
        # High-order differentiation matrices (D^k for k >= 4) become severely
        # ill-conditioned at large n when computed as matrix powers.
        # Condition numbers: D^4 at n=128 has cond ~ 1.5e20, at n=256 has cond ~ 2.5e22
        if diff_order is not None and diff_order >= 4:
            self.max_n = 128  # Conservative limit for 4th+ order operators
        else:
            self.max_n = 4096  # Standard limit for spectral collocation

        self.min_n = 8  # Minimum grid size

        # Current solution for Newton iteration (used to linearize callable BCs)
        self.u_current = None

        # Domain splitting and continuity specifications
        # These are public attributes accessed by op_discretization during the
        # discretization phase. They store specifications (metadata) that get
        # compiled into concrete constraint matrices.
        self.blocks = None  # List of block specifications (one per subinterval)
        self.continuity_constraints = None  # List of continuity constraint specifications

    @property
    def is_periodic(self) -> bool:
        """Check if this operator uses periodic boundary conditions.

        Returns:
            True if bc is the string "periodic", False otherwise.
        """
        return isinstance(self.bc, str) and self.bc.lower() == "periodic"

    @staticmethod
    def _filter_eigenvalues(vals_all, vecs_all, k, sigma=None):
        """Filter and sort eigenvalues from dense solver output.

        Filters out infinite/NaN eigenvalues and returns the k eigenvalues
        closest to the target (sigma if provided, else smallest by magnitude).

        Args:
            vals_all: All eigenvalues from solver
            vecs_all: All eigenvectors from solver
            k: Number of eigenvalues to return
            sigma: Target value (if None, use smallest magnitude)

        Returns:
            tuple: (vals, vecs) - filtered eigenvalues and eigenvectors, or (None, None) if no finite eigenvalues
        """
        # Filter out infinite/nan eigenvalues
        finite_mask = np.isfinite(vals_all)
        vals_finite = vals_all[finite_mask]
        vecs_finite = vecs_all[:, finite_mask]

        # Return None if no finite eigenvalues
        if len(vals_finite) == 0:
            return None, None

        # Sort by magnitude or distance from sigma and take k smallest/nearest
        if sigma is not None and sigma != 0:
            idx_sort = np.argsort(np.abs(vals_finite - sigma))
        else:
            idx_sort = np.argsort(np.abs(vals_finite))

        k_actual = min(k, len(vals_finite))
        vals = vals_finite[idx_sort[:k_actual]]
        vecs = vecs_finite[:, idx_sort[:k_actual]]

        return vals, vecs

    @staticmethod
    def _assemble_constraint_rows(rows, rhs, total_size):
        """Assemble constraint rows into sparse matrix and RHS vector.

        Args:
            rows: List of constraint row matrices
            rhs: List of constraint RHS values
            total_size: Total number of columns in assembled matrix

        Returns:
            tuple: (A, b) - sparse matrix and RHS vector (empty if no constraints)
        """
        n = len(rows)
        if n > 0:
            A = sparse.vstack(rows, format="csr")
            b_vec = np.array(rhs)
        else:
            A = sparse.csr_matrix((0, total_size))
            b_vec = np.array([])
        return A, b_vec

    def _count_bc_conditions(self, bc):
        """Count how many conditions a BC specification represents.

        For callable BCs that return lists (e.g., lambda u: [u, u.diff()]), we need
        to trace the BC to determine how many conditions it represents.

        Args:
            bc: Boundary condition (can be callable, list, tuple, or None)

        Returns:
            int: Number of BC conditions
        """
        if bc is None:
            return 0

        if isinstance(bc, (list, tuple)):
            return sum(1 for item in bc if item is not None)

        if callable(bc):
            try:
                # Trace the BC to see if it returns a list
                tracer = OrderTracerAST("u", domain=self.domain)
                traced_result = bc(tracer)

                # If BC returns a list/tuple, count the conditions
                if isinstance(traced_result, (list, tuple)):
                    return len(traced_result)
                else:
                    return 1
            except Exception:
                # If tracing fails, assume single condition
                return 1

        # For other BC types (e.g., constants), count as single condition
        return 1

    def _check_well_posedness(self):
        """Check if the problem is well-posed (correct number of boundary conditions).

        For a differential operator of order z on a single interval, we need exactly
        z boundary conditions (counting both left and right BCs).

        For multiple intervals, we additionally need (z * (n_intervals - 1)) continuity
        constraints, which are automatically generated.

        Raises:
            Warning: If the number of BCs does not match the differential order.
        """
        len(list(self.domain.intervals))

        # Count user-provided boundary conditions using proper tracing
        n_lbc = self._count_bc_conditions(self.lbc)
        n_rbc = self._count_bc_conditions(self.rbc)

        # Handle periodic BCs separately
        if self.is_periodic:
            # Periodic BCs provide diff_order constraints
            n_general_bc = self.diff_order
        else:
            n_general_bc = len(self.bc) if self.bc else 0

        # Count integral constraints
        n_integral = 0
        if self.integral_constraint is not None:
            if isinstance(self.integral_constraint, dict):
                n_integral = 1
            else:
                n_integral = len(self.integral_constraint)

        total_bcs = n_lbc + n_rbc + n_general_bc + n_integral

        # For a differential operator of order z, we need z boundary conditions
        # (assuming single interval; multiple intervals have automatic continuity)
        required_bcs = self.diff_order if self.diff_order is not None else 0

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

        For periodic BCs, enforces:
            u^(k)(a) - u^(k)(b) = 0, k = 0, ..., z-1

        Sets:
            self.blocks: List of block specifications (one per subinterval)
                Each block_spec is a dict with:
                    - 'interval': Interval object [a, b]
                    - 'index': Block index
                    - 'coeffs': List of coefficient functions
                    - 'diff_order': Differential order

            self.continuity_constraints: List of continuity constraint specifications
                Each constraint is a dict with:
                    - 'type': 'continuity' or 'periodic'
                    - 'location': Breakpoint location
                    - 'left_block': Index of left block
                    - 'right_block': Index of right block
                    - 'derivative_order': Order of derivative (0, 1, ..., z-1)
        """
        intervals = list(self.domain.intervals)
        n_intervals = len(intervals)

        # Validate that periodic BCs are not mixed with lbc/rbc
        if self.is_periodic:
            if self.lbc is not None or self.rbc is not None:
                raise ValueError(
                    "Periodic boundary conditions cannot be used together with lbc or rbc. "
                    "Periodic BCs already constrain both endpoints."
                )

        # Periodic BCs require single interval
        if self.is_periodic and n_intervals > 1:
            raise ValueError("Periodic boundary conditions require a single interval domain")

        # Create block specifications for each subinterval
        self.blocks = []
        for i, interval in enumerate(intervals):
            block_spec = {
                "interval": interval,
                "index": i,
                "coeffs": self.coeffs,  # Coefficient functions (global)
                "diff_order": self.diff_order,
            }
            self.blocks.append(block_spec)

        # Create continuity constraint specifications
        # At each internal breakpoint, enforce continuity of u and its derivatives up to diff order - 1.
        self.continuity_constraints = []

        if self.is_periodic:
            # For periodic BCs, enforce u^(k)(a) = u^(k)(b) for k = 0, ..., diff_order-1
            # This is similar to continuity but connects endpoints of the single interval
            interval = intervals[0]
            a, b = interval[0], interval[-1]

            for deriv_order in range(self.diff_order):
                constraint = {
                    "type": "periodic",
                    "location_left": a,
                    "location_right": b,
                    "block": 0,  # Single block for periodic
                    "derivative_order": deriv_order,
                }
                self.continuity_constraints.append(constraint)
        elif n_intervals > 1:
            # For each internal breakpoint, add continuity constraints
            for i in range(n_intervals - 1):
                # Breakpoint between interval i and interval i+1
                left_idx = i
                right_idx = i + 1
                bp = self.domain[i + 1]  # The (i+1)-th breakpoint

                # Add continuity constraints for u and derivatives
                for deriv_order in range(self.diff_order):
                    constraint = {
                        "type": "continuity",
                        "location": bp,
                        "left_block": left_idx,
                        "right_block": right_idx,
                        "derivative_order": deriv_order,
                    }
                    self.continuity_constraints.append(constraint)

        # Check well-posedness
        self._check_well_posedness()

    def _build_discretization_from_jacobian(self, n: int, bc_enforcement: str = "append") -> dict:
        """Build discretization dictionary using AdChebfun Jacobian computer.

        This method is used when AdChebfun can compute the Jacobian matrix
        at any requested discretization size. This avoids coefficient explosion
        by working directly with sampled operators instead of constructing
        explicit coefficient functions.

        Args:
            n: Discretization size to use
            bc_enforcement: BC enforcement strategy ('append' or 'replace')

        Returns:
            Discretization dictionary compatible with assemble_system()
        """
        # Compute the Jacobian matrix at the requested discretization size
        # Note: AdChebfun uses n to mean n+1 collocation points
        J = self._jacobian_computer(n)

        # Convert to sparse matrix if needed (AdChebfun returns dense numpy array)
        if not sparse.issparse(J):
            J = sparse.csr_matrix(J)

        # The Jacobian should be square with size matching n+1 collocation points
        actual_size = J.shape[0]
        if J.shape[0] != J.shape[1]:
            raise ValueError(f"Jacobian matrix is not square: {J.shape}")

        logger.debug("=" * 60)
        logger.debug("AdChebfun Discretization Debug (n=%d)", n)
        logger.debug("=" * 60)
        logger.debug("Jacobian matrix shape: %s", J.shape)
        logger.debug("Actual size (n+1 collocation points): %d", actual_size)

        # For single-interval problems, use the matrix as a single block
        # For multi-interval, this approach needs extension (TODO)
        n_intervals = len(list(self.domain.intervals))
        if n_intervals > 1:
            raise NotImplementedError("AdChebfun-based solving not yet supported for multi-interval problems")

        # Build BC rows using op_discretization
        # We still need to discretize BCs normally
        logger.debug("Calling OpDiscretization._build_bc_rows...")
        try:
            bc_rows, bc_rhs = OpDiscretization._build_bc_rows(self, [actual_size], self.u_current)
            logger.debug("  BC discretization succeeded")
        except Exception as e:
            logger.error("  BC discretization FAILED: %s: %s", type(e).__name__, e)
            traceback.print_exc()
            raise

        logger.debug("Boundary Conditions:")
        logger.debug("  Number of BC rows: %d", len(bc_rows))
        if bc_rows:
            for i, row in enumerate(bc_rows):
                if hasattr(row, "shape"):
                    logger.debug("  BC row %d shape: %s", i, row.shape)
                elif hasattr(row, "size"):
                    logger.debug("  BC row %d size: %s", i, row.size)
                else:
                    logger.debug(
                        "  BC row %d type: %s, len: %s", i, type(row), len(row) if hasattr(row, "__len__") else "N/A"
                    )
        logger.debug("  BC RHS length: %s", len(bc_rhs) if hasattr(bc_rhs, "__len__") else "scalar")

        # Build RHS: sample the RHS function at collocation points
        # For Newton iteration, self.rhs is the negative residual (-residual) set by Chebop
        # We need to evaluate this chebfun at the collocation points
        logger.debug("Building RHS...")

        try:
            # Check if we have a residual evaluator function (from AdChebfun path)
            if hasattr(self, "_residual_evaluator"):
                logger.debug("  Getting collocation points for residual evaluator...")
                x_pts = cheb_points_scaled(n, Interval(self.domain.support[0], self.domain.support[1]))

                logger.debug("  Calling residual evaluator at %d points...", len(x_pts))

                # Use the residual evaluator to get accurate RHS values at collocation points
                # This evaluates N(u) directly at the requested points rather than
                # using a coarse chebfun representation
                rhs_vals = self._residual_evaluator(x_pts)
                b_block = rhs_vals

                logger.debug("RHS Construction (from residual evaluator):")
                logger.debug("  x_pts length: %d", len(x_pts))
                logger.debug("  rhs_vals shape: %s", rhs_vals.shape if hasattr(rhs_vals, "shape") else len(rhs_vals))
                logger.debug("  rhs_vals min/max: [%.2e, %.2e]", np.min(rhs_vals), np.max(rhs_vals))

            elif self.rhs is not None:
                logger.debug("  Getting collocation points...")
                x_pts = cheb_points_scaled(n, Interval(self.domain.support[0], self.domain.support[1]))

                logger.debug("  Evaluating RHS chebfun at %d points...", len(x_pts))

                # The RHS is a chebfun (set to -residual by Newton iteration)
                # Evaluate it at collocation points
                rhs_vals = self.rhs(x_pts)
                b_block = rhs_vals

                logger.debug("RHS Construction (from chebfun):")
                logger.debug("  x_pts length: %d", len(x_pts))
                logger.debug("  rhs_vals shape: %s", rhs_vals.shape if hasattr(rhs_vals, "shape") else len(rhs_vals))
                logger.debug("  rhs_vals min/max: [%.2e, %.2e]", np.min(rhs_vals), np.max(rhs_vals))
            else:
                b_block = np.zeros(actual_size)
                logger.debug("RHS Construction (zero):")
                logger.debug("  Using zero RHS with size %d", actual_size)

            logger.debug("  RHS construction succeeded")
        except Exception as e:
            logger.error("  RHS construction FAILED: %s: %s", type(e).__name__, e)
            traceback.print_exc()
            raise

        # Create discretization dictionary
        discretization = {
            "blocks": [J],  # Single block with Jacobian matrix
            "bc_rows": bc_rows,
            "bc_rhs": bc_rhs,
            "continuity_rows": [],  # No continuity for single interval
            "continuity_rhs": [],
            "integral_rows": [],  # TODO: Handle integral constraints
            "integral_rhs": [],
            "point_rows": [],  # TODO: Handle point constraints
            "point_rhs": [],
            "rhs_blocks": [b_block],
            "n_per_block": [actual_size],  # Use actual matrix size, not n
            "bc_enforcement": bc_enforcement,
        }

        logger.debug("Discretization Dictionary Summary:")
        logger.debug("  blocks[0] shape: %s", J.shape)
        logger.debug("  rhs_blocks[0] shape: %s", b_block.shape if hasattr(b_block, "shape") else len(b_block))
        logger.debug("  n_per_block: %s", [actual_size])
        logger.debug("  bc_enforcement: %s", bc_enforcement)
        logger.debug("=" * 60)

        return discretization

    def assemble_system(self, discretization: dict) -> tuple[np.ndarray, np.ndarray]:
        """Assemble global linear system from discretization.

        Two strategies for BC enforcement:

        1. 'append' (default, backward compatible):
            [A_blocks]      [b_blocks]
            [A_bc    ]  u = [b_bc    ]
            [A_cont  ]      [b_cont  ]
           Overdetermined system, solved by least squares. BCs satisfied approximately.

        2. 'replace' (MATLAB Chebfun approach for spectral accuracy):
            [A_bc    ]      [b_bc    ]
            [A_proj  ]  u = [b_proj  ]
            [A_cont  ]      [b_cont  ]
           Square system where BC rows replace operator rows. BCs satisfied to machine precision.

        Args:
            discretization: Dictionary with:
                - 'blocks': List of discretized operator blocks (matrices)
                - 'bc_rows': List of boundary condition rows
                - 'continuity_rows': List of continuity constraint rows
                - 'rhs_blocks': List of RHS vectors for each block
                - 'bc_rhs': RHS values for boundary conditions
                - 'continuity_rhs': RHS values for continuity (usually 0)
                - 'n_per_block': List of sizes for each block
                - 'bc_enforcement': 'append' or 'replace'

        Returns:
            (A, b): Global system matrix and RHS vector
        """
        blocks = discretization["blocks"]
        bc_rows = discretization["bc_rows"]
        continuity_rows = discretization["continuity_rows"]
        integral_rows = discretization["integral_rows"]
        point_rows = discretization["point_rows"]
        mean_zero_rows = discretization.get("mean_zero_rows", [])
        rhs_blocks = discretization["rhs_blocks"]
        bc_rhs = discretization["bc_rhs"]
        continuity_rhs = discretization["continuity_rhs"]
        integral_rhs = discretization["integral_rhs"]
        point_rhs = discretization["point_rhs"]
        mean_zero_rhs = discretization.get("mean_zero_rhs", [])
        n_per_block = discretization["n_per_block"]
        bc_enforcement = discretization.get("bc_enforcement", "append")

        total_size = sum(n_per_block)

        # Stack operator blocks into block-diagonal structure
        A_blocks = sparse.block_diag(blocks, format="csr")

        # Stack RHS blocks
        b_blocks = np.concatenate(rhs_blocks)

        # Convert BC, continuity, integral, point, and mean-zero constraint rows to full-width sparse matrices
        n_bc = len(bc_rows)
        n_cont = len(continuity_rows)
        n_int = len(integral_rows)
        n_point = len(point_rows)
        n_mean_zero = len(mean_zero_rows)

        if n_bc > 0:
            A_bc = sparse.vstack(bc_rows, format="csr")
            b_bc_vec = np.array(bc_rhs)
        else:
            A_bc = sparse.csr_matrix((0, total_size))
            b_bc_vec = np.array([])

        if n_cont > 0:
            A_cont = sparse.vstack(continuity_rows, format="csr")
            b_cont_vec = np.array(continuity_rhs)
        else:
            A_cont = sparse.csr_matrix((0, total_size))
            b_cont_vec = np.array([])

        if n_int > 0:
            A_int = sparse.vstack(integral_rows, format="csr")
            b_int_vec = np.array(integral_rhs)
        else:
            A_int = sparse.csr_matrix((0, total_size))
            b_int_vec = np.array([])

        if n_point > 0:
            A_point = sparse.vstack(point_rows, format="csr")
            b_point_vec = np.array(point_rhs)
        else:
            A_point = sparse.csr_matrix((0, total_size))
            b_point_vec = np.array([])

        if n_mean_zero > 0:
            A_mean_zero = sparse.vstack(mean_zero_rows, format="csr")
            b_mean_zero_vec = np.array(mean_zero_rhs)
        else:
            A_mean_zero = sparse.csr_matrix((0, total_size))
            b_mean_zero_vec = np.array([])

        # Assemble system based on BC enforcement strategy
        if bc_enforcement == "replace" and n_bc > 0:
            # MATLAB Chebfun approach: Replace operator rows with BC rows
            # This creates a square system where BCs are satisfied to machine precision

            # Total number of constraints to replace
            n_constraints = n_bc + n_int + n_point + n_mean_zero

            # Get dimensions
            n_rows_op = A_blocks.shape[0]
            A_blocks.shape[1]

            if n_constraints > n_rows_op:
                raise ValueError(
                    f"Too many constraints ({n_constraints}) for system size ({n_rows_op}). System is over-constrained."
                )

            # Strategy: Remove rows uniformly from operator matrix
            # MATLAB approach: Use projection matrices to reduce dimension while
            # maintaining spectral accuracy. We approximate this by uniform sampling.

            # Calculate how many rows to keep from operator
            n_rows_to_keep = n_rows_op - n_constraints

            # Remove rows uniformly to maintain spectral accuracy
            if n_rows_to_keep < n_rows_op:
                # Uniform sampling strategy: keep evenly spaced rows
                # This maintains good approximation properties across the domain
                rows_to_keep = np.linspace(0, n_rows_op - 1, n_rows_to_keep, dtype=int)

                # Project operator matrix and RHS
                A_proj = A_blocks[rows_to_keep, :]
                b_proj = b_blocks[rows_to_keep]
            else:
                A_proj = A_blocks
                b_proj = b_blocks

            # Stack: BC rows first, then projected operator, then continuity
            A = sparse.vstack([A_bc, A_int, A_point, A_mean_zero, A_proj, A_cont], format="csr")
            b = np.concatenate([b_bc_vec, b_int_vec, b_point_vec, b_mean_zero_vec, b_proj, b_cont_vec])
        else:
            # Default 'append' strategy: Stack everything (overdetermined, least squares)
            A = sparse.vstack([A_blocks, A_bc, A_cont, A_int, A_point, A_mean_zero], format="csr")
            b = np.concatenate([b_blocks, b_bc_vec, b_cont_vec, b_int_vec, b_point_vec, b_mean_zero_vec])

        return A, b

    def solve_linear_system(self, mat: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system A*u = b using appropriate method based on system shape.

        For square systems (m == n):
        - Row scaling: s = 1 / max(1, max(abs(A), [], 2))
        - LU factorization with partial pivoting (MATLAB Chebfun approach)
        - Solve using forward/backward substitution

        For overdetermined systems (m > n) from rectangularization:
        - Use scipy.sparse.linalg.lsmr (iterative sparse least squares)
        - LSMR is stable, efficient, and handles ill-conditioned systems well
        - Provides improved accuracy for eigenvalue problems (~5 orders of magnitude)

        The row scaling improves numerical stability for ill-conditioned systems,
        which is critical for higher-order spectral methods.

        Args:
            mat: System matrix (sparse or dense)
            b: Right-hand side vector

        Returns:
            Solution vector u

        References:
            - Driscoll & Hale (2016), "Rectangular spectral collocation"
            - Fong & Saunders (2011), "LSMR: An iterative algorithm for sparse least-squares problems"
        """
        m, n = mat.shape

        # Overdetermined system: choose solver based on size
        # For systems with n < 5000, use direct lstsq for reliability
        # For very large systems (n >= 5000), use iterative LSMR for memory efficiency
        if m > n:
            # Small to moderately large systems: use direct lstsq (more reliable)
            # Dense lstsq uses LAPACK's optimized least-squares solver.
            if n < 5000:
                # Convert to dense if sparse
                A_dense = sparse_to_dense(mat)

                # Direct least squares for best accuracy
                # Use rcond=None (machine precision default) for better handling of ill-conditioned systems
                # Using rcond=self.tol (e.g., 1e-14) is too strict for systems with condition number > 1e14
                u, _, rank, _ = np.linalg.lstsq(A_dense, b, rcond=None)
                return u

            # Very large systems: use iterative LSMR for memory efficiency
            else:
                # Convert to sparse if not already
                if not hasattr(mat, "toarray"):
                    mat_sp = sparse.csr_matrix(mat)
                else:
                    mat_sp = mat

                # LSMR for sparse overdetermined least squares
                # atol, btol: convergence tolerances for ||A'r|| and ||r||
                m_sp, n_sp = mat_sp.shape
                maxiter = 4 * min(m_sp, n_sp)
                result = lsmr(mat_sp, b, atol=self.tol, btol=self.tol, maxiter=maxiter, show=False)
                u, istop, itn, normr, normar, normA, condA, normx = result

                # Check convergence
                # istop codes: 1-3 indicate successful convergence
                if istop not in [1, 2, 3]:
                    warnings.warn(
                        f"LSMR convergence issue (istop={istop}). Residual norm: {normr:.2e}, ||A'r||: {normar:.2e}"
                    )

                return u

        # Square or underdetermined system: use existing approach
        # Convert to dense for LU decomposition
        A_dense = sparse_to_dense(mat)

        # Row scaling to improve accuracy (MATLAB @valsDiscretization/mldivide.m:19-20)
        # s = 1 / max(1, max(abs(A), [], 2))
        row_maxes = np.max(np.abs(A_dense), axis=1)
        s = 1.0 / np.maximum(1.0, row_maxes)

        # Apply row scaling
        A_scaled = A_dense * s[:, np.newaxis]
        sb = s * b

        # For square systems, use LU decomposition (MATLAB approach) - fast for most cases
        if m == n:
            try:
                lu, piv = lu_factor(A_scaled)
                u = lu_solve((lu, piv), sb)
                return u
            except np.linalg.LinAlgError:
                # If LU fails, fall through to least squares
                pass

        # For rectangular or singular systems, use least squares on scaled system
        # The row scaling still improves conditioning compared to unscaled lstsq
        u, _, rank, _ = np.linalg.lstsq(A_scaled, sb, rcond=self.tol)

        # Check for rank deficiency
        if rank < min(m, n):
            # For periodic BC systems, rank deficiency is expected due to constant nullspace
            # (adding any constant to a periodic solution gives another periodic solution)
            # The least-squares solver correctly finds the minimum-norm solution.
            # Suppress warning for periodic case to avoid confusion.
            is_periodic = isinstance(self.bc, str) and self.bc.lower() == "periodic"
            if not is_periodic:
                warnings.warn(f"System is rank deficient: rank={rank}, min(m,n)={min(m, n)}")

        return u

    def reconstruct_solution(self, u: np.ndarray, n_per_block: list[int]) -> Chebfun:
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

        # Check if we're using periodic BCs (Fourier collocation)
        use_fourier = OpDiscretization._is_periodic(self) and len(self.blocks) == 1

        # Split solution vector into per-block pieces
        pieces = []
        offset = 0
        for i, n_block in enumerate(n_per_block):
            u_block = u[offset : offset + n_block]
            interval = self.blocks[i]["interval"]

            # Create tech from values, then wrap in Bndfun
            # Use Trigtech for periodic (Fourier), Chebtech otherwise
            if use_fourier:
                tech_piece = Trigtech.initvalues(u_block, interval=interval)
                piece = Bndfun(tech_piece, interval)
            else:
                tech_piece = Chebtech.initvalues(u_block, interval=interval)
                piece = Bndfun(tech_piece, interval)
            pieces.append(piece)

            offset += n_block

        solution = Chebfun(pieces)
        return solution

    def solve(
        self,
        rhs: Chebfun | None = None,
        n: int | None = None,
        discretization: str = "collocation",
    ) -> Chebfun:
        """Solve the linear BVP with adaptive refinement.

        Args:
            rhs: Optional right-hand side (default: use self.rhs from constructor)
            n: Optional fixed discretization size (default: adaptive)
            discretization: Discretization method to use. Options:
                - 'collocation' (default): Standard Chebyshev collocation (Driscoll-Hale style)
                - 'ultraspherical': Ultraspherical spectral method (Olver-Townsend)
                  Works in coefficient space with banded matrices for O(n) solve.
                  Currently supports constant-coefficient 2nd order ODEs.

        Returns:
            Chebfun solution to the BVP
        """
        # Route to ultraspherical method if requested
        if discretization == "ultraspherical":
            return self._solve_ultraspherical(rhs=rhs, n=n)
        # Allow rhs to be provided as parameter for convenience
        if rhs is not None:
            original_rhs = self.rhs
            self.rhs = rhs
        else:
            original_rhs = None

        try:
            # Specification phase
            if self.blocks is None:
                self.prepare_domain()

            # Run diagnostics to detect potential issues
            # This will warn about singularities, oscillatory coefficients, etc.
            # Skip diagnostics if disabled (e.g., during Newton iteration for performance)
            if not self._disable_diagnostics:
                diagnose_linop(self, verbose=True)

            # Determine discretization sequence
            # Match MATLAB's dimensionValues: powers of 2 up to 512, then half powers
            # See _remove/chebfun/@valsDiscretization/valsDiscretization.m
            if n is not None:
                n_values = [n]
            else:
                min_pow = np.log2(self.min_n)
                max_pow = np.log2(self.max_n)

                if max_pow <= 9:
                    # Up to 512: use powers of 2
                    pow_vec = np.arange(min_pow, max_pow + 1, 1.0)
                elif min_pow >= 9:
                    # Above 512: use half powers of 2 (sqrt(2) growth)
                    pow_vec = np.arange(min_pow, max_pow + 0.5, 0.5)
                else:
                    # Hybrid: powers of 2 up to 512, then half powers
                    pow_vec = np.concatenate([np.arange(min_pow, 9 + 1, 1.0), np.arange(9.5, max_pow + 0.5, 0.5)])

                n_values = np.round(2.0**pow_vec).astype(int).tolist()

            prev_solution = None
            solution = None  # Initialize to handle early loop exits

            for n_current in n_values:
                self.n_current = n_current

                # Use 'append' mode for all operators. Later, maybe switch to driscoll_hale
                # although currently having some problems with this while append is working best.
                # should test when to use replace as well, if ever.
                bc_enforcement = "append"

                # Check if we have a Jacobian computer function from AdChebfun
                if hasattr(self, "_jacobian_computer"):
                    # Use AdChebfun-based Jacobian computation
                    # This bypasses coefficient-based discretization and avoids coefficient explosion
                    discretization = self._build_discretization_from_jacobian(n_current, bc_enforcement)
                else:
                    # Standard coefficient-based discretization
                    discretization = OpDiscretization.build_discretization(
                        self, n_current, self.u_current, bc_enforcement=bc_enforcement
                    )

                A, b = self.assemble_system(discretization)

                try:
                    u = self.solve_linear_system(A, b)

                except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
                    if n is not None:
                        raise RuntimeError(f"Failed to solve at n={n}: {e}") from e
                    continue

                solution = self.reconstruct_solution(u, discretization["n_per_block"])

                # Skip simplify() for periodic problems with Trigtech, as it uses
                # linear interpolation which destroys spectral accuracy.
                # For Trigtech, the solution is already at the correct resolution.
                is_periodic = OpDiscretization._is_periodic(self)
                if not is_periodic:
                    solution = solution.simplify()

                # Residual check
                residual = A @ u - b
                relres = np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-16)

                # Accept solution if residual is small enough
                is_last_n = n_current == n_values[-1]

                # MATLAB Chebfun approach: happinessCheck via standardCheck
                # Checks if cutoff < n where cutoff = standardChop(coeffs, tol)
                # NO residual check - only coefficient decay!

                solution_is_happy = False
                for fun in solution.funs:
                    # Get coefficients of the solution piece
                    coeffs = fun.onefun.coeffs
                    if len(coeffs) > 0:
                        # For Trigtech (Fourier basis), pair k and -k modes before checking decay
                        # Without pairing, standard_chop sees large values at both ends with
                        # noise in between, confusing the plateau detection

                        if isinstance(fun.onefun, Trigtech):
                            coeffs_to_check = Trigtech._pair_fourier_coeffs(coeffs)
                        else:
                            coeffs_to_check = coeffs

                        # MATLAB: cutoff = standardChop(coeffs, tol)
                        # Use the linop tolerance (matches MATLAB's pref.chebfuneps)
                        cutoff = standard_chop(coeffs_to_check, tol=self.tol)
                        # MATLAB: ishappy = (cutoff < n)
                        # Accept if cutoff < current discretization size
                        if cutoff < n_current:
                            solution_is_happy = True
                            break

                # MATLAB returns immediately when happy, no residual check needed
                # However, for problems with oscillatory coefficients, we need to also
                # verify that BCs are satisfied to avoid premature convergence
                if solution_is_happy:
                    # Check BC satisfaction for Dirichlet BCs
                    bc_satisfied = True
                    bc_tol = max(10 * self.tol, 1e-8)  # Stricter than residual, but not machine precision

                    if self.lbc is not None and not callable(self.lbc):
                        # Check left BC(s)
                        lbc_list = [self.lbc] if not isinstance(self.lbc, (list, tuple)) else self.lbc
                        for deriv_order, bc_val in enumerate(lbc_list):
                            if bc_val is None:
                                continue
                            if deriv_order == 0:  # Dirichlet BC
                                a_val = self.domain.support[0]
                                u_at_a = solution(np.array([a_val]))[0]
                                if abs(u_at_a - bc_val) > bc_tol:
                                    bc_satisfied = False
                                    break

                    if bc_satisfied and self.rbc is not None and not callable(self.rbc):
                        # Check right BC(s)
                        rbc_list = [self.rbc] if not isinstance(self.rbc, (list, tuple)) else self.rbc
                        for deriv_order, bc_val in enumerate(rbc_list):
                            if bc_val is None:
                                continue
                            if deriv_order == 0:  # Dirichlet BC
                                b_val = self.domain.support[1]
                                u_at_b = solution(np.array([b_val]))[0]
                                if abs(u_at_b - bc_val) > bc_tol:
                                    bc_satisfied = False
                                    break

                    if bc_satisfied:
                        return solution
                    # else: continue to next resolution

                # Original stricter check for very good residuals
                if relres < max(100 * self.tol, 1e-8):
                    # Good residual - check adaptive convergence
                    if prev_solution is not None:
                        diff = (solution - prev_solution).norm()
                        if diff < self.tol:
                            return solution
                    # For algebraic equations at last n with good residual, accept
                    if is_last_n and self.diff_order == 0 and relres < 1e-6:
                        return solution

                # Note: We don't warn about large residuals during intermediate adaptive
                # refinement iterations, since higher resolution often fixes this.
                # Final warning (if needed) is issued at max_n.
                # elif relres > 10 * self.tol:
                #     warnings.warn(f"Large residual: {relres:.2e}")

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
                        if self.rhs is not None and hasattr(self.rhs, "funs"):
                            for fun in self.rhs.funs:
                                if fun.size > 1000:  # Needed many points to represent
                                    rhs_is_nonsmooth = True
                                    break

                        # Be more lenient for non-smooth RHS
                        threshold = 0.01 if rhs_is_nonsmooth else 1e-4

                    if relres < threshold:
                        if relres > 1e-4:
                            warnings.warn(
                                f"Returning solution at max n={self.max_n} with relative residual {relres:.2e}"
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
            if solution is None:
                raise RuntimeError(
                    "Failed to find a solution at any discretization level. "
                    "Try increasing max_n or checking boundary conditions."
                )
            return solution
        finally:
            # Restore original rhs if it was temporarily overridden
            if original_rhs is not None:
                self.rhs = original_rhs

    def _solve_ultraspherical(self, rhs: Chebfun | None = None, n: int | None = None) -> Chebfun:
        """Solve using the ultraspherical spectral method (Olver-Townsend).

        The ultraspherical method works in coefficient space rather than at
        collocation points. Key advantages:
        - Banded matrices giving O(n) complexity for banded operators
        - Better conditioning than collocation for some problems
        - Well-suited for constant-coefficient problems
        - Achieves machine precision accuracy

        Current limitations:
        - Only supports 2nd order ODEs
        - Only supports constant coefficients (not variable coefficient a(x)*u)
        - Single interval only (no domain splitting)

        Reference: Olver & Townsend, "A Fast and Well-Conditioned Spectral Method",
        SIAM Review, 2013.

        Args:
            rhs: Optional right-hand side (default: use self.rhs)
            n: Optional fixed discretization size (default: adaptive)

        Returns:
            Chebfun solution
        """
        # Use provided rhs or fall back to self.rhs
        actual_rhs = rhs if rhs is not None else self.rhs

        # Validate: ultraspherical currently only supports 2nd order
        if self.diff_order != 2:
            raise NotImplementedError(
                f"Ultraspherical method currently only supports 2nd order ODEs, "
                f"got order {self.diff_order}. Use discretization='collocation' for other orders."
            )

        # Validate: single interval only
        intervals_list = list(self.domain.intervals)
        if len(intervals_list) > 1:
            raise NotImplementedError(
                "Ultraspherical method does not support domain splitting. "
                "Use discretization='collocation' for multi-interval problems."
            )

        # Extract interval
        interval = intervals_list[0]
        a, b = interval

        # Extract coefficient values (must be constant)
        # coeffs list is [a_0(x), a_1(x), a_2(x)] for a_2*u'' + a_1*u' + a_0*u
        # Evaluate at midpoint to get constant value
        x_mid = (a + b) / 2
        coeffs_vals = []
        for i, coeff in enumerate(self.coeffs):
            if coeff is None:
                coeffs_vals.append(None)
            elif callable(coeff):
                val = coeff(np.array([x_mid]))[0]
                coeffs_vals.append(val)
            elif isinstance(coeff, Chebfun):
                val = coeff(np.array([x_mid]))[0]
                # Check if coefficient is approximately constant
                x_test = np.linspace(a, b, 10)
                vals_test = coeff(x_test)
                if np.max(np.abs(vals_test - val)) > 1e-10:
                    raise NotImplementedError(
                        f"Ultraspherical method requires constant coefficients. "
                        f"Coefficient a_{i}(x) varies. Use discretization='collocation'."
                    )
                coeffs_vals.append(val)
            else:
                coeffs_vals.append(float(coeff))

        # Get RHS coefficients
        if actual_rhs is None:
            rhs_coeffs = np.array([0.0])
        elif isinstance(actual_rhs, Chebfun):
            if len(actual_rhs.funs) > 0:
                rhs_coeffs = actual_rhs.funs[0].onefun.coeffs.copy()
            else:
                rhs_coeffs = np.array([0.0])
        else:
            rhs_coeffs = np.array([float(actual_rhs)])

        # Determine discretization size
        if n is None:
            # Start with reasonable default, increase adaptively
            n_values = [16, 32, 64, 128, 256, 512]
        else:
            n_values = [n]

        solution = None
        prev_solution = None

        for n_current in n_values:
            # Call ultraspherical_solve from spectral module
            try:
                sol_coeffs = ultraspherical_solve(
                    coeffs=coeffs_vals,
                    rhs_coeffs=rhs_coeffs,
                    n=n_current,
                    interval=interval,
                    lbc=self.lbc,
                    rbc=self.rbc,
                )
            except Exception as e:
                if n is not None:
                    raise RuntimeError(f"Ultraspherical solve failed at n={n_current}: {e}") from e
                continue

            # Convert coefficient solution to Chebfun
            # sol_coeffs are Chebyshev coefficients
            onefun = Chebtech(sol_coeffs)
            bndfun = Bndfun(onefun, interval)
            solution = Chebfun([bndfun])

            # Check convergence via coefficient decay
            cutoff = standard_chop(sol_coeffs, tol=self.tol)
            if cutoff < n_current:
                return solution

            # Check convergence vs previous
            if prev_solution is not None:
                diff = (solution - prev_solution).norm()
                if diff < self.tol:
                    return solution

            prev_solution = solution

            if n is not None:
                return solution

        if solution is None:
            raise RuntimeError("Ultraspherical solve failed to converge")

        return solution

    def _discretization_size(self, n: int | None = None) -> int:
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

    def _discretize(self, n: int | None = None) -> tuple[np.ndarray, dict]:
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
        # Use BC row replacement for high-order operators where exact BCs are critical
        bc_enforcement = "replace" if self.diff_order >= 4 else "append"
        disc = OpDiscretization.build_discretization(self, n_actual, self.u_current, bc_enforcement=bc_enforcement)
        A, _ = self.assemble_system(disc)
        return sparse_to_dense(A), disc

    def _discretize_operator_only(self, n: int | None = None) -> tuple[np.ndarray, dict]:
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
        # Use BC row replacement for high-order operators where exact BCs are critical
        bc_enforcement = "replace" if self.diff_order >= 4 else "append"
        disc = OpDiscretization.build_discretization(self, n_actual, self.u_current, bc_enforcement=bc_enforcement)
        n_per_block = disc["n_per_block"]
        total_n = sum(n_per_block)

        A_op = sparse_to_dense(sparse.block_diag(disc["blocks"], format="csr"))
        # Ensure square by taking only the operator part
        A_op = A_op[:total_n, :total_n]
        return A_op, disc

    def _check_eigenvalue_spurious(
        self, val: float, efun: Chebfun, mass_matrix: Optional["LinOp"] = None
    ) -> tuple[bool, str]:
        """Check if an eigenvalue/eigenfunction pair is spurious.

        An eigenvalue is considered potentially spurious if:
        1. The eigenfunction's Chebyshev coefficients have not decayed sufficiently
           (indicating the function is not well-resolved)
        2. The residual ||L[u] - λ*M[u]|| is large (equation not satisfied accurately)

        Args:
            val: Eigenvalue
            efun: Eigenfunction (Chebfun)
            mass_matrix: Optional mass matrix for generalized problem

        Returns:
            is_spurious: True if eigenvalue appears spurious
            reason: Description of why it's spurious (empty if not spurious)
        """
        # Check 1: Coefficient tail decay
        # If the last 10 coefficients are a significant fraction of the total,
        # the eigenfunction is not well-resolved
        try:
            if hasattr(efun, "funs") and len(efun.funs) > 0:
                for fun in efun.funs:
                    if hasattr(fun, "onefun") and hasattr(fun.onefun, "coeffs"):
                        coeffs = fun.onefun.coeffs
                        n = len(coeffs)
                        if n > 10:
                            # Measure relative size of tail coefficients
                            tail_size = np.linalg.norm(coeffs[-10:])
                            total_size = np.linalg.norm(coeffs)
                            tail_ratio = tail_size / (total_size + 1e-14)

                            if tail_ratio > 0.01:  # 1% threshold
                                return True, f"Large tail ratio: {tail_ratio:.2e} (coefficients not decayed)"
        except Exception:
            pass  # If coefficient check fails, continue to residual check

        # Check 2: Residual check ||L[u] - λ*M[u]||
        try:
            Lu = self(efun)
            if mass_matrix is not None:
                Mu = mass_matrix(efun)
                residual = Lu - val * Mu
            else:
                residual = Lu - val * efun

            res_norm = residual.norm(2)
            efun_norm = efun.norm(2)
            Lu_norm = Lu.norm(2)

            # Use relative residual: ||L*u - λ*u|| / (||L*u|| + |λ|*||u||)
            # This correctly handles both small and large eigenvalues:
            # - For small λ: denominator ≈ ||L*u||, so we check ||L*u - λ*u|| / ||L*u||
            # - For large λ: denominator ≈ |λ|*||u||, so we check ||L*u - λ*u|| / |λ|*||u||
            # - Intermediate λ: balanced contribution from both terms
            denominator = Lu_norm + abs(val) * efun_norm + 1e-14
            rel_residual = res_norm / denominator

            # Threshold is more lenient for higher-order differential operators
            # because differentiation amplifies coefficient errors.
            # For second-order operators (most common), threshold ~ 2.0 is reasonable.
            threshold = 2.0 if self.diff_order >= 2 else 1e-3
            if rel_residual > threshold:
                return True, f"Large residual: {rel_residual:.2e}"
        except Exception as e:
            # If residual computation fails, warn about that
            return True, f"Could not compute residual: {e}"

        return False, ""

    def eigs(
        self,
        k: int = 6,
        sigma: float | None = None,
        mass_matrix: Optional["LinOp"] = None,
        rectangularization: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, list[Chebfun]]:
        """Compute k eigenvalues and eigenfunctions of the operator.

        Solves the eigenvalue problem L[u] = lambda * M[u] subject to boundary conditions.
        If mass_matrix is None, solves the standard problem L[u] = lambda * u.

        Uses adaptive refinement to ensure eigenvalue convergence.

        Args:
            k: Number of eigenvalues to compute (default 6)
            sigma: Target eigenvalue for shift-invert mode (default None = smallest magnitude)
            mass_matrix: Optional mass matrix operator M for generalized problem L[u] = λ*M[u]
                        If None, solves standard eigenvalue problem L[u] = λ*u
            rectangularization: If True, use rectangular (overdetermined) discretization
                               for improved eigenvalue accuracy (~5 orders of magnitude).
                               Uses m = min(2*n, n+50) collocation points per n coefficients.
                               Default False for backward compatibility.
            **kwargs: Additional arguments passed to scipy.sparse.linalg.eigs

        Returns:
            eigenvalues: Array of k eigenvalues
            eigenfunctions: List of k Chebfun eigenfunctions (L2-normalized)

        Examples:
            # Standard eigenvalue problem: -u'' = λu
            L = LinOp([a0, a1, a2], domain, diff_order=2)
            L.lbc = L.rbc = 0
            evals, efuns = L.eigs(k=5)

            # With rectangularization for improved accuracy
            evals, efuns = L.eigs(k=5, rectangularization=True)

            # Generalized eigenvalue problem: -u'' = λ * x * u (weighted)
            M = LinOp([weight_func], domain, diff_order=0)  # M[u] = x*u
            M.lbc = M.rbc = 0
            evals, efuns = L.eigs(k=5, mass_matrix=M, rectangularization=True)

        References:
            Driscoll & Hale (2016), "Rectangular spectral collocation"
        """
        if self.blocks is None:
            self.prepare_domain()

        # Adaptive refinement sequence scaled by operator order
        # Go in powers of 2 from base up to max_n (like MATLAB Chebfun)
        base = self._discretization_size()
        n_list = []
        n = base
        while n <= self.max_n:
            n_list.append(n)
            n *= 2
        if not n_list:
            n_list = [base]

        prev_vals = None
        final_disc = None
        final_vecs = None

        for n in n_list:
            # Use BC row replacement for high-order operators where exact BCs are critical
            bc_enforcement = "replace" if self.diff_order >= 4 else "append"
            disc = OpDiscretization.build_discretization(
                self,
                n,
                rectangularization=rectangularization,
                for_eigenvalue_problem=True,
                bc_enforcement=bc_enforcement,
            )
            n_per_block = disc["n_per_block"]
            m_per_block = disc.get("m_per_block", n_per_block)
            total_dofs = sum(n_per_block)
            is_rectangular = disc.get("rectangularization", False)

            # Build operator matrix
            # For rectangular: blocks are (m+1) x (n+1), need to extract (n+1) x (n+1) subspace
            # For square: blocks are (n+1) x (n+1), use as-is
            A_op_sparse = sparse.block_diag(disc["blocks"], format="csr")

            if is_rectangular:
                # Rectangular case: operator is (sum(m_per_block)) x (sum(n_per_block))
                # Need to project onto n-dimensional subspace via QR
                # This is handled below after BC projection
                total_rows = sum(m_per_block)
                A_op_rect = sparse_to_dense(A_op_sparse)[:total_rows, :total_dofs]
            else:
                # Square case: extract (n+1) x (n+1) square operator
                A_op = sparse_to_dense(A_op_sparse)[:total_dofs, :total_dofs]

            # Get BC rows and continuity rows (includes periodic constraints)
            bc_rows_list = disc.get("bc_rows", [])
            continuity_rows_list = disc.get("continuity_rows", [])

            # Combine all constraint rows (BCs + continuity/periodic constraints)
            all_constraint_rows = bc_rows_list + continuity_rows_list

            if len(all_constraint_rows) == 0:
                # No constraints - eigenvalue problem on full operator
                if is_rectangular:
                    # Rectangular without BCs: Solve generalized eigenvalue problem
                    # A_rect @ v = λ * PS.T @ v
                    # Following MATLAB's approach where PS projects solution space
                    PS_matrices = disc.get("projection_matrices", [])
                    if not PS_matrices:
                        raise RuntimeError("Rectangular discretization missing projection matrices")

                    # Build block-diagonal PS matrix
                    PS = sparse.block_diag(PS_matrices, format="csr")

                    # A_rect is the rectangular operator (m+1) x (n+1)
                    # PS is the projection matrix (n+1) x (m+1)
                    # We solve: A_rect @ v = λ * PS.T @ v
                    # This is equivalent to: (PS @ A_rect) @ v = λ * (PS @ PS.T) @ v
                    # The projected system is square (n+1) x (n+1)
                    A_eig_sparse = PS @ A_op_rect
                    A_eig = sparse_to_dense(A_eig_sparse)

                    # Mass matrix: PS @ PS.T projects the identity
                    M_eig_sparse = PS @ PS.T
                    M_eig = sparse_to_dense(M_eig_sparse)

                    # No BC projection needed
                    bc_proj = np.eye(total_dofs)
                else:
                    A_eig = A_op
                    M_eig = None
                    bc_proj = np.eye(total_dofs)
            else:
                # Assemble constraint matrix from all constraint rows
                BC = []
                for bc_row in all_constraint_rows:
                    BC.append(sparse_to_dense(bc_row).ravel()[:total_dofs])
                BC = np.array(BC)

                # For eigenproblems L[u] = λu, we need homogeneous BCs: BC @ u = 0
                # Find null space of BC matrix
                try:
                    # Use QR decomposition for numerical stability
                    # Q, R, P = qr(BC.T, pivoting=True)
                    # Nullspace is last (n - rank) columns of Q

                    Q, R, P = qr(BC.T, mode="full", pivoting=True)

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
                if is_rectangular:
                    # Rectangular discretization: Full implementation
                    # Get projection matrices PS (barycentric interpolation m+1 -> n+1)
                    PS_matrices = disc.get("projection_matrices", [])
                    if not PS_matrices:
                        raise RuntimeError("Rectangular discretization missing projection matrices")

                    # Build block-diagonal PS matrix
                    PS = sparse.block_diag(PS_matrices, format="csr")

                    # Project rectangular operator: PA = PS @ A_rect
                    # A_rect is (m+1) x (n+1), PS is (n+1) x (m+1)
                    # Result: PA is (n+1) x (n+1) - square!
                    A_projected_sparse = PS @ A_op_rect
                    A_projected = sparse_to_dense(A_projected_sparse)

                    # Now project onto BC-satisfying subspace
                    # bc_proj is the nullspace of BCs in the coefficient space
                    A_eig = bc_proj.T @ A_projected @ bc_proj

                else:
                    # Square: A_op is (total_dofs) x (total_dofs)
                    # A_eig = Z.T @ A_op @ Z where Z are nullspace basis vectors
                    A_eig = bc_proj.T @ A_op @ bc_proj

                # Handle mass matrix for generalized eigenvalue problem
                if mass_matrix is not None:
                    # Discretize mass matrix at same resolution
                    # Use BC row replacement for high-order operators where exact BCs are critical
                    bc_enforcement = "replace" if mass_matrix.diff_order >= 4 else "append"
                    M_disc = OpDiscretization.build_discretization(
                        mass_matrix, n, rectangularization=rectangularization, bc_enforcement=bc_enforcement
                    )
                    M_op_sparse = sparse.block_diag(M_disc["blocks"], format="csr")

                    if is_rectangular:
                        # Get projection matrices for mass matrix
                        M_PS_matrices = M_disc.get("projection_matrices", [])
                        if not M_PS_matrices:
                            raise RuntimeError("Mass matrix rectangular discretization missing projection matrices")

                        M_PS = sparse.block_diag(M_PS_matrices, format="csr")
                        M_total_rows = sum(M_disc.get("m_per_block", M_disc["n_per_block"]))
                        M_op_rect = sparse_to_dense(M_op_sparse)[:M_total_rows, :total_dofs]

                        # Project mass matrix: M_projected = M_PS @ M_rect
                        M_projected_sparse = M_PS @ M_op_rect
                        M_projected = sparse_to_dense(M_projected_sparse)

                        # Project onto BC-satisfying subspace
                        M_eig = bc_proj.T @ M_projected @ bc_proj
                    else:
                        M_op = sparse_to_dense(M_op_sparse)[:total_dofs, :total_dofs]
                        # Project mass matrix onto same BC-satisfying subspace
                        M_eig = bc_proj.T @ M_op @ bc_proj
                else:
                    # No explicit mass matrix: use PS @ PS.T as mass matrix for rectangular
                    if is_rectangular:
                        PS_matrices = disc.get("projection_matrices", [])
                        PS = sparse.block_diag(PS_matrices, format="csr")
                        # Mass matrix in coefficient space: PS @ PS.T (projects identity)
                        M_projected_sparse = PS @ PS.T
                        M_projected = sparse_to_dense(M_projected_sparse)
                        # Project onto BC-satisfying subspace
                        M_eig = bc_proj.T @ M_projected @ bc_proj
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
                    # For large problems (>500 DOFs), use sparse solver with shift-invert
                    # Otherwise use dense solver

                    if A_eig.shape[0] > 500:
                        # Try sparse solver with shift-invert mode
                        # Convert to standard form: M^{-1} A v = λ v
                        try:
                            # Try Cholesky factorization if M is positive definite

                            try:
                                # Attempt Cholesky factorization (assumes M is positive definite)
                                cholesky(M_eig, lower=True)
                                # Solve M v = A v for eigenvalue λ by converting to standard form
                                # M^{-1} A v = λ v
                                # Use shift-invert: (A - σM)^{-1} M v = (λ - σ)^{-1} v
                                # This targets eigenvalues near σ

                                # For smallest magnitude eigenvalues, use σ = 0
                                sigma_shift = 0.0 if sigma is None else sigma

                                # Create operator for shift-invert: solve (A - σM)x = b

                                def matvec(v):
                                    # Solve (A - σM)x = M*v for x
                                    # Then return M*x
                                    rhs = M_eig @ v
                                    A_shifted = A_eig - sigma_shift * M_eig
                                    x = np.linalg.solve(A_shifted, rhs)
                                    return M_eig @ x

                                op = LinearOperator(A_eig.shape, matvec=matvec)

                                # Use sparse eigensolver
                                # Extract 'which' from kwargs if provided, default to 'LM' for shift-invert
                                which = kwargs.pop("which", "LM")
                                vals_inv, vecs = sp_eigs(op, k=k_actual, which=which, **kwargs)

                                # Convert back: λ = σ + 1/μ where μ are the computed eigenvalues
                                vals = sigma_shift + 1.0 / vals_inv

                            except np.linalg.LinAlgError:
                                # Cholesky failed - M might not be positive definite
                                # Fall back to LU factorization approach

                                # Solve M^{-1} A as standard eigenvalue problem
                                # For small k, it's better to use dense solver on M^{-1}A
                                M_inv_A = np.linalg.solve(M_eig, A_eig)

                                # Use sparse eigensolver on the standard form
                                # Extract 'which' from kwargs if provided, default to 'SM'
                                which = kwargs.pop("which", "SM")
                                vals, vecs = sp_eigs(M_inv_A, k=k_actual, which=which, **kwargs)

                        except Exception as e:
                            # Sparse solver failed, fall back to dense
                            warnings.warn(
                                f"Sparse solver failed for generalized eigenvalue problem "
                                f"({A_eig.shape[0]} DOFs): {e}. Falling back to dense solver."
                            )

                            vals_all, vecs_all = eig(A_eig, M_eig)
                            vals, vecs = self._filter_eigenvalues(vals_all, vecs_all, k_actual)
                            if vals is None:
                                continue
                    else:
                        # Small problem - use dense solver

                        vals_all, vecs_all = eig(A_eig, M_eig)
                        vals, vecs = self._filter_eigenvalues(vals_all, vecs_all, k_actual)
                        if vals is None:
                            continue
                else:
                    # Standard eigenvalue problem

                    # For rectangular discretization, the projected matrix Q.T @ A_rect @ bc_proj
                    # is generally NON-SYMMETRIC, which causes ARPACK convergence issues.
                    # MATLAB Chebfun uses dense eig() for rectangular discretization (see eigs.m line 362-363).
                    # We follow the same approach: use dense solver for all eigenvalue problems
                    # or when matrix is small enough.

                    if A_eig.shape[0] <= 500 or is_rectangular:
                        # Use dense solver for small matrices or rectangular discretization

                        vals_all, vecs_all = eig(A_eig)
                        vals, vecs = self._filter_eigenvalues(vals_all, vecs_all, k_actual, sigma)
                        if vals is None:
                            continue
                    else:
                        # Use sparse solver for large square matrices
                        # Always use shift-invert mode (sigma) for smallest eigenvalues
                        # ARPACK's which='SM' fails on ill-conditioned spectral matrices
                        sigma_val = sigma if sigma is not None else 0.0
                        vals, vecs = sp_eigs(A_eig, k=k_actual, sigma=sigma_val, **kwargs)
            except Exception:
                continue

            # Sort by magnitude
            idx = np.argsort(np.abs(vals))
            vals, vecs = vals[idx], vecs[:, idx]

            # Store for reconstruction
            final_disc = disc
            final_vecs = vecs
            final_bc_proj = bc_proj

            # Check convergence by testing eigenfunction resolution (like MATLAB)
            # Create a linear combination of eigenfunctions and check if coefficients have decayed
            # This ensures eigenfunctions are well-resolved, not just eigenvalues
            converged = False
            if prev_vals is not None and len(prev_vals) == len(vals):
                # MATLAB's approach: combine eigenfunctions with nontrivial coefficients
                # to avoid accidental cancellations
                coeff_vec = 1.0 / (2.0 * np.arange(1, len(vals) + 1))

                # Reconstruct the combined eigenfunction
                combined_eigvec = vecs @ coeff_vec
                u_combined = final_bc_proj @ np.real_if_close(combined_eigvec)
                ef_combined = self.reconstruct_solution(u_combined, n_per_block)

                # Check if Chebyshev/Fourier coefficients have decayed sufficiently
                # This is the key difference from checking eigenvalue convergence
                try:
                    # Get the function values and convert to Chebyshev/Fourier coefficients
                    if hasattr(ef_combined, "funs") and len(ef_combined.funs) > 0:
                        # For piecewise functions, check each piece using standard_chop
                        all_resolved = True
                        for fun_idx, fun in enumerate(ef_combined.funs):
                            if hasattr(fun, "onefun"):
                                # Check for Chebyshev coefficients (Chebtech)
                                if hasattr(fun.onefun, "coeffs"):
                                    coeffs = fun.onefun.coeffs
                                    # Use standard_chop to determine if coefficients have decayed
                                    # If cutoff < len(coeffs), we're happy (coefficients have decayed)
                                    # If cutoff == len(coeffs), we're unhappy (need more resolution)
                                    cutoff = standard_chop(coeffs, self.tol)
                                    if cutoff >= len(coeffs):
                                        # Not happy - coefficients haven't decayed enough
                                        all_resolved = False
                                        break
                                # Check for Fourier coefficients (Trigtech)
                                elif hasattr(fun.onefun, "fourier_coeffs"):
                                    # For Fourier collocation, use eigenvalue convergence instead
                                    # Fourier coefficients don't decay the same way as Chebyshev
                                    rel_err = np.abs(vals - prev_vals) / (np.abs(prev_vals) + 1e-14)
                                    eigs_tol = kwargs.get("tol", max(self.tol, 1e-8))
                                    converged = np.max(rel_err) < eigs_tol
                                    # Exit the loop early - we're using eigenvalue convergence
                                    break
                        else:
                            # If we didn't break, check if all_resolved is still True
                            converged = all_resolved
                    else:
                        # Fallback to eigenvalue convergence if can't check coefficients
                        rel_err = np.abs(vals - prev_vals) / (np.abs(prev_vals) + 1e-14)
                        eigs_tol = kwargs.get("tol", max(self.tol, 1e-8))
                        converged = np.max(rel_err) < eigs_tol
                except Exception:
                    # Fallback to eigenvalue convergence on error
                    rel_err = np.abs(vals - prev_vals) / (np.abs(prev_vals) + 1e-14)
                    eigs_tol = kwargs.get("tol", max(self.tol, 1e-8))
                    converged = np.max(rel_err) < eigs_tol

                if converged:
                    break
            prev_vals = vals.copy()

        if prev_vals is None or final_disc is None:
            raise RuntimeError("eigs failed to converge")

        # Reconstruct eigenfunctions
        n_per_block = final_disc["n_per_block"]
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

        # Check for spurious eigenvalues
        for j, (val, efun) in enumerate(zip(prev_vals, eigenfunctions)):
            is_spurious, reason = self._check_eigenvalue_spurious(val, efun, mass_matrix)
            if is_spurious:
                warnings.warn(
                    f"Eigenvalue {j} (λ={val:.6f}) may be spurious: {reason}. "
                    f"Consider increasing discretization size or checking problem formulation."
                )

        return np.real_if_close(prev_vals), eigenfunctions

    def expm(self, t: float = 1.0, u0: Chebfun | None = None, num_eigs: int = 50) -> Chebfun:
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
            u0 = Chebfun.initfun(lambda x: np.ones_like(x), list(self.domain))

        # Special case: t=0 means identity
        if abs(t) < 1e-15:
            return u0.copy() if hasattr(u0, "copy") else u0

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

        # Check truncation: sum of squared coefficients should be close to ||u0||^2
        u0_norm_sq = (u0 * u0).sum()
        coeffs_norm_sq = sum(c**2 for c in coeffs)
        rel_truncation = abs(u0_norm_sq - coeffs_norm_sq) / (u0_norm_sq + 1e-14)

        if rel_truncation > 0.01:  # 1% truncation error
            warnings.warn(
                f"Eigenfunction expansion may be incomplete: {rel_truncation * 100:.1f}% of energy missing. "
                f"Only {len(evals)} eigenfunctions used. Consider increasing num_eigs or check if operator "
                f"is non-self-adjoint (eigenfunctions may not span the space)."
            )

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

    def null(self, tol: float | None = None) -> list[Chebfun]:
        """Compute a basis for the nullspace of the operator.

        Returns functions u such that L[u] = 0 (with boundary conditions).

        Args:
            tol: Tolerance for determining nullspace (default: self.tol)

        Returns:
            List of Chebfun functions forming a basis for the nullspace
        """
        tol = tol if tol is not None else self.tol
        A, disc = self._discretize()
        n_per_block = disc["n_per_block"]
        total_n = sum(n_per_block)

        # SVD to find nullspace
        _, s, vh = np.linalg.svd(A, full_matrices=True)

        # Find numerical rank: count singular values above tolerance
        if len(s) > 0 and s[0] > 0:
            rank = np.sum(s >= tol * s[0])
        else:
            rank = 0

        # Nullspace is spanned by right singular vectors beyond the rank
        null_vecs = vh[rank:].T if rank < vh.shape[0] else np.empty((total_n, 0))

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

    def svds(self, k: int = 6) -> tuple[np.ndarray, list[Chebfun], list[Chebfun]]:
        """Compute k largest singular values and singular functions.

        Args:
            k: Number of singular values to compute

        Returns:
            s: Array of k largest singular values
            u_funcs: List of k left singular functions
            v_funcs: List of k right singular functions
        """
        A, disc = self._discretize()
        n_per_block = disc["n_per_block"]
        total_n = sum(n_per_block)

        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        k_actual = min(k, len(S))

        # Right singular functions (input space)
        v_funcs = [self.reconstruct_solution(Vh[j, :total_n], n_per_block) for j in range(k_actual)]

        # Left singular functions (output space)
        # NOTE: For rectangular systems (m rows > n cols due to BCs), the left singular
        # vectors U live in R^m (output/constraint space), not R^n (DOF/input space).
        # We can only reconstruct them as functions if m == n (square system).
        u_funcs = []
        if A.shape[0] == A.shape[1]:
            # Square system: left singular vectors can be reconstructed as functions
            for j in range(k_actual):
                u_funcs.append(self.reconstruct_solution(U[:, j], n_per_block))
        else:
            # Rectangular system: left singular vectors are in constraint space
            # Cannot meaningfully reconstruct as functions in the domain
            # Return the raw vectors instead (they represent combinations of residuals/BCs)

            warnings.warn(
                "Left singular vectors for rectangular system (with BCs) are in constraint space, "
                "not function space. Returning None for u_funcs. Only v_funcs (right singular vectors) "
                "represent functions in the domain."
            )
            u_funcs = None

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

    def rank(self, tol: float | None = None) -> int:
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
        A, _ = self._discretize()
        plt.figure()
        plt.spy(A, **kwargs)
        plt.title(f"LinOp sparsity pattern (order {self.diff_order})")
        plt.show()

    def __add__(self, other):
        """Add two linear operators: (L1 + L2)[u] = L1[u] + L2[u].

        Args:
            other: Another LinOp

        Returns:
            New LinOp representing the sum

        Raises:
            ValueError: If operators have incompatible domains
            TypeError: If other is not a LinOp

        Example:
            Addition of compatible LinOp objects combines their domains and orders.
        """
        if not isinstance(other, LinOp):
            raise TypeError(f"Cannot add LinOp and {type(other)}")

        # Check domain compatibility (both endpoints and internal breakpoints)
        if self.domain.support != other.domain.support:
            raise ValueError(f"Operators must have same domain: {self.domain.support} vs {other.domain.support}")

        # Check internal breakpoints match
        # Interval objects are tuple-like (a, b)
        self_breaks = sorted([tuple(iv) for iv in self.domain.intervals])
        other_breaks = sorted([tuple(iv) for iv in other.domain.intervals])

        if self_breaks != other_breaks:
            warnings.warn(
                f"Operators have different internal breakpoint structures. "
                f"This may cause issues in discretization. "
                f"Self breakpoints: {self_breaks}, Other breakpoints: {other_breaks}"
            )

        # Result has max differential order
        new_order = max(self.diff_order, other.diff_order)

        # Pad coefficient lists to same length
        new_coeffs = []
        for k in range(new_order + 1):
            # Get coefficients from both operators (0 if not present)
            if k < len(self.coeffs):
                c1 = self.coeffs[k]
            else:
                c1 = Chebfun.initfun(lambda x: 0 * x, list(self.domain))

            if k < len(other.coeffs):
                c2 = other.coeffs[k]
            else:
                c2 = Chebfun.initfun(lambda x: 0 * x, list(self.domain))

            # Add coefficients
            new_coeffs.append(c1 + c2)

        return LinOp(coeffs=new_coeffs, domain=self.domain, diff_order=new_order, tol=min(self.tol, other.tol))

    def __sub__(self, other):
        """Subtract two linear operators: (L1 - L2)[u] = L1[u] - L2[u].

        Args:
            other: Another LinOp

        Returns:
            New LinOp representing the difference
        """
        return self + (-other)

    def __mul__(self, scalar):
        """Multiply operator by scalar (right multiplication): (L * c)[u] = c * L[u].

        Args:
            scalar: Scalar value

        Returns:
            New LinOp with scaled coefficients

        Raises:
            TypeError: If scalar is not numeric
        """
        if not isinstance(scalar, (int, float, np.number)):
            raise TypeError(f"Can only multiply LinOp by scalar, not {type(scalar)}")

        # Scale all coefficients
        new_coeffs = [c * scalar for c in self.coeffs]

        return LinOp(coeffs=new_coeffs, domain=self.domain, diff_order=self.diff_order, tol=self.tol)

    def __rmul__(self, scalar):
        """Multiply operator by scalar (left multiplication): (c * L)[u] = c * L[u].

        Args:
            scalar: Scalar value

        Returns:
            New LinOp with scaled coefficients
        """
        return self * scalar

    def __neg__(self):
        """Negate operator: (-L)[u] = -L[u].

        Returns:
            New LinOp with negated coefficients
        """
        return self * (-1)

    def __truediv__(self, scalar):
        """Divide operator by scalar: L / c = (1/c) * L.

        Args:
            scalar: Scalar divisor

        Returns:
            New LinOp with coefficients divided by scalar
        """
        if not isinstance(scalar, (int, float, np.number)):
            raise TypeError(f"Can only divide LinOp by scalar, not {type(scalar)}")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide operator by zero")
        return self * (1.0 / scalar)

    def __pow__(self, power):
        """Raise operator to a power: L ** n means applying L n times.

        Note: This is operator composition, not matrix power.
        L**2 means L ∘ L (apply L, then apply L again).

        Args:
            power: Non-negative integer power

        Returns:
            Composed operator (for power > 1)
            Identity-like operator (for power == 0)
            Self (for power == 1)

        Raises:
            NotImplementedError: Operator composition not yet implemented
            ValueError: For negative or non-integer powers
        """
        if not isinstance(power, (int, np.integer)):
            raise ValueError("Operator power must be an integer")
        if power < 0:
            raise ValueError("Negative powers not supported (would require inverse)")

        if power == 0:
            # L^0 = I (identity operator)
            # Create identity operator: 1*u + 0*u'

            a0 = Chebfun.initfun(lambda x: 1 + 0 * x, list(self.domain))
            return LinOp(coeffs=[a0], domain=self.domain, diff_order=0, tol=self.tol)
        elif power == 1:
            return self
        else:
            # L^n for n > 1: compose with self n times
            result = self
            for _ in range(power - 1):
                result = result @ self
            return result

    def __matmul__(self, other):
        """Compose two linear operators: (L1 @ L2)[u] = L1[L2[u]].

        This implements operator composition where L1 @ L2 means "apply L2 first,
        then apply L1 to the result". This follows Python's matrix multiplication
        convention where A @ B means "multiply A times B".

        Mathematical Definition:
        -----------------------
        For linear operators L1 = sum_i a_i(x) D^i and L2 = sum_j b_j(x) D^j,
        the composition (L1 ∘ L2) is computed by:

            (L1 @ L2)[u] = L1[L2[u]]

        The composed operator is generally a higher-order operator whose
        coefficients are determined by applying the product rule and chain rule.

        Algorithm:
        ---------
        For L1 = sum_i a_i D^i and L2 = sum_j b_j D^j:

        1. Compute L2[u] symbolically as a linear combination of u and its derivatives
        2. Apply L1 to this result, using Leibniz's product rule for each term:
           D^k[b_j(x) D^j u] = sum_{m=0}^k (k choose m) b_j^(m)(x) D^(j+k-m) u
        3. Collect coefficients of each derivative order

        The resulting operator has order = order(L1) + order(L2).

        Args:
            other: Another LinOp to compose with (applied first)

        Returns:
            New LinOp representing the composition L1 @ L2

        Raises:
            TypeError: If other is not a LinOp
            ValueError: If operators have incompatible domains

        Examples:
            Composition of differentiation operators: D @ D = D^2
            composes operators by function composition, combining their orders and coefficients.

        Notes:
            - Composition is associative but not commutative: (L1 @ L2) @ L3 = L1 @ (L2 @ L3)
            - Generally L1 @ L2 ≠ L2 @ L1 unless operators commute
            - The composed operator inherits the tighter tolerance of the two inputs
            - Boundary conditions are NOT automatically composed - they must be set separately
        """
        if not isinstance(other, LinOp):
            raise TypeError(f"Cannot compose LinOp with {type(other)}")

        # Check domain compatibility
        if self.domain.support != other.domain.support:
            raise ValueError(
                f"Operators must have same domain for composition: {self.domain.support} vs {other.domain.support}"
            )

        # Result order is sum of orders
        result_order = self.diff_order + other.diff_order

        # Initialize coefficient list for result
        result_coeffs = []

        # For each derivative order k in the result (0 to result_order):
        # We need to find all ways that L1 and L2 can produce D^k u
        #
        # L2[u] = sum_j b_j(x) D^j u
        # L1[L2[u]] = sum_i a_i(x) D^i [sum_j b_j(x) D^j u]
        #
        # Applying D^i to b_j(x) D^j u using Leibniz rule:
        # D^i[b_j(x) D^j u] = sum_m (i choose m) b_j^(m)(x) D^(i+j-m) u
        #
        # So coefficient of D^k u in result is:
        # c_k = sum_{i,j,m: i+j-m=k} a_i(x) (i choose m) b_j^(m)(x)

        for k in range(result_order + 1):
            c_k = Chebfun.initfun(lambda x: 0 * x, list(self.domain))

            # Sum over all (i, j, m) such that i + j - m = k
            for i in range(len(self.coeffs)):
                if self.coeffs[i] is None:
                    continue
                a_i = self.coeffs[i]

                for j in range(len(other.coeffs)):
                    if other.coeffs[j] is None:
                        continue
                    b_j = other.coeffs[j]

                    # For this i and j, we need m such that i + j - m = k
                    # So m = i + j - k
                    m = i + j - k

                    # m must be in valid range: 0 <= m <= i
                    if 0 <= m <= i:
                        # Compute m-th derivative of b_j
                        b_j_deriv_m = b_j
                        for _ in range(m):
                            b_j_deriv_m = b_j_deriv_m.diff()

                        # Binomial coefficient
                        binom = comb(i, m, exact=True)

                        # Add contribution
                        c_k = c_k + (binom * a_i * b_j_deriv_m)

            result_coeffs.append(c_k)

        return LinOp(coeffs=result_coeffs, domain=self.domain, diff_order=result_order, tol=min(self.tol, other.tol))

    def adjoint(self) -> "LinOp":
        """Compute the adjoint (formal adjoint) of this operator.

        The adjoint operator L* is the unique operator satisfying the formal
        inner product property:
            ⟨Lu, v⟩ = ⟨u, L*v⟩ + boundary terms

        where ⟨u, v⟩ = ∫ u(x) v(x) dx is the L² inner product.

        Mathematical Formula:
        --------------------
        For a linear differential operator L = sum_k a_k(x) D^k, the adjoint is:
            L* = sum_k (-1)^k D^k [a_k(x) ·]

        Expanding D^k [a_k(x) u] using Leibniz's product rule:
            D^k [a_k(x) u] = sum_j (k choose j) a_k^(j)(x) D^(k-j) u

        Therefore:
            L* = sum_k (-1)^k sum_j (k choose j) a_k^(j)(x) D^(k-j)

        Collecting coefficients of D^m (where m = k-j, giving k = m+j):
            b_m = sum_j (-1)^(m+j) (m+j choose j) a_(m+j)^(j)(x)

        where a_k^(j) denotes the j-th derivative of coefficient a_k.

        Key Properties:
        --------------
        - Adjoint of adjoint: (L*)* = L
        - Self-adjoint operators: L* = L (e.g., -d²/dx² with Dirichlet BCs)
        - Eigenvalue problems: If Lu = λu, then L*v = λ̄v (for eigenvectors v)
        - Least squares: Normal equations (L*L)u = L*f arise from min ||Lu - f||²

        Applications:
        ------------
        - Eigenvalue problems and spectral analysis
        - Least-squares fitting and optimization
        - Adjoint-based sensitivity analysis
        - Self-adjoint problems (Sturm-Liouville theory)
        - Green's functions and fundamental solutions

        Returns:
            New LinOp representing the adjoint operator L*

        Notes:
            The boundary terms in the inner product identity depend on the
            specific boundary conditions. For proper BCs (e.g., homogeneous
            Dirichlet), these terms vanish.

        Examples:
            >>> from chebpy import chebfun
            >>> from chebpy.linop import LinOp
            >>> from chebpy.utilities import Domain

            >>> # Example 1: Adjoint of first derivative
            >>> # d/dx has adjoint -d/dx
            >>> domain = Domain([0, 1])
            >>> a0 = chebfun(lambda x: 0*x, [0, 1])
            >>> a1 = chebfun(lambda x: 1 + 0*x, [0, 1])
            >>> L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)
            >>> L_adj = L.adjoint()
            >>> # L_adj has coeffs [0, -1], so L_adj = -d/dx

            >>> # Example 2: Self-adjoint second derivative
            >>> # -d²/dx² is self-adjoint
            >>> a0 = chebfun(lambda x: 0*x, [0, 1])
            >>> a1 = chebfun(lambda x: 0*x, [0, 1])
            >>> a2 = chebfun(lambda x: -1 + 0*x, [0, 1])
            >>> L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
            >>> L_adj = L.adjoint()
            >>> # L_adj == L (self-adjoint)

            >>> # Example 3: Variable coefficient operator
            >>> # L = -d/dx[x·d/dx] (Sturm-Liouville form)
            >>> # This is self-adjoint
            >>> a0 = chebfun(lambda x: 0*x, [0, 1])
            >>> a1 = chebfun(lambda x: -x, [0, 1])  # coefficient -x
            >>> L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)
            >>> L_adj = L.adjoint()

        See Also:
            expm, eigs, svds
        """
        # Compute the adjoint coefficients
        # L* = sum_k (-1)^k D^k a_k(x)
        #
        # The coefficient of D^m in L* is:
        #    b_m = sum_j (-1)^(m+j) (m+j choose j) a_(m+j)^(j)(x)

        n = self.diff_order
        new_coeffs = []

        for m in range(n + 1):
            # Initialize as zero
            b_m = Chebfun.initfun(lambda x: 0 * x, list(self.domain))

            # Sum over j such that m+j <= n (i.e., j <= n-m)
            for j in range(n - m + 1):
                k = m + j
                if k < len(self.coeffs):
                    # Get a_k
                    a_k = self.coeffs[k]

                    # Compute j-th derivative of a_k
                    a_k_deriv = a_k
                    for _ in range(j):
                        a_k_deriv = a_k_deriv.diff()

                    # Binomial coefficient and sign
                    binom_coeff = comb(k, j, exact=True)
                    sign = (-1) ** k

                    # Add to b_m
                    b_m = b_m + (sign * binom_coeff) * a_k_deriv

            new_coeffs.append(b_m)

        # Create adjoint operator
        L_adj = LinOp(coeffs=new_coeffs, domain=self.domain, diff_order=n, tol=self.tol)

        return L_adj

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
