"""OpDiscretization: Chebyshev spectral discretization of differential operators.

This module provides the OpDiscretization class for discretizing linear differential
operators on Chebyshev grids.

TWO-PHASE DESIGN PATTERN:
-------------------------
The LinOp/OpDiscretization system uses a two-phase approach:

1. SPECIFICATION PHASE (LinOp.prepare_domain):
   - LinOp creates metadata describing WHAT needs to be discretized
   - linop.blocks: List of block specifications (one per subinterval)
   - linop.continuity_constraints: List of continuity constraint specifications
   - These are abstract descriptions, not yet discretized

2. DISCRETIZATION PHASE (OpDiscretization.build_discretization):
   - OpDiscretization compiles specifications into concrete matrices
   - Generates Chebyshev collocation points on each subinterval
   - Builds differentiation matrices D^k
   - Builds diagonal coefficient multiplication matrices M_{a_k}
   - Assembles discrete operator blocks: A_block = sum_k M_{a_k} D^k
   - Constructs boundary condition constraint rows from linop.lbc/rbc
   - Constructs continuity constraint rows from linop.continuity_constraints

This separation of concerns allows LinOp to focus on mathematical specifications
while OpDiscretization handles numerical implementation details.

Responsibilities:
- Generate Chebyshev collocation points on each subinterval
- Build differentiation matrices D^k
- Build diagonal coefficient multiplication matrices M_{a_k}
- Assemble discrete operator blocks: A_block = sum_k M_{a_k} D^k
- Construct boundary condition constraint rows
- Construct interface continuity constraint rows
"""

from typing import Dict, List, Optional, Tuple

import warnings

import numpy as np
from scipy import sparse

from .spectral import cheb_points_scaled, diff_matrix, mult_matrix, identity_matrix


class OpDiscretization:
    """Discretization builder for linear differential operators.

    This class provides static methods to discretize LinOp objects
    into matrices suitable for linear algebra solvers.
    """

    @staticmethod
    def build_discretization(linop, n: int) -> Dict:
        """Build complete discretization of a LinOp at resolution n.

        NAMING CONVENTION:
            n = "number of interior degrees of freedom"
            Each block uses n+1 collocation points (including endpoints)
            Matrices are (n+1) × (n+1)

            This matches Chebyshev collocation: n degrees of freedom → n+1 points

        This is the main entry point. It:
        1. Discretizes each block interval
        2. Builds BC constraint rows
        3. Builds continuity constraint rows
        4. Packages everything into a dictionary

        Args:
            linop: LinOp instance to discretize
            n: Number of interior degrees of freedom per interval (yields n+1 points)

        Returns:
            Dictionary with:
                - 'blocks': List of operator matrices (one per interval)
                - 'bc_rows': List of BC constraint matrices
                - 'continuity_rows': List of continuity constraint matrices
                - 'rhs_blocks': List of RHS vectors for each block
                - 'bc_rhs': RHS values for BCs
                - 'continuity_rhs': RHS values for continuity
                - 'n_per_block': List of grid sizes (each = n+1) for each block
        """
        # Ensure domain is prepared
        if linop.blocks is None:
            linop.prepare_domain()

        blocks = []
        rhs_blocks = []
        n_per_block = []

        # Discretize each block
        for block_spec in linop.blocks:
            interval = block_spec['interval']
            coeffs = block_spec['coeffs']
            diff_order = block_spec['diff_order']

            # Build operator matrix for this block
            A_block = OpDiscretization._build_block_operator(
                interval, coeffs, diff_order, n
            )
            blocks.append(A_block)

            # Build RHS for this block
            if linop.rhs is not None:
                # Evaluate RHS at collocation points
                x_pts = cheb_points_scaled(n, interval)
                rhs_vals = linop.rhs(x_pts)
                rhs_blocks.append(rhs_vals)
            else:
                rhs_blocks.append(np.zeros(n + 1))

            n_per_block.append(n + 1)

        # Build boundary condition rows
        bc_rows, bc_rhs = OpDiscretization._build_bc_rows(
            linop, n_per_block
        )

        # Build continuity constraint rows
        continuity_rows, continuity_rhs = OpDiscretization._build_continuity_rows(
            linop, n_per_block
        )

        return {
            'blocks': blocks,
            'bc_rows': bc_rows,
            'continuity_rows': continuity_rows,
            'rhs_blocks': rhs_blocks,
            'bc_rhs': bc_rhs,
            'continuity_rhs': continuity_rhs,
            'n_per_block': n_per_block
        }

    @staticmethod
    def _build_block_operator(
        interval,
        coeffs: List,
        diff_order: int,
        n: int
    ) -> sparse.spmatrix:
        """Build discretized operator matrix for a single interval.

        NAMING CONVENTION:
            n = "number of interior degrees of freedom"
            Uses n+1 collocation points
            Returns (n+1) × (n+1) matrix

        Constructs:
            A = sum_{k=0}^z M_{a_k} D^k

        where:
        - M_{a_k} is diagonal matrix with a_k(x) on diagonal
        - D^k is k-th order differentiation matrix

        Args:
            interval: Interval object [a, b]
            coeffs: List of coefficient Chebfuns [a_0(x), ..., a_z(x)]
            diff_order: Maximum derivative order
            n: Number of interior degrees of freedom (yields n+1 points)

        Returns:
            Sparse (n+1) × (n+1) matrix representing discretized operator
        """
        # Get points on this interval
        x_pts = cheb_points_scaled(n, interval)
        size = n + 1

        # Initialize accumulator
        A = sparse.csr_matrix((size, size))

        # Add each term: a_k(x) * D^k
        for k in range(diff_order + 1):
            if k < len(coeffs) and coeffs[k] is not None:
                # Get coefficient function
                coeff_fun = coeffs[k]

                # Build multiplication matrix M_{a_k}
                M_k = mult_matrix(coeff_fun, n)

                # Build differentiation matrix D^k
                if k == 0:
                    D_k = identity_matrix(n)
                else:
                    D_k = diff_matrix(n, interval, order=k)

                # Add contribution
                A = A + M_k @ D_k

        return A

    @staticmethod
    def _build_bc_rows(linop, n_per_block: List[int]) -> Tuple[List, List]:
        """Build constraint rows for boundary conditions.

        POINT ORDERING:
            Chebyshev points are in ASCENDING order (left to right):
            pts[0] = left endpoint (a), pts[-1] = right endpoint (b)

            Therefore:
            - Left BC: row[0, 0] = 1.0 enforces u(a) = bc_value
            - Right BC: row[0, -1] = 1.0 enforces u(b) = bc_value

        Creates sparse row matrices that enforce:
        - Left boundary conditions (at first point = left endpoint)
        - Right boundary conditions (at last point = right endpoint)
        - Derivative boundary conditions using differentiation matrices
        - General boundary conditions

        Args:
            linop: LinOp with BC specifications
            n_per_block: Sizes of each block (each = n+1 points)

        Returns:
            (bc_rows, bc_rhs): List of constraint matrices and RHS values
        """
        bc_rows = []
        bc_rhs = []

        total_size = sum(n_per_block)
        n_blocks = len(n_per_block)

        # Get first block info for left BCs
        first_block_n = n_per_block[0]
        first_block_interval = linop.blocks[0]['interval'] if linop.blocks else linop.domain.support

        # Get last block info for right BCs
        last_block_n = n_per_block[-1]
        last_block_offset = sum(n_per_block[:-1])
        last_block_interval = linop.blocks[-1]['interval'] if linop.blocks else linop.domain.support

        # Left boundary condition
        if linop.lbc is not None:
            bc_list = linop.lbc if isinstance(linop.lbc, (list, tuple)) else [linop.lbc]

            for deriv_order, bc_value in enumerate(bc_list):
                # Skip None values (no constraint for this derivative order)
                if bc_value is None:
                    continue

                # Handle callable BC
                if callable(bc_value):
                    # Callable BC: general functional bc(u)
                    # Use linearization to extract constraint row
                    from .api import chebfun
                    import numpy as np
                    from .algorithms import chebpts2
                    a_val, b_val = linop.domain.support

                    try:
                        # Linearize BC functional around zero
                        # For linear functional L: L(u) = integral of l(x)*u(x) or point evaluation
                        # Strategy: Test BC on basis functions to extract constraint row

                        # Create interpolating polynomials at collocation points
                        # Each basis function e_i is 1 at point i, 0 elsewhere
                        constraint_row = np.zeros(first_block_n)

                        # Evaluate BC on each basis function
                        pts = chebpts2(first_block_n - 1)  # Chebyshev points on [-1, 1]
                        # Map to [a_val, b_val]
                        pts_scaled = 0.5 * (b_val - a_val) * pts + 0.5 * (a_val + b_val)

                        for i in range(first_block_n):
                            # Create basis function: 1 at point i, 0 elsewhere
                            vals = np.zeros(first_block_n)
                            vals[i] = 1.0

                            # Build chebfun from these values
                            # Use Chebyshev interpolation
                            from .chebtech import Chebtech
                            from .bndfun import Bndfun
                            from .utilities import Interval
                            from .chebfun import Chebfun
                            tech = Chebtech.initvalues(vals)
                            interval = Interval(a_val, b_val)
                            bndfun = Bndfun(tech, interval)
                            u_basis = Chebfun([bndfun])

                            # Apply BC functional
                            bc_result = bc_value(u_basis)

                            # Extract scalar value
                            if hasattr(bc_result, '__call__'):
                                # Result is a function - evaluate at left endpoint
                                val = bc_result(np.array([a_val]))[0]
                            elif isinstance(bc_result, (list, np.ndarray)):
                                # Result is array-like
                                val = float(bc_result[0] if len(bc_result) > 0 else 0.0)
                            else:
                                # Scalar
                                val = float(bc_result)

                            constraint_row[i] = val

                        # Check if BC is essentially zero (constant functional)
                        if np.linalg.norm(constraint_row) < 1e-12:
                            # Zero functional - skip this BC
                            continue

                        # Build sparse row for full system
                        row = sparse.lil_matrix((1, total_size))
                        row[0, :first_block_n] = constraint_row
                        bc_rows.append(row.tocsr())

                        # Get RHS: bc(0)
                        u_zero = chebfun(lambda x: np.zeros_like(x), [a_val, b_val])
                        bc_zero = bc_value(u_zero)
                        if hasattr(bc_zero, '__call__'):
                            rhs_val = bc_zero(np.array([a_val]))[0]
                        elif isinstance(bc_zero, (list, np.ndarray)):
                            rhs_val = float(bc_zero[0] if len(bc_zero) > 0 else 0.0)
                        else:
                            rhs_val = float(bc_zero)
                        bc_rhs.append(rhs_val)

                    except Exception as e:
                        # Fallback: simple Dirichlet at endpoint
                        warnings.warn(f"BC linearization failed: {e}. Using Dirichlet fallback.")
                        row = sparse.lil_matrix((1, total_size))
                        row[0, 0] = 1.0  # u(a) = 0
                        bc_rows.append(row.tocsr())
                        bc_rhs.append(0.0)
                else:
                    # Numeric value: u^(k)(a) = bc_value
                    row = sparse.lil_matrix((1, total_size))
                    if deriv_order == 0:
                        # Points in ascending order: pts[0] = a (left endpoint)
                        row[0, 0] = 1.0  # u(a) = bc_value
                    else:
                        # Derivative BC: u^(k)(a) = bc_value
                        # Note: first_block_n = n+1, so we call diff_matrix(n, ...)
                        D = diff_matrix(first_block_n - 1, first_block_interval, order=deriv_order)
                        # First row of D corresponds to left endpoint (pts[0] = a)
                        row[0, :first_block_n] = D[0, :].toarray()  # First row (left endpoint)
                    bc_rows.append(row.tocsr())
                    bc_rhs.append(bc_value)

        # Right boundary condition
        if linop.rbc is not None:
            bc_list = linop.rbc if isinstance(linop.rbc, (list, tuple)) else [linop.rbc]

            for deriv_order, bc_value in enumerate(bc_list):
                # Skip None values (no constraint for this derivative order)
                if bc_value is None:
                    continue

                # Handle callable BC
                if callable(bc_value):
                    # Callable BC: use linearization
                    from .api import chebfun
                    import numpy as np
                    from .algorithms import chebpts2
                    a_val, b_val = linop.domain.support

                    try:
                        # Linearize BC functional by testing on basis functions
                        constraint_row = np.zeros(last_block_n)

                        # Evaluate BC on each basis function of last block
                        pts = chebpts2(last_block_n - 1)
                        pts_scaled = 0.5 * (b_val - a_val) * pts + 0.5 * (a_val + b_val)

                        for i in range(last_block_n):
                            vals = np.zeros(last_block_n)
                            vals[i] = 1.0

                            from .chebtech import Chebtech
                            from .bndfun import Bndfun
                            from .utilities import Interval
                            from .chebfun import Chebfun
                            tech = Chebtech.initvalues(vals)
                            # Note: basis function is on last block's interval
                            # For single interval, this is the full domain
                            # For multiple intervals, need to handle carefully
                            interval = Interval(a_val, b_val)
                            bndfun = Bndfun(tech, interval)
                            u_basis = Chebfun([bndfun])

                            bc_result = bc_value(u_basis)

                            # Extract scalar value
                            if hasattr(bc_result, '__call__'):
                                val = bc_result(np.array([b_val]))[0]
                            elif isinstance(bc_result, (list, np.ndarray)):
                                val = float(bc_result[0] if len(bc_result) > 0 else 0.0)
                            else:
                                val = float(bc_result)

                            constraint_row[i] = val

                        if np.linalg.norm(constraint_row) < 1e-12:
                            continue

                        row = sparse.lil_matrix((1, total_size))
                        row[0, last_block_offset:last_block_offset+last_block_n] = constraint_row
                        bc_rows.append(row.tocsr())

                        # Get RHS
                        u_zero = chebfun(lambda x: np.zeros_like(x), [a_val, b_val])
                        bc_zero = bc_value(u_zero)
                        if hasattr(bc_zero, '__call__'):
                            rhs_val = bc_zero(np.array([b_val]))[0]
                        elif isinstance(bc_zero, (list, np.ndarray)):
                            rhs_val = float(bc_zero[0] if len(bc_zero) > 0 else 0.0)
                        else:
                            rhs_val = float(bc_zero)
                        bc_rhs.append(rhs_val)

                    except Exception as e:
                        warnings.warn(f"Right BC linearization failed: {e}. Using Dirichlet fallback.")
                        row = sparse.lil_matrix((1, total_size))
                        row[0, total_size - 1] = 1.0
                        bc_rows.append(row.tocsr())
                        bc_rhs.append(0.0)
                else:
                    # Numeric value: u^(k)(b) = bc_value
                    row = sparse.lil_matrix((1, total_size))
                    if deriv_order == 0:
                        # Points in ascending order: pts[-1] = b (right endpoint)
                        row[0, total_size - 1] = 1.0  # u(b) = bc_value
                    else:
                        # Derivative BC: u^(k)(b) = bc_value
                        # Note: last_block_n = n+1, so we call diff_matrix(n, ...)
                        D = diff_matrix(last_block_n - 1, last_block_interval, order=deriv_order)
                        # Last row of D corresponds to right endpoint (pts[-1] = b)
                        row[0, last_block_offset:last_block_offset+last_block_n] = D[-1, :].toarray()  # Last row (right endpoint)
                    bc_rows.append(row.tocsr())
                    bc_rhs.append(bc_value)

        # General boundary conditions (.bc list)
        if hasattr(linop, 'bc') and linop.bc:
            # linop.bc is a list of callable BCs: [bc1, bc2, ...]
            # Each bc is a functional bc(u) that should equal 0
            # Use functional linearization to extract constraint rows
            from .api import chebfun
            from .chebtech import Chebtech
            from .bndfun import Bndfun
            from .utilities import Interval
            from .chebfun import Chebfun

            a_val, b_val = linop.domain.support

            for bc_functional in linop.bc:
                if not callable(bc_functional):
                    # Skip non-callable BCs
                    continue

                try:
                    # Linearize BC functional by evaluating on basis functions
                    # We use full DOF space since general BCs may involve any point
                    constraint_row = np.zeros(total_size)

                    # For each DOF, evaluate BC on corresponding basis function
                    for i in range(first_block_n):
                        # Create basis function: 1 at collocation point i, 0 elsewhere
                        vals = np.zeros(first_block_n)
                        vals[i] = 1.0

                        # Build chebfun from these values
                        tech = Chebtech.initvalues(vals)
                        interval = Interval(a_val, b_val)
                        bndfun = Bndfun(tech, interval)
                        u_basis = Chebfun([bndfun])

                        # Apply BC functional
                        bc_result = bc_functional(u_basis)

                        # Extract scalar value
                        if hasattr(bc_result, '__call__'):
                            val = bc_result(np.array([a_val]))[0]
                        elif isinstance(bc_result, (list, np.ndarray)):
                            val = float(bc_result[0] if len(bc_result) > 0 else 0.0)
                        else:
                            val = float(bc_result)

                        constraint_row[i] = val

                    # Check if BC is essentially zero
                    if np.linalg.norm(constraint_row) < 1e-12:
                        continue

                    # Build sparse row for full system
                    row = sparse.lil_matrix((1, total_size))
                    row[0, :first_block_n] = constraint_row[:first_block_n]
                    bc_rows.append(row.tocsr())

                    # Get RHS: bc(0)
                    u_zero = chebfun(lambda x: np.zeros_like(x), [a_val, b_val])
                    bc_zero = bc_functional(u_zero)
                    if hasattr(bc_zero, '__call__'):
                        rhs_val = bc_zero(np.array([a_val]))[0]
                    elif isinstance(bc_zero, (list, np.ndarray)):
                        rhs_val = float(bc_zero[0] if len(bc_zero) > 0 else 0.0)
                    else:
                        rhs_val = float(bc_zero)
                    bc_rhs.append(rhs_val)

                except Exception as e:
                    warnings.warn(f"General BC linearization failed: {e}. Skipping this BC.")
                    continue

        return bc_rows, bc_rhs

    @staticmethod
    def _build_continuity_rows(linop, n_per_block: List[int]) -> Tuple[List, List]:
        """Build constraint rows for continuity at interfaces.

        This implements the DISCRETIZATION PHASE for continuity constraints.
        It compiles the abstract constraint specifications from
        linop.continuity_constraints into concrete sparse matrix rows.

        For each internal breakpoint, enforces:
            u_left(x_break) - u_right(x_break) = 0
            u'_left(x_break) - u'_right(x_break) = 0
            ...
            u^(z-1)_left(x_break) - u^(z-1)_right(x_break) = 0

        Each constraint specification from linop.continuity_constraints is a dict:
            {
                'type': 'continuity',
                'location': bp,
                'left_block': left_idx,
                'right_block': right_idx,
                'derivative_order': deriv_order
            }

        This is compiled into a sparse matrix row that enforces the constraint
        on the global nodal solution vector.

        Args:
            linop: LinOp with continuity_constraints specifications
            n_per_block: Sizes of each block

        Returns:
            (continuity_rows, continuity_rhs): List of constraint matrices and RHS
        """
        continuity_rows = []
        continuity_rhs = []

        if linop.continuity_constraints is None:
            return continuity_rows, continuity_rhs

        total_size = sum(n_per_block)

        for constraint in linop.continuity_constraints:
            left_block = constraint['left_block']
            right_block = constraint['right_block']
            deriv_order = constraint['derivative_order']

            # Compute offsets to find where each block starts in global vector
            offset_left = sum(n_per_block[:left_block])
            offset_right = sum(n_per_block[:right_block])

            n_left = n_per_block[left_block]
            n_right = n_per_block[right_block]

            if deriv_order == 0:
                # Continuity of function value
                # u_left[last] - u_right[first] = 0
                row = sparse.lil_matrix((1, total_size))
                row[0, offset_left + n_left - 1] = 1.0  # Last point of left block
                row[0, offset_right] = -1.0  # First point of right block
                continuity_rows.append(row.tocsr())
                continuity_rhs.append(0.0)

            else:
                # Continuity of derivatives (more complex)
                # Need to compute derivative at boundary using differentiation matrix
                # For now, simplified implementation
                interval_left = linop.blocks[left_block]['interval']
                interval_right = linop.blocks[right_block]['interval']

                # Get differentiation matrices
                D_left = diff_matrix(n_left - 1, interval_left, order=deriv_order)
                D_right = diff_matrix(n_right - 1, interval_right, order=deriv_order)

                # Extract rows corresponding to boundary points
                # Left block: last row
                # Right block: first row
                row = sparse.lil_matrix((1, total_size))

                # Set coefficients from differentiation matrix
                # This is simplified; full implementation needs careful indexing
                row[0, offset_left:offset_left + n_left] = D_left[-1, :].toarray()
                row[0, offset_right:offset_right + n_right] = -D_right[0, :].toarray()

                continuity_rows.append(row.tocsr())
                continuity_rhs.append(0.0)

        return continuity_rows, continuity_rhs

    @staticmethod
    def evaluate_at_points(fun, points: np.ndarray) -> np.ndarray:
        """Evaluate a Chebfun at given points.

        Helper method for RHS evaluation.

        Args:
            fun: Chebfun to evaluate
            points: Points at which to evaluate

        Returns:
            Array of function values
        """
        return fun(points)
