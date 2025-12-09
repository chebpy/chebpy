"""Chebyshev spectral discretization of differential operators.

This module provides the op_discretization class for discretizing linear differential
operators on Chebyshev grids.

Responsibilities:
- Generate Chebyshev collocation points on each subinterval
- Build differentiation matrices D^k
- Build diagonal coefficient multiplication matrices M_{a_k}
- Assemble discrete operator blocks: A_block = sum_k M_{a_k} D^k
- Construct constraint rows
"""

import numpy as np
from scipy import sparse

from .adchebfun import linearize_bc_matrix
from .algorithms import clencurt_weights
from .bndfun import Bndfun
from .chebfun import Chebfun
from .chebtech import Chebtech
from .sparse_utils import extract_scalar, is_nearly_zero, prune_sparse, sparse_to_dense
from .spectral import (
    barycentric_matrix,
    cheb_points_scaled,
    diff_matrix,
    diff_matrix_driscoll_hale,
    diff_matrix_rectangular,
    fourier_diff_matrix,
    fourier_points_scaled,
    identity_matrix,
    mult_matrix,
    projection_matrix_rectangular,
)
from .utilities import Interval


class OpDiscretization:
    """Discretization builder for linear differential operators.

    This class provides static methods to discretize LinOp objects
    into matrices suitable for linear algebra solvers.
    """

    @staticmethod
    def _create_sparse_row(total_size: int, indices: list, values: list) -> sparse.spmatrix:
        """Create a sparse row matrix with specified indices and values.

        Args:
            total_size: Total number of columns in the row
            indices: List of column indices to set
            values: List of values to set at those indices

        Returns:
            Sparse CSR matrix (1 x total_size)
        """
        row = sparse.lil_matrix((1, total_size))
        for idx, val in zip(indices, values):
            row[0, idx] = val
        return row.tocsr()

    @staticmethod
    def _setup_linearization_point(linop, u_current, block_interval_obj=None):
        """Setup linearization point for callable BC evaluation.

        Args:
            linop: LinOp instance
            u_current: Current solution for linearization (or None)
            block_interval_obj: Interval object for the block (or None to use domain)

        Returns:
            Tuple of (a_val, b_val, u_lin) where u_lin is the linearization Chebfun
        """
        if block_interval_obj is not None:
            a_val, b_val = block_interval_obj
        else:
            a_val, b_val = linop.domain.support

        if u_current is not None:
            u_lin = u_current
        else:
            u_lin = Chebfun.initfun(lambda x: np.zeros_like(x), [a_val, b_val])

        return a_val, b_val, u_lin

    @staticmethod
    def _is_periodic(linop) -> bool:
        """Check if the LinOp has periodic boundary conditions.

        Periodic BCs are detected if continuity_constraints contain 'periodic' type entries.
        This follows MATLAB Chebfun's approach where periodic BCs trigger Fourier collocation.

        Args:
            linop: LinOp instance

        Returns:
            True if periodic BCs are used, False otherwise
        """
        if linop.continuity_constraints is None:
            return False

        # Check if any constraint is periodic
        for constraint in linop.continuity_constraints:
            if constraint.get("type") == "periodic":
                return True

        return False

    @staticmethod
    def build_discretization(
        linop,
        n: int,
        u_current=None,
        m: int = None,
        rectangularization: bool = False,
        for_eigenvalue_problem: bool = False,
        bc_enforcement: str = "append",
    ) -> dict:
        """Build discretization of a LinOp at resolution n.

        This is the main entry point. It:
        1. Discretizes each block interval
        2. Builds constraint rows
        3. Packages everything into a dictionary

        Args:
            linop: LinOp instance to discretize
            n: Number of interior degrees of freedom per interval (yields n+1 points)
            u_current: Current solution for linearizing callable BCs (for Newton iteration)
            m: Number of collocation points for rectangular discretization (yields m+1 points).
               If None and rectangularization=True, uses heuristic: min(2*n, n+50).
               Must satisfy m >= n.
            rectangularization: If True, use rectangular (overdetermined) discretization
                               for improved eigenvalue accuracy and numerical stability.
            for_eigenvalue_problem: If True, prepare discretization for eigenvalue solving.
            bc_enforcement: Strategy for enforcing boundary conditions:
                          - 'append': Append BC rows to operator (overdetermined system, least squares)
                          - 'replace': Replace operator rows with BC rows (square system, exact BCs)
                                     This is the MATLAB Chebfun approach for spectral accuracy.
                          - 'driscoll_hale': Use Driscoll-Hale rectangular collocation where the
                                     operator matrix is (n-k+1) × (n+1) for k-th order ODE, then
                                     k BC rows are added to form a square (n+1) × (n+1) system.
                                     This preserves polynomial accuracy without row deletion.
                                     Reference: Driscoll & Hale (2016), IMA J. Numer. Anal.

        Returns:
            Dictionary with:
                - 'blocks': List of operator matrices (one per interval)
                - 'bc_rows': List of BC constraint matrices
                - 'continuity_rows': List of continuity constraint matrices
                - 'rhs_blocks': List of RHS vectors for each block
                - 'bc_rhs': RHS values for BCs
                - 'continuity_rhs': RHS values for continuity
                - 'n_per_block': List of coefficient grid sizes (each = n+1) for each block
                - 'm_per_block': List of collocation grid sizes (each = m+1 if rectangular, else n+1)
                - 'rectangularization': Boolean indicating if rectangular discretization was used
                - 'bc_enforcement': The BC enforcement strategy used
        """
        if linop.blocks is None:
            linop.prepare_domain()

        # Check if periodic BCs are used (triggers Fourier collocation)
        use_fourier = OpDiscretization._is_periodic(linop) and len(linop.blocks) == 1

        # Determine m for rectangularization
        if rectangularization and m is None:
            m = min(2 * n, n + 50)
        elif not rectangularization:
            m = n  # Square discretization

        if m < n:
            raise ValueError(f"Rectangular discretization requires m >= n, got m={m}, n={n}")

        blocks = []
        rhs_blocks = []
        n_per_block = []
        m_per_block = []

        # Check for Driscoll-Hale mode
        use_driscoll_hale = bc_enforcement == "driscoll_hale"

        # Discretize each block
        for block_spec in linop.blocks:
            interval = block_spec["interval"]
            coeffs = block_spec["coeffs"]
            diff_order = block_spec["diff_order"]

            A_block = OpDiscretization._build_block_operator(
                interval,
                coeffs,
                diff_order,
                n,
                m=m,
                use_fourier=use_fourier,
                rectangularization=rectangularization,
                bc_enforcement=bc_enforcement,
            )
            blocks.append(A_block)

            # Build RHS for this block
            if linop.rhs is not None:
                # Evaluate RHS at collocation points
                if use_fourier:
                    # Fourier for periodic problems: use n points (not affected by rectangularization)
                    x_pts = fourier_points_scaled(n, interval)
                elif use_driscoll_hale:
                    # Driscoll-Hale: n-diff_order+1 points
                    m_dh = n - diff_order
                    x_pts = cheb_points_scaled(m_dh, interval)
                else:
                    # Chebyshev otherwise: use m+1 collocation points for rectangular
                    x_pts = cheb_points_scaled(m, interval)
                rhs_vals = linop.rhs(x_pts)
                rhs_blocks.append(rhs_vals)
            else:
                # Periodic: n points, Driscoll-Hale: n-diff_order+1, Other: m+1 points
                if use_fourier:
                    rhs_size = n
                elif use_driscoll_hale:
                    rhs_size = n - diff_order + 1
                else:
                    rhs_size = m + 1
                rhs_blocks.append(np.zeros(rhs_size))

            # Store grid sizes
            # n_per_block: coefficient DOFs (n+1 for Chebyshev)
            # m_per_block: collocation/output points
            if use_fourier:
                n_per_block.append(n)
                m_per_block.append(n)
            elif use_driscoll_hale:
                n_per_block.append(n + 1)
                m_per_block.append(n - diff_order + 1)
            else:
                n_per_block.append(n + 1)
                m_per_block.append(m + 1)

        # Build boundary condition rows
        # For periodic problems with Fourier collocation, no BC rows needed
        # Periodicity is implicit in the toeplitz structure of Fourier diff matrices
        if use_fourier:
            bc_rows, bc_rhs = [], []
        else:
            bc_rows, bc_rhs = OpDiscretization._build_bc_rows(linop, n_per_block, u_current)

        # Build continuity constraint rows
        # For periodic problems with Fourier collocation, no continuity rows needed (unless eigenvalue problem)
        continuity_rows, continuity_rhs = OpDiscretization._build_continuity_rows(
            linop, n_per_block, use_fourier=use_fourier, for_eigenvalue_problem=for_eigenvalue_problem
        )

        # Build integral constraint rows
        integral_rows, integral_rhs = OpDiscretization._build_integral_constraint_rows(linop, n_per_block)

        # Build point constraint rows
        point_rows, point_rhs = OpDiscretization._build_point_constraint_rows(linop, n_per_block)

        # Add mean-zero constraint for periodic Fourier problems with even order >= 2
        # This resolves the nullspace issue where constants are in the kernel of
        # higher-order differential operators (e.g., u'''' = f has solution u + C)
        mean_zero_rows, mean_zero_rhs = OpDiscretization._build_mean_zero_constraint(
            linop, n_per_block, use_fourier=use_fourier
        )

        # Build projection matrices PS for rectangular discretization
        # PS projects from m+1 collocation points to n+1 coefficient points
        projection_matrices = []
        if rectangularization and not use_fourier:
            for block_spec in linop.blocks:
                interval = block_spec["interval"]
                PS_block = projection_matrix_rectangular(n, m, interval)
                projection_matrices.append(PS_block)

        return {
            "blocks": blocks,
            "bc_rows": bc_rows,
            "continuity_rows": continuity_rows,
            "integral_rows": integral_rows,
            "point_rows": point_rows,
            "mean_zero_rows": mean_zero_rows,
            "rhs_blocks": rhs_blocks,
            "bc_rhs": bc_rhs,
            "continuity_rhs": continuity_rhs,
            "integral_rhs": integral_rhs,
            "point_rhs": point_rhs,
            "mean_zero_rhs": mean_zero_rhs,
            "n_per_block": n_per_block,
            "m_per_block": m_per_block,
            "rectangularization": rectangularization,
            "projection_matrices": projection_matrices,
            "bc_enforcement": bc_enforcement,
        }

    @staticmethod
    def _build_block_operator(
        interval,
        coeffs: list,
        diff_order: int,
        n: int,
        m: int = None,
        use_fourier: bool = False,
        rectangularization: bool = False,
        bc_enforcement: str = "append",
    ) -> sparse.spmatrix:
        """Build discretized operator matrix for a single interval.

        Constructs:
            A = sum_{k=0}^z M_{a_k} D^k

        where:
        - M_{a_k} is diagonal matrix with a_k(x) on diagonal
        - D^k is k-th order differentiation matrix

        Args:
            interval: Interval object [a, b]
            coeffs: List of coefficient Chebfuns [a_0(x), ..., a_z(x)]
            diff_order: Maximum derivative order
            n: Number of interior degrees of freedom (Chebyshev: yields n+1 points, Fourier: n points)
            m: Number of collocation points for rectangular discretization (yields m+1 points)
            use_fourier: If True, use Fourier collocation; otherwise use Chebyshev
            rectangularization: If True, build rectangular (overdetermined) matrix
            bc_enforcement: BC enforcement strategy ('append', 'replace', or 'driscoll_hale')

        Returns:
            Sparse matrix representing discretized operator
            - Chebyshev square: (n+1) x (n+1)
            - Chebyshev rectangular (overdetermined): (m+1) x (n+1)
            - Chebyshev Driscoll-Hale: (n-diff_order+1) x (n+1)
            - Fourier: n x n (not affected by rectangularization)
        """
        # Determine grid sizes and mode
        use_driscoll_hale = bc_enforcement == "driscoll_hale"
        max_order = diff_order if diff_order is not None else (len(coeffs) - 1 if coeffs else 0)

        if use_fourier:
            # Fourier: n x n (rectangularization not applicable)
            row_size = n
            col_size = n
        elif use_driscoll_hale:
            # Driscoll-Hale: (n-diff_order+1) x (n+1) for operator rows
            # BCs will be added separately to make it square
            row_size = n - max_order + 1
            col_size = n + 1
        elif rectangularization and m is not None:
            # Chebyshev rectangular (overdetermined): (m+1) x (n+1)
            row_size = m + 1
            col_size = n + 1
        else:
            # Chebyshev square: (n+1) x (n+1)
            row_size = n + 1
            col_size = n + 1

        # Initialize accumulator
        A = sparse.csr_matrix((row_size, col_size))

        # Add each term: a_k(x) * D^k
        for k in range(max_order + 1):
            if k < len(coeffs) and coeffs[k] is not None:
                # Get coefficient function
                coeff_fun = coeffs[k]

                # Build multiplication matrix M_{a_k}
                # Need to evaluate at correct grid points for Fourier or Chebyshev
                if use_fourier:
                    x_pts = fourier_points_scaled(n, interval)
                    coeff_vals = coeff_fun(x_pts)
                    coeff_vals = np.atleast_1d(coeff_vals).ravel()
                    M_k = sparse.diags(coeff_vals, 0, format="csr")
                elif use_driscoll_hale:
                    # Driscoll-Hale: evaluate at n-max_order+1 output points
                    m_dh = n - max_order
                    x_pts = cheb_points_scaled(m_dh, interval)
                    coeff_vals = coeff_fun(x_pts)
                    coeff_vals = np.atleast_1d(coeff_vals).ravel()
                    M_k = sparse.diags(coeff_vals, 0, format="csr")
                else:
                    # For rectangular, evaluate at m+1 collocation points
                    if rectangularization and m is not None:
                        x_pts = cheb_points_scaled(m, interval)
                        coeff_vals = coeff_fun(x_pts)
                        coeff_vals = np.atleast_1d(coeff_vals).ravel()
                        M_k = sparse.diags(coeff_vals, 0, format="csr")
                    else:
                        M_k = mult_matrix(coeff_fun, n, interval=interval)

                # Build differentiation matrix D^k
                if use_fourier:
                    if k == 0:
                        D_k = sparse.eye(n, format="csr")
                    else:
                        D_k = sparse.csr_matrix(fourier_diff_matrix(n, interval, order=k))
                elif use_driscoll_hale:
                    # Driscoll-Hale rectangular differentiation
                    # D^k maps n+1 values to (n-max_order+1) values
                    if k == 0:
                        # Identity needs to resample from n+1 to n-max_order+1 points
                        m_dh = n - max_order
                        x_output = cheb_points_scaled(m_dh, interval)
                        D_k = barycentric_matrix(x_output, n, interval)
                        D_k = sparse.csr_matrix(D_k)
                    else:
                        D_k = diff_matrix_driscoll_hale(n, interval, order=k)
                        # Resample to match output size if k < max_order
                        if k < max_order:
                            # D_k is (n-k+1) × (n+1), need (n-max_order+1) × (n+1)
                            m_dh = n - max_order
                            m_k = n - k
                            if m_k != m_dh:
                                # Resample from m_k+1 to m_dh+1 points
                                x_output = cheb_points_scaled(m_dh, interval)
                                P = barycentric_matrix(x_output, m_k, interval)
                                D_k = sparse.csr_matrix(P @ sparse_to_dense(D_k))
                elif rectangularization and m is not None:
                    # Rectangular (overdetermined): use interpolation (k=0) or diff matrix (k>0)
                    D_k = diff_matrix_rectangular(n, m, interval, order=k)
                else:
                    # Square Chebyshev
                    if k == 0:
                        D_k = identity_matrix(n)
                    else:
                        D_k = diff_matrix(n, interval, order=k)

                # Add contribution
                A = A + M_k @ D_k

        return A

    @staticmethod
    def _build_endpoint_bc(
        is_left: bool,
        bc_value,
        deriv_order: int,
        linop,
        n_per_block: list[int],
        u_current,
        total_size: int,
    ) -> tuple[list, list]:
        """Build boundary condition rows for left or right endpoint.

        Args:
            is_left: True for left BC, False for right BC
            bc_value: BC value (numeric or callable)
            deriv_order: Derivative order for this BC
            linop: LinOp instance
            n_per_block: Sizes of each block
            u_current: Current solution for linearization
            total_size: Total system size

        Returns:
            Tuple of (bc_rows, bc_rhs) for this BC
        """
        bc_rows = []
        bc_rhs = []

        if is_left:
            block_n = n_per_block[0]
            block_offset = 0
            block_interval = linop.blocks[0]["interval"] if linop.blocks else linop.domain.support
            endpoint_idx = 0
        else:
            block_n = n_per_block[-1]
            block_offset = sum(n_per_block[:-1])
            block_interval = linop.blocks[-1]["interval"] if linop.blocks else linop.domain.support
            endpoint_idx = total_size - 1

        # Handle callable BC
        if callable(bc_value):
            # Get interval and setup linearization point
            interval_obj = linop.blocks[0 if is_left else -1]["interval"] if linop.blocks else None
            a_val, b_val, u_lin = OpDiscretization._setup_linearization_point(linop, u_current, interval_obj)

            # Linearize BC at appropriate endpoint
            x_bc = a_val if is_left else b_val
            residual, constraint_row = linearize_bc_matrix(bc_value, u_lin, n=block_n - 1, x_bc=x_bc)

            # Check if BC returned multiple constraints
            if isinstance(residual, (list, tuple)):
                for res_i, row_i in zip(residual, constraint_row):
                    if is_nearly_zero(row_i):
                        continue

                    row = sparse.lil_matrix((1, total_size))
                    row[0, block_offset : block_offset + block_n] = row_i
                    bc_rows.append(row.tocsr())
                    bc_rhs.append(extract_scalar(res_i, negate=True))
            else:
                # Single constraint
                if is_nearly_zero(constraint_row):
                    return bc_rows, bc_rhs

                row = sparse.lil_matrix((1, total_size))
                row[0, block_offset : block_offset + block_n] = constraint_row
                bc_rows.append(row.tocsr())
                bc_rhs.append(extract_scalar(residual, negate=True))
        else:
            # Numeric value: u^(k)(endpoint) = bc_value
            row = sparse.lil_matrix((1, total_size))
            if deriv_order == 0:
                row[0, endpoint_idx] = 1.0
            else:
                # Derivative BC
                D = diff_matrix(block_n - 1, block_interval, order=deriv_order)
                row_idx = 0 if is_left else -1
                row[0, block_offset : block_offset + block_n] = sparse_to_dense(D[row_idx, :])
            bc_rows.append(row.tocsr())
            bc_rhs.append(bc_value)

        return bc_rows, bc_rhs

    @staticmethod
    def _build_bc_rows(linop, n_per_block: list[int], u_current=None) -> tuple[list, list]:
        """Build constraint rows for boundary conditions.

        Point ordering:
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
            n_per_block: Sizes of each block (each = n+1 points for Chebyshev, n for Fourier)
            u_current: Current solution for linearizing callable BCs (Newton iteration)
            use_fourier: If True and problem is periodic, skip BC rows (periodicity implicit in Fourier)

        Returns:
            (bc_rows, bc_rhs): List of constraint matrices and RHS values
        """
        bc_rows = []
        bc_rhs = []
        total_size = sum(n_per_block)

        # Left boundary condition
        if linop.lbc is not None:
            bc_list = linop.lbc if isinstance(linop.lbc, (list, tuple)) else [linop.lbc]

            for deriv_order, bc_value in enumerate(bc_list):
                if bc_value is None:
                    continue

                rows, rhs = OpDiscretization._build_endpoint_bc(
                    is_left=True,
                    bc_value=bc_value,
                    deriv_order=deriv_order,
                    linop=linop,
                    n_per_block=n_per_block,
                    u_current=u_current,
                    total_size=total_size,
                )
                bc_rows.extend(rows)
                bc_rhs.extend(rhs)

        # Right boundary condition
        if linop.rbc is not None:
            bc_list = linop.rbc if isinstance(linop.rbc, (list, tuple)) else [linop.rbc]

            for deriv_order, bc_value in enumerate(bc_list):
                if bc_value is None:
                    continue

                rows, rhs = OpDiscretization._build_endpoint_bc(
                    is_left=False,
                    bc_value=bc_value,
                    deriv_order=deriv_order,
                    linop=linop,
                    n_per_block=n_per_block,
                    u_current=u_current,
                    total_size=total_size,
                )
                bc_rows.extend(rows)
                bc_rhs.extend(rhs)

        # General boundary conditions (.bc list)
        if hasattr(linop, "bc") and linop.bc:
            # linop.bc is a list of callable BCs: [bc1, bc2, ...]
            # Each bc is a functional bc(u) that should equal 0
            # Use functional linearization to extract constraint rows

            a_val, b_val = linop.domain.support
            first_block_n = n_per_block[0]

            for bc_functional in linop.bc:
                if not callable(bc_functional):
                    continue

                # Linearize BC functional by evaluating on basis functions
                constraint_row = np.zeros(total_size)

                # For each DOF, evaluate BC on corresponding basis function
                for i in range(first_block_n):
                    vals = np.zeros(first_block_n)
                    vals[i] = 1.0

                    tech = Chebtech.initvalues(vals)
                    interval = Interval(a_val, b_val)
                    bndfun = Bndfun(tech, interval)
                    u_basis = Chebfun([bndfun])

                    bc_result = bc_functional(u_basis)

                    # Extract scalar value
                    if hasattr(bc_result, "__call__"):
                        val = bc_result(np.array([a_val]))[0]
                    else:
                        val = extract_scalar(bc_result)

                    constraint_row[i] = val

                # Check if BC is essentially zero
                if is_nearly_zero(constraint_row):
                    continue

                # Build sparse row for full system
                row = sparse.lil_matrix((1, total_size))
                row[0, :first_block_n] = constraint_row[:first_block_n]
                bc_rows.append(row.tocsr())

                # Get RHS: bc(0)
                u_zero = Chebfun.initfun(lambda x: np.zeros_like(x), [a_val, b_val])
                bc_zero = bc_functional(u_zero)
                if hasattr(bc_zero, "__call__"):
                    rhs_val = bc_zero(np.array([a_val]))[0]
                else:
                    rhs_val = extract_scalar(bc_zero)
                bc_rhs.append(rhs_val)

        return bc_rows, bc_rhs

    @staticmethod
    def _build_continuity_rows(
        linop, n_per_block: list[int], use_fourier: bool = False, for_eigenvalue_problem: bool = False
    ) -> tuple[list, list]:
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

        Note on derivative orders:
            The derivative_order must satisfy deriv_order < max_operator_order.
            This validation is expected to be performed upstream during constraint
            generation (e.g., in LinOp construction). If the highest derivative
            in the operator is z, continuity is enforced for orders 0, ..., z-1.

        Args:
            linop: LinOp with continuity_constraints specifications
            n_per_block: Sizes of each block
            use_fourier: Whether Fourier collocation is being used
            for_eigenvalue_problem: If True, always generate constraint rows even for
                                   Fourier periodic problems (eigenvalue problems need
                                   explicit constraints, not implicit periodicity)

        Returns:
            (continuity_rows, continuity_rhs): List of constraint matrices and RHS
        """
        # For periodic BVPs with Fourier collocation, no continuity rows needed
        # Periodicity is implicit in the toeplitz structure of Fourier diff matrices
        # However, for eigenvalue problems we ALWAYS need explicit constraint rows
        # to project onto the periodic subspace
        if use_fourier and OpDiscretization._is_periodic(linop) and not for_eigenvalue_problem:
            return [], []

        continuity_rows = []
        continuity_rhs = []

        if linop.continuity_constraints is None:
            return continuity_rows, continuity_rhs

        total_size = sum(n_per_block)

        for constraint in linop.continuity_constraints:
            constraint_type = constraint.get("type", "continuity")

            if constraint_type == "periodic":
                # Periodic BC: u^(k)(a) = u^(k)(b)
                block_idx = constraint["block"]
                deriv_order = constraint["derivative_order"]

                offset = sum(n_per_block[:block_idx])
                n_block = n_per_block[block_idx]

                if deriv_order == 0:
                    # u(a) - u(b) = 0
                    # First point minus last point
                    row = sparse.lil_matrix((1, total_size))
                    row[0, offset] = 1.0  # First point (a)
                    row[0, offset + n_block - 1] = -1.0  # Last point (b)
                    continuity_rows.append(row.tocsr())
                    continuity_rhs.append(0.0)
                else:
                    # u'^(k)(a) - u'^(k)(b) = 0
                    interval = linop.blocks[block_idx]["interval"]
                    D = diff_matrix(n_block - 1, interval, order=deriv_order)

                    row = sparse.lil_matrix((1, total_size))
                    # Derivative at first point minus derivative at last point
                    row[0, offset : offset + n_block] = D[0, :] - D[-1, :]

                    continuity_rows.append(prune_sparse(row))
                    continuity_rhs.append(0.0)

                continue  # Skip to next constraint

            # Regular continuity constraint
            left_block = constraint["left_block"]
            right_block = constraint["right_block"]
            deriv_order = constraint["derivative_order"]

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
                # Continuity of derivatives
                # Enforce u^(k)_left(x_break) - u^(k)_right(x_break) = 0
                # using differentiation matrix boundary rows

                # Note: diff_matrix() already includes interval scaling,
                # so these matrices are properly scaled for their physical intervals
                interval_left = linop.blocks[left_block]["interval"]
                interval_right = linop.blocks[right_block]["interval"]

                # Get differentiation matrices (already scaled for physical intervals)
                D_left = diff_matrix(n_left - 1, interval_left, order=deriv_order)
                D_right = diff_matrix(n_right - 1, interval_right, order=deriv_order)

                # Extract boundary rows: last row of left block, first row of right block
                # This implements the evaluation: E_{x=b} · D^k for left, E_{x=a} · D^k for right
                row = sparse.lil_matrix((1, total_size))

                # Assign sparse rows directly (no .toarray() conversion)
                row[0, offset_left : offset_left + n_left] = D_left[-1, :]
                row[0, offset_right : offset_right + n_right] = -D_right[0, :]

                continuity_rows.append(prune_sparse(row))
                continuity_rhs.append(0.0)

        return continuity_rows, continuity_rhs

    @staticmethod
    def _build_integral_constraint_rows(linop, n_per_block: list[int]) -> tuple[list, list]:
        """Build constraint rows for integral constraints.

        Integral constraints have the form:
            ∫ g(x) u(x) dx = c

        where g(x) is a weight function (or 1 for unweighted integral).

        This is discretized using Clenshaw-Curtis quadrature:
            ∑_i w_i g(x_i) u(x_i) = c

        where w_i are quadrature weights and x_i are collocation points.

        Args:
            linop: LinOp with integral_constraint specification
            n_per_block: Sizes of each block

        Returns:
            (integral_rows, integral_rhs): List of constraint matrices and RHS values
        """
        integral_rows = []
        integral_rhs = []

        if linop.integral_constraint is None:
            return integral_rows, integral_rhs

        # Handle single constraint or list of constraints
        constraints = linop.integral_constraint
        if isinstance(constraints, dict):
            constraints = [constraints]

        total_size = sum(n_per_block)

        for constraint in constraints:
            weight_fun = constraint.get("weight", None)
            target_value = constraint.get("value", 0.0)

            # Build constraint row using Clenshaw-Curtis quadrature
            row = sparse.lil_matrix((1, total_size))

            offset = 0
            for block_idx, n_block in enumerate(n_per_block):
                # Get interval for this block
                interval = linop.blocks[block_idx]["interval"] if linop.blocks else linop.domain.support

                # Get collocation points on this interval
                x_pts = cheb_points_scaled(n_block - 1, interval)

                # Get Clenshaw-Curtis quadrature weights
                # Standard CC formula for n+1 Chebyshev points on [-1, 1]
                a, b = interval
                n = n_block - 1  # n_block = n+1 points

                # Clenshaw-Curtis weights using FFT-based O(n log n) algorithm
                weights = clencurt_weights(n)

                # Scale from [-1, 1] to [a, b]
                weights *= (b - a) / 2.0

                # Apply weight function if specified
                if weight_fun is not None:
                    g_vals = weight_fun(x_pts)
                    # Ensure g_vals is 1D (MATLAB Chebfun returns column vectors)
                    g_vals = np.atleast_1d(g_vals).ravel()
                    weights = weights * g_vals

                # Set row entries
                row[0, offset : offset + n_block] = weights

                offset += n_block

            integral_rows.append(row.tocsr())
            integral_rhs.append(target_value)

        return integral_rows, integral_rhs

    @staticmethod
    def _build_point_constraint_rows(linop, n_per_block) -> tuple[list, list]:
        """Build point constraint rows for interior BCs.

        Point constraints enforce conditions at specific interior points, e.g.:
            u(0.5) = 0.5
            u'(0.3) = 1.0
            u''(0.7) = -0.2

        Args:
            linop: LinOp with point_constraints attribute
            n_per_block: List of grid sizes for each block

        Returns:
            Tuple of (constraint_rows, constraint_rhs) where:
                - constraint_rows: List of sparse matrices (one row per constraint)
                - constraint_rhs: List of RHS values
        """
        if not linop.point_constraints:
            return [], []

        constraint_rows = []
        constraint_rhs = []
        total_dofs = sum(n_per_block)

        for pc in linop.point_constraints:
            x_loc = pc["location"]
            deriv_order = pc["derivative_order"]
            value = pc["value"]

            # Find which block contains x_loc
            block_idx = None
            for idx, block_spec in enumerate(linop.blocks):
                interval = block_spec["interval"]
                a, b = interval
                if a <= x_loc <= b:
                    block_idx = idx
                    break

            if block_idx is None:
                raise ValueError(f"Point constraint location {x_loc} is outside domain {linop.domain}")

            # Compute offset to this block
            offset = sum(n_per_block[:block_idx])
            n_block = n_per_block[block_idx]
            n = n_block - 1  # Number of intervals

            # Get interval for this block
            interval = linop.blocks[block_idx]["interval"]

            # Build evaluation row for derivative of order deriv_order at x_loc
            row = sparse.lil_matrix((1, total_dofs))

            if deriv_order == 0:
                # Just evaluate at x_loc using barycentric interpolation
                E = barycentric_matrix(np.array([x_loc]), n, interval)
                row[0, offset : offset + n_block] = E[0, :]
            else:
                # Evaluate derivative at x_loc
                # Strategy: D^k maps coefficients to derivative values at collocation points
                # Then evaluate at x_loc using barycentric interpolation

                # Get differentiation matrix D^k: (n+1) x (n+1)
                D = diff_matrix(n, interval, order=deriv_order)

                # Evaluate at x_loc: barycentric interpolation from collocation points
                # E: 1 x (n+1) - evaluates at x_loc from values at collocation points
                E = barycentric_matrix(np.array([x_loc]), n, interval)

                # Combined: E @ D evaluates k-th derivative at x_loc
                ED = E @ D
                row[0, offset : offset + n_block] = ED[0, :]

            constraint_rows.append(row.tocsr())
            constraint_rhs.append(value)

        return constraint_rows, constraint_rhs

    @staticmethod
    def _build_mean_zero_constraint(linop, n_per_block: list[int], use_fourier: bool = False) -> tuple[list, list]:
        """Build mean-zero constraint for periodic Fourier problems.

        For periodic problems with Fourier collocation and even-order derivatives >= 2,
        the differentiation matrix has constants in its nullspace. This causes the
        system to be singular. Following MATLAB Chebfun's approach, we add a mean-zero
        constraint to make the system well-determined:

            ∫ u(x) dx = 0   (mean-zero constraint)

        This is discretized using Fourier quadrature with trapezoidal weights.

        Mathematical background:
            For u^(2k) = f with periodic BCs (k >= 1), both u(x) and u(x) + C satisfy
            the equation. The mean-zero constraint uniquely determines the solution.

        Args:
            linop: LinOp with differential operator specification
            n_per_block: Sizes of each block (n points for Fourier)
            use_fourier: If True and conditions are met, add mean-zero constraint

        Returns:
            (mean_zero_rows, mean_zero_rhs): Constraint matrix rows and RHS (0.0)
        """
        # Only apply to periodic Fourier problems
        if not (use_fourier and OpDiscretization._is_periodic(linop)):
            return [], []

        # Check if we have even-order derivative >= 2
        # This is when the nullspace issue occurs
        if linop.blocks is None or len(linop.blocks) == 0:
            return [], []

        # Find maximum differentiation order in the operator
        max_diff_order = 0
        for block_spec in linop.blocks:
            diff_order = block_spec.get("diff_order", 0)
            if diff_order is not None and diff_order > max_diff_order:
                max_diff_order = diff_order

        # Only add constraint for even orders >= 2
        # (Second order actually works without it, but we add for robustness)
        if max_diff_order < 2 or max_diff_order % 2 != 0:
            return [], []

        # Build the mean-zero constraint: ∫ u dx = 0
        # For Fourier collocation on [0, 2π], this is discretized using trapezoidal rule
        mean_zero_rows = []
        mean_zero_rhs = []

        total_size = sum(n_per_block)
        row = sparse.lil_matrix((1, total_size))

        # Get interval (should be single interval for periodic)
        if linop.blocks:
            interval = linop.blocks[0]["interval"]
        else:
            interval = linop.domain.support

        # For Fourier points, trapezoidal rule gives uniform weights
        # ∫_a^b f(x) dx ≈ (b-a)/n * Σ f(x_i)
        a, b = interval
        n = n_per_block[0]
        weight = (b - a) / n

        # Set all weights equal (trapezoidal rule for equally-spaced points)
        row[0, :n] = weight

        mean_zero_rows.append(row.tocsr())
        mean_zero_rhs.append(0.0)

        return mean_zero_rows, mean_zero_rhs
