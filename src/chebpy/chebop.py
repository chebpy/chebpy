"""Chebop: Differential operators with boundary conditions.

This module provides the Chebop class for representing and solving
differential equations using spectral collocation methods. It follows
the MATLAB Chebfun Chebop design.

The Chebop class supports:
- Linear and nonlinear differential operators
- Boundary conditions (endpoint, periodic, integral constraints)
- Automatic operator analysis (linearity detection, order, coefficients)
- Adaptive spectral discretization and solving
- Newton iteration for nonlinear problems
- Eigenvalue problems

Example:
    >>> from chebpy import chebop
    >>> N = chebop([-1, 1])
    >>> N.domain is not None
    True
"""

import inspect
import math
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator
from scipy.sparse.linalg import spsolve

from .adchebfun import AdChebfun
from .algorithms import standard_chop
from .bndfun import Bndfun
from .chebfun import Chebfun
from .chebtech import Chebtech
from .chebyshev import vals2coeffs2
from .linop import LinOp
from .operator_compiler import OperatorCompiler
from .order_detection_ast import OrderTracerAST
from .settings import _preferences
from .sparse_utils import sparse_to_dense
from .spectral import cheb_points_scaled, diff_matrix
from .utilities import Domain, Interval, InvalidDomain


class BoundaryCondition:
    """Represents a boundary condition for a differential operator.

    Attributes:
        func: Callable that takes solution u and returns residual
        type: Type of BC ('left', 'right', 'periodic', 'integral', 'continuity')
        location: Location of BC (for endpoint conditions)
    """

    def __init__(self, func: Callable, bc_type: str = "endpoint", location: float | None = None):
        """Initialize a boundary condition.

        Args:
            func: Function that evaluates the BC. Should accept solution u (and possibly x).
            bc_type: Type of boundary condition.
            location: Location for endpoint conditions.
        """
        self.func = func
        self.type = bc_type
        self.location = location

    def __call__(self, *args, **kwargs):
        """Evaluate the boundary condition."""
        return self.func(*args, **kwargs)


class Chebop:
    """Differential operator with boundary conditions.

    This class represents a differential operator that can be linear or nonlinear,
    along with boundary conditions and constraints. It provides methods to:
    - Analyze the operator structure
    - Detect linearity
    - Extract coefficient functions for linear operators
    - Solve linear and nonlinear boundary value problems
    - Compute eigenvalues and eigenfunctions

    Attributes:
        domain: Domain on which the operator is defined
        op: Operator functional N(x, u, u', u'', ...)
        lbc: Left boundary condition functional
        rbc: Right boundary condition functional
        bc: General boundary conditions (list)
        rhs: Right-hand side function f(x)
        init: Initial guess for nonlinear problems
        numbc: Number of boundary conditions

    Internal attributes (populated by analyze_operator):
        _is_linear: Whether operator is detected to be linear
        _diff_order: Highest derivative order
        _coeffs: List of coefficient functions [a_0(x), a_1(x), ..., a_z(x)]
        _linop: Cached LinOp object
    """

    def __init__(self, *args, **kwargs):
        """Initialize a Chebop.

        Usage:
            Chebop([a, b])                     # domain only
            Chebop(a, b)                       # domain as separate args
            Chebop(domain, op=func)            # with operator
            Chebop(domain, op=func, lbc=lbc)   # with BCs

        Args:
            *args: Positional arguments (domain specification)
            **kwargs: Keyword arguments (op, lbc, rbc, bc, rhs, init, etc.)
        """
        # Parse domain and operator from positional args
        # Support both Chebop([a,b], op=f) and Chebop(f, [a,b]) styles
        if len(args) == 1:
            # Single argument: must be domain
            domain = args[0]
        elif len(args) == 2:
            # Two arguments: could be (a, b) or (op, domain) or (domain, op)
            if callable(args[0]) and isinstance(args[1], (list, tuple, np.ndarray)):
                # Chebop(op, domain) style
                kwargs.setdefault("op", args[0])
                domain = args[1]
            elif callable(args[1]) and isinstance(args[0], (list, tuple, np.ndarray)):
                # Chebop(domain, op) style
                domain = args[0]
                kwargs.setdefault("op", args[1])
            else:
                # Both scalars: (a, b) domain endpoints
                domain = [args[0], args[1]]
        else:
            domain = kwargs.get("domain", [-1, 1])

        # Store domain
        if isinstance(domain, Domain):
            self.domain = domain
        else:
            self.domain = Domain(domain)

        # Validate domain is not empty
        if len(self.domain) == 0:
            raise InvalidDomain("Domain cannot be empty")

        # Store operator and boundary conditions
        self.op = kwargs.get("op", None)  # Operator functional
        self._lbc = kwargs.get("lbc", None)  # Left BC (internal storage)
        self._rbc = kwargs.get("rbc", None)  # Right BC (internal storage)
        self.bc = kwargs.get("bc", [])  # General BCs
        if not isinstance(self.bc, list):
            self.bc = [self.bc] if self.bc is not None else []

        # Store RHS and initial guess
        self.rhs = kwargs.get("rhs", None)  # RHS function
        self.init = kwargs.get("init", None)  # Initial guess

        # Store point constraints
        self.point_constraints = kwargs.get("point_constraints", [])
        if not isinstance(self.point_constraints, list):
            self.point_constraints = [self.point_constraints] if self.point_constraints is not None else []

        # Boundary condition count
        self.numbc = kwargs.get("numbc", None)

        # Analysis cache
        self._is_linear = None
        self._bc_is_linear = None  # True if all callable BCs are linear
        self._diff_order = None
        self._coeffs = None
        self._linop = None
        self._analyzed = False

        # System properties (for coupled ODE systems)
        self._is_system = None  # True if operator returns list/tuple
        self._num_variables = None  # Number of dependent variables
        self._num_equations = None  # Number of equations

        # Solver options
        self.tol = kwargs.get("tol", 1e-10)  # Convergence tolerance
        self.maxiter = kwargs.get("maxiter", 50)  # Max iterations for nonlinear solver
        self.damping = kwargs.get("damping", 1.0)  # Damping parameter for Newton
        self.discretization_size = kwargs.get("discretization_size", 16)  # Initial grid size
        self.verbose = kwargs.get("verbose", False)  # Print Newton iteration info

    @property
    def lbc(self):
        """Get left boundary condition."""
        return self._lbc

    @lbc.setter
    def lbc(self, value):
        """Set left boundary condition.

        Accepts:
        - Callable: lambda u: u - value
        - List: [u(a), u'(a), u''(a), ...] values at left boundary
        - Scalar: equivalent to [value]
        """
        self._lbc = value

    @property
    def rbc(self):
        """Get right boundary condition."""
        return self._rbc

    @rbc.setter
    def rbc(self, value):
        """Set right boundary condition.

        Accepts:
        - Callable: lambda u: u - value
        - List: [u(b), u'(b), u''(b), ...] values at right boundary
        - Scalar: equivalent to [value]
        """
        self._rbc = value

    def __repr__(self):
        """String representation of the Chebop."""
        domain_str = f"domain: {self.domain.support}"
        op_str = "op: " + ("defined" if self.op is not None else "not defined")
        bc_str = f"BCs: lbc={self.lbc is not None}, rbc={self.rbc is not None}, general={len(self.bc)}"
        return f"Chebop({domain_str}, {op_str}, {bc_str})"

    def _is_ivp(self) -> bool:
        """Check if this is an Initial Value Problem (IVP).

        An IVP has initial values (numbers) at one endpoint only, not
        functional constraints. This distinguishes IVPs from BVPs.

        IVP: lbc = [1, 0] (values of u, u' at left)
        BVP: lbc = lambda u: u - 1 (functional constraint)

        Returns:
            True if this is an IVP/FVP (use time-stepping), False if BVP (use collocation)
        """
        # Check if lbc is numerical values (scalar, list, or array), not a callable
        lbc_is_values = (
            self.lbc is not None
            and not callable(self.lbc)
            and isinstance(self.lbc, (int, float, list, np.ndarray, np.number))
        )

        # Check if rbc is numerical values, not a callable
        rbc_is_values = (
            self.rbc is not None
            and not callable(self.rbc)
            and isinstance(self.rbc, (int, float, list, np.ndarray, np.number))
        )

        has_bc = len(self.bc) > 0

        # Left IVP: lbc is numerical values, no rbc or general bc
        left_ivp = lbc_is_values and not rbc_is_values and not has_bc and self.rbc is None
        # Right IVP (FVP): rbc is numerical values, no lbc or general bc
        right_ivp = rbc_is_values and not lbc_is_values and not has_bc and self.lbc is None

        return left_ivp or right_ivp

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "Chebop":
        """Create a Chebop from a dictionary specification.

        Args:
            spec: Dictionary with keys 'domain', 'op', 'lbc', 'rbc', etc.

        Returns:
            Chebop instance
        """
        domain = spec.get("domain", [-1, 1])
        return cls(domain, **{k: v for k, v in spec.items() if k != "domain"})

    @staticmethod
    def _maxnorm(f):
        """Compute maximum norm of a Chebfun.

        Args:
            f: Chebfun object

        Returns:
            Maximum absolute value of coefficients, or inf if f is None
        """
        if f is None:
            return np.inf
        return max(np.max(np.abs(fun.coeffs)) for fun in f)

    @staticmethod
    def _create_zero_funs(a, b, num_vars):
        """Create array of zero functions for linearization.

        Args:
            a: Left endpoint
            b: Right endpoint
            num_vars: Number of variables

        Returns:
            List of zero Chebfun objects
        """
        return [Chebfun.initfun(lambda x: np.zeros_like(x), [a, b]) for _ in range(num_vars)]

    @staticmethod
    def _create_ad_variables(zero_funs, n, num_vars, block_size, total_cols):
        """Create AdChebfun variables with proper Jacobian structure.

        Args:
            zero_funs: List of zero Chebfun objects
            n: Discretization size
            num_vars: Number of variables
            block_size: Size of each block (n+1)
            total_cols: Total number of columns in Jacobian

        Returns:
            List of AdChebfun variables
        """
        ad_vars = []
        for j in range(num_vars):
            jac = sparse.lil_matrix((block_size, total_cols))
            jac[:, j * block_size : (j + 1) * block_size] = sparse.identity(block_size)
            jac = jac.tocsr()
            ad_vars.append(AdChebfun(zero_funs[j], n=n, jacobian=jac))
        return ad_vars

    @staticmethod
    def _normalize_bc(bc):
        """Normalize boundary condition to (count, values_list) format.

        Args:
            bc: Boundary condition (scalar, list, tuple, or None)

        Returns:
            Tuple of (count, values_list)
        """
        if bc is None:
            return 0, []
        if isinstance(bc, (list, tuple)):
            return len(bc), list(bc)
        return 1, [bc]

    def _process_system_bc(
        self, is_left, bc, n, num_vars, num_eqs, block_size, total_cols, a, b, bc_rows, bc_cols, bc_vals, rhs_vec
    ):
        """Process system boundary conditions (left or right).

        Args:
            is_left: True for left BC, False for right BC
            bc: Boundary condition (callable, list/tuple, or scalar)
            n: Discretization size
            num_vars: Number of variables
            num_eqs: Number of equations
            block_size: Size of each block (n+1)
            total_cols: Total number of columns
            a: Left endpoint
            b: Right endpoint
            bc_rows: List to append BC row indices
            bc_cols: List to append BC column indices
            bc_vals: List to append BC values
            rhs_vec: RHS vector to modify
        """
        if bc is None:
            return

        if callable(bc):
            zero_funs = self._create_zero_funs(a, b, num_vars)
            ad_vars = self._create_ad_variables(zero_funs, n, num_vars, block_size, total_cols)

            bc_results = bc(*ad_vars)
            if not isinstance(bc_results, (list, tuple)):
                bc_results = [bc_results]

            bc_row_counters = [0] * num_eqs
            for bc_idx, bc_constraint in enumerate(bc_results):
                if bc_constraint is None:
                    continue
                if not isinstance(bc_constraint, AdChebfun):
                    raise TypeError("Functional BC must return AdChebfun")

                # Extract constraint row from appropriate end
                constraint_row = sparse_to_dense(bc_constraint.jacobian[0 if is_left else -1, :]).ravel()
                eq_idx = bc_idx % num_eqs
                row_within_eq = bc_row_counters[eq_idx]
                bc_row_counters[eq_idx] += 1

                # Compute row index based on side
                if is_left:
                    eq_row_idx = eq_idx * block_size + row_within_eq
                    if eq_row_idx >= (eq_idx + 1) * block_size:
                        raise ValueError("Too many BCs")
                else:
                    eq_row_idx = (eq_idx + 1) * block_size - 1 - row_within_eq
                    if eq_row_idx < eq_idx * block_size:
                        raise ValueError("Too many BCs")

                for col_idx in range(total_cols):
                    if constraint_row[col_idx] != 0:
                        bc_rows.append(eq_row_idx)
                        bc_cols.append(col_idx)
                        bc_vals.append(constraint_row[col_idx])

                rhs_vec[eq_row_idx] = -bc_constraint.func(a if is_left else b)

        elif isinstance(bc, (list, tuple)):
            for var_idx, bc_val in enumerate(bc):
                if bc_val is None:
                    continue
                if is_left:
                    eq_row_idx = var_idx * block_size
                    var_col_idx = var_idx * block_size
                else:
                    eq_row_idx = (var_idx + 1) * block_size - 1
                    var_col_idx = (var_idx + 1) * block_size - 1
                bc_rows.append(eq_row_idx)
                bc_cols.append(var_col_idx)
                bc_vals.append(1.0)
                rhs_vec[eq_row_idx] = float(bc_val) if np.isscalar(bc_val) else bc_val
        else:
            # Scalar BC
            for var_idx in range(num_vars):
                if is_left:
                    eq_row_idx = var_idx * block_size
                    var_col_idx = var_idx * block_size
                else:
                    eq_row_idx = (var_idx + 1) * block_size - 1
                    var_col_idx = (var_idx + 1) * block_size - 1
                bc_rows.append(eq_row_idx)
                bc_cols.append(var_col_idx)
                bc_vals.append(1.0)
                rhs_vec[eq_row_idx] = float(bc)

    def analyze_operator(self):
        """Analyze the operator to determine linearity, order, and coefficients.

        This method:
        1. Detects whether the operator is a system (returns list/tuple)
        2. Detects whether the operator is linear or nonlinear
        3. Determines the differential order (highest derivative)
        4. Extracts coefficient functions a_k(x) for linear operators
        5. Builds representation L = sum_{k=0}^z a_k(x) D^k
        6. Parses boundary condition types

        Sets:
            self._is_system: bool
            self._num_variables: int (for systems)
            self._num_equations: int (for systems)
            self._is_linear: bool
            self._diff_order: int
            self._coeffs: list of coefficient functions (for linear case)
            self._analyzed: bool

        Raises:
            ValueError: If operator cannot be analyzed
        """
        if self._analyzed:
            return

        if self.op is None:
            raise ValueError("Cannot analyze: operator not defined")

        # Detect if this is a system (operator returns list/tuple)
        self._detect_system()

        # Determine differential order by testing with random functions
        self._diff_order = self._detect_order()

        # Test for linearity (operator)
        self._is_linear = self._test_linearity()

        # Test for linearity (boundary conditions)
        self._bc_is_linear = self._test_bc_linearity()

        # Extract coefficients for linear operators (scalar only)
        # Only extract if both operator AND BCs are linear
        if self._is_linear and self._bc_is_linear and not self._is_system:
            self._coeffs = self._extract_coefficients()
        else:
            self._coeffs = None

        self._analyzed = True

    def _detect_system(self):
        """Detect if operator represents a system of coupled equations.

        A system operator has signature like lambda u, v: [eq1, eq2] where:
        - Multiple input variables (u, v, w, ...)
        - Returns list or tuple of equations

        Sets:
            self._is_system: True if system, False if scalar
            self._num_variables: Number of dependent variables
            self._num_equations: Number of equations

        Raises:
            ValueError: If system dimensions are inconsistent
        """
        try:
            # Get operator signature
            sig = inspect.signature(self.op)
            params = list(sig.parameters.values())

            # Count required parameters (excluding x, which is sometimes first arg)
            required_params = [p for p in params if p.default == inspect.Parameter.empty]
            num_params = len(required_params)

            # Test operator with dummy chebfuns to see what it returns

            # Create test functions
            test_funs = [Chebfun.initfun(lambda x: x + i, self.domain) for i in range(num_params)]

            # Call operator
            if num_params == 0:
                result = self.op()
            elif num_params == 1:
                result = self.op(test_funs[0])
            else:
                result = self.op(*test_funs)

            # Check if result is a list/tuple (system) or single value (scalar)
            if isinstance(result, (list, tuple)):
                self._is_system = True
                self._num_equations = len(result)
                self._num_variables = num_params

                # Validate dimensions
                if self._num_equations != self._num_variables:
                    raise ValueError(
                        f"System dimension mismatch: {self._num_variables} variables "
                        f"but {self._num_equations} equations. System must be square."
                    )
            else:
                # Scalar operator
                self._is_system = False
                self._num_variables = 1
                self._num_equations = 1

        except ValueError as e:
            # Dimension mismatch and other validation errors should propagate
            if "mismatch" in str(e).lower() or "inconsistent" in str(e).lower():
                raise
            # Other errors: assume scalar

            warnings.warn(f"System detection failed: {e}. Assuming scalar operator.")
            self._is_system = False
            self._num_variables = 1
            self._num_equations = 1
        except Exception as e:
            # Generic exceptions: assume scalar

            warnings.warn(f"System detection failed: {e}. Assuming scalar operator.")
            self._is_system = False
            self._num_variables = 1
            self._num_equations = 1

    def _detect_order(self) -> int:
        """Detect highest derivative order via AST-based symbolic tracing.

        Uses an OrderTracerAST that builds an expression tree during evaluation.
        This correctly handles:
        - Chained derivatives: u.diff().diff().diff()
        - Direct derivatives: u.diff(3)
        - Derivatives in numpy functions: sin(u.diff(2))
        - Variable coefficients: x*u.diff(2)
        - Multiple branches: sin(u') + exp(u'')

        The order is the maximum derivative order found anywhere in the tree.

        Falls back to numerical probing if AST tracing fails (e.g., due to
        operators using Chebfun coefficient functions like `x_fun * u`).
        TODO: Ideally, this should never fail.
        """
        # Create tracers for u and x with domain for Chebfun compatibility
        u_tracer = OrderTracerAST("u", domain=self.domain)
        x_tracer = OrderTracerAST("x", domain=self.domain)

        try:
            # Get the operator signature to know how many arguments it expects
            sig = inspect.signature(self.op)
            params = list(sig.parameters.values())

            # Count required parameters (those without defaults)
            required_params = sum(1 for p in params if p.default == inspect.Parameter.empty)

            # Call operator with tracers and capture the result
            if required_params == 1:
                result = self.op(u_tracer)
            elif required_params == 0:
                # All params have defaults
                result = self.op()
            else:
                # nargs >= 2: first arg is usually x or space variable
                result = self.op(x_tracer, u_tracer)

            # Extract order from the result's expression tree
            # The result should be an OrderTracerAST with the full expression
            if isinstance(result, OrderTracerAST):
                order = result.get_max_order()
            elif hasattr(result, "get_max_order"):
                order = result.get_max_order()
            else:
                # Fallback: shouldn't happen but try u_tracer
                order = u_tracer.get_max_order()

            return int(order)

        except Exception:
            # AST tracing failed - fall back to numerical probing
            return self._detect_order_numerical()

    def _detect_order_numerical(self) -> int:
        """Detect differential order via numerical probing.

        This is a fallback method when AST-based tracing fails (e.g., when the
        operator uses Chebfun coefficient functions). It uses linearity probing:

        For a linear operator L = sum_k a_k(x) D^k of order z:
        - L(u + ε*v) - L(u) = ε * L(v) for any u, v and small ε
        - The difference quotient approximates L(v)
        - By choosing v with specific derivative properties, we detect z

        For an order-z operator with smooth test functions:
        - f(x) = sin(ωx) has ||f^(k)|| ~ ω^k
        - Higher derivatives grow faster with ω
        - The operator's response to high-frequency functions reveals order
        """
        a, b = self.domain.support
        max_test_order = 10  # Maximum order to test

        # Use a frequency-based detection approach
        # For operator of order k, |L(sin(ωx))| ~ ω^k for large ω
        omega_low = 2.0
        omega_high = 20.0

        try:
            # Test function at low frequency
            u_low = Chebfun.initfun(lambda x: np.sin(omega_low * (x - a)), [a, b])
            L_low = self._evaluate_operator_safe(u_low)

            # Test function at high frequency
            u_high = Chebfun.initfun(lambda x: np.sin(omega_high * (x - a)), [a, b])
            L_high = self._evaluate_operator_safe(u_high)

            if L_low is None or L_high is None:
                return 2  # Default fallback

            # Compute norms using L2 norm
            norm_low = self._function_norm(L_low)
            norm_high = self._function_norm(L_high)

            if norm_low < 1e-14 or norm_high < 1e-14:
                # One of the norms is too small - likely operator is in kernel
                # Try different test functions
                u_low = Chebfun.initfun(lambda x: np.exp(0.5 * (x - a)), [a, b])
                u_high = Chebfun.initfun(lambda x: np.exp(2.0 * (x - a)), [a, b])
                L_low = self._evaluate_operator_safe(u_low)
                L_high = self._evaluate_operator_safe(u_high)

                if L_low is None or L_high is None:
                    return 2

                norm_low = self._function_norm(L_low)
                norm_high = self._function_norm(L_high)

                if norm_low < 1e-14:
                    return 0  # Identity-like operator

                # For exponential test: L(e^(ax)) ~ a^k * e^(ax) for order k
                ratio = np.log(norm_high / (norm_low + 1e-14)) / np.log(4.0)
                detected_order = int(round(ratio))
            else:
                # Ratio of norms should be ~ (omega_high/omega_low)^k
                ratio = np.log(norm_high / norm_low) / np.log(omega_high / omega_low)
                detected_order = int(round(ratio))

            # Clamp to reasonable range
            detected_order = max(0, min(detected_order, max_test_order))

            return detected_order

        except Exception:  # pragma: no cover
            return 2  # Default to second order

    def _evaluate_operator_safe(self, u: Chebfun, x_override: Chebfun | None = None) -> Chebfun | None:
        """Safely evaluate operator using signature introspection with fallbacks for nonstandard user call styles.

        Supports:
        op(u)
        op(x, u)
        op(x, u, u', u'', ...)
        """
        try:
            sig = inspect.signature(self.op)
            # Count only REQUIRED parameters (without defaults)
            # This handles cases like lambda u, eps=0.5: ... correctly
            params = sig.parameters
            n_required = sum(
                1
                for p in params.values()
                if p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            )
            nargs = n_required
        except Exception:  # pragma: no cover
            nargs = None

        try:
            # Best guess: use introspection based on required args only
            if nargs == 1:
                return self.op(u)

            elif nargs >= 2:
                x = x_override or Chebfun.initidentity(self.domain)
                args = [x, u]

                # supply derivatives if requested
                for k in range(1, nargs - 1):
                    args.append(u.diff(k))

                return self.op(*args)

        except Exception:
            pass

        # --- fallback attempts with different calling conventions ---
        for attempt in [
            lambda: self.op(u),
            lambda: self.op(x_override or Chebfun.initidentity(self.domain), u),
        ]:
            try:
                return attempt()
            except Exception:
                pass

        return None

    def _test_linearity(self) -> bool:
        """Probe linearity via homogeneity + additivity tests.

        Uses test functions designed to avoid being in the kernel of the operator,
        and applies both relative and absolute tolerance checks to handle cases
        where the operator output is very small.

        Also detects AFFINE operators (linear + constant) and treats them as linear
        for solving purposes by extracting the constant term separately.

        For systems, tests linearity for all variables simultaneously.

        Optimization: Uses a cheap "quick check" with simpler polynomials first
        to detect obvious nonlinearity early, avoiding expensive chebfun construction.
        """
        if self.op is None:
            return False

        # Check if operator has extra parameters (for continuation, etc.)
        # Parametric operators like lambda u, eps: ... cannot be tested for linearity
        # in the standard way since the parameter changes behavior
        #
        # However, we need to distinguish:
        # 1. Variable coefficient operators: lambda x, u: x*u (linear in u, x is spatial variable)
        # 2. Parametric operators: lambda u, eps: eps*u'' + u (nonlinear for continuation)
        try:
            sig = inspect.signature(self.op)
            params = list(sig.parameters.values())
            param_names = [p.name for p in params]
            required_params = sum(1 for p in params if p.default == inspect.Parameter.empty)

            # If operator takes 2+ required parameters, check if first param is 'x'
            # If first param is 'x', it's a variable coefficient operator (potentially linear)
            # Otherwise, it's a parametric operator (nonlinear for continuation)
            if required_params >= 2 and not self._is_system:
                # Check if first parameter is named 'x' (spatial variable)
                if param_names[0] != "x":
                    # Parametric operator like lambda u, eps: ...
                    return False
                # If first param is 'x' and there are 3+ params, it's still parametric
                # Example: lambda x, u, eps: ...
                if required_params >= 3:
                    return False
                # If we get here, it's lambda x, u: ... (variable coefficient, potentially linear)
        except Exception:
            # If we can't inspect, proceed with testing
            pass

        a, b = self.domain.support
        rel_tol = 1e-6
        abs_tol = 1e-10  # Absolute tolerance for near-zero outputs

        try:
            # Quick check with simple low-degree polynomials
            # This catches obvious nonlinearity (like y^2, y*diff(y), etc.) cheaply
            if not self._is_system:
                # Use functions with non-zero derivatives to catch nonlinearities like (u')^2
                # u1 = x (linear function with u' = 1, u'' = 0)
                # u2 = x^2 (quadratic with u' = 2x, u'' = 2)
                u1_simple = Chebfun.initfun(lambda x: x - a, [a, b])  # Shifted so u(a) = 0
                u2_simple = Chebfun.initfun(lambda x: (x - a) ** 2, [a, b])
                alpha = 2.0

                try:
                    N_u1 = self._evaluate_operator_safe(u1_simple)
                    N_u2 = self._evaluate_operator_safe(u2_simple)

                    if N_u1 is not None and N_u2 is not None:
                        # Quick homogeneity test: N(2*u1) vs 2*N(u1)
                        N_scaled = self._evaluate_operator_safe(alpha * u1_simple)
                        err_quick = self._maxnorm(N_scaled - alpha * N_u1)
                        denom_quick = self._maxnorm(alpha * N_u1)

                        # If this fails, might be affine - test that below
                        if err_quick > abs_tol and err_quick / (denom_quick + 1e-14) > rel_tol:
                            # Could be affine (linear + constant)
                            # Test: N(αu) - N(0) vs α(N(u) - N(0))
                            zero_func = Chebfun.initfun(lambda x: 0 * x, [a, b])
                            N_zero = self._evaluate_operator_safe(zero_func)

                            if N_zero is not None:
                                # Check if N(0) is reasonable (not NaN, not infinite)
                                # If N(0) has NaN or Inf, operator is likely nonlinear (e.g., 1/u)
                                N_zero_vals = N_zero(np.linspace(a, b, 20))
                                if np.any(np.isnan(N_zero_vals)) or np.any(np.isinf(N_zero_vals)):
                                    return False  # Nonlinear operator with singularities

                                # Test if N(u) - N(0) is linear
                                N_u1_shifted = N_u1 - N_zero
                                N_scaled_shifted = N_scaled - N_zero

                                err_affine = self._maxnorm(N_scaled_shifted - alpha * N_u1_shifted)
                                denom_affine = self._maxnorm(alpha * N_u1_shifted)

                                if err_affine > abs_tol and err_affine / (denom_affine + 1e-14) > rel_tol:
                                    return False  # Truly nonlinear
                                else:
                                    # For affine operators, test additivity of N(u) - N(0)
                                    N_u2_shifted = N_u2 - N_zero
                                    N_sum = self._evaluate_operator_safe(u1_simple + u2_simple)
                                    N_sum_shifted = N_sum - N_zero

                                    err_add_shifted = self._maxnorm(N_sum_shifted - (N_u1_shifted + N_u2_shifted))
                                    denom_add_shifted = self._maxnorm(N_u1_shifted + N_u2_shifted)

                                    if (
                                        err_add_shifted > abs_tol
                                        and err_add_shifted / (denom_add_shifted + 1e-14) > rel_tol
                                    ):
                                        return False
                                    else:
                                        # Affine operator confirmed, skip further tests
                                        return True

                        # Quick additivity test: N(u1+u2) vs N(u1)+N(u2) (only if not affine)
                        N_sum = self._evaluate_operator_safe(u1_simple + u2_simple)
                        err_add = self._maxnorm(N_sum - (N_u1 + N_u2))
                        denom_add = self._maxnorm(N_u1 + N_u2)

                        # If this fails, definitely nonlinear - exit early
                        if err_add > abs_tol and err_add / (denom_add + 1e-14) > rel_tol:
                            return False
                except Exception:
                    # Quick test failed, fall through to full test
                    pass

            if self._is_system:
                # For systems, create test function vectors
                # Each test is a vector of functions, one per variable
                num_vars = self._num_variables

                # Create test vectors u1, u2
                u1 = [
                    Chebfun.initfun(lambda x, i=i: np.exp(x - a) * np.sin((2.7 + 0.3 * i) * (x - a)), [a, b])
                    for i in range(num_vars)
                ]
                u2 = [
                    Chebfun.initfun(lambda x, i=i: np.cos((1.3 + 0.2 * i) * (x - a)) + 0.5 * (x - a) ** (i + 1), [a, b])
                    for i in range(num_vars)
                ]
                alpha = 2.5

                # Evaluate operator on test vectors
                N_u1 = self.op(*u1)
                N_u2 = self.op(*u2)

                if N_u1 is None or N_u2 is None:
                    return False

                # Test homogeneity and additivity for each equation
                # Homogeneity: N(α*u) = α*N(u)
                u1_scaled = [alpha * u for u in u1]
                N_u1_scaled = self.op(*u1_scaled)
                err1_list = [N_u1_scaled[i] - alpha * N_u1[i] for i in range(self._num_equations)]

                # Additivity: N(u1 + u2) = N(u1) + N(u2)
                u_sum = [u1[i] + u2[i] for i in range(num_vars)]
                N_sum = self.op(*u_sum)
                err2_list = [N_sum[i] - (N_u1[i] + N_u2[i]) for i in range(self._num_equations)]

                # Check all equations pass linearity test
                for i in range(self._num_equations):
                    err1_norm = self._maxnorm(err1_list[i])
                    err2_norm = self._maxnorm(err2_list[i])
                    denom1 = self._maxnorm(alpha * N_u1[i])
                    denom2 = self._maxnorm(N_u1[i] + N_u2[i])

                    pass1 = (err1_norm < abs_tol) or (err1_norm / (denom1 + 1e-14) < rel_tol)
                    pass2 = (err2_norm < abs_tol) or (err2_norm / (denom2 + 1e-14) < rel_tol)

                    if not (pass1 and pass2):
                        return False

                return True

            else:
                # Scalar operator - original logic
                u1 = Chebfun.initfun(lambda x: np.exp(x - a) * np.sin(2.7 * (x - a)), [a, b])
                u2 = Chebfun.initfun(lambda x: np.cos(1.3 * (x - a)) + 0.5 * (x - a) ** 2, [a, b])
                alpha = 2.5

                N_u1 = self._evaluate_operator_safe(u1)
                N_u2 = self._evaluate_operator_safe(u2)

                if N_u1 is None or N_u2 is None:
                    return False

                # Homogeneity: N(α*u) = α*N(u)
                N_alpha_u1 = self._evaluate_operator_safe(alpha * u1)
                err1 = N_alpha_u1 - alpha * N_u1

                # Additivity: N(u1 + u2) = N(u1) + N(u2)
                err2 = self._evaluate_operator_safe(u1 + u2) - (N_u1 + N_u2)

                err1_norm = self._maxnorm(err1)
                err2_norm = self._maxnorm(err2)
                denom1 = self._maxnorm(alpha * N_u1)
                denom2 = self._maxnorm(N_u1 + N_u2)

                # Check both relative and absolute errors
                # If denominator is very small, use absolute tolerance
                pass1 = (err1_norm < abs_tol) or (err1_norm / (denom1 + 1e-14) < rel_tol)
                pass2 = (err2_norm < abs_tol) or (err2_norm / (denom2 + 1e-14) < rel_tol)

                # If homogeneity fails, test for affine
                if not pass1:
                    zero_func = Chebfun.initfun(lambda x: 0 * x, [a, b])
                    N_zero = self._evaluate_operator_safe(zero_func)

                    if N_zero is not None:
                        # Check if N(0) is reasonable (not NaN, not infinite)
                        N_zero_vals = N_zero(np.linspace(a, b, 20))
                        if not (np.any(np.isnan(N_zero_vals)) or np.any(np.isinf(N_zero_vals))):
                            # Test if N(u) - N(0) is linear
                            N_u1_shifted = N_u1 - N_zero
                            N_alpha_u1_shifted = N_alpha_u1 - N_zero

                            err1_affine = N_alpha_u1_shifted - alpha * N_u1_shifted
                            err1_affine_norm = self._maxnorm(err1_affine)
                            denom1_affine = self._maxnorm(alpha * N_u1_shifted)

                            pass1_affine = (err1_affine_norm < abs_tol) or (
                                err1_affine_norm / (denom1_affine + 1e-14) < rel_tol
                            )

                            if pass1_affine:
                                # Affine operator detected, treat as linear
                                pass1 = True

                                # Also need to test additivity of shifted operator
                                N_u2_shifted = N_u2 - N_zero
                                N_sum = self._evaluate_operator_safe(u1 + u2)
                                N_sum_shifted = N_sum - N_zero

                                err2_affine = N_sum_shifted - (N_u1_shifted + N_u2_shifted)
                                err2_affine_norm = self._maxnorm(err2_affine)
                                denom2_affine = self._maxnorm(N_u1_shifted + N_u2_shifted)

                                pass2_affine = (err2_affine_norm < abs_tol) or (
                                    err2_affine_norm / (denom2_affine + 1e-14) < rel_tol
                                )

                                if pass2_affine:
                                    pass2 = True

                return bool(pass1 and pass2)

        except Exception:
            return False

    def _test_bc_linearity(self) -> bool:
        """Test if all boundary conditions are linear.

        A BC is linear if bc(alpha*u) = alpha*bc(u) for scalar alpha.
        Non-callable BCs (constants, lists of constants) are always linear.

        Returns:
            True if all BCs are linear, False if any BC is nonlinear
        """
        a, b = self.domain.support

        # Collect all callable BCs to test
        callable_bcs = []

        # Check lbc
        if callable(self.lbc):
            callable_bcs.append(("lbc", self.lbc, a))
        elif isinstance(self.lbc, list):
            for i, bc in enumerate(self.lbc):
                if callable(bc):
                    callable_bcs.append((f"lbc[{i}]", bc, a))

        # Check rbc
        if callable(self.rbc):
            callable_bcs.append(("rbc", self.rbc, b))
        elif isinstance(self.rbc, list):
            for i, bc in enumerate(self.rbc):
                if callable(bc):
                    callable_bcs.append((f"rbc[{i}]", bc, b))

        # Check general BCs
        for i, bc_obj in enumerate(self.bc):
            if isinstance(bc_obj, BoundaryCondition) and callable(bc_obj.func):
                callable_bcs.append((f"bc[{i}]", bc_obj.func, bc_obj.location))
            elif callable(bc_obj):
                callable_bcs.append((f"bc[{i}]", bc_obj, None))

        # If no callable BCs, they're all linear (constants)
        if not callable_bcs:
            return True

        # Test each callable BC for linearity using homogeneity test
        # bc(alpha*u) should equal alpha*bc(u) for linear BCs
        rel_tol = 1e-6
        abs_tol = 1e-10
        alpha = 2.0

        # Create test functions
        u1 = Chebfun.initfun(lambda x: np.sin(np.pi * (x - a) / (b - a)), [a, b])
        u2 = Chebfun.initfun(lambda x: np.cos(2 * np.pi * (x - a) / (b - a)), [a, b])

        for name, bc_func, location in callable_bcs:
            try:
                # Test homogeneity: bc(alpha*u) = alpha*bc(u)
                bc_u1 = bc_func(u1)
                bc_alpha_u1 = bc_func(alpha * u1)

                # Handle both scalar and chebfun returns
                if isinstance(bc_u1, Chebfun):
                    val1 = bc_u1(location if location is not None else (a + b) / 2)
                    val2 = bc_alpha_u1(location if location is not None else (a + b) / 2)
                else:
                    val1 = float(bc_u1)
                    val2 = float(bc_alpha_u1)

                err = abs(val2 - alpha * val1)
                denom = abs(alpha * val1)

                if err > abs_tol and (denom < 1e-14 or err / denom > rel_tol):
                    # Failed homogeneity test - BC is nonlinear
                    return False

                # Also test additivity: bc(u1 + u2) = bc(u1) + bc(u2)
                bc_u2 = bc_func(u2)
                bc_sum = bc_func(u1 + u2)

                if isinstance(bc_u1, Chebfun):
                    loc = location if location is not None else (a + b) / 2
                    val_sum = bc_sum(loc)
                    val_expected = bc_u1(loc) + bc_u2(loc)
                else:
                    val_sum = float(bc_sum)
                    val_expected = float(bc_u1) + float(bc_u2)

                err_add = abs(val_sum - val_expected)
                denom_add = abs(val_expected)

                if err_add > abs_tol and (denom_add < 1e-14 or err_add / denom_add > rel_tol):
                    # Failed additivity test - BC is nonlinear
                    return False

            except Exception:
                # If we can't evaluate the BC, assume it might be nonlinear
                # to be safe
                return False

        return True

    def _extract_coefficients(self) -> list[Chebfun]:
        """Extract coefficient functions a_k(x) for linear operator.

            L = Σ a_k(x) D^k

        Uses triangular polynomial probing and Chebfun algebra.
        """
        if not self._is_linear:
            return None

        a, b = self.domain.support
        z = self._diff_order

        x = Chebfun.initidentity(self.domain)

        # Construct polynomial basis:
        #   p_k(x) = (x-a)^k / k!
        polys = [(x - a) ** k / math.factorial(k) for k in range(z + 1)]

        results = []
        for p in polys:
            results.append(self._evaluate_operator_safe(p))

        coeffs = []

        for j in range(z + 1):
            if j == 0:
                a_j = results[0]
            else:
                residual = results[j]

                # subtract known lower-order contributions
                for i in range(j):
                    deg = j - i
                    contrib = coeffs[i] * (x - a) ** deg / math.factorial(deg)
                    residual = residual - contrib

                a_j = residual

            coeffs.append(a_j)

        return coeffs

    def _expand_callable_bc(self, bc):
        """Expand callable BC that returns a list into a list of callables.

        The key insight: we need to expand at the AST/definition level, not
        at the evaluation level. If BC is `lambda u: [u, u.diff()]`, we cannot
        just do `lambda u: bc(u)[0]` because that breaks AD during Newton iteration.

        Instead, we trace the BC to understand its structure, then create new
        lambdas that directly compute each element.

        Args:
            bc: Boundary condition (can be callable, list, or None)

        Returns:
            Expanded BC (list of callables if BC returns list, otherwise unchanged)
        """
        if bc is None or not callable(bc):
            return bc

        try:
            # Trace the BC to see if it returns a list

            tracer = OrderTracerAST("u", domain=self.domain)
            traced_result = bc(tracer)

            # If BC returns a list/tuple, we need to understand what each element is
            if isinstance(traced_result, (list, tuple)):
                # traced_result is like [ADTracer_for_u, ADTracer_for_u.diff()]
                # We need to create lambdas that compute each element directly

                # The challenge: we can't easily decompose a lambda that returns a list
                # Solution: Keep the lambda as-is but ensure discretization handles it
                # The list handling code should work if we ensure it's called correctly
                # during both linear solve and Newton iteration

                # Mark this BC as multi-valued by converting to a list format
                # that linop can recognize and process correctly
                expanded = []
                for i in range(len(traced_result)):
                    # Each lambda calls the original BC and extracts element i
                    # This works for both regular chebfun and ADChebfun
                    def make_bc(idx):
                        def bc_i(u):
                            result = bc(u)
                            if isinstance(result, (list, tuple)):
                                return result[idx]
                            return result

                        return bc_i

                    expanded.append(make_bc(i))
                return expanded
            else:
                # Single condition, return as-is
                return bc
        except Exception:
            # If tracing fails, assume single condition
            return bc

    def to_linop(self):
        """Convert this Chebop to a LinOp object.

        Requires that the operator has been analyzed and determined to be linear.
        Creates a LinOp instance with:
        - Coefficient functions (or pre-computed Jacobian matrix)
        - Domain
        - Boundary conditions
        - RHS

        Returns:
            LinOp instance

        Raises:
            ValueError: If operator is not linear or not analyzed
        """
        if not self._analyzed:
            self.analyze_operator()

        if not self._is_linear:
            raise ValueError("Cannot convert nonlinear operator to LinOp")

        # For affine operators, separate the constant term from the linear part
        # Affine operators have form: N(u) = L(u) + c where L is linear
        # The user writes: lambda u: u.diff(2) + u - 1 (meaning u'' + u - 1 = 0)
        # We need to convert to: L(u) = f where L(u) = u'' + u and f = 1
        #
        # To detect the constant term c, evaluate op(0):
        # N(0) = L(0) + c = 0 + c = c (since L is linear, L(0) = 0)
        coeffs_for_linop = self._coeffs
        rhs_for_linop = self.rhs

        # Check if operator has affine constant term by evaluating at zero
        zero_func = Chebfun.initfun(lambda x: 0 * x, self.domain)
        N_zero = self._evaluate_operator_safe(zero_func)

        if N_zero is not None:
            c_max = np.max(np.abs(N_zero(np.linspace(*self.domain.support, 20))))

            if c_max > 1e-14:
                # Operator has form N(u) = L(u) + c
                # User equation: N(u) = 0 means L(u) + c = 0, so L(u) = -c
                # If rhs was explicitly set, keep it: L(u) = rhs - c
                # If rhs was not set (None), default is: L(u) = -c

                if rhs_for_linop is None:
                    # No explicit RHS, so equation is N(u) = 0
                    # Convert to L(u) = -c
                    rhs_for_linop = -N_zero
                else:
                    # Explicit RHS was set: N(u) = rhs
                    # Convert to L(u) + c = rhs, so L(u) = rhs - c
                    rhs_for_linop = rhs_for_linop - N_zero

                # Create a new operator that subtracts the constant
                original_op = self.op

                # Determine the signature of the original operator to preserve it
                try:
                    sig = inspect.signature(original_op)
                    params = list(sig.parameters.values())
                    n_required = sum(1 for p in params if p.default == inspect.Parameter.empty)
                except Exception:
                    n_required = 1

                # Create linear_op with matching signature
                if n_required == 1:

                    def linear_op(u):
                        return original_op(u) - N_zero
                else:
                    # For operators like lambda x, u: x*u, preserve (x, u) signature
                    def linear_op(x, u):
                        return original_op(x, u) - N_zero

                # Temporarily replace operator to extract pure linear coefficients
                self.op = linear_op
                coeffs_for_linop = self._extract_coefficients()
                self.op = original_op  # Restore

        # Create LinOp with extracted information
        # Note: We don't cache this because rhs can change (e.g., in Newton iteration)
        # and we need to create a fresh LinOp each time
        # Handle periodic boundary conditions
        # If Chebop has periodic attribute set, pass it to LinOp as bc='periodic'
        bc_for_linop = self.bc
        if hasattr(self, "periodic") and self.periodic:
            bc_for_linop = "periodic"

        linop = LinOp(
            coeffs=coeffs_for_linop,
            domain=self.domain,
            diff_order=self._diff_order,
            lbc=self.lbc,
            rbc=self.rbc,
            bc=bc_for_linop,
            rhs=rhs_for_linop,
            tol=self.tol,
            point_constraints=self.point_constraints,
        )

        # Pass current solution for BC linearization (Newton iteration)
        if hasattr(self, "_u_current"):
            linop.u_current = self._u_current

        # Pass Jacobian computer function if available (from AdChebfun)
        if hasattr(self, "_jacobian_computer"):
            linop._jacobian_computer = self._jacobian_computer

        return linop

    def to_linop_system(self, n=None):
        """Convert this system Chebop to a LinOpSystem object.

        For a system operator like lambda u, v: [L11[u] + L12[v], L21[u] + L22[v]],
        extracts individual operator blocks L_ij and assembles them into a LinOpSystem.

        Uses automatic differentiation to extract blocks by linearizing around zero.

        Args:
            n: Discretization size (default: 16)

        Returns:
            dict: Dictionary with keys 'ops', 'n', 'domain', 'num_equations', 'num_variables'

        Raises:
            ValueError: If operator is not a system or not linear
        """
        if not self._analyzed:
            self.analyze_operator()

        if not self._is_system:
            raise ValueError("Cannot convert scalar operator to LinOpSystem. Use to_linop() instead.")

        if not self._is_linear:
            raise ValueError("Nonlinear systems not yet supported")

        # For linear systems, the Jacobian IS the operator itself
        # We extract operator blocks L_ij by linearizing around zero functions
        a, b = self.domain.support

        # Create array of zero functions (linearization point)
        zero_funs = self._create_zero_funs(a, b, self._num_variables)

        # Choose discretization size
        if n is None:
            n = 16  # Default discretization

        # Extract operator blocks L_ij
        # The key insight: when we evaluate with AdChebfun variables, each equation's
        # Jacobian represents derivatives w.r.t. ALL input variables.
        # We need to split this into blocks corresponding to each variable.

        # Create AdChebfun variables with proper Jacobian structure for systems
        # Each variable's Jacobian should be a block in the full system Jacobian
        # Variable j has Jacobian = [0, 0, ..., I_j, ..., 0] where I_j is at position j
        block_size = n + 1
        total_size = self._num_variables * block_size

        ad_vars = self._create_ad_variables(zero_funs, n, self._num_variables, block_size, total_size)

        # Evaluate operator to get equations
        try:
            equations = self.op(*ad_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate operator with AD variables: {e}")

        # Extract operator blocks from Jacobians
        ops = [[None for _ in range(self._num_variables)] for _ in range(self._num_equations)]

        for i in range(self._num_equations):
            if not isinstance(equations[i], AdChebfun):
                raise ValueError(f"Equation {i} did not return an AdChebfun. Got {type(equations[i])}.")

            # The Jacobian for equation i is (n+1) × (num_vars * (n+1))
            # It represents derivatives w.r.t. [u_0, ..., u_n, v_0, ..., v_n, ...]
            # We need to split it into blocks: [L_i0, L_i1, ...]
            full_jac = equations[i].jacobian

            # Split Jacobian into blocks by column ranges
            block_size = n + 1
            for j in range(self._num_variables):
                col_start = j * block_size
                col_end = (j + 1) * block_size
                # Extract block L_ij: equation i, variable j
                ops[i][j] = full_jac[:, col_start:col_end]

        # We now have Jacobian matrices for each operator block
        # These ARE the discretized operators - we can directly assemble the block system
        # Return the ops array (Jacobian matrices) along with metadata
        return {
            "ops": ops,  # 2D list of Jacobian matrices
            "n": n,  # Discretization size
            "domain": self.domain,
            "num_equations": self._num_equations,
            "num_variables": self._num_variables,
        }

    def solve(self, **kwargs):
        """Solve the boundary value problem.

        This is the main entry point for solving. It:
        1. Analyzes the operator
        2. Routes to linear or nonlinear solver
        3. Returns the solution as a Chebfun

        Args:
            **kwargs: Solver options (tol, maxiter, init, etc.)

        Returns:
            Chebfun: Solution to the BVP

        Raises:
            ValueError: If operator or BCs not properly defined
        """
        # Validate operator is defined
        if self.op is None:
            raise ValueError("Operator not defined. Set N.op = lambda u: ... before calling solve().")

        # Update solver options
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Analyze if not yet done
        if not self._analyzed:
            self.analyze_operator()

        # Validate boundary conditions for systems
        if self._is_system:
            # Check if scalar BCs are being applied to a system
            # Scalar boundary conditions for systems are only supported for first-order systems
            lbc_is_scalar = (
                self.lbc is not None and not callable(self.lbc) and isinstance(self.lbc, (int, float, np.number))
            )
            rbc_is_scalar = (
                self.rbc is not None and not callable(self.rbc) and isinstance(self.rbc, (int, float, np.number))
            )

            if lbc_is_scalar or rbc_is_scalar:
                raise ValueError(
                    "Specifying boundary conditions as a scalar for systems is not supported.\n"
                    "For systems with multiple variables, boundary conditions must be:\n"
                    "  - A list/array with one condition per variable, or\n"
                    "  - A callable function that returns conditions for all variables\n"
                    "Example: For a 2-variable system, use N.lbc = [1.0, 0.5] instead of N.lbc = 1.0"
                )

        # Check if this is a system
        if self._is_system:
            # System solving - returns tuple of chebfuns
            if self._is_linear:
                solution = self._solve_linear_system()
            else:
                # Nonlinear systems not implemented
                raise NotImplementedError(
                    "Nonlinear systems not implemented. Only linear coupled ODE systems are supported."
                )
            return solution
        else:
            # Scalar solving - returns single chebfun
            # Check if continuation is requested (validate before routing)
            use_continuation = kwargs.get("continuation", False)
            if use_continuation and self._is_linear:
                raise ValueError(
                    "Continuation method requires a nonlinear operator with a parameter.\n"
                    "Your operator appears to be linear. Use solve() without continuation=True."
                )

            # Check if this is an IVP (Initial Value Problem)
            # IVPs should use time-stepping, not global BVP solvers
            if self._is_ivp():
                if self.verbose:
                    print("Detected IVP - using time-stepping solver")
                return self._solve_ivp()

            # Route to appropriate BVP solver
            # Problem is linear only if BOTH operator AND BCs are linear
            is_fully_linear = self._is_linear and self._bc_is_linear
            if is_fully_linear:
                return self._solve_linear()
            else:
                if use_continuation:
                    continuation_range = kwargs.get("continuation_range", None)
                    return self._solve_with_continuation(continuation_range)
                else:
                    return self._solve_nonlinear()

    def _solve_with_continuation(self, continuation_range=None):
        """Solve a stiff nonlinear BVP using continuation/homotopy method.

        For very stiff problems, direct Newton iteration may fail to converge.
        Continuation gradually transitions from an easier problem to the target problem.

        Strategy:
        - Start with a parameter value that makes the problem easier
        - Solve the easier problem
        - Gradually adjust parameter toward target value
        - Use previous solution as initial guess for next step

        Args:
            continuation_range: List/array of parameter values from easy to hard
                If None, uses automatic continuation strategy

        Returns:
            Chebfun: Solution to the target problem

        Example:
            For Van der Pol with small epsilon (stiff):
                N = Chebop([0, 1])
                N.op = lambda u, eps: eps*u.diff(2) - (1-u**2)*u.diff() + u
                # Start with easier problem (eps=1.0) and continue to target (eps=0.001)
                u = N.solve(continuation=True, continuation_range=[1.0, 0.1, 0.01, 0.001])
        """
        if continuation_range is None:
            # Auto-detect continuation parameter from operator signature
            # This is a simplified implementation - user should provide range
            raise ValueError(
                "continuation_range must be specified.\n"
                "Example: solve(continuation=True, continuation_range=[1.0, 0.1, 0.01, 0.001])"
            )

        # Store original operator
        original_op = self.op

        # Detect if operator has a parameter

        sig = inspect.signature(original_op)
        params = list(sig.parameters.keys())

        # Determine calling convention
        if len(params) == 1:
            # op(u) form - can't do continuation without parameter
            raise ValueError(
                "Operator must accept a continuation parameter for continuation method.\n"
                "Use form: lambda u, param: ... or lambda x, u, param: ...\n"
                f"Current operator has signature: {sig}"
            )
        elif len(params) == 2 and params[0] in ["x", "t"]:
            # op(x, u) form - need to modify to op(x, u, param)
            needs_x = True
            has_param = False
        elif len(params) == 2:
            # op(u, param) form
            needs_x = False
            has_param = True
        elif len(params) == 3:
            # op(x, u, param) form
            needs_x = True
            has_param = True
        else:
            # Unexpected parameter count
            raise ValueError(
                f"Operator must have 2 or 3 parameters for continuation method.\n"
                "Use form: lambda u, param: ... or lambda x, u, param: ...\n"
                f"Current operator has signature: {sig}"
            )

        solutions = []
        for i, param_value in enumerate(continuation_range):
            # Create modified operator with fixed parameter
            if has_param:
                if needs_x:
                    # op(x, u, param) -> op(x, u) with param fixed
                    def parameterized_op(x, u, p=param_value, orig=original_op):
                        return orig(x, u, p)
                else:
                    # op(u, param) -> op(u) with param fixed
                    def parameterized_op(u, p=param_value, orig=original_op):
                        return orig(u, p)
            else:
                # Operator doesn't explicitly have parameter, use original
                parameterized_op = original_op

            # Set operator for this continuation step
            self.op = parameterized_op

            # Use previous solution as initial guess (if available)
            if solutions:
                self.init = solutions[-1]

            # Re-analyze operator for this continuation step
            self._analyzed = False
            self.analyze_operator()

            # Solve at this parameter value
            try:
                u = self._solve_nonlinear()
                solutions.append(u)
            except Exception as e:
                warnings.warn(
                    f"Continuation failed at parameter = {param_value}: {e}\n"
                    f"Try adding more intermediate parameter values.",
                    UserWarning,
                )
                # Restore original operator before raising
                self.op = original_op
                raise

        # Restore original operator
        self.op = original_op

        # Return final solution
        return solutions[-1]

    def eigs(self, k: int = 6, **kwargs):
        """Compute k eigenvalues and eigenfunctions of the operator.

        Solves the eigenvalue problem L[u] = lambda * u subject to boundary conditions.

        Args:
            k: Number of eigenvalues to compute (default 6)
            **kwargs: Additional arguments passed to LinOp.eigs (sigma, mass_matrix, etc.)

        Returns:
            eigenvalues: Array of k eigenvalues
            eigenfunctions: List of k Chebfun eigenfunctions (L2-normalized)

        Examples:
            >>> # Eigenvalue problem: -u'' = λu, u(0) = u(1) = 0
            >>> N = Chebop([0, 1])
            >>> N.op = lambda u: -u.diff(2)
            >>> N.lbc = 0
            >>> N.rbc = 0
            >>> evals, efuns = N.eigs(k=5)
        """
        linop = self.to_linop()
        return linop.eigs(k=k, **kwargs)

    def _solve_linear(self):
        """Solve a linear boundary value problem.

        Converts to LinOp and calls its solve method.

        Returns:
            Chebfun: Solution
        """
        linop = self.to_linop()
        return linop.solve()

    def _solve_linear_system(self, n_target=None):
        """Solve a linear system of coupled ODEs with adaptive refinement.

        Adaptively increases discretization size until solution converges.
        Uses happiness check based on coefficient decay (standard_chop).

        Args:
            n_target: Optional fixed discretization size (default: adaptive)

        Returns:
            tuple: Tuple of Chebfun objects (u, v, ...) representing the solution
        """
        # Early validation: check system structure and BC count at minimal cost
        # This avoids expensive iteration through all discretization sizes
        # when the problem is fundamentally ill-posed
        try:
            # Try solving at minimal n to catch structural errors early
            _ = self._solve_system_at_n(8)
        except ValueError as e:
            # Structural errors (wrong BC count, etc.) won't be fixed by changing n
            # Re-raise immediately to fail fast
            if "boundary conditions" in str(e).lower() or "too many" in str(e).lower():
                raise
            # Other errors might be n-dependent, continue to main loop
        except RuntimeError as e:
            # Singular matrix errors usually indicate fundamental issues (overdetermined, etc.)
            # that won't be fixed by changing n
            if "singular" in str(e).lower():
                raise
            # Other runtime errors might be n-dependent
        except Exception:
            # Other exceptions might be n-dependent, continue to main loop
            pass

        # Determine discretization sequence (match LinOp.solve())
        min_n = 8
        max_n = 4096

        if n_target is not None:
            n_values = [n_target]
        else:
            # Generate sequence of discretization sizes
            min_pow = np.log2(min_n)
            max_pow = np.log2(max_n)

            if max_pow <= 9:
                pow_vec = np.arange(min_pow, max_pow + 1, 1.0)
            elif min_pow >= 9:
                pow_vec = np.arange(min_pow, max_pow + 0.5, 0.5)
            else:
                pow_vec = np.concatenate([np.arange(min_pow, 9 + 1, 1.0), np.arange(9.5, max_pow + 0.5, 0.5)])

            n_values = np.round(2.0**pow_vec).astype(int).tolist()

        prev_solutions = None
        tol = 1e-14

        for n in n_values:
            try:
                solutions = self._solve_system_at_n(n)
            except Exception as e:
                if n_target is not None:
                    raise RuntimeError(f"Failed to solve system at n={n}: {e}")
                continue

            # Happiness check: coefficients should decay
            solution_is_happy = False
            for sol_fun in solutions:
                if len(sol_fun.funs) > 0:
                    coeffs = sol_fun.funs[0].onefun.coeffs
                    if len(coeffs) > 0:
                        cutoff = standard_chop(coeffs, tol=tol)
                        if cutoff < n:
                            solution_is_happy = True
                            break

            if solution_is_happy:
                return solutions

            # Convergence check
            if prev_solutions is not None:
                max_diff = 0.0
                for prev_sol, curr_sol in zip(prev_solutions, solutions):
                    diff_norm = (curr_sol - prev_sol).norm()
                    max_diff = max(max_diff, diff_norm)
                if max_diff < tol:
                    return solutions

            prev_solutions = solutions

            if n_target is not None:
                return solutions

        # Return last solution with warning
        if prev_solutions is not None:
            warnings.warn(f"System solver did not fully converge. Returning solution at n={n_values[-1]}.")
            return prev_solutions

        raise RuntimeError("Failed to solve system")

    def _solve_system_at_n(self, n):
        """Solve the linear system at a fixed discretization size n."""
        system_data = self.to_linop_system(n=n)
        ops = system_data["ops"]
        num_eqs = system_data["num_equations"]
        num_vars = system_data["num_variables"]
        a, b = system_data["domain"].support

        block_size = n + 1
        total_rows = num_eqs * block_size
        total_cols = num_vars * block_size

        block_matrix = sparse.bmat([[ops[i][j] for j in range(num_vars)] for i in range(num_eqs)], format="csr")

        rhs_vec = np.zeros(total_rows)
        if self.rhs is not None:
            interval = Interval(a, b)
            x_pts = cheb_points_scaled(n, interval)
            if callable(self.rhs):
                rhs_vals = self.rhs(x_pts)
                for i in range(num_eqs):
                    rhs_vec[i * block_size : (i + 1) * block_size] = rhs_vals
            elif isinstance(self.rhs, (list, tuple)):
                for i, rhs_func in enumerate(self.rhs):
                    if callable(rhs_func):
                        rhs_vals = rhs_func(x_pts)
                        rhs_vec[i * block_size : (i + 1) * block_size] = rhs_vals

        bc_rows = []
        bc_vals = []
        bc_cols = []

        # Process left and right BCs using helper
        self._process_system_bc(
            True, self.lbc, n, num_vars, num_eqs, block_size, total_cols, a, b, bc_rows, bc_cols, bc_vals, rhs_vec
        )
        self._process_system_bc(
            False, self.rbc, n, num_vars, num_eqs, block_size, total_cols, a, b, bc_rows, bc_cols, bc_vals, rhs_vec
        )

        # Apply BCs

        bc_by_row = defaultdict(list)
        for row_idx, col_idx, bc_val in zip(bc_rows, bc_cols, bc_vals):
            bc_by_row[row_idx].append((col_idx, bc_val))

        block_matrix = block_matrix.tolil()
        for row_idx, entries in bc_by_row.items():
            block_matrix[row_idx, :] = 0
            for col_idx, bc_val in entries:
                block_matrix[row_idx, col_idx] = bc_val
        block_matrix = block_matrix.tocsr()

        if len(bc_by_row) < num_vars:
            raise ValueError("Insufficient boundary conditions")

        solution_vec = spsolve(block_matrix, rhs_vec)

        if np.any(np.isnan(solution_vec)) or np.any(np.isinf(solution_vec)):
            raise RuntimeError("System matrix is singular")

        solutions = self._reconstruct_system_solution(solution_vec)
        return tuple(solutions)

    def _reconstruct_system_solution(self, flat_solution):
        """Reconstruct a flat solution vector into tuple of individual Chebfuns.

        Splits concatenated solution vector [u_vals, v_vals, ...] back into separate functions.

        Args:
            flat_solution: numpy array with concatenated values for all variables at collocation points

        Returns:
            tuple: Tuple of Chebfun objects (u, v, w, ...), one per variable

        Example:
            For a 2x2 system with n=16 (17 points per variable):
            flat_solution has 34 values: [u0,...,u16, v0,...,v16]
            Returns (u_chebfun, v_chebfun)
        """
        # Handle both numpy arrays and Chebfun inputs
        if isinstance(flat_solution, np.ndarray):
            coeffs = flat_solution
        else:
            # Chebfun input
            if len(flat_solution.funs) != 1:
                raise ValueError("System reconstruction with breakpoints not supported.")
            coeffs = flat_solution.funs[0].onefun.coeffs

        total_size = len(coeffs)

        # Each variable gets equal share of coefficients/values
        n_per_var = total_size // self._num_variables

        if total_size % self._num_variables != 0:
            raise ValueError(
                f"Cannot evenly split {total_size} values into "
                f"{self._num_variables} variables. Expected multiple of {self._num_variables}."
            )

        # Split values and reconstruct individual chebfuns
        # The values are at Chebyshev collocation points

        a, b = self.domain.support
        interval = Interval(a, b)

        solutions = []
        for i in range(self._num_variables):
            start_idx = i * n_per_var
            end_idx = (i + 1) * n_per_var
            var_vals = coeffs[start_idx:end_idx]

            # Convert values at Chebyshev points to Chebyshev coefficients
            cheb_coeffs = vals2coeffs2(var_vals)

            # Create Chebtech from coefficients
            chebtech = Chebtech(cheb_coeffs, interval=Interval(-1, 1))

            # Wrap in Bndfun (maps to actual domain [a,b])
            bndfun = Bndfun(chebtech, interval)

            # Wrap in Chebfun
            var_fun = Chebfun([bndfun])
            solutions.append(var_fun)

        return tuple(solutions)

    def _solve_nonlinear(self):
        """Solve a nonlinear BVP using damped Newton iteration with robustness improvements.

        Implements Deuflhard's affine-invariant damped Newton method with additional
        robustness for very stiff problems like Van der Pol and Carrier equations.

        Returns:
            Chebfun: Solution
        """
        # Get initial guess
        if self.init is not None:
            u = self.init
            # Project user-provided initial guess to satisfy BCs if needed
            u = self._project_to_satisfy_bcs(u)
        else:
            # Default initial guess: Create one that satisfies BCs from the start
            # This is much better than creating a random guess and projecting
            a, b = self.domain.support

            # Build initial guess from BC values using polynomial interpolation
            u = self._create_initial_guess_from_bcs(a, b)

        # Force minimum resolution for nonlinear problems
        # Coarse initial guesses (e.g., 2-3 points) trigger finite difference path
        # instead of AdChebfun, preventing proper convergence
        MIN_RESOLUTION_FOR_NONLINEAR = 16
        u_size = max(fun.size for fun in u.funs)
        if u_size < MIN_RESOLUTION_FOR_NONLINEAR:
            # Refine initial guess to have adequate resolution
            # This ensures AdChebfun can be used from the start
            a, b = self.domain.support
            # Create a copy with minimum resolution
            u_copy = u  # Save reference to evaluate

            def eval_u(x):
                return u_copy(x) if np.isscalar(x) or x.size == 1 else u_copy(x)

            u = Chebfun.initfun(eval_u, [a, b], n=MIN_RESOLUTION_FOR_NONLINEAR)

        # Relaxed tolerance for nonlinear problems
        nonlinear_tol = 200 * self.tol

        # Damping parameters
        lambda_val = 1.0
        lambda_min = 1e-6  # Minimum damping factor
        damping = True
        pref_damping = True

        # Newton tracking variables
        norm_delta_old = None
        norm_delta_bar = None
        delta_bar = None
        err_est = np.inf
        success = False
        give_up = 0

        # Newton iteration
        if self.verbose:
            print(f"Starting Newton iteration (maxiter={self.maxiter}, tol={self.tol}, nonlinear_tol={nonlinear_tol})")

        # Track for stagnation detection
        norm_delta_history = []
        stagnation_threshold = 1e-15  # If norm_delta stops decreasing below this, we're stuck

        # Performance optimization: Use lower maxpow2 for Jacobian coefficient construction
        # This provides 5x speedup by preventing coefficients from hitting 65537-point limit
        # Default maxpow2=16 (65537 pts), use maxpow2=12 (4097 pts) for intermediate Jacobian coefficients
        # This still provides excellent accuracy (residuals ~1e-16) but much faster

        original_maxpow2 = _preferences.maxpow2
        _preferences.maxpow2 = 12  # Use 4097 max points instead of 65537 for Jacobian coefficients

        try:
            for iteration in range(self.maxiter):
                # Compute residual
                residual = self._compute_residual(u)

                # Compute Jacobian
                jacobian_op = self._compute_jacobian(u)
                jacobian_op.rhs = -residual
                jacobian_op._u_current = u  # Store current solution for BC linearization

                # Convert to LinOp for solving
                jacobian_linop = jacobian_op.to_linop()
                jacobian_linop._u_current = u  # Preserve for callable BC linearization

                # Disable diagnostics during Newton iteration for performance (2x speedup)
                # Diagnostics are expensive and don't provide useful info during iteration
                jacobian_linop._disable_diagnostics = True
                if hasattr(jacobian_linop, "_jacobian_computer"):
                    # Store a residual evaluator function that will be called at discretization time
                    # This ensures we get accurate residual values at the collocation points
                    u_current = u
                    rhs_current = self.rhs
                    op_current = self.op

                    def residual_evaluator(x_pts):
                        """Evaluate residual at given points by recomputing operator."""
                        # Evaluate N(u) at the requested points
                        Nu = op_current(u_current)
                        Nu_vals = Nu(x_pts)

                        # Evaluate RHS at the requested points
                        if rhs_current is not None:
                            rhs_vals = rhs_current(x_pts)
                        else:
                            rhs_vals = np.zeros_like(x_pts)

                        # Return negative residual: -(N(u) - rhs) = rhs - N(u)
                        return rhs_vals - Nu_vals

                if iteration > 0:
                    current_size = max(fun.size for fun in u.funs)
                    # Start adaptive search from current size (or slightly smaller to allow reduction)
                    jacobian_linop.min_n = max(jacobian_linop.min_n, current_size // 2)
                else:
                    diff_order = jacobian_linop.diff_order
                    if diff_order >= 4:
                        # Fourth-order and higher: need fine discretization
                        recommended_min_n = max(12, diff_order * 3)
                        jacobian_linop.min_n = max(jacobian_linop.min_n, recommended_min_n)
                    elif diff_order >= 2:
                        # Second-order: moderate discretization
                        jacobian_linop.min_n = max(jacobian_linop.min_n, 8)

                # Solve for Newton correction
                try:
                    delta = jacobian_linop.solve()
                except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
                    if self.verbose:
                        print(f"  Iteration {iteration}: Failed to solve for Newton correction: {e}")
                    give_up = True
                    break

                norm_delta = self._function_norm(delta)
                norm_delta_history.append(norm_delta)

                # Check if initial guess solves problem
                if iteration == 0:
                    u_scale = max(self._function_norm(u), 1.0)
                    if norm_delta / u_scale < self.tol:
                        return u

                # Stagnation detection: If norm_delta isn't decreasing, we're stuck
                if iteration >= 3:
                    recent_deltas = norm_delta_history[-3:]
                    # Check if we're making essentially no progress
                    if all(
                        abs(recent_deltas[i] - recent_deltas[i - 1]) < stagnation_threshold * norm_delta
                        for i in range(1, len(recent_deltas))
                    ):
                        if self.verbose:
                            print(f"  Iteration {iteration}: Stagnation detected (no progress in last 3 iterations)")
                        # Stagnation detected - try to break out
                        if give_up >= 0.5:
                            # Already tried desperate measures, give up
                            if self.verbose:
                                print(f"  Iteration {iteration}: Already attempted recovery, giving up")
                            break
                        else:
                            # Try one more time with give_up increment
                            give_up += 0.5
                            if self.verbose:
                                print(f"  Iteration {iteration}: Attempting stagnation recovery")

                # Damped Newton phase
                if damping:
                    u, lambda_val, c_factor, success, give_up, norm_delta_bar, delta_bar, stay_damped = (
                        self._damping_step(
                            u,
                            delta,
                            jacobian_linop,
                            iteration,
                            lambda_val,
                            lambda_min,
                            norm_delta,
                            norm_delta_old,
                            norm_delta_bar,
                            delta_bar,
                            nonlinear_tol,
                            give_up,
                        )
                    )

                    # Update damping mode based on _damping_step's decision
                    damping = stay_damped

                    if self.verbose:
                        c_str = f"{c_factor:.4f}" if not np.isnan(c_factor) else "NaN"
                        print(
                            f"  Iteration {iteration}: norm_delta={norm_delta:.2e}, lambda={lambda_val:.4f}, "
                            f"c_factor={c_str}, damping={damping}, success={success}, give_up={give_up}"
                        )

                    # Early termination if give_up count is too high
                    if give_up >= 1:
                        if self.verbose:
                            print(f"  Iteration {iteration}: give_up={give_up} >= 1, terminating")
                        break

                    if success or (give_up > 0 and iteration > 0):
                        break

                else:  # Undamped phase
                    # Compute contraction factor
                    if norm_delta_old is not None:
                        c_factor = norm_delta / norm_delta_old

                        # Check for divergence - switch back to damping
                        if c_factor >= 1.0:
                            damping = pref_damping
                            if damping:
                                if self.verbose:
                                    print(
                                        f"  Iteration {iteration}: norm_delta={norm_delta:.2e}, "
                                        f"c_factor={c_factor:.4f}, diverging - switching back to damped mode"
                                    )
                                continue
                            else:
                                u = u + delta

                        else:
                            # Error estimate
                            err_est = norm_delta / (1.0 - c_factor**2)
                            lambda_val = 1.0

                            if self.verbose:
                                print(
                                    f"  Iteration {iteration}: norm_delta={norm_delta:.2e}, c_factor={c_factor:.4f}, "
                                    f"err_est={err_est:.2e}, undamped phase"
                                )

                            # Check convergence
                            if err_est < nonlinear_tol:
                                u = u + delta
                                success = True
                                break

                            u = u + delta

                    else:
                        u = u + delta
                        if self.verbose:
                            print(f"  Iteration {iteration}: norm_delta={norm_delta:.2e}, first undamped step")

                #  Ensure u maintains minimum resolution after update (Bug #6 fix)
                # If u gets simplified to very low resolution (e.g., 3 points), the next
                # Jacobian computation will fail. Re-refine if needed.
                MIN_RESOLUTION_FOR_NONLINEAR = 16
                u_size = max(fun.size for fun in u.funs)
                if u_size < MIN_RESOLUTION_FOR_NONLINEAR:
                    a, b = self.domain.support
                    u_copy = u

                    def eval_u_refine(x):
                        return u_copy(x) if np.isscalar(x) or x.size == 1 else u_copy(x)

                    u = Chebfun.initfun(eval_u_refine, [a, b], n=MIN_RESOLUTION_FOR_NONLINEAR)

                norm_delta_old = norm_delta

                if success or give_up:
                    break

            if not success:
                warnings.warn(
                    f"Newton iteration did not converge in {self.maxiter} iterations. "
                    f"Final error estimate: {err_est:.2e}"
                )

            return u
        finally:
            # Restore original maxpow2
            _preferences.maxpow2 = original_maxpow2

    def _damping_step(
        self,
        u,
        delta,
        jacobian_linop,
        iteration,
        lambda_val,
        lambda_min,
        norm_delta,
        norm_delta_old,
        norm_delta_bar,
        delta_bar,
        err_tol,
        give_up,
    ):
        """Perform one damped Newton step using error-based strategy with backtracking.

        This implements predictor-corrector damping with robustness improvements
        for stiff problems.

        Key improvements:
        1. Backtracking line search when simplified Newton fails
        2. Detection of repeated failures and aggressive damping reduction
        3. Better handling of LSMR convergence issues
        4. Timeout protection against infinite loops

        Returns:
            Tuple of (u_new, lambda, c_factor, success, give_up, norm_delta_bar, delta_bar, stay_damped)
            where stay_damped indicates whether to stay in damped mode for next iteration
        """
        success = False
        c_factor = np.nan
        init_prediction = True
        stay_damped = True  # Default: stay in damped mode
        lambda_prime = 1.0  # Will be computed when we have contraction

        # Track consecutive failures for aggressive damping
        consecutive_failures = 0

        # Damping iteration to find acceptable step size
        max_damping_iter = 20
        for damping_iter in range(max_damping_iter):
            # Predictor: estimate good lambda based on previous step
            if iteration > 0 and init_prediction and norm_delta_old is not None:
                if delta_bar is not None and norm_delta_bar is not None:
                    # Predictor uses deltaBar - delta
                    diff_norm = self._function_norm(delta_bar - delta)
                    if diff_norm > 1e-14:
                        mu = (norm_delta_old * norm_delta_bar) / (diff_norm * norm_delta) * lambda_val
                        lambda_val = min(1.0, mu)

                init_prediction = False

            # Check if lambda too small - try multiple fallback strategies
            if lambda_val < lambda_min:
                # Strategy 1: Try full Newton step
                u_trial = u + delta
                lambda_val = 1.0
                give_up += 0.5
                c_factor = np.nan
                stay_damped = True
                break

            # Take trial step
            u_trial = u + lambda_val * delta
            if self.verbose:
                x_test = np.linspace(u.domain.support[0], u.domain.support[1], 10)
                u_trial(x_test)

            # Evaluate operator at trial point
            try:
                residual_trial = self._compute_residual(u_trial)
                if self.verbose:
                    residual_trial(x_test)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Residual computation failed - reduce lambda and try again
                if self.verbose:
                    print(f"    Damping iter {damping_iter}: Residual eval failed at lambda={lambda_val:.4f}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    # Too many failures - give up on this Newton step
                    give_up = True
                    # Return current best guess
                    u_trial = u + lambda_val * delta if lambda_val > lambda_min else u
                    break
                lambda_val *= 0.5
                continue

            # Solve for simplified Newton step (reuse Jacobian)
            jacobian_linop.rhs = -residual_trial
            try:
                delta_bar = jacobian_linop.solve()
                if self.verbose:
                    delta_bar(x_test)
                consecutive_failures = 0  # Reset on success
            except Exception:
                # Use backtracking: try smaller lambda values
                if self.verbose:
                    print(
                        f"    Damping iter {damping_iter}: Simplified Newton failed "
                        f"at lambda={lambda_val:.4f}, backtracking"
                    )

                consecutive_failures += 1

                # If we've failed too many times, give up
                if consecutive_failures >= 3:
                    give_up = True
                    # Return best available solution
                    u_trial = u + lambda_val * delta if lambda_val > lambda_min else u
                    break

                # Backtracking: Try sequence of smaller lambda values
                backtrack_factors = [0.5, 0.25, 0.125, 0.0625]
                found_good_lambda = False

                for factor in backtrack_factors:
                    lambda_test = lambda_val * factor
                    if lambda_test < lambda_min:
                        continue

                    u_test = u + lambda_test * delta
                    try:
                        res_test = self._compute_residual(u_test)
                        jacobian_linop.rhs = -res_test
                        delta_bar_test = jacobian_linop.solve()
                        # Success! Use this lambda
                        lambda_val = lambda_test
                        u_trial = u_test
                        delta_bar = delta_bar_test
                        residual_trial = res_test
                        found_good_lambda = True
                        consecutive_failures = 0
                        if self.verbose:
                            print(
                                f"    Damping iter {damping_iter}: Backtracking succeeded with lambda={lambda_val:.4f}"
                            )
                        break
                    except Exception:
                        continue

                if not found_good_lambda:
                    # All backtracking failed - reduce lambda more aggressively and continue
                    lambda_val *= 0.25
                    if self.verbose:
                        print(
                            f"    Damping iter {damping_iter}: All backtracking failed, setting lambda={lambda_val:.4f}"
                        )
                    continue

                # If we found a good lambda via backtracking, proceed with that

            norm_delta_bar = self._function_norm(delta_bar)
            # Contraction factor
            c_factor = norm_delta_bar / norm_delta
            # Correction factor for step-size
            diff_norm = self._function_norm(delta_bar - (1.0 - lambda_val) * delta)
            if diff_norm > 1e-14:
                mu_prime = (0.5 * norm_delta * lambda_val**2) / diff_norm
            else:
                mu_prime = lambda_val
            # If not contracting, decrease lambda
            if c_factor >= 1.0:
                lambda_val = min(mu_prime, 0.5 * lambda_val)
                if self.verbose:
                    print(
                        f"    Damping iter {damping_iter}: No contraction (c={c_factor:.4f}), "
                        f"reducing lambda to {lambda_val:.4f}"
                    )
                continue

            # New potential candidate for lambda
            lambda_prime = min(1.0, mu_prime)

            # Check for convergence within damped phase
            if lambda_prime == 1.0 and norm_delta_bar < err_tol:
                success = True
                give_up = 0
                if self.verbose:
                    print(f"    Damping iter {damping_iter}: Converged in damped phase!")
                break

            # Switch to pure Newton if experiencing good convergence
            if lambda_prime == 1.0 and c_factor < 0.5:
                stay_damped = False
            else:
                stay_damped = True

            # If lambda_prime >= 4*lambda, increase lambda and continue
            if lambda_prime >= 4.0 * lambda_val:
                lambda_val = lambda_prime
                if self.verbose:
                    print(f"    Damping iter {damping_iter}: Increasing lambda to {lambda_val:.4f}")
                continue

            # Accept this step
            give_up = 0
            if self.verbose:
                print(
                    f"    Damping iter {damping_iter}: Step accepted with lambda={lambda_val:.4f}, "
                    f"c_factor={c_factor:.4f}"
                )
            break

        # If we exhausted all damping iterations without accepting a step, give up
        if damping_iter == max_damping_iter - 1 and give_up == 0:
            if self.verbose:
                print(f"    Damping: Exhausted {max_damping_iter} iterations without acceptable step")
            give_up = True

        return u_trial, lambda_val, c_factor, success, give_up, norm_delta_bar, delta_bar, stay_damped

    def _validate_operator_values(self, f: Chebfun, name: str = "Function") -> None:
        """Validate that function values don't contain NaN or Inf.

        Args:
            f: Chebfun to validate
            name: Name for error messages

        Raises:
            ValueError: If function contains NaN or Inf values
        """
        if f is None:
            return

        # Check coefficients for NaN/Inf
        for fun in f:
            coeffs = fun.coeffs
            if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
                raise ValueError(
                    f"{name} contains NaN or Inf values. "
                    "This usually indicates:\n"
                    "- Division by zero or near-zero values\n"
                    "- Negative values in fractional powers (e.g., (-1)^0.5)\n"
                    "- Overflow in exponentials or other functions\n"
                    "Try adjusting initial guess or checking operator definition."
                )

    def _solve_ivp(self):
        """Solve Initial Value Problem (IVP) using time-stepping.

        For IVPs with conditions only at one endpoint, time-stepping
        is much more efficient than global BVP solvers.

        Converts higher-order ODE to first-order system and uses
        scipy.integrate.solve_ivp for time integration.

        The method automatically extracts the ODE structure by evaluating
        the operator at test points, assuming the highest derivative
        appears linearly (standard form for explicit ODEs).

        Returns:
            Chebfun: Solution to the IVP

        Raises:
            RuntimeError: If IVP solver fails to converge
            ValueError: If highest derivative has zero coefficient
        """
        # Default tolerances for IVP solver
        # ivpAbsTol = 1e5*eps ≈ 2.22e-11
        # ivpRelTol = 100*eps ≈ 2.22e-14
        eps = np.finfo(float).eps
        ivp_abstol = 1e5 * eps
        ivp_reltol = 100 * eps

        a, b = self.domain.support[0], self.domain.support[-1]

        # Determine if this is left IVP (forward) or right IVP (backward)
        is_left_ivp = self.lbc is not None
        t_span = (a, b) if is_left_ivp else (b, a)

        # Get initial conditions
        if is_left_ivp:
            ic = self.lbc if isinstance(self.lbc, (list, np.ndarray)) else [self.lbc]
        else:
            ic = self.rbc if isinstance(self.rbc, (list, np.ndarray)) else [self.rbc]

        ic = np.array(ic, dtype=float)
        order = len(ic)  # Order of the ODE = number of initial conditions

        # Determine operator signature
        try:
            sig = inspect.signature(self.op)
            params = sig.parameters
            n_required = sum(
                1
                for p in params.values()
                if p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            )
            op_nargs = n_required
        except Exception:
            op_nargs = 1  # Default to single argument

        # Try to use operator pre-compilation
        use_precompiled = False
        compiled_rhs = None

        try:
            # Only try pre-compilation for single-argument operators (op(y))
            # Multi-argument operators (op(x, y)) need PointEval approach
            # Only try pre-compilation for single-argument operators (op(y))
            if op_nargs == 1:
                # Try to extract RHS value
                rhs_val = 0.0
                can_compile = True

                if self.rhs is None:
                    rhs_val = 0.0
                elif isinstance(self.rhs, (int, float)):
                    rhs_val = float(self.rhs)
                elif hasattr(self.rhs, "domain"):
                    # RHS is a chebfun - check if it's constant
                    # For constant chebfuns, we can still use pre-compilation
                    if len(self.rhs) <= 2:  # Constant chebfun has length 1
                        # Evaluate at midpoint to get constant value
                        mid = (self.domain.support[0] + self.domain.support[-1]) / 2
                        rhs_val = float(self.rhs(mid))
                    else:
                        # Non-constant RHS - can't pre-compile (would need time-dependent RHS support)
                        can_compile = False
                elif callable(self.rhs):
                    # Callable but not chebfun - assume scalar
                    rhs_val = 0.0
                else:
                    can_compile = False

                if can_compile:
                    compiler = OperatorCompiler()
                    compiled_rhs = compiler.compile_ivp_operator(self.op, self.domain, order, rhs_val)
                    use_precompiled = True

                    if self.verbose:
                        print("Using pre-compiled operator")
        except Exception as e:
            # If pre-compilation fails, fall back to PointEval approach
            if self.verbose:
                print(f"Pre-compilation failed ({e}), using PointEval approach")
            use_precompiled = False

        # Store operator's domain for PointEval to use
        op_domain = self.domain

        # Pre-evaluate chebfun coefficients on a dense grid for fast interpolation
        # This is the key optimization: evaluate once, interpolate many times
        a, b = self.domain.support[0], self.domain.support[-1]
        n_grid = min(500, int((b - a) * 5) + 100)  # Adaptive grid size
        t_grid = np.linspace(a, b, n_grid)

        # Extract chebfun coefficients from operator closure
        grid_cache = {}
        if hasattr(self.op, "__closure__") and self.op.__closure__:
            for cell in self.op.__closure__:
                try:
                    obj = cell.cell_contents
                    if hasattr(obj, "__call__") and hasattr(obj, "domain"):
                        # It's a chebfun - pre-evaluate on grid
                        grid_cache[id(obj)] = (t_grid, obj(t_grid))
                except (ValueError, AttributeError):
                    continue

        # Fallback dict cache for chebfuns not in closure
        chebfun_eval_cache = {}

        # PointEval: lightweight class for evaluating operator at a single point
        class PointEval:
            """Evaluate ODE operator at a single point with known derivatives.

            This class mimics a Chebfun but only stores derivative values at one point.
            When the operator is applied, arithmetic operations return scalars.
            """

            def __init__(self, derivs, t_val=None):
                """Constructor for PointEval class.

                Args:
                derivs: List of derivative values [u, u', u'', ...]
                t_val: Value of independent variable (for x-dependent operators).
                """
                self.derivs = list(derivs)
                self.t = t_val
                self._value = derivs[0]
                # Use operator's domain for compatibility with Chebfun operations
                self.domain = op_domain

            def diff(self, k=1):
                """Return k-th derivative value."""
                if k < len(self.derivs):
                    return self.derivs[k]
                raise ValueError(f"Derivative order {k} not available (max: {len(self.derivs) - 1})")

            def __call__(self, x):
                """Evaluate at point (returns u value)."""
                return self._value

            # Arithmetic operations return scalars based on u(t) value
            def __float__(self):
                return float(self._value)

            def __index__(self):
                """Make PointEval appear as scalar-like to numpy."""
                return int(self._value)

            @property
            def ndim(self):
                """Make PointEval appear as 0-dimensional (scalar) to numpy."""
                return 0

            @property
            def shape(self):
                """Make PointEval appear as scalar to numpy."""
                return ()

            def _eval_other(self, other):
                """Helper to evaluate other operand (handles chebfuns)."""
                if isinstance(other, PointEval):
                    return other._value
                elif hasattr(other, "__call__") and hasattr(other, "domain"):
                    # It's a chebfun-like object
                    obj_id = id(other)

                    # First, check if we have pre-evaluated grid values
                    if obj_id in grid_cache:
                        t_grid_vals, y_grid_vals = grid_cache[obj_id]
                        # Fast linear interpolation
                        return np.interp(self.t, t_grid_vals, y_grid_vals)

                    # Fallback: evaluate at t with caching
                    cache_key = (obj_id, self.t)
                    if cache_key not in chebfun_eval_cache:
                        chebfun_eval_cache[cache_key] = other(self.t)
                    return chebfun_eval_cache[cache_key]
                else:
                    return other

            def __add__(self, other):
                return self._value + self._eval_other(other)

            def __radd__(self, other):
                return self._eval_other(other) + self._value

            def __sub__(self, other):
                return self._value - self._eval_other(other)

            def __rsub__(self, other):
                return self._eval_other(other) - self._value

            def __mul__(self, other):
                return self._value * self._eval_other(other)

            def __rmul__(self, other):
                return self._eval_other(other) * self._value

            def __truediv__(self, other):
                return self._value / self._eval_other(other)

            def __rtruediv__(self, other):
                return self._eval_other(other) / self._value

            def __pow__(self, exp):
                return self._value**exp

            def __rpow__(self, base):
                return base**self._value

            def __neg__(self):
                return -self._value

            def __pos__(self):
                return self._value

            def __abs__(self):
                return abs(self._value)

            # NumPy ufunc support - convert to value and apply
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                new_inputs = tuple(x._value if isinstance(x, PointEval) else x for x in inputs)
                return getattr(ufunc, method)(*new_inputs, **kwargs)

            def __array__(self, dtype=None):
                return np.array(self._value, dtype=dtype)

        def evaluate_op_at_point(t_val, derivs):
            """Evaluate operator N(u) at point t with given derivative values.

            Args:
                t_val: Value of independent variable
                derivs: List [u, u', u'', ..., u^(n)] of derivative values at t

            Returns:
                Scalar value of N(u) at t
            """
            u_point = PointEval(derivs, t_val)

            if op_nargs == 1:
                # op(u)
                result = self.op(u_point)
            else:
                # op(x, u) - x is also a PointEval returning t
                x_point = PointEval([t_val], t_val)
                result = self.op(x_point, u_point)

            # Handle different result types
            if isinstance(result, PointEval):
                return result._value
            elif hasattr(result, "__call__") and hasattr(result, "domain"):
                # Result is a chebfun - evaluate at t_val
                return result(t_val)
            else:
                return float(result)

        def compute_highest_derivative(t_val, y):
            """Compute u^(n) such that N(u) = 0 at point t.

            Uses linear extraction: assumes N(u) = a*u^(n) + b where
            a and b can depend on t, u, u', ..., u^(n-1).

            Solves: u^(n) = -b/a
            """
            # Evaluate with u^(n) = 0 to get constant term b
            derivs_0 = list(y) + [0.0]
            N0 = evaluate_op_at_point(t_val, derivs_0)

            # Evaluate with u^(n) = 1 to get coefficient a + b
            derivs_1 = list(y) + [1.0]
            N1 = evaluate_op_at_point(t_val, derivs_1)

            # Extract linear coefficients: N = a*u^(n) + b
            a_coeff = N1 - N0  # Coefficient of u^(n)
            b_coeff = N0  # Constant term (when u^(n)=0)

            # Check for degenerate case
            if abs(a_coeff) < 1e-14:
                raise ValueError(
                    f"Highest derivative u^({order}) has near-zero coefficient ({a_coeff:.2e}).\n"
                    "This may indicate:\n"
                    "  1. The ODE is actually lower order than expected\n"
                    "  2. The highest derivative appears nonlinearly\n"
                    "  3. A singular point in the coefficient function\n"
                    f"At t={t_val}, y={y}"
                )

            # Solve: a*u^(n) + b = 0 => u^(n) = -b/a
            u_n = -b_coeff / a_coeff

            # Handle RHS if present
            if self.rhs is not None:
                if callable(self.rhs):
                    rhs_val = self.rhs(t_val)
                else:
                    rhs_val = float(self.rhs)
                # N(u) = rhs => a*u^(n) + b = rhs => u^(n) = (rhs - b)/a
                u_n = (rhs_val - b_coeff) / a_coeff

            return u_n

        def first_order_system(t_val, y):
            """Convert N-th order ODE to first-order system.

            State: y = [u, u', u'', ..., u^(n-1)]
            Returns: [u', u'', ..., u^(n)]
            """
            # Compute the highest derivative
            u_n = compute_highest_derivative(t_val, y)

            # Build output: [u', u'', ..., u^(n)]
            return np.array(list(y[1:]) + [u_n])

        # Solve the IVP
        if self.verbose:
            print(f"Solving IVP (order {order}) from t={t_span[0]} to t={t_span[1]}")
            print(f"Initial conditions: {ic}")
            print(f"Tolerances: rtol={ivp_reltol:.2e}, atol={ivp_abstol:.2e}")

        # Use pre-compiled RHS if available, otherwise use PointEval approach
        rhs_function = compiled_rhs if use_precompiled else first_order_system

        sol = solve_ivp(
            rhs_function,
            t_span,
            ic,
            method="BDF",  # BDF is robust for stiff problems
            dense_output=True,
            rtol=ivp_reltol,
            atol=ivp_abstol,
        )

        if not sol.success:
            raise RuntimeError(f"IVP solver failed: {sol.message}")

        # Create chebfun from solution using dense output with breakpoints
        # Use scipy's solution points as breakpoints for better accuracy

        # Select subset of scipy's time points as breakpoints
        # Too many breaks = slow, too few = poor accuracy
        # Aim for 20-50 breaks depending on problem size
        n_scipy_points = len(sol.t)
        n_breaks = min(50, max(10, n_scipy_points // 100))

        # Ensure we always include endpoints
        if n_breaks >= n_scipy_points:
            # Use all scipy points if we don't have many
            breakpoints = sol.t.tolist()
        else:
            # Sample uniformly from scipy's adaptive points
            indices = np.linspace(0, n_scipy_points - 1, n_breaks, dtype=int)
            breakpoints = sol.t[indices].tolist()
            # Ensure exact endpoints (numerical precision)
            breakpoints[0] = a
            breakpoints[-1] = b

        u_solution = Chebfun.initfun(lambda x: sol.sol(x)[0], breakpoints)

        # Validate solution for NaN/Inf values
        testpts = np.linspace(a, b, min(100, len(u_solution)))
        vals = u_solution(testpts)
        if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
            raise ValueError(
                "IVP solver produced NaN or Inf values. This typically indicates:\n"
                "  1. Invalid initial conditions (e.g., log of negative number)\n"
                "  2. Numerical instability or singularity in the ODE\n"
                "  3. Division by zero in the operator"
            )

        if self.verbose:
            print(f"IVP solved successfully. Solution length: {len(u_solution)}")
            print(f"Solver used {len(sol.t)} time steps, {len(breakpoints)} breakpoints")

        return u_solution

    def _compute_residual(self, u: Chebfun) -> Chebfun:
        """Compute residual N(u) - f for nonlinear operator.

        Args:
            u: Current solution estimate

        Returns:
            Residual as Chebfun

        Raises:
            RuntimeError: If operator evaluation fails
            ValueError: If operator produces NaN/Inf values
        """
        # Evaluate operator
        Nu = self._evaluate_operator_safe(u)

        if Nu is None:
            raise RuntimeError("Operator evaluation returned None")

        # Validate operator output for NaN/Inf
        self._validate_operator_values(Nu, "Operator N(u)")

        # Subtract RHS
        if self.rhs is not None:
            # RHS should already be a Chebfun
            residual = Nu - self.rhs
        else:
            residual = Nu

        # Validate residual
        self._validate_operator_values(residual, "Residual N(u) - f")

        return residual

    def _compute_jacobian(self, u: Chebfun) -> "Chebop":
        """Compute the Fréchet derivative (Jacobian) of the operator at u.

        For operator N(u), the Fréchet derivative J(u) is defined by:
            N(u + εv) = N(u) + J(u)[v] + O(ε²)

        This uses AdChebfun-based automatic differentiation with an operator
        block approach. The Jacobian is stored as a sparse matrix that operates
        on discretized collocation point values, not as explicit coefficient functions.

        For operators of the form N(x, u), x is the independent variable
        and should be kept fixed when computing the derivative with respect to u.

        IMPORTANT: For Newton iteration, the Jacobian BCs must include the
        BC residuals. For a BC like u(a) = c, the linearized BC is δu(a) = c - u(a),
        NOT δu(a) = 0. This ensures that after one Newton step, the boundary
        condition is exactly satisfied: u_new(a) = u(a) + δu(a) = u(a) + (c - u(a)) = c.

        Args:
            u: Current solution estimate

        Returns:
            Chebop representing the linearized operator with BC residuals
        """
        # Determine operator calling convention
        x = Chebfun.initidentity(self.domain)
        needs_x = False

        try:
            sig = inspect.signature(self.op)
            # Count only parameters WITHOUT defaults (required parameters)
            # Operators like lambda u, e=eps: ... should be treated as single-arg
            params = sig.parameters
            n_required = sum(
                1
                for p in params.values()
                if p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            )
            if n_required >= 2:
                needs_x = True
        except (ValueError, TypeError):
            try:
                self.op(u)
                needs_x = False
            except (TypeError, AttributeError):
                needs_x = True

        # Compute BC residuals for Newton iteration
        # For BC u(a) = c, the residual is c - u(a)
        # This will be the RHS of the linearized BC: δu(a) = c - u(a)
        a_val, b_val = self.domain.support
        lbc_residual, rbc_residual = self._compute_bc_residuals(u, a_val, b_val)
        u_size = max(fun.size for fun in u.funs)

        # If u is too coarse for AdChebfun, refine it first
        # This ensures we can always use the more accurate AdChebfun path
        min_size_for_ad = 16
        if u_size < min_size_for_ad:
            # Refine u by evaluating at more points
            a, b = self.domain.support
            target_size = min_size_for_ad
            x_pts = cheb_points_scaled(target_size, Interval(a, b))
            u_vals = u(x_pts)

            # Create refined chebfun from values
            interp = BarycentricInterpolator(x_pts, u_vals)
            u = Chebfun.initfun(lambda x: interp(x), [a, b])
            u_size = max(fun.size for fun in u.funs)

        use_adchebfun = True  # Always use AdChebfun after refinement

        # Use AdChebfun-based automatic differentiation
        if use_adchebfun:
            try:
                # Store a function that computes the Jacobian at any discretization size n
                # This allows LinOp's adaptive loop to recompute at different sizes
                def compute_jacobian_at_n(n: int):
                    """Compute Jacobian matrix at discretization size n."""
                    try:
                        u_size = max(fun.size for fun in u.funs)

                        # If u is too coarse relative to target n, refine it
                        # Use at least n points (better: 2*n for safety margin)
                        target_size = max(n, 16)  # At minimum 16 points for stability

                        if u_size < target_size:
                            # Refine u by evaluating at more points and reconstructing
                            # This gives AdChebfun a better-resolved function to work with
                            # We create an interpolating function from the sampled values
                            x_pts = cheb_points_scaled(
                                target_size, Interval(self.domain.support[0], self.domain.support[1])
                            )
                            u_vals = u(x_pts)

                            # Create an interpolating chebfun from these values
                            # Map points from [a,b] to [-1,1] for interpolation
                            a, b = self.domain.support[0], self.domain.support[1]
                            2 * (x_pts - a) / (b - a) - 1  # Map to [-1,1]

                            # Use numpy's polyfit to get Chebyshev coefficients
                            # Then construct chebfun from the fitted polynomial

                            interp = BarycentricInterpolator(x_pts, u_vals)
                            u_refined = Chebfun.initfun(lambda x: interp(x), [a, b])
                        else:
                            u_refined = u

                        # Create AdChebfun variable at requested discretization
                        ad_u = AdChebfun(u_refined, n=n)

                        # Evaluate operator with proper signature
                        if needs_x:
                            # Operator needs independent variable: op(x, u)
                            # Create identity chebfun for independent variable with ZERO Jacobian
                            # (x does not depend on u, so ∂x/∂u = 0)
                            x_chebfun = Chebfun.initidentity(self.domain)
                            zero_jacobian = sparse.csr_matrix((n + 1, n + 1))
                            x_ad = AdChebfun(x_chebfun, n=n, jacobian=zero_jacobian)
                            ad_result = self.op(x_ad, ad_u)
                        else:
                            # Operator only needs u: op(u)
                            ad_result = self.op(ad_u)

                        return ad_result.jacobian
                    except Exception as e:
                        # If AdChebfun fails, raise with context
                        raise RuntimeError(
                            f"AdChebfun Jacobian computation failed at n={n}: {type(e).__name__}: {e}"
                        ) from e

                # Create new Chebop with linearized operator
                # Store the Jacobian computation function (not a fixed matrix)
                jac_op = Chebop(self.domain)
                jac_op._jacobian_computer = compute_jacobian_at_n

                # Handle BCs for Newton iteration
                # For differential equations (diffOrder > 0), BCs include residuals
                # For algebraic equations (diffOrder == 0), no BCs needed in linearization
                if self._diff_order is not None and self._diff_order == 0:
                    # Algebraic equation: no BC constraints in Jacobian
                    jac_op.lbc = None
                    jac_op.rbc = None
                else:
                    # Differential equation: use BC residuals
                    jac_op.lbc = lbc_residual  # BC residual value (c - u(a))
                    jac_op.rbc = rbc_residual  # BC residual value (c - u(b))

                # Handle periodic and general BCs
                # For periodic BCs, preserve the 'periodic' string
                # For nonlinear periodic problems, the Jacobian must also be periodic
                if self.bc == "periodic":
                    jac_op.bc = "periodic"
                else:
                    jac_op.bc = []  # General BCs need separate handling if used

                jac_op.tol = self.tol

                # Mark as linear and analyzed (no need to re-analyze)
                jac_op._is_linear = True
                jac_op._analyzed = True
                jac_op._diff_order = self._diff_order

                # Extract coefficient functions for analysis/testing purposes
                # Create a wrapper operator that applies the AdChebfun Jacobian
                def adchebfun_jacobian_op(v):
                    """Apply AdChebfun Jacobian to a chebfun v."""
                    # Choose a reasonable discretization size
                    n = 64
                    jac_matrix = compute_jacobian_at_n(n)

                    # Discretize v at Chebyshev points
                    interval = Interval(self.domain.support[0], self.domain.support[1])
                    x_pts = cheb_points_scaled(n, interval)
                    v_vals = v(x_pts)

                    # Apply Jacobian matrix
                    result_vals = jac_matrix @ v_vals

                    # Reconstruct chebfun from result
                    result = Chebfun.initfun(
                        lambda x, xs=x_pts.copy(), ys=result_vals.copy(): np.interp(x, xs, ys),
                        [self.domain.support[0], self.domain.support[1]],
                    )
                    return result

                # Set the operator for testing/debugging purposes
                jac_op.op = adchebfun_jacobian_op

                # Extract coefficients using the operator wrapper
                jac_op._coeffs = self._extract_jacobian_coefficients(adchebfun_jacobian_op, u)

                return jac_op

            except Exception as e:
                # Fall back to finite differences if AD fails
                warnings.warn(
                    f"Automatic differentiation failed: {type(e).__name__}: {e}\n"
                    f"Falling back to finite differences. This may reduce accuracy.\n"
                    f"To improve AD support, consider:\n"
                    f"  - Simplifying operator expressions\n"
                    f"  - Avoiding complex nested lambdas\n"
                    f"  - Using supported numpy functions (exp, sin, cos, abs, etc.)\n"
                    f"  - Providing a custom Jacobian via manual linearization",
                    UserWarning,
                    stacklevel=2,
                )
                use_adchebfun = False  # Mark that AD failed
        else:
            # Use finite differences for coarse initial guesses
            warnings.warn(
                f"Using finite differences for Jacobian (u has only {u_size} points). "
                f"AdChebfun requires at least 16 points for numerical stability.",
                UserWarning,
                stacklevel=2,
            )

        # Finite difference fallback (used when use_adchebfun is False or AdChebfun failed)
        if not use_adchebfun:
            # Use adaptive epsilon based on solution magnitude
            u_scale = max(self._function_norm(u), 1.0)
            epsilon = max(1e-7, 1e-6 * u_scale)

            if needs_x:

                def jacobian_op(v):
                    Nu = self.op(x, u)
                    Nu_plus_eps_v = self.op(x, u + epsilon * v)
                    return (Nu_plus_eps_v - Nu) / epsilon
            else:

                def jacobian_op(v):
                    Nu = self.op(u)
                    Nu_plus_eps_v = self.op(u + epsilon * v)
                    return (Nu_plus_eps_v - Nu) / epsilon

            # Extract coefficients by probing the Jacobian
            # The extraction uses its own fine discretization (n=64) and is independent
            # of u's resolution, so we always attempt it for consistency
            coeffs = self._extract_jacobian_coefficients(jacobian_op, u)

            jac_op = Chebop(self.domain)
            jac_op.op = jacobian_op
            # Handle BCs same as AD path
            if self._diff_order is not None and self._diff_order == 0:
                jac_op.lbc = None
                jac_op.rbc = None
            else:
                jac_op.lbc = lbc_residual
                jac_op.rbc = rbc_residual

            # Handle periodic and general BCs (same as AD path)
            if self.bc == "periodic":
                jac_op.bc = "periodic"
            else:
                jac_op.bc = []

            jac_op.tol = self.tol
            jac_op._is_linear = True
            jac_op._analyzed = True
            jac_op._diff_order = self._diff_order
            jac_op._coeffs = coeffs

            return jac_op

    def _extract_jacobian_coefficients(self, jacobian_op, u):
        """Extract coefficient functions from a Jacobian operator.

        Given a linear operator J(v), find coefficients c_k(x) such that:
            J(v) = c_0(x)*v + c_1(x)*v' + c_2(x)*v'' + ...

        Uses probing with polynomial test functions.

        Args:
            jacobian_op: Callable that applies Jacobian to a chebfun
            u: Current solution (for domain info)

        Returns:
            List of coefficient chebfuns [c_0, c_1, ...]
        """
        a, b = self.domain.support
        max_order = self._diff_order if self._diff_order is not None else 2

        # Use fine discretization
        n = 64
        interval = Interval(a, b)
        x_pts = cheb_points_scaled(n, interval)
        N = n + 1

        # Build differentiation matrices
        D = [np.eye(N)]  # D^0 = I
        for k in range(1, max_order + 1):
            D.append(sparse_to_dense(diff_matrix(n, interval, order=k)))

        # Apply Jacobian to polynomial test functions
        # and solve least squares for coefficients at each point
        num_tests = max_order + 2  # Need at least max_order+1 tests
        test_results = []
        test_derivs = []

        for j in range(num_tests):
            # Test function v_j = x^j (on mapped domain)
            # Map x from [a,b] to [-1, 1] for polynomials
            def test_fn(x, power=j, aa=a, bb=b):
                # Map to [-1, 1]
                t = 2 * (x - aa) / (bb - aa) - 1
                return t**power

            v_test = Chebfun.initfun(test_fn, [a, b])
            Jv = jacobian_op(v_test)
            test_results.append(Jv(x_pts))

            # Compute derivatives of test function at collocation points
            v_derivs = [v_test(x_pts)]  # v
            for k in range(1, max_order + 1):
                v_k = v_test.diff(k)
                v_derivs.append(v_k(x_pts))
            test_derivs.append(v_derivs)

        # At each collocation point, solve for coefficients
        # J(v_j)(x_i) = sum_k c_k(x_i) * v_j^(k)(x_i)
        coeffs_at_pts = []

        for k in range(max_order + 1):
            coeffs_at_pts.append(np.zeros(N))

        for i in range(N):
            # Build system for point i
            A = np.zeros((num_tests, max_order + 1))
            b_vec = np.zeros(num_tests)

            for j in range(num_tests):
                for k in range(max_order + 1):
                    A[j, k] = test_derivs[j][k][i]
                b_vec[j] = test_results[j][i]

            # Solve least squares
            c, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            for k in range(max_order + 1):
                coeffs_at_pts[k][i] = c[k]

        # Create coefficient chebfuns from values
        # x_pts from cheb_points_scaled are already in ascending order
        coeffs = []
        for k in range(max_order + 1):
            c_vals = coeffs_at_pts[k]
            c_k = Chebfun.initfun(lambda x, xs=x_pts.copy(), vs=c_vals.copy(): np.interp(x, xs, vs), [a, b])
            coeffs.append(c_k)

        return coeffs

    def _create_initial_guess_from_bcs(self, a: float, b: float) -> Chebfun:
        """Create an initial guess that satisfies the boundary conditions.

        Constructs a low-order polynomial that exactly satisfies the BCs.

        This is used when no initial guess is provided by the user.

        Args:
            a: Left endpoint
            b: Right endpoint

        Returns:
            Chebfun that satisfies the BCs
        """
        # Extract BC values using helper
        num_lbc, lbc_vals = self._normalize_bc(self.lbc)
        num_rbc, rbc_vals = self._normalize_bc(self.rbc)

        # If no BCs, return zero function
        if num_lbc == 0 and num_rbc == 0:
            return Chebfun.initfun(lambda x: np.zeros_like(x), [a, b])

        # Build Hermite interpolating polynomial based on BC configuration
        # lbc=[c0, c1, ...] means u(a)=c0, u'(a)=c1, etc.
        # rbc=[d0, d1, ...] means u(b)=d0, u'(b)=d1, etc.

        try:
            if num_lbc == 1 and num_rbc == 1:
                # Simple linear interpolation: u(a) = c0, u(b) = d0
                c0 = float(lbc_vals[0])
                d0 = float(rbc_vals[0])
                return Chebfun.initfun(lambda x: c0 + (d0 - c0) * (x - a) / (b - a), [a, b])

            elif num_lbc == 2 and num_rbc == 0:
                # Left BCs only: u(a) = c0, u'(a) = c1
                # Polynomial: c0 + c1*(x-a)
                c0 = float(lbc_vals[0])
                c1 = float(lbc_vals[1])
                return Chebfun.initfun(lambda x: c0 + c1 * (x - a), [a, b])

            elif num_lbc == 0 and num_rbc == 2:
                # Right BCs only: u(b) = d0, u'(b) = d1
                # Polynomial: d0 + d1*(x-b)
                d0 = float(rbc_vals[0])
                d1 = float(rbc_vals[1])
                return Chebfun.initfun(lambda x: d0 + d1 * (x - b), [a, b])

            elif num_lbc == 2 and num_rbc == 1:
                # Cubic Hermite: u(a)=c0, u'(a)=c1, u(b)=d0
                c0 = float(lbc_vals[0])
                c1 = float(lbc_vals[1])
                d0 = float(rbc_vals[0])

                # Use cubic Hermite interpolation
                def poly(x):
                    t = (x - a) / (b - a)
                    h = b - a
                    # Cubic Hermite basis functions
                    H00 = (1 + 2 * t) * (1 - t) ** 2  # u(a)
                    H10 = t * (1 - t) ** 2 * h  # u'(a)
                    H01 = t**2 * (3 - 2 * t)  # u(b)
                    return c0 * H00 + c1 * H10 + d0 * H01

                return Chebfun.initfun(poly, [a, b])

            elif num_lbc == 1 and num_rbc == 2:
                # Cubic Hermite: u(a)=c0, u(b)=d0, u'(b)=d1
                c0 = float(lbc_vals[0])
                d0 = float(rbc_vals[0])
                d1 = float(rbc_vals[1])

                def poly(x):
                    t = (x - a) / (b - a)
                    h = b - a
                    # Cubic Hermite basis functions
                    H00 = (1 - t) ** 2 * (1 + 2 * t)  # u(a)
                    H01 = t**2 * (3 - 2 * t)  # u(b)
                    H11 = t**2 * (t - 1) * h  # u'(b)
                    return c0 * H00 + d0 * H01 + d1 * H11

                return Chebfun.initfun(poly, [a, b])

            elif num_lbc == 2 and num_rbc == 2:
                # Full cubic Hermite: u(a)=c0, u'(a)=c1, u(b)=d0, u'(b)=d1
                c0 = float(lbc_vals[0])
                c1 = float(lbc_vals[1])
                d0 = float(rbc_vals[0])
                d1 = float(rbc_vals[1])

                def poly(x):
                    t = (x - a) / (b - a)
                    h = b - a
                    # Standard cubic Hermite basis
                    H00 = (1 + 2 * t) * (1 - t) ** 2  # u(a)
                    H10 = t * (1 - t) ** 2 * h  # u'(a)
                    H01 = t**2 * (3 - 2 * t)  # u(b)
                    H11 = t**2 * (t - 1) * h  # u'(b)
                    return c0 * H00 + c1 * H10 + d0 * H01 + d1 * H11

                return Chebfun.initfun(poly, [a, b])

            elif num_lbc == 1 and num_rbc == 0:
                # Only left value BC: u(a) = c0
                # Use constant function
                c0 = float(lbc_vals[0])
                return Chebfun.initfun(lambda x: c0 + 0 * x, [a, b])

            elif num_lbc == 0 and num_rbc == 1:
                # Only right value BC: u(b) = d0
                # Use constant function
                d0 = float(rbc_vals[0])
                return Chebfun.initfun(lambda x: d0 + 0 * x, [a, b])

            else:
                # Fall back to zero function for other cases
                return Chebfun.initfun(lambda x: np.zeros_like(x), [a, b])

        except Exception as e:
            # If construction fails, use zero function

            warnings.warn(f"Failed to construct initial guess from BCs: {e}. Using zero.", UserWarning)
            return Chebfun.initfun(lambda x: np.zeros_like(x), [a, b])

    def _project_to_satisfy_bcs(self, u_init: Chebfun) -> Chebfun:
        """Project initial guess to satisfy boundary conditions.

        For nonlinear problems with derivative BCs, do not project the initial guess.
        Newton iteration handles BC violations. Projection can destroy good guesses
        (e.g., turning a reasonable linear function into a constant).

        Only do minimal correction for value BCs if violations are very large.

        Args:
            u_init: Initial guess (may violate BCs)

        Returns:
            u_proj: Possibly corrected guess (or original if BCs involve derivatives)
        """

        # Helper to check if BC involves derivatives
        def has_derivative_bc(bc):
            """Check if BC involves derivatives (list with len > 1)."""
            if bc is None:
                return False
            if isinstance(bc, (list, tuple)) and len(bc) > 1:
                return True
            # Callable BCs might involve derivatives, be conservative
            if callable(bc):
                return True
            return False

        # If ANY BC involves derivatives, do NOT project
        # Newton will handle all BCs correctly
        if has_derivative_bc(self.lbc) or has_derivative_bc(self.rbc):
            return u_init

        # No BCs or only simple value BCs
        # Still, be conservative - only correct large violations
        # Trust the user's initial guess
        return u_init

    def _function_norm(self, f: Chebfun) -> float:
        """Compute a norm of a chebfun.

        Args:
            f: Chebfun

        Returns:
            L2 norm of f
        """
        return np.sqrt((f * f).sum())

    def _validate_operator_values(self, f: Chebfun, description: str = "Operator output"):
        """Validate that a chebfun contains no NaN or Inf values.

        Args:
            f: Chebfun to validate
            description: Description for error message

        Raises:
            ValueError: If NaN or Inf values are detected
        """
        # Convert intervals generator to list for subscripting
        intervals_list = list(f.domain.intervals) if hasattr(f.domain, "intervals") else None

        # Check each fun piece
        for i, fun in enumerate(f.funs):
            coeffs = fun.coeffs
            if np.any(np.isnan(coeffs)):
                a, b = self.domain.support
                interval = intervals_list[i] if intervals_list is not None else (a, b)
                raise ValueError(
                    f"{description} produced NaN values on interval {interval}.\n"
                    f"Possible causes:\n"
                    f"  - Division by zero in the operator\n"
                    f"  - Negative values in sqrt() or log()\n"
                    f"  - Invalid fractional powers (e.g., negative base with non-integer exponent)\n"
                    f"  - Domain issues in special functions\n"
                    f"Check your operator definition for singularities at the current solution."
                )
            if np.any(np.isinf(coeffs)):
                a, b = self.domain.support
                interval = intervals_list[i] if intervals_list is not None else (a, b)
                raise ValueError(
                    f"{description} produced Inf values on interval {interval}.\n"
                    f"Possible causes:\n"
                    f"  - Division by zero in the operator\n"
                    f"  - Exponential growth in the solution\n"
                    f"  - Overflow in numerical operations\n"
                    f"Check your operator definition and initial guess."
                )

    def _compute_bc_residuals(self, u: Chebfun, a: float, b: float):
        """Compute boundary condition residuals for Newton iteration.

        For a Dirichlet BC u(a) = c, the residual is c - u(a).
        This residual becomes the RHS of the linearized BC: δu(a) = c - u(a).

        After Newton update u_new = u + δu:
            u_new(a) = u(a) + (c - u(a)) = c

        So the BC is exactly satisfied after one Newton step.

        For list BCs like [c0, c1, ...] representing [u(a), u'(a), ...]:
        The residual is [c0 - u(a), c1 - u'(a), ...].

        For callable BCs like lambda u: u - c (representing u(a) = c):
        We evaluate at the current u to get the residual u(a) - c,
        then negate to get c - u(a).

        Args:
            u: Current solution estimate
            a: Left boundary point
            b: Right boundary point

        Returns:
            (lbc_residual, rbc_residual): Residual values for left and right BCs
        """
        lbc_residual = self._compute_single_bc_residual(self.lbc, u, a, is_left=True)
        rbc_residual = self._compute_single_bc_residual(self.rbc, u, b, is_left=False)
        return lbc_residual, rbc_residual

    def _compute_single_bc_residual(self, bc, u: Chebfun, x_bc: float, is_left: bool):
        """Compute residual for a single boundary condition.

        Args:
            bc: Boundary condition (scalar, list, or callable)
            u: Current solution
            x_bc: Boundary point x value
            is_left: True for left BC, False for right BC

        Returns:
            BC residual value(s) - what δu should equal at the boundary
            For callable BCs that involve derivatives (e.g., lambda f: f.diff() - 1),
            returns a list [None, ..., None, value] to preserve derivative order information.
        """
        if bc is None:
            return None

        if callable(bc):
            # Callable BC: detect derivative order to preserve BC structure
            # For example, lambda f: f.diff() - 1 should give [None, value] (Neumann BC)
            # NOT just a scalar value (which would be treated as Dirichlet)

            try:
                # First, detect the derivative order using AST tracing

                tracer = OrderTracerAST("u", domain=self.domain)
                traced_result = bc(tracer)

                # Handle case where BC returns a list of conditions (e.g., lambda u: [u, u.diff()])
                if isinstance(traced_result, (list, tuple)):
                    # Multiple BCs returned - find max derivative order across all
                    derivative_order = max(
                        (item.get_max_order() if hasattr(item, "get_max_order") else 0) for item in traced_result
                    )
                else:
                    derivative_order = traced_result.get_max_order()

                # Now evaluate the BC at current solution
                bc_eval = bc(u)

                # Handle case where BC returns a list of conditions (fourth-order clamped beam)
                if isinstance(bc_eval, (list, tuple)):
                    # Multiple BCs returned - extract each residual value
                    residuals = []
                    for item in bc_eval:
                        if hasattr(item, "__call__"):
                            # Chebfun result - evaluate at boundary
                            res = float(item(np.array([x_bc]))[0])
                        else:
                            res = float(item)
                        residuals.append(-res)  # Negative residual for BC format
                    return residuals
                else:
                    # Single BC condition
                    # Extract residual value
                    if hasattr(bc_eval, "__call__"):
                        # Chebfun result - evaluate at boundary
                        residual_at_bc = float(bc_eval(np.array([x_bc]))[0])
                    else:
                        residual_at_bc = float(bc_eval)

                    # Return in list format that preserves derivative order
                    # For a BC like f.diff() - 1 (derivative_order=1), return [None, -residual]
                    # For a BC like f - 1 (derivative_order=0), return [-residual]
                    residuals = [None] * derivative_order + [-residual_at_bc]

                    # If derivative_order == 0, return scalar (Dirichlet BC)
                    if derivative_order == 0:
                        return -residual_at_bc
                    else:
                        # Return list preserving derivative order structure
                        return residuals

            except Exception as e:
                # Fallback: evaluate BC and return as scalar (assume Dirichlet)
                warnings.warn(
                    f"Could not detect derivative order for callable BC: {e}. "
                    f"Assuming Dirichlet BC (derivative order 0). "
                    f"For Neumann or higher-order BCs, use list format like [None, value].",
                    UserWarning,
                )
                bc_eval = bc(u)
                if hasattr(bc_eval, "__call__"):
                    residual_at_bc = float(bc_eval(np.array([x_bc]))[0])
                else:
                    residual_at_bc = float(bc_eval)
                return -residual_at_bc

        elif isinstance(bc, (list, tuple)):
            # List BC: [c0, c1, ...] for [u(x), u'(x), u''(x), ...]
            residuals = []
            u_derivs = [u]  # u^(0) = u
            for k in range(1, len(bc)):
                u_derivs.append(u_derivs[-1].diff())

            for k, c_k in enumerate(bc):
                if c_k is None:
                    residuals.append(None)
                elif callable(c_k):
                    # c_k is a callable BC functional
                    bc_eval = c_k(u)
                    if hasattr(bc_eval, "__call__"):
                        # Chebfun result - evaluate at boundary
                        residual_at_bc = float(bc_eval(np.array([x_bc]))[0])
                    else:
                        residual_at_bc = float(bc_eval)
                    # Return negated residual
                    residuals.append(-residual_at_bc)
                else:
                    # c_k is desired numeric value, u^(k)(x_bc) is current value
                    u_k_at_bc = float(u_derivs[k](np.array([x_bc]))[0])
                    residuals.append(c_k - u_k_at_bc)
            return residuals

        else:
            # Scalar BC: u(x_bc) = bc
            u_at_bc = float(u(np.array([x_bc]))[0])
            return bc - u_at_bc  # Residual: desired - current

    def _make_homogeneous_bc(self, bc):
        """Convert a boundary condition to its homogeneous version.

        For Newton iteration, the Jacobian needs homogeneous BCs because
        we solve J[u](δu) = -F(u) for the correction δu, and the correction
        should vanish at the boundary.

        - For scalar BCs like lbc=1, the homogeneous version is 0
        - For list BCs like [1, 0], the homogeneous version is [0, 0]
        - For callable BCs like lambda u: u - 1, the homogeneous version
          is lambda u: u (i.e., the same functional but with zero RHS)

        Args:
            bc: Original boundary condition (scalar, list, or callable)

        Returns:
            Homogeneous version of the boundary condition
        """
        if bc is None:
            return None

        if callable(bc):
            # For callable BC, we want to keep the same structure but
            # make it homogeneous. Most BCs are of the form:
            #   bc(u) = L(u) - c where L is linear and c is constant
            # The homogeneous version is:
            #   bc_hom(u) = L(u)
            # We can extract this by probing the BC
            try:
                # Test on u=0 and u=1 to extract linear part
                a_val, b_val = self.domain.support
                u_zero = Chebfun.initfun(lambda x: np.zeros_like(x), [a_val, b_val])
                u_one = Chebfun.initfun(lambda x: np.ones_like(x), [a_val, b_val])

                bc_zero = bc(u_zero)
                bc_one = bc(u_one)

                # Get scalar values
                if hasattr(bc_zero, "__call__"):
                    c0 = float(bc_zero(np.array([a_val]))[0])
                else:
                    c0 = float(bc_zero)

                if hasattr(bc_one, "__call__"):
                    float(bc_one(np.array([a_val]))[0])
                else:
                    float(bc_one)

                # For affine BC: bc(u) = a*u + b
                # bc(0) = b, bc(1) = a + b
                # So: a = bc(1) - bc(0), b = bc(0)
                # Homogeneous version has b=0: bc_hom(u) = a*u

                if abs(c0) < 1e-14:
                    # Already homogeneous
                    return bc
                else:
                    # Return homogeneous version: bc(u) - c0
                    def homogeneous_bc(v, original_bc=bc, offset=c0):
                        result = original_bc(v)
                        if hasattr(result, "__call__"):
                            # Chebfun result
                            return result - offset
                        else:
                            return result - offset

                    return homogeneous_bc
            except Exception:
                # Fallback: return identity BC (u = 0)
                return lambda u: u

        elif isinstance(bc, (list, tuple)):
            # List of BCs for u, u', u'', ... at boundary
            # Homogeneous version: replace non-None values with 0, keep None as None
            return [0 if val is not None else None for val in bc]

        else:
            # Scalar BC: homogeneous version is 0
            return 0

    @classmethod
    def identity(cls, domain):
        """Create the identity operator I.

        Args:
            domain: Domain for the operator

        Returns:
            Chebop representing the identity operator
        """
        op = cls(domain)
        op.op = lambda u: u
        return op

    @classmethod
    def diff(cls, domain, order=1):
        """Create a differentiation operator D^order.

        Args:
            domain: Domain for the operator
            order: Order of differentiation (default: 1)

        Returns:
            Chebop representing the differentiation operator
        """
        op = cls(domain)
        if order == 1:
            op.op = lambda u: u.diff()
        elif order == 2:
            op.op = lambda u: u.diff().diff()
        else:

            def diff_n(u):
                result = u
                for _ in range(order):
                    result = result.diff()
                return result

            op.op = diff_n
        return op
