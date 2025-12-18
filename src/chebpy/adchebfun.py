"""Automatic differentiation for chebfuns with operator block Jacobians.

This module tracks the Jacobian as discrete operator blocks (matrices).
"""

from collections.abc import Callable

import numpy as np
from scipy import sparse

from .chebfun import Chebfun
from .sparse_utils import extract_scalar, jacobian_to_row, sparse_to_dense
from .spectral import barycentric_matrix, diff_matrix, identity_matrix, mult_matrix
from .utilities import ensure_interval, is_scalar_type


def _extract_residual_jacobian(bc_elem, x_bc, n):
    """Extract residual and jacobian row from a BC element.

    Handles AdChebfun, AdChebfunScalar, and scalar constants uniformly.

    Args:
        bc_elem: BC element (AdChebfun, AdChebfunScalar, or scalar)
        x_bc: Boundary point for evaluation
        n: Discretization size

    Returns:
        (residual, jacobian_row) tuple
    """
    if isinstance(bc_elem, AdChebfun):
        # Still a function - evaluate at boundary point
        bc_at_pt = bc_elem(np.array([x_bc]))
        return extract_scalar(bc_at_pt.value), jacobian_to_row(bc_at_pt.jacobian)
    elif isinstance(bc_elem, AdChebfunScalar):
        # Already evaluated
        return extract_scalar(bc_elem.value), jacobian_to_row(bc_elem.jacobian)
    else:
        # Constant
        residual = float(bc_elem) if np.isscalar(bc_elem) else 0.0
        return residual, np.zeros(n + 1)


class AdChebfun:
    """Chebfun with automatic differentiation using operator blocks.

    An adchebfun tracks:
    - func: The chebfun value
    - jacobian: Sparse matrix representing the Fréchet derivative operator
    - n: Discretization size (number of collocation points - 1)
    - domain: Interval [a, b]

    The Jacobian is a discrete linear operator that acts on perturbations.
    For u represented by coefficients c, the Jacobian J acts on perturbation
    coefficients δc to give the linearized change in the output.

    Example:
        u = AdChebfun(chebfun(lambda x: x**2, [0, 1]), n=16)
        v = u.diff()  # v.jacobian is now the differentiation matrix D
        w = v(np.array([1.0]))  # w.jacobian is evaluation of D at x=1
    """

    def __init__(self, func: Chebfun, n: int | None = None, jacobian: sparse.spmatrix | None = None):
        """Initialize an adchebfun.

        Args:
            func: The chebfun value
            n: Discretization size (collocation points - 1). If None, inferred from func.
            jacobian: Sparse matrix Jacobian. If None, initializes as identity.
        """
        # Convert Domain to Interval (support is a Domain, we need an Interval)
        self.domain = ensure_interval(func.support)

        # Infer discretization size from func if not provided
        if n is None:
            # Get the actual number of coefficients being used
            # For a chebfun, this is the length of its underlying chebtech
            if hasattr(func, "funs") and len(func.funs) > 0:
                first_fun = func.funs[0]
                if hasattr(first_fun, "onefun") and hasattr(first_fun.onefun, "coeffs"):
                    n = len(first_fun.onefun.coeffs) - 1
                else:
                    n = 16  # Default
            else:
                n = 16  # Default

        self.n = n

        # Store the function as-is
        # The Jacobian matrices will operate on discretized values at n+1 points
        # The function itself can have any representation - we only use it for evaluation
        self.func = func

        if jacobian is None:
            # Identity: J[u](v) = v (represented as identity matrix)
            self.jacobian = identity_matrix(n)
        else:
            self.jacobian = jacobian

    def diff(self, order: int = 1) -> "AdChebfun":
        """Differentiate the adchebfun.

        For f with Jacobian J, after diff:
            new_func = f'
            new_jacobian = D * J  (differentiation matrix times old Jacobian)

        Args:
            order: Order of differentiation

        Returns:
            New adchebfun with differentiated func and updated Jacobian
        """
        new_func = self.func.diff(order)

        # Chain rule: J_new = D^k * J_old
        D = diff_matrix(self.n, self.domain, order=order)
        new_jacobian = D @ self.jacobian

        return AdChebfun(new_func, self.n, new_jacobian)

    def __call__(self, x: float | np.ndarray) -> "AdChebfunScalar":
        """Evaluate at points.

        The Jacobian becomes an evaluation operator: rows of J corresponding
        to the evaluation points.

        Args:
            x: Points to evaluate at (scalar or array)

        Returns:
            AdChebfunScalar with evaluated values and evaluation Jacobian
        """
        # Evaluate the function
        x_array = np.atleast_1d(x)
        values = self.func(x_array)

        # Build evaluation matrix: maps collocation point values to evaluation at x
        # This uses barycentric interpolation
        E = barycentric_matrix(x_array, self.n, self.domain)

        # Chain rule: J_new = E * J_old
        new_jacobian = E @ self.jacobian

        return AdChebfunScalar(values, self.n, self.domain, new_jacobian)

    def __add__(self, other) -> "AdChebfun":
        """Add: (f + g).jacobian = f.jacobian + g.jacobian."""
        if isinstance(other, AdChebfun):
            if self.n != other.n:
                raise ValueError(f"Cannot add adchebfuns with different discretizations: {self.n} vs {other.n}")
            new_func = self.func + other.func
            new_jacobian = self.jacobian + other.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif isinstance(other, Chebfun):
            # Treat regular Chebfun as constant (zero Jacobian contribution)
            new_func = self.func + other
            return AdChebfun(new_func, self.n, self.jacobian)
        elif is_scalar_type(other) or isinstance(other, np.ndarray):
            # Adding constant: Jacobian unchanged
            new_func = self.func + other
            return AdChebfun(new_func, self.n, self.jacobian)
        else:
            return NotImplemented

    def __radd__(self, other) -> "AdChebfun":
        """Add from right: other + self."""
        return self.__add__(other)

    def __sub__(self, other) -> "AdChebfun":
        """Subtract: (f - g).jacobian = f.jacobian - g.jacobian."""
        if isinstance(other, AdChebfun):
            if self.n != other.n:
                raise ValueError("Cannot subtract adchebfuns with different discretizations")
            new_func = self.func - other.func
            new_jacobian = self.jacobian - other.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif isinstance(other, Chebfun):
            # Treat regular Chebfun as constant (zero Jacobian contribution)
            new_func = self.func - other
            return AdChebfun(new_func, self.n, self.jacobian)
        elif is_scalar_type(other) or isinstance(other, np.ndarray):
            new_func = self.func - other
            return AdChebfun(new_func, self.n, self.jacobian)
        else:
            return NotImplemented

    def __rsub__(self, other) -> "AdChebfun":
        """Right subtract: c - f."""
        if isinstance(other, Chebfun):  # pragma: no cover
            # Treat regular Chebfun as constant (zero Jacobian contribution)
            # Note: This branch is unreachable because Chebfun.__sub__ doesn't
            # return NotImplemented - it tries to handle everything directly.
            new_func = other - self.func
            new_jacobian = -self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif is_scalar_type(other) or isinstance(other, np.ndarray):
            new_func = other - self.func
            new_jacobian = -self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        else:
            return NotImplemented

    def __mul__(self, other) -> "AdChebfun":
        """Multiply: product rule (f*g)' = f'*g + f*g'.

        J_{f*g} = diag(g) * J_f + diag(f) * J_g
        """
        if isinstance(other, AdChebfun):
            if self.n != other.n:
                raise ValueError("Cannot multiply adchebfuns with different discretizations")
            new_func = self.func * other.func

            # Product rule: d/dε[(u+εv)(w+εz)] = v*w + u*z
            # In matrix form: J_new = M_g * J_f + M_f * J_g
            # where M_f, M_g are diagonal multiplication matrices
            M_f = mult_matrix(self.func, self.n, self.domain)
            M_g = mult_matrix(other.func, self.n, self.domain)

            new_jacobian = M_g @ self.jacobian + M_f @ other.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif isinstance(other, Chebfun):
            # Treat regular Chebfun as constant (zero Jacobian)
            # Product rule: (f * c)' = f' * c (since c' = 0)
            new_func = self.func * other
            M_c = mult_matrix(other, self.n, self.domain)
            new_jacobian = M_c @ self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif is_scalar_type(other):
            # Multiply by constant: (c*f)' = c*f'
            new_func = self.func * other
            new_jacobian = other * self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        else:
            return NotImplemented

    def __rmul__(self, other) -> "AdChebfun":
        """Multiply from right: other * self."""
        return self.__mul__(other)

    def __truediv__(self, other) -> "AdChebfun":
        """Divide: quotient rule (f/g)' = (f'*g - f*g') / g^2.

        J_{f/g} = (1/g) * J_f - (f/g^2) * J_g
        """
        if isinstance(other, AdChebfun):
            if self.n != other.n:
                raise ValueError("Cannot divide adchebfuns with different discretizations")
            new_func = self.func / other.func

            # Quotient rule in matrix form
            g_inv = Chebfun.initfun(lambda x: 1.0 / other.func(x), self.domain)
            f_over_g2 = self.func / (other.func**2)

            M_g_inv = mult_matrix(g_inv, self.n, self.domain)
            M_f_over_g2 = mult_matrix(f_over_g2, self.n, self.domain)

            new_jacobian = M_g_inv @ self.jacobian - M_f_over_g2 @ other.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif isinstance(other, Chebfun):
            # Treat regular Chebfun as constant (zero Jacobian)
            # Quotient rule: (f / c)' = f' / c (since c' = 0)
            new_func = self.func / other
            c_inv = Chebfun.initfun(lambda x: 1.0 / other(x), self.domain)
            M_c_inv = mult_matrix(c_inv, self.n, self.domain)
            new_jacobian = M_c_inv @ self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        elif is_scalar_type(other):
            new_func = self.func / other
            new_jacobian = self.jacobian / other
            return AdChebfun(new_func, self.n, new_jacobian)
        else:
            return NotImplemented

    def __pow__(self, exponent) -> "AdChebfun":
        """Power: chain rule (f^n)' = n * f^(n-1) * f'.

        J_{f^n} = n * f^(n-1) * J_f
        """
        if is_scalar_type(exponent):
            new_func = self.func**exponent

            # Chain rule: d/dε[(u+εv)^n] = n*u^(n-1)*v
            f_pow_n_minus_1 = self.func ** (exponent - 1)
            M = mult_matrix(exponent * f_pow_n_minus_1, self.n, self.domain)

            new_jacobian = M @ self.jacobian
            return AdChebfun(new_func, self.n, new_jacobian)
        else:
            return NotImplemented

    def __neg__(self) -> "AdChebfun":
        """Negate: (-f)' = -f'."""
        return AdChebfun(-self.func, self.n, -self.jacobian)

    def _apply_unary_op(self, func, derivative_func) -> "AdChebfun":
        """Apply a unary operation with chain rule.

        Args:
            func: Function to apply (e.g., np.sin)
            derivative_func: Derivative of func (e.g., np.cos)

        Returns:
            New AdChebfun with chain rule applied: J_new = diag(derivative) * J_old
        """
        new_func = Chebfun.initfun(lambda x: func(self.func(x)), self.domain)
        derivative = Chebfun.initfun(lambda x: derivative_func(self.func(x)), self.domain)
        M = mult_matrix(derivative, self.n, self.domain)
        new_jacobian = M @ self.jacobian
        return AdChebfun(new_func, self.n, new_jacobian)

    def sin(self) -> "AdChebfun":
        """Compute sine with autodifferentiation."""
        return self._apply_unary_op(np.sin, np.cos)

    def cos(self) -> "AdChebfun":
        """Compute cosine with autodifferentiation."""
        return self._apply_unary_op(np.cos, lambda x: -np.sin(x))

    def exp(self) -> "AdChebfun":
        """Compute exponential with autodifferentiation."""
        return self._apply_unary_op(np.exp, np.exp)

    def log(self) -> "AdChebfun":
        """Compute natural logarithm with autodifferentiation."""
        return self._apply_unary_op(np.log, lambda x: 1.0 / x)

    def sqrt(self) -> "AdChebfun":
        """Compute square root with autodifferentiation."""
        return self._apply_unary_op(np.sqrt, lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def sinh(self) -> "AdChebfun":
        """Compute hyperbolic sine with autodifferentiation."""
        return self._apply_unary_op(np.sinh, np.cosh)

    def cosh(self) -> "AdChebfun":
        """Compute hyperbolic cosine with autodifferentiation."""
        return self._apply_unary_op(np.cosh, np.sinh)

    def tanh(self) -> "AdChebfun":
        """Compute hyperbolic tangent with autodifferentiation."""
        return self._apply_unary_op(np.tanh, lambda x: 1.0 / (np.cosh(x) ** 2))

    def log1p(self) -> "AdChebfun":
        """Compute log(1+x) with autodifferentiation."""
        return self._apply_unary_op(np.log1p, lambda x: 1.0 / (1.0 + x))

    def abs(self) -> "AdChebfun":
        """Absolute value with autodifferentiation.

        Uses sign(u) as the derivative (valid almost everywhere).
        At u=0, we use 0 as the derivative by convention.
        """
        return self._apply_unary_op(np.abs, np.sign)

    def __abs__(self) -> "AdChebfun":
        """Support abs() and np.abs()."""
        return self.abs()


class AdChebfunScalar:
    """Scalar result from evaluating an adchebfun at points.

    Tracks:
    - value: Evaluated values (array)
    - jacobian: Evaluation operator matrix (E * original_jacobian)
    """

    def __init__(self, value: np.ndarray, n: int, domain: tuple, jacobian: sparse.spmatrix):
        """Initialize scalar adchebfun.

        Args:
            value: Evaluated values
            n: Discretization size
            domain: Interval [a, b]
            jacobian: Sparse matrix Jacobian
        """
        self.value = value
        self.func = value  # Alias for compatibility
        self.n = n
        self.domain = domain
        self.jacobian = jacobian

    def __add__(self, other):
        """Add: self + other."""
        if isinstance(other, (AdChebfunScalar, AdChebfun)):
            new_value = self.value + (other.value if hasattr(other, "value") else other.func)
            new_jacobian = self.jacobian + other.jacobian
            return AdChebfunScalar(new_value, self.n, self.domain, new_jacobian)
        elif isinstance(other, (int, float, np.ndarray)):
            return AdChebfunScalar(self.value + other, self.n, self.domain, self.jacobian)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Add from right: other + self."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract: self - other."""
        if isinstance(other, (AdChebfunScalar, AdChebfun)):
            new_value = self.value - (other.value if hasattr(other, "value") else other.func)
            new_jacobian = self.jacobian - other.jacobian
            return AdChebfunScalar(new_value, self.n, self.domain, new_jacobian)
        elif isinstance(other, (int, float, np.ndarray)):
            return AdChebfunScalar(self.value - other, self.n, self.domain, self.jacobian)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Subtract from right: other - self."""
        if isinstance(other, (int, float, np.ndarray)):
            return AdChebfunScalar(other - self.value, self.n, self.domain, -self.jacobian)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiply: self * other."""
        if is_scalar_type(other):
            return AdChebfunScalar(self.value * other, self.n, self.domain, other * self.jacobian)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Multiply from right: other * self."""
        return self.__mul__(other)

    def __neg__(self):
        """Negate: -self."""
        return AdChebfunScalar(-self.value, self.n, self.domain, -self.jacobian)

    def __getitem__(self, index):
        """Support subscripting: adchebfun_scalar[0] returns value[0]."""
        if isinstance(self.value, np.ndarray):
            indexed_value = self.value[index]
            # Extract corresponding row(s) of Jacobian
            J_dense = sparse_to_dense(self.jacobian)
            indexed_jacobian = J_dense[index] if J_dense.ndim > 1 else J_dense
            # Ensure 2D for sparse matrix
            if indexed_jacobian.ndim == 1:
                indexed_jacobian = indexed_jacobian.reshape(1, -1)
            return AdChebfunScalar(indexed_value, self.n, self.domain, sparse.csr_matrix(indexed_jacobian))
        else:
            raise TypeError("AdChebfunScalar value is not subscriptable")

    def __float__(self):
        """Convert to float for scalar operations."""
        if np.isscalar(self.value):
            return float(self.value)
        elif isinstance(self.value, np.ndarray) and self.value.size == 1:
            return float(self.value.item())
        else:
            raise ValueError("Cannot convert non-scalar to float")


def linearize_bc_matrix(bc_func: Callable, u: Chebfun, n: int, x_bc: float | None = None) -> tuple:
    """Linearize a boundary condition functional using AD with operator blocks.

    Returns the discrete Jacobian matrix directly, not a functional.

    Args:
        bc_func: Boundary condition functional
        u: Current solution chebfun
        n: Discretization size (collocation points - 1)
        x_bc: Boundary point where BC should be evaluated. If None, inferred from domain.

    Returns:
        (residual, jacobian_matrix):
            residual: BC residual value(s) - scalar for single BC, list for multiple BCs
            jacobian_matrix: Sparse matrix (or dense row) representing linearization
                            For multiple BCs, returns stacked rows (one per BC)
    """
    # Wrap u in adchebfun with matrix Jacobian
    u_ad = AdChebfun(u, n=n)

    # Evaluate BC - automatically computes matrix Jacobian
    result = bc_func(u_ad)

    # If x_bc not provided, use rightmost point of domain
    if x_bc is None:
        domain = u_ad.domain
        x_bc = domain[-1] if hasattr(domain, "__getitem__") else domain

    # Check if BC returns a list (multiple constraints)
    if isinstance(result, (list, tuple)):
        # BC returned multiple constraints: [u, u.diff(), ...] for fourth-order etc.
        residuals = []
        jacobian_rows = []

        for bc_elem in result:
            res, jac_row = _extract_residual_jacobian(bc_elem, x_bc, n)
            residuals.append(res)
            jacobian_rows.append(jac_row)

        # Return list of residuals and stacked Jacobian rows
        return residuals, np.vstack(jacobian_rows)

    # Single constraint: extract and return
    return _extract_residual_jacobian(result, x_bc, n)
