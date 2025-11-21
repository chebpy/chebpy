"""Operator function evaluation and automatic differentiation for chebop.

This module provides the infrastructure for evaluating operator functions
like @(y) diff(y,2)+y and computing their Jacobians for Newton iteration.
"""

import inspect
import numpy as np


class OperatorFunction:
    """Wrapper for operator functions that provides evaluation and differentiation.

    Handles operator functions of the form:
    - @(y) diff(y,2) + y         # scalar, no independent variable
    - @(x,y) diff(y,2) + x*y     # scalar with independent variable
    - @(t,y) diff(y) - f(t)*y    # IVP form
    - @(t,u,v) [diff(u)-...; diff(v)-...]  # systems
    """

    def __init__(self, func, domain):
        """Initialize operator function wrapper.

        Args:
            func: Callable operator function
            domain: Domain [a, b] on which operator is defined
        """
        self.func = func
        self.domain = domain

        # Introspect function signature
        sig = inspect.signature(func)
        self.nargs = len(sig.parameters)
        self.param_names = list(sig.parameters.keys())

        # Determine if this is a system (multiple dependent variables)
        # For now, we'll detect this based on number of args > 2
        self.is_system = self.nargs > 2

        # Determine if independent variable is included
        self.has_indep_var = self.nargs >= 2

    def __call__(self, *args, **kwargs):
        """Evaluate the operator function."""
        return self.func(*args, **kwargs)

    def evaluate_on_grid(self, x_pts, u_coeffs, rhs_func=None):
        """Evaluate operator function on Chebyshev grid points.

        Args:
            x_pts: Chebyshev collocation points (n+1 points)
            u_coeffs: Current solution coefficients (n+1 values)
            rhs_func: Right-hand side function (optional)

        Returns:
            residual: Operator evaluation L(u) - rhs at each grid point
        """
        from .chebfun import Chebfun

        # Create chebfun from coefficients for differentiation
        u = Chebfun.from_data(x_pts, u_coeffs)

        # Evaluate operator based on signature
        if self.nargs == 1:
            # @(y) form
            op_eval = self.func(u)
        elif self.nargs == 2:
            # @(x,y) or @(t,y) form
            x = Chebfun(lambda t: t, self.domain)
            op_eval = self.func(x, u)
        else:
            raise NotImplementedError(f"Systems not yet supported (nargs={self.nargs})")

        # Convert result to values at grid points
        if isinstance(op_eval, Chebfun):
            op_values = op_eval(x_pts)
        else:
            # Could be a constant
            op_values = np.full_like(x_pts, float(op_eval))

        # Subtract RHS
        if rhs_func is not None:
            if isinstance(rhs_func, Chebfun):
                rhs_values = rhs_func(x_pts)
            elif callable(rhs_func):
                rhs_values = rhs_func(x_pts)
            else:
                rhs_values = np.full_like(x_pts, float(rhs_func))
            residual = op_values - rhs_values
        else:
            residual = op_values

        return residual

    def jacobian_fd(self, x_pts, u_coeffs, eps=1e-7):
        """Compute Jacobian using finite differences.

        Args:
            x_pts: Chebyshev collocation points
            u_coeffs: Current solution coefficients
            eps: Finite difference step size

        Returns:
            J: Jacobian matrix (n+1 x n+1)
        """
        n = len(u_coeffs)
        J = np.zeros((n, n))

        # Compute F(u)
        F0 = self.evaluate_on_grid(x_pts, u_coeffs)

        # Finite difference approximation
        for j in range(n):
            u_perturbed = u_coeffs.copy()
            u_perturbed[j] += eps
            F1 = self.evaluate_on_grid(x_pts, u_perturbed)
            J[:, j] = (F1 - F0) / eps

        return J


def diff(u, order=1):
    """Differentiation operator for use in operator functions.

    This function is meant to be used inside operator function definitions
    to compute derivatives. It mimics MATLAB's diff() function.

    Args:
        u: Chebfun or array to differentiate
        order: Order of differentiation (default 1)

    Returns:
        Differentiated chebfun or array

    Examples:
        >>> N.op = lambda y: diff(y, 2) + y
        >>> N.op = lambda x, y: diff(y, 2) + x * diff(y)
    """
    if hasattr(u, 'diff'):
        # It's a chebfun
        result = u
        for _ in range(order):
            result = result.diff()
        return result
    else:
        # It's an array - use numerical differentiation
        # This is a placeholder - in practice we'd use spectral differentiation
        raise NotImplementedError("Array differentiation not yet implemented")


class DifferentialOp:
    """Helper class for building differential operators symbolically.

    This allows constructing operators like:
        L = D2 + x*D + I
    where D2, D, I are DifferentialOp instances.
    """

    def __init__(self, name, order=0, coeff=None):
        """Initialize differential operator.

        Args:
            name: Name of operator ('D', 'I', 'M', etc.)
            order: Order of differentiation
            coeff: Coefficient function (for multiplication)
        """
        self.name = name
        self.order = order
        self.coeff = coeff
        self.terms = [(1.0, order, coeff)]  # List of (scalar, diff_order, coeff_func)

    def __add__(self, other):
        """Add two differential operators."""
        if isinstance(other, DifferentialOp):
            result = DifferentialOp('composite', 0, None)
            result.terms = self.terms + other.terms
            return result
        elif isinstance(other, (int, float)):
            # Add constant (which is like adding c*I)
            result = DifferentialOp('composite', 0, None)
            result.terms = self.terms + [(float(other), 0, None)]
            return result
        else:
            return NotImplemented

    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply differential operator by scalar or function."""
        if isinstance(other, (int, float)):
            result = DifferentialOp('scaled', self.order, self.coeff)
            result.terms = [(scalar * other, order, coeff)
                           for scalar, order, coeff in self.terms]
            return result
        else:
            # Function multiplication
            return NotImplemented

    def __rmul__(self, other):
        """Right multiplication by scalar."""
        return self.__mul__(other)

    def to_chebop(self, domain):
        """Convert to Chebop operator.

        Args:
            domain: Domain for the operator

        Returns:
            Chebop instance representing this operator
        """
        from .chebop import Chebop

        # Start with zero operator
        op = None

        for scalar, order, coeff in self.terms:
            if order == 0 and coeff is None:
                # Scalar * Identity
                term = scalar * Chebop.identity(domain)
            elif order > 0 and coeff is None:
                # Scalar * D^k
                term = scalar * Chebop.diff(domain, order=order)
            elif order == 0 and coeff is not None:
                # Multiplication by function
                term = scalar * Chebop.diag(coeff)
            else:
                # coeff * D^k
                D_k = Chebop.diff(domain, order=order)
                M = Chebop.diag(coeff)
                term = scalar * (M * D_k)

            if op is None:
                op = term
            else:
                op = op + term

        return op


# Standard differential operators
def D(domain, order=1):
    """Create differentiation operator D^k.

    Args:
        domain: Domain for operator
        order: Order of differentiation

    Returns:
        DifferentialOp instance
    """
    return DifferentialOp('D', order=order)


def I(domain):
    """Create identity operator I.

    Args:
        domain: Domain for operator

    Returns:
        DifferentialOp instance
    """
    return DifferentialOp('I', order=0)
