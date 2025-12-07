"""AST-based order detection for differential operators.

This module provides an expression tree (AST) implementation for detecting
the highest derivative order in differential operators. It correctly handles:

1. Chained derivatives: u.diff().diff().diff()
2. Direct derivatives: u.diff(3)
3. Derivatives inside numpy functions: sin(u.diff(2))
4. Variable coefficients: x*u.diff(2), sin(x)*u.diff()
5. Multiple branches: sin(u') + exp(u'')
6. Complex nonlinear operators: u*u' + u''

The order is determined by the maximum derivative order that
appears in any branch of the expression tree.
"""

from abc import ABC, abstractmethod
from typing import Any


class ASTNode(ABC):
    """Base class for expression tree nodes."""

    @abstractmethod
    def get_max_order(self) -> int:
        """Return the maximum derivative order in this subtree."""
        pass  # pragma: no cover

    def _wrap_operand(self, other):
        """Convert operand to ASTNode. Can be overridden in subclasses."""
        if isinstance(other, ASTNode):
            return other
        # Treat anything else as a constant
        return ConstNode(0)

    # Binary operations - concrete implementations using BinOpNode
    def __add__(self, other):
        """Add: self + other."""
        return BinOpNode("+", self, self._wrap_operand(other))

    def __radd__(self, other):
        """Add from right: other + self."""
        return BinOpNode("+", self._wrap_operand(other), self)

    def __sub__(self, other):
        """Subtract: self - other."""
        return BinOpNode("-", self, self._wrap_operand(other))

    def __rsub__(self, other):
        """Subtract from right: other - self."""
        return BinOpNode("-", self._wrap_operand(other), self)

    def __mul__(self, other):
        """Multiply: self * other."""
        return BinOpNode("*", self, self._wrap_operand(other))

    def __rmul__(self, other):
        """Multiply from right: other * self."""
        return BinOpNode("*", self._wrap_operand(other), self)

    def __truediv__(self, other):
        """Divide: self / other."""
        return BinOpNode("/", self, self._wrap_operand(other))

    def __rtruediv__(self, other):
        """Divide from right: other / self."""
        return BinOpNode("/", self._wrap_operand(other), self)

    def __pow__(self, other):
        """Power: self ** other."""
        return BinOpNode("**", self, self._wrap_operand(other))

    def __neg__(self):
        """Negate: -self."""
        return UnaryOpNode("-", self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Function call (for numpy functions)."""
        pass  # pragma: no cover

    @abstractmethod
    def diff(self, n: int = 1):
        """Differentiate n times."""
        pass  # pragma: no cover


class ConstNode(ASTNode):
    """Represents a constant value (doesn't contribute to order)."""

    def __init__(self, value: Any = 0):
        """Initialize constant node.

        Args:
            value: Constant value (default 0).
        """
        self.value = value

    def get_max_order(self) -> int:
        """Constants have no derivative terms."""
        return 0

    def __call__(self, *args, **kwargs):
        """Apply function to constant node."""
        # Constants evaluate to themselves under any function
        return self

    def diff(self, n: int = 1):
        """Differentiate constant node."""
        # Derivative of a constant is 0
        return ConstNode(0)


class VarNode(ASTNode):
    """Represents a variable (u or x) - no inherent derivative order."""

    def __init__(self, name: str):
        """Initialize variable node.

        Args:
            name: Variable name (e.g., 'u' or 'x').
        """
        self.name = name

    def get_max_order(self) -> int:
        """Variables themselves have order 0."""
        return 0

    def __call__(self, *args, **kwargs):
        """Apply function to variable node."""
        # Variable under a function call - preserves the variable
        return self

    def diff(self, n: int = 1):
        """Taking derivative of a variable u creates a DiffNode."""
        return DiffNode(self, n)


class DiffNode(ASTNode):
    """Represents a derivative operation."""

    def __init__(self, expr: ASTNode, order: int = 1):
        """Create a derivative node.

        Args:
            expr: The expression being differentiated
            order: The number of times to differentiate (default 1)
        """
        self.expr = expr
        self.order = order

    def get_max_order(self) -> int:
        """Maximum order is the derivative order plus any in the subexpression.

        For example:
        - u' has order 1
        - (u')' (which is u'') has order 2
        - (sin(u'))' has order 2 (chain rule produces u'' in the derivative)
        """
        # The order of this derivative node is at least self.order
        # plus any derivatives in the subexpression
        sub_order = self.expr.get_max_order()
        return self.order + sub_order

    def __call__(self, *args, **kwargs):
        """Apply function to derivative node."""
        # Derivative under a function call preserves the order
        # e.g., sin(u') has order 1 from the u'
        return self

    def diff(self, n: int = 1):
        """Taking another derivative increases the order.

        u'.diff() = u''.
        We can chain this: accumulate orders through DiffNode construction.
        """
        # Create a new DiffNode wrapping this one
        return DiffNode(self, n)


class UnaryOpNode(ASTNode):
    """Represents a unary operation like negation."""

    def __init__(self, op: str, expr: ASTNode):
        """Initialize unary operation node.

        Args:
            op: Operation symbol (e.g., '-').
            expr: Expression to apply operation to.
        """
        self.op = op
        self.expr = expr

    def get_max_order(self) -> int:
        """Unary ops don't change the maximum derivative order."""
        return self.expr.get_max_order()

    def __neg__(self):
        """Negate: -self."""
        if self.op == "-":
            # Double negation
            return self.expr
        return UnaryOpNode("-", self)

    def __call__(self, *args, **kwargs):
        """Apply function to unary operation node."""
        # Function call on unary op
        return FunctionNode("call", self)

    def diff(self, n: int = 1):
        """Derivative of negation is negation of derivative."""
        return UnaryOpNode("-", DiffNode(self.expr, n))


class BinOpNode(ASTNode):
    """Represents a binary operation like +, -, *, /."""

    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        """Initialize binary operation node.

        Args:
            op: Operation symbol (e.g., '+', '-', '*', '/').
            left: Left operand.
            right: Right operand.
        """
        self.op = op
        self.left = left
        self.right = right

    def get_max_order(self) -> int:
        """Maximum order is max of both sides.

        For a + b, order = max(order(a), order(b))
        For a * b, order = max(order(a), order(b))
        For a / b, order = max(order(a), order(b))

        This is correct because:
        - (u' + u'') has order 2 (from u'')
        - (u' * u'') has order 2 (both terms have derivatives)
        - u.diff(2) / u.diff() has order 2 (numerator dominates)
        """
        left_order = self.left.get_max_order()
        right_order = self.right.get_max_order()
        return max(left_order, right_order)

    def __call__(self, *args, **kwargs):
        """Apply function to binary operation node."""
        # Binary operation under a function call
        return FunctionNode("call", self)

    def diff(self, n: int = 1):
        """Derivative of a binary operation.

        This is somewhat artificial since we're in symbolic tracing,
        but we should propagate correctly.
        """
        # Both sides get differentiated
        left_diff = DiffNode(self.left, n) if not isinstance(self.left, (ConstNode, VarNode)) else self.left.diff(n)
        right_diff = DiffNode(self.right, n) if not isinstance(self.right, (ConstNode, VarNode)) else self.right.diff(n)

        if self.op == "+":
            return BinOpNode("+", left_diff, right_diff)
        elif self.op == "-":
            return BinOpNode("-", left_diff, right_diff)
        elif self.op == "*":
            # Product rule
            return BinOpNode("+", BinOpNode("*", left_diff, self.right), BinOpNode("*", self.left, right_diff))
        else:
            # For other operations, take max order
            return BinOpNode(self.op, left_diff, right_diff)


class FunctionNode(ASTNode):
    """Represents application of a function (e.g., sin, cos, exp)."""

    def __init__(self, func_name: str, expr: ASTNode):
        """Initialize function node.

        Args:
            func_name: Name of the function (e.g., 'sin', 'exp').
            expr: Expression to apply function to.
        """
        self.func_name = func_name
        self.expr = expr

    def get_max_order(self) -> int:
        """Function application doesn't change derivative order.

        sin(u') has order 1 (from u')
        exp(u'') has order 2 (from u'')
        """
        return self.expr.get_max_order()

    def __call__(self, *args, **kwargs):
        """Apply function to function node."""
        # Nested function calls
        return FunctionNode("call", self)

    def diff(self, n: int = 1):
        """Derivative of a function application.

        d/dx sin(u') = cos(u') * u''
        This has order at least 2 (from u'').
        """
        # Differentiate the inner expression
        inner_diff = DiffNode(self.expr, n)
        # Wrap in function (chain rule)
        return FunctionNode(self.func_name, inner_diff)


class OrderTracerAST(ASTNode):
    """Main tracer class that builds an AST during symbolic evaluation.

    This class builds an expression tree during symbolic tracing to detect
    the maximum derivative order in a differential operator. It implements
    a minimal Chebfun-compatible interface to allow operations like
    `chebfun_coeff * tracer` to work during tracing.
    """

    # Marker for Chebfun to recognize and defer to __rmul__
    _is_order_tracer = True

    # High priority ensures __rmul__ is called before Chebfun.__mul__
    __array_priority__ = 1000

    def __init__(self, name: str = "u", domain=None):
        """Initialize the tracer.

        Args:
            name: Name of the variable ("u" or "x")
            domain: Optional domain for Chebfun compatibility
        """
        self.name = name
        self._root: ASTNode = VarNode(name)
        # Chebfun compatibility attributes
        self._domain = domain
        self.isempty = False

    @property
    def domain(self):
        """Domain property for Chebfun compatibility."""
        return self._domain

    def _break(self, targetdomain):
        """Stub for Chebfun compatibility - returns iterable of self."""
        return [self]

    def __iter__(self):
        """Iterate: iter(self)."""
        return iter([self])

    def get_max_order(self) -> int:
        """Get the maximum derivative order in the expression tree."""
        return self._root.get_max_order()

    def _wrap_operand(self, other):
        """Convert Python values to AST nodes."""
        if isinstance(other, OrderTracerAST):
            return other._root
        elif isinstance(other, ASTNode):
            return other
        else:
            # Wrap in ConstNode but preserve the actual object
            # This allows operator_compiler to extract chebfun coefficients
            return ConstNode(other)

    def _create_result(self, root: ASTNode) -> "OrderTracerAST":
        """Create a new OrderTracerAST with the given root."""
        new_tracer = OrderTracerAST(self.name, domain=self._domain)
        new_tracer._root = root
        return new_tracer

    def __add__(self, other):
        """Add: self + other."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("+", self._root, other_node))

    def __radd__(self, other):
        """Add from right: other + self."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("+", other_node, self._root))

    def __sub__(self, other):
        """Subtract: self - other."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("-", self._root, other_node))

    def __rsub__(self, other):
        """Subtract from right: other - self."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("-", other_node, self._root))

    def __mul__(self, other):
        """Multiply: self * other."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("*", self._root, other_node))

    def __rmul__(self, other):
        """Multiply from right: other * self."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("*", other_node, self._root))

    def __truediv__(self, other):
        """Divide: self / other."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("/", self._root, other_node))

    def __rtruediv__(self, other):
        """Divide from right: other / self."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("/", other_node, self._root))

    def __pow__(self, other):
        """Power: self ** other."""
        other_node = self._wrap_operand(other)
        return self._create_result(BinOpNode("**", self._root, other_node))

    def __neg__(self):
        """Negate: -self."""
        return self._create_result(UnaryOpNode("-", self._root))

    def __call__(self, *args, **kwargs):
        """Apply function to tracer."""
        # For numpy function calls
        return self._create_result(FunctionNode("call", self._root))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions (ufuncs).

        This is called when a numpy ufunc like np.sin, np.cos, np.exp, etc.
        is applied to an OrderTracerAST. We wrap the function call in our AST.

        Args:
            ufunc: The numpy ufunc being applied
            method: The ufunc method ('__call__', 'reduce', etc.)
            inputs: Input arrays
            kwargs: Keyword arguments

        Returns:
            An OrderTracerAST with the function node in the expression tree.
        """
        if method != "__call__":
            # Only handle regular function calls, not reduce/accumulate/etc.
            return NotImplemented

        # Find which input is the OrderTracerAST and build result
        result_root = None

        if len(inputs) == 1:
            # Unary function: sin, cos, exp, log, abs, etc.
            if isinstance(inputs[0], OrderTracerAST):
                func_name = ufunc.__name__
                result_root = FunctionNode(func_name, inputs[0]._root)
            else:
                return NotImplemented

        elif len(inputs) == 2:
            # Binary function: add, multiply, etc.
            # But these should be handled by __add__, __mul__, etc.
            # This is a fallback for numpy's implementation
            left = inputs[0]._root if isinstance(inputs[0], OrderTracerAST) else ConstNode(inputs[0])
            right = inputs[1]._root if isinstance(inputs[1], OrderTracerAST) else ConstNode(inputs[1])
            func_name = ufunc.__name__
            result_root = FunctionNode(func_name, BinOpNode("+", left, right))  # Simplification, logic stays the same

        # ufuncs have 1 or 2 inputs, so no need to check for 3+

        if result_root is None:
            return NotImplemented

        return self._create_result(result_root)

    def diff(self, n: int = 1):
        """Differentiate n times."""
        return self._create_result(DiffNode(self._root, n))
