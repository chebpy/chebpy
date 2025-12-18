"""Operator pre-compilation for IVP solving.

This module provides operator pre-compilation, converting differential
operators into optimized functions for ODE solvers. This eliminates
per-step overhead from dynamic operator evaluation.

The approach:
1. Trace operator with AST to extract structure
2. Identify and pre-evaluate chebfun coefficients
3. Generate optimized Python function with coefficients in closure
4. Pass directly to scipy's solve_ivp

Example:
    >>> from chebpy import chebop, chebfun
    >>> from numpy import sin
    >>> L = chebop([0, 200])
    >>> coef = chebfun(lambda t: sin(t/10), [0, 200])
    >>> L.op = lambda y: y.diff(2) + 0.06*coef*y.diff() + y
    >>>
    >>> # Without pre-compilation: slower due to operator overhead at every time step
    >>> # With pre-compilation: much faster performance
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .order_detection_ast import BinOpNode, ConstNode, DiffNode, FunctionNode, OrderTracerAST, UnaryOpNode, VarNode


class CoefficientExtractor:
    """Extract coefficients from AST by walking the tree."""

    def __init__(self, max_order: int):
        """Initialize coefficient extractor.

        Args:
            max_order: Maximum derivative order in the operator.
        """
        self.max_order = max_order
        self.chebfuns = []  # List of chebfun coefficients found

    def extract(self, ast_root) -> tuple[Any, Any]:
        """Extract coefficient of highest derivative and remaining expression.

        Walk the tree to identify the term containing the highest derivative,
        extract its coefficient, and return the rest of the expression.

        Args:
            ast_root: Root node of the AST.

        Returns:
            (highest_deriv_coeff, remaining_expr)
            - highest_deriv_coeff: Coefficient multiplying u^(n)
            - remaining_expr: Everything else (moved to RHS)
        """
        # Start by finding terms with highest derivative
        highest_deriv_coeff = 1.0
        remaining_terms = []

        if isinstance(ast_root, BinOpNode) and ast_root.op == "+":
            # Expression is a sum: split into terms
            terms = self._split_sum(ast_root)

            for term in terms:
                if self._has_highest_deriv(term):
                    # Extract coefficient from this term
                    coeff = self._extract_coeff_from_term(term)
                    highest_deriv_coeff = coeff
                else:
                    remaining_terms.append(term)
        else:
            # Single term - check if it's the highest derivative
            if self._has_highest_deriv(ast_root):
                highest_deriv_coeff = self._extract_coeff_from_term(ast_root)
            else:
                remaining_terms.append(ast_root)

        # Combine remaining terms
        if len(remaining_terms) == 0:
            remaining_expr = ConstNode(0)
        elif len(remaining_terms) == 1:
            remaining_expr = remaining_terms[0]
        else:
            # Sum up all remaining terms
            remaining_expr = remaining_terms[0]
            for term in remaining_terms[1:]:
                remaining_expr = BinOpNode("+", remaining_expr, term)

        return highest_deriv_coeff, remaining_expr

    def _split_sum(self, node) -> list:
        """Split a sum into individual terms."""
        if isinstance(node, BinOpNode) and node.op == "+":
            return self._split_sum(node.left) + self._split_sum(node.right)
        elif isinstance(node, BinOpNode) and node.op == "-":
            # Subtraction: left + (-right)

            return self._split_sum(node.left) + [UnaryOpNode("-", node.right)]
        else:
            return [node]

    def _has_highest_deriv(self, node) -> bool:
        """Check if this node contains the highest derivative."""
        if isinstance(node, DiffNode):
            # Calculate total derivative order
            order = node.get_max_order()
            return order == self.max_order
        elif hasattr(node, "left") and hasattr(node, "right"):
            return self._has_highest_deriv(node.left) or self._has_highest_deriv(node.right)
        elif hasattr(node, "expr"):
            return self._has_highest_deriv(node.expr)
        return False

    def _extract_coeff_from_term(self, term):
        """Extract coefficient from a term containing highest derivative.

        For example:
        - 0.06*coef*u.diff() -> returns 0.06*coef
        - 5*u.diff(2) -> returns 5
        - u.diff(2) -> returns 1
        """
        if isinstance(term, DiffNode):
            # Just the derivative, no coefficient
            return 1.0
        elif isinstance(term, BinOpNode) and term.op == "*":
            # Multiplication: check which side has the derivative
            if self._has_highest_deriv(term.right):
                # Coefficient is on the left
                return term.left
            else:
                # Coefficient is on the right
                return term.right
        else:
            # Default: coefficient is 1
            return 1.0


class CodeGenerator:
    """Generate optimized Python functions from AST and pre-evaluated coefficients."""

    def __init__(self, domain, order: int):
        """Initialize code generator.

        Args:
            domain: Domain of the problem.
            order: Maximum derivative order.
        """
        self.domain = domain
        self.order = order
        self.grid_cache = {}  # Will store pre-evaluated chebfun coefficients

    def preeval_coefficients(self, coeff_ast, n_grid: int = 500):
        """Pre-evaluate chebfun coefficients on a dense grid.

        Args:
            coeff_ast: AST node representing the coefficient.
            n_grid: Number of grid points.

        Returns:
            Evaluator function that uses interpolation.
        """
        # Check if coefficient is a chebfun (has domain attribute)
        if hasattr(coeff_ast, "value") and hasattr(coeff_ast.value, "domain"):
            chebfun_obj = coeff_ast.value
            # Pre-evaluate on grid
            a, b = self.domain.support[0], self.domain.support[-1]
            t_grid = np.linspace(a, b, n_grid)
            values = chebfun_obj(t_grid)

            # Store in cache
            self.grid_cache[id(chebfun_obj)] = (t_grid, values)

            # Return interpolator
            def evaluator(t):
                return np.interp(t, t_grid, values)

            return evaluator
        elif isinstance(coeff_ast, ConstNode):
            # Constant coefficient
            val = coeff_ast.value if hasattr(coeff_ast, "value") else 1.0
            return lambda t: val
        else:
            # Unknown coefficient type - try to evaluate if callable
            if callable(coeff_ast):
                return coeff_ast
            else:
                return lambda t: 1.0

    def generate_rhs_function(self, coeff_evaluator: Callable, expr_evaluator: Callable, rhs: float = 0.0) -> Callable:
        """Generate optimized RHS function for ODE solver.

        Creates a function of the form:
            lambda t, u: [u[1], u[2], ..., (rhs - expr) / coeff]

        Args:
            coeff_evaluator: Function to evaluate coefficient at time t.
            expr_evaluator: Function to evaluate remaining expression.
            rhs: Right-hand side value (default 0).

        Returns:
            Compiled function suitable for solve_ivp.
        """
        order = self.order

        def compiled_rhs(t, u):
            """Optimized ODE RHS function.

            Args:
                t: Current time.
                u: State vector [u, u', u'', ..., u^(n-1)].

            Returns:
                Derivative vector [u', u'', ..., u^(n)].
            """
            # First-order reformulation: u'_i = u_{i+1}
            result = np.zeros(order)
            for i in range(order - 1):
                result[i] = u[i + 1]

            # Highest derivative: u^(n) = (rhs - expr) / coeff
            coeff_val = coeff_evaluator(t)
            expr_val = expr_evaluator(t, u)
            result[order - 1] = (rhs - expr_val) / coeff_val

            return result

        return compiled_rhs


class ExpressionEvaluator:
    """Evaluate AST expressions at runtime using pre-evaluated coefficients."""

    def __init__(self, grid_cache: dict):
        """Initialize expression evaluator.

        Args:
            grid_cache: Dictionary mapping chebfun IDs to (t_grid, values).
        """
        self.grid_cache = grid_cache

    def create_evaluator(self, expr_ast) -> Callable:
        """Create a function that evaluates the AST at given (t, u).

        Args:
            expr_ast: AST node representing the expression.

        Returns:
            Function (t, u) -> scalar value.
        """

        def evaluate(t, u):
            return self._eval_node(expr_ast, t, u)

        return evaluate

    def _eval_node(self, node, t, u):
        """Recursively evaluate an AST node."""
        if isinstance(node, ConstNode):
            # Check if it's a wrapped chebfun
            if hasattr(node, "value") and hasattr(node.value, "domain"):
                # It's a chebfun coefficient
                chebfun_obj = node.value
                obj_id = id(chebfun_obj)
                if obj_id in self.grid_cache:
                    t_grid, values = self.grid_cache[obj_id]
                    return np.interp(t, t_grid, values)
                else:
                    return chebfun_obj(t)
            else:
                return node.value if hasattr(node, "value") else 0.0

        elif isinstance(node, VarNode):
            # Variable u (not a derivative) - return u[0]
            return u[0]

        elif isinstance(node, DiffNode):
            # Derivative: access u array
            order = node.order
            if order < len(u):
                return u[order]
            else:
                raise ValueError(f"Derivative order {order} exceeds state vector size")

        elif isinstance(node, BinOpNode):
            left_val = self._eval_node(node.left, t, u)
            right_val = self._eval_node(node.right, t, u)

            if node.op == "+":
                return left_val + right_val
            elif node.op == "-":
                return left_val - right_val
            elif node.op == "*":
                return left_val * right_val
            elif node.op == "/":
                return left_val / right_val
            elif node.op == "**":
                return left_val**right_val
            else:
                raise ValueError(f"Unknown binary operator: {node.op}")

        elif isinstance(node, UnaryOpNode):
            # Unary operation
            expr_val = self._eval_node(node.expr, t, u)
            if node.op == "-":
                return -expr_val
            else:
                return expr_val

        elif isinstance(node, FunctionNode):
            # Function application (sin, cos, exp, log, etc.)
            expr_val = self._eval_node(node.expr, t, u)

            # Map function names to numpy functions
            func_map = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
            }

            func = func_map.get(node.func_name)
            if func is None:
                raise ValueError(f"Unsupported function in operator compiler: {node.func_name}")

            result = func(expr_val)

            # Check for NaN/Inf in function result
            if np.isnan(result) or np.isinf(result):
                raise ValueError(
                    f"Function {node.func_name}({expr_val}) produced NaN or Inf. "
                    "This typically indicates invalid input (e.g., log of negative number)."
                )

            return result

        elif hasattr(node, "expr"):
            # Generic node with expr attribute
            return self._eval_node(node.expr, t, u)

        else:
            return 0.0


class OperatorCompiler:
    """Main class for compiling differential operators into optimized functions."""

    def compile_ivp_operator(self, op: Callable, domain, max_order: int, rhs: float = 0.0) -> Callable:
        """Compile a differential operator into an optimized ODE RHS function.

        Process:
        1. Trace operator to build AST
        2. Extract coefficient of highest derivative
        3. Pre-evaluate chebfun coefficients on grid
        4. Generate optimized function with coefficients in closure

        Args:
            op: Differential operator function (e.g., lambda y: y.diff(2) + y).
            domain: Domain of the problem.
            max_order: Maximum derivative order in the operator.
            rhs: Right-hand side value (default 0).

        Returns:
            Optimized function (t, u) -> du/dt suitable for solve_ivp.
        """
        # 1. Trace operator to build AST
        tracer = OrderTracerAST(domain=domain)
        result = op(tracer)

        # Extract the root AST node
        if hasattr(result, "_root"):
            ast_root = result._root
        else:
            ast_root = result

        # 2. Extract coefficients and structure
        extractor = CoefficientExtractor(max_order)
        highest_deriv_coeff, remaining_expr = extractor.extract(ast_root)

        # 3. Set up code generator
        codegen = CodeGenerator(domain, max_order)

        # Pre-evaluate coefficient
        coeff_evaluator = codegen.preeval_coefficients(highest_deriv_coeff)

        # Create expression evaluator
        expr_eval = ExpressionEvaluator(codegen.grid_cache)
        expr_evaluator = expr_eval.create_evaluator(remaining_expr)

        # 4. Generate optimized RHS function
        compiled_fn = codegen.generate_rhs_function(coeff_evaluator, expr_evaluator, rhs)

        return compiled_fn
