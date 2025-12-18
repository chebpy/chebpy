"""Tests for AST-based order detection in operators.

This test suite is designed to validate that the order detection properly
handles all legitimate differential operator use cases. Tests are organized
by complexity and realistic operator patterns.

Key principle: An operator's derivative order is the maximum derivative
order that appears anywhere in its expression tree, regardless of whether
the derivative is inside numpy functions, multiplied by variables, or
part of complex expressions.

All tests assume an AST-based implementation that:
1. Builds an expression tree during symbolic tracing
2. Tracks derivative nodes in the tree
3. Returns the maximum derivative order found in any branch
"""

import numpy as np

from chebpy import chebfun, chebop
from chebpy.order_detection_ast import (
    BinOpNode,
    ConstNode,
    DiffNode,
    FunctionNode,
    OrderTracerAST,
    UnaryOpNode,
    VarNode,
)


class TestOrderDetectionBasic:
    """Basic order detection that must work with any implementation."""

    def test_identity_has_order_zero(self):
        """Identity operator u has order 0."""
        N = chebop([-1, 1])
        N.op = lambda u: u
        assert N._detect_order() == 0

    def test_first_derivative_direct(self):
        """Direct first derivative u' has order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff()
        assert N._detect_order() == 1

    def test_second_derivative_direct(self):
        """Direct second derivative u'' has order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)
        assert N._detect_order() == 2

    def test_third_derivative_direct(self):
        """Direct third derivative u''' has order 3."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(3)
        assert N._detect_order() == 3

    def test_fourth_derivative_direct(self):
        """Direct fourth derivative u'''' has order 4."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(4)
        assert N._detect_order() == 4


class TestOrderDetectionChaining:
    """Tests for chained derivative notation, which must work."""

    def test_first_derivative_chained(self):
        """Chained u.diff() has order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff()
        assert N._detect_order() == 1

    def test_second_derivative_chained(self):
        """Chained u.diff().diff() has order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff().diff()
        assert N._detect_order() == 2

    def test_third_derivative_chained(self):
        """Chained u.diff().diff().diff() has order 3."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff().diff().diff()
        assert N._detect_order() == 3

    def test_fourth_derivative_chained(self):
        """Chained u.diff().diff().diff().diff() has order 4."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff().diff().diff().diff()
        assert N._detect_order() == 4

    def test_mixed_direct_and_chained(self):
        """Mixed: u.diff(2) + u.diff() should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2) + u.diff()
        assert N._detect_order() == 2

    def test_deep_chain_ten_derivatives(self):
        """Very deep chain: 10 successive derivatives should have order 10."""
        N = chebop([-1, 1])

        def op(u):
            result = u
            for _ in range(10):
                result = result.diff()
            return result

        N.op = op
        assert N._detect_order() == 10


class TestOrderDetectionWithNumpy:
    """Tests for derivatives inside numpy functions.

    The order should be determined by the derivative, not affected by the
    numpy function wrapping it.
    """

    def test_sin_of_first_derivative(self):
        """sin(u') should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u.diff())
        assert N._detect_order() == 1

    def test_cos_of_second_derivative(self):
        """cos(u'') should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.cos(u.diff(2))
        assert N._detect_order() == 2

    def test_exp_of_first_derivative(self):
        """exp(u') should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: np.exp(u.diff())
        assert N._detect_order() == 1

    def test_exp_of_third_derivative(self):
        """exp(u''') should have order 3."""
        N = chebop([-1, 1])
        N.op = lambda u: np.exp(u.diff(3))
        assert N._detect_order() == 3

    def test_log_of_derivative_squared(self):
        """log(u'**2) should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: np.log(u.diff() ** 2)
        assert N._detect_order() == 1

    def test_sqrt_of_derivative_squared(self):
        """sqrt(u''**2) should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sqrt(u.diff(2) ** 2)
        assert N._detect_order() == 2

    def test_abs_of_first_derivative(self):
        """abs(u') should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: np.abs(u.diff())
        assert N._detect_order() == 1

    def test_nested_functions(self):
        """sin(cos(u.diff(2))) should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(np.cos(u.diff(2)))
        assert N._detect_order() == 2

    def test_sum_of_functions_of_different_orders(self):
        """sin(u') + exp(u'') should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u.diff()) + np.exp(u.diff(2))
        assert N._detect_order() == 2

    def test_product_of_functions_with_derivatives(self):
        """sin(u') * cos(u.diff(2)) should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u.diff()) * np.cos(u.diff(2))
        assert N._detect_order() == 2


class TestOrderDetectionVariableCoefficients:
    """Tests for operators with variable coefficients.

    These are standard differential operators of the form:
    L = a₀(x) + a₁(x)D + a₂(x)D² + ...

    The order is determined by the highest derivative term, regardless of
    what coefficient it's multiplied by.
    """

    def test_constant_times_first_derivative(self):
        """3*u' should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: 3 * u.diff()
        assert N._detect_order() == 1

    def test_x_times_second_derivative(self):
        """x*u'' should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u, x: x * u.diff(2)
        assert N._detect_order() == 2

    def test_sin_x_times_first_derivative(self):
        """sin(x)*u' should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u, x: np.sin(x) * u.diff()
        assert N._detect_order() == 1

    def test_polynomial_coefficient_second_derivative(self):
        """(1+x²)*u'' should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u, x: (1 + x**2) * u.diff(2)
        assert N._detect_order() == 2

    def test_exp_x_times_third_derivative(self):
        """exp(x)*u''' should have order 3."""
        N = chebop([-1, 1])
        N.op = lambda u, x: np.exp(x) * u.diff(3)
        assert N._detect_order() == 3

    def test_zero_coefficient_high_derivative(self):
        """0*u.diff(5) should still have order 5."""
        N = chebop([-1, 1])
        N.op = lambda u: 0 * u.diff(5)
        assert N._detect_order() == 5

    def test_complex_variable_coefficient(self):
        """np.sin(x) * u.diff(2) + np.exp(x) * u.diff() + u should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u, x: np.sin(x) * u.diff(2) + np.exp(x) * u.diff() + u
        assert N._detect_order() == 2


class TestOrderDetectionNonlinear:
    """Tests for nonlinear operators.

    Important: u itself (not its derivative) appearing nonlinearly should NOT
    affect the order. Only derivatives matter for order.

    Order is about the differential structure, not the function values.
    """

    def test_u_squared_has_order_zero(self):
        """u² is nonlinear but has order 0 (no derivatives)."""
        N = chebop([-1, 1])
        N.op = lambda u: u**2
        assert N._detect_order() == 0

    def test_sin_of_u_has_order_zero(self):
        """sin(u) is nonlinear but has order 0."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u)
        assert N._detect_order() == 0

    def test_exp_u_plus_derivative_has_order_one(self):
        """exp(u) + u' should have order 1 (from derivative)."""
        N = chebop([-1, 1])
        N.op = lambda u: np.exp(u) + u.diff()
        assert N._detect_order() == 1

    def test_u_times_derivative_has_order_one(self):
        """U * u' should have order 1 (from derivative, nonlinear in u)."""
        N = chebop([-1, 1])
        N.op = lambda u: u * u.diff()
        assert N._detect_order() == 1

    def test_u_squared_times_second_derivative_has_order_two(self):
        """u² * u'' should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u**2 * u.diff(2)
        assert N._detect_order() == 2

    def test_sin_u_times_derivative_has_order_one(self):
        """sin(u) * u' should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u) * u.diff()
        assert N._detect_order() == 1

    def test_derivative_of_u_squared_conceptual(self):
        """(u**2).diff() should have order 1.

        When we trace through (u**2).diff():
        - u**2 creates a trace of u*u
        - .diff() on that creates a derivative
        - So the order is 1
        """
        N = chebop([-1, 1])
        N.op = lambda u: (u**2).diff()
        assert N._detect_order() == 1

    def test_complex_nonlinear_burgers_like(self):
        """u*u.diff() + u.diff(2) (Burgers-like) should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u * u.diff() + u.diff(2)
        assert N._detect_order() == 2


class TestOrderDetectionMixedComplexity:
    """Tests for complex real-world operator patterns."""

    def test_sturm_liouville(self):
        """-(p*u')' + q*u: should correctly identify order 2.

        This is: -p*u'' - p'*u' + q*u
        Highest derivative is u'', so order 2.
        """
        N = chebop([-1, 1])
        N.op = lambda u, x: -(np.exp(x) * u.diff()).diff() + u
        assert N._detect_order() == 2

    def test_fourth_order_beam_equation(self):
        """U'''' (fourth order) should be detected as 4."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(4)
        assert N._detect_order() == 4

    def test_second_order_with_variable_coefficient_complex(self):
        """(1+x²)*u'' + (1-x²)*u' + u should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u, x: (1 + x**2) * u.diff(2) + (1 - x**2) * u.diff() + u
        assert N._detect_order() == 2

    def test_third_order_with_nonlinearity(self):
        """u*u''' + u'² + u should have order 3."""
        N = chebop([-1, 1])
        N.op = lambda u: u * u.diff(3) + u.diff() ** 2 + u
        assert N._detect_order() == 3

    def test_highly_nonlinear_second_order(self):
        """sin(u)*cos(u') + u'' should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(u) * np.cos(u.diff()) + u.diff(2)
        assert N._detect_order() == 2


class TestOrderDetectionEdgeCases:
    """Edge cases and boundary conditions."""

    def test_addition_chain_many_terms(self):
        """U + u' + u'' + u''' + u'''' should have order 4."""
        N = chebop([-1, 1])
        N.op = lambda u: u + u.diff() + u.diff(2) + u.diff(3) + u.diff(4)
        assert N._detect_order() == 4

    def test_deeply_nested_functions(self):
        """Multiple nested functions should still detect correctly."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(np.cos(np.exp(u.diff(2))))
        assert N._detect_order() == 2

    def test_power_of_derivative(self):
        """(u')^5 should have order 1."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff() ** 5
        assert N._detect_order() == 1

    def test_division_by_derivative(self):
        """(u''/u') should have order 2."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2) / u.diff()
        assert N._detect_order() == 2

    def test_sum_with_division(self):
        """U'' + 1/(u.diff(3)) should have order 3."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2) + 1 / u.diff(3)
        assert N._detect_order() == 3


class TestOrderDetectionConsistency:
    """Tests ensuring consistent behavior across different representations."""

    def test_direct_and_chained_equivalence_order_two(self):
        """u.diff(2) and u.diff().diff() should have same order."""
        N1 = chebop([-1, 1])
        N1.op = lambda u: u.diff(2)
        order1 = N1._detect_order()

        N2 = chebop([-1, 1])
        N2.op = lambda u: u.diff().diff()
        order2 = N2._detect_order()

        assert order1 == order2 == 2

    def test_direct_and_chained_equivalence_order_three(self):
        """u.diff(3) and u.diff().diff().diff() should have same order."""
        N1 = chebop([-1, 1])
        N1.op = lambda u: u.diff(3)
        order1 = N1._detect_order()

        N2 = chebop([-1, 1])
        N2.op = lambda u: u.diff().diff().diff()
        order2 = N2._detect_order()

        assert order1 == order2 == 3

    def test_numpy_wrapped_and_direct_same_order(self):
        """np.sin(u.diff(2)) and u.diff(2) should both have order 2."""
        N1 = chebop([-1, 1])
        N1.op = lambda u: u.diff(2)
        order1 = N1._detect_order()

        N2 = chebop([-1, 1])
        N2.op = lambda u: np.sin(u.diff(2))
        order2 = N2._detect_order()

        assert order1 == order2 == 2

    def test_multiple_representations_same_order(self):
        """Multiple equivalent forms should give same order."""
        forms = [
            lambda u: u.diff(2),
            lambda u: u.diff().diff(),
            lambda u: np.sin(u.diff(2)),
            lambda u: (1 + 0 * u) * u.diff(2),
            lambda u: u.diff(2) + 0 * u,
        ]

        orders = []
        for form in forms:
            N = chebop([-1, 1])
            N.op = form
            orders.append(N._detect_order())

        # All should be 2
        assert all(o == 2 for o in orders), f"Orders: {orders}"


class TestOrderDetectionDoesNotBreakSolving:
    """Integration tests: order detection must work correctly for actual BVP solving."""

    def test_second_order_solve_with_numpy_function_in_coeff(self):
        """Solving a second-order BVP with numpy functions in coefficients."""
        N = chebop([-1, 1])
        N.op = lambda u, x: np.exp(x) * u.diff(2) + np.sin(x) * u.diff() + u
        N.lbc = lambda u: u + u.diff()  # u(-1) + u'(-1) = 0
        N.rbc = lambda u: u  # u(1) = 0

        # Should be able to create BVP without error
        assert N._detect_order() == 2

    def test_third_order_solve_with_derivatives(self):
        """Solving a third-order BVP."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(3) + u.diff() + u
        N.lbc = lambda u: u
        N.bc = lambda u: u.diff()
        N.rbc = lambda u: u.diff(2)

        assert N._detect_order() == 3

    def test_nonlinear_second_order_burgers(self):
        """Nonlinear Burgers-like equation u*u' + u''."""
        N = chebop([-1, 1])
        N.op = lambda u: u * u.diff() + u.diff(2)
        N.lbc = lambda u: u + 1
        N.rbc = lambda u: u - 1

        assert N._detect_order() == 2


class TestOrderDetectionChallenging:
    """Challenging edge cases designed to stress-test the AST tracer."""

    def test_derivative_inside_multiple_numpy_functions(self):
        """Multiple numpy function calls: np.tanh(np.exp(u.diff(2)))."""
        N = chebop([-1, 1])
        N.op = lambda u: np.tanh(np.exp(u.diff(2)))
        assert N._detect_order() == 2

    def test_derivative_in_denominator(self):
        """Derivative in denominator: 1 / (1 + u.diff())."""
        N = chebop([-1, 1])
        N.op = lambda u: 1 / (1 + u.diff())
        assert N._detect_order() == 1

    def test_absolute_value_of_derivative(self):
        """Absolute value: np.abs(u.diff(3))."""
        N = chebop([-1, 1])
        N.op = lambda u: np.abs(u.diff(3))
        assert N._detect_order() == 3

    def test_sign_of_derivative(self):
        """Sign function: np.sign(u.diff())."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sign(u.diff())
        assert N._detect_order() == 1

    def test_arctan_of_derivative(self):
        """Arctan: np.arctan(u.diff(2))."""
        N = chebop([-1, 1])
        N.op = lambda u: np.arctan(u.diff(2))
        assert N._detect_order() == 2

    def test_sinh_cosh_mix(self):
        """Hyperbolic functions: np.sinh(u.diff()) + np.cosh(u.diff(2))."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sinh(u.diff()) + np.cosh(u.diff(2))
        assert N._detect_order() == 2

    def test_derivative_raised_to_power(self):
        """Power: (u.diff())**3 + (u.diff(2))**2."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff() ** 3 + u.diff(2) ** 2
        assert N._detect_order() == 2

    def test_floor_division_like_operation(self):
        """Integer-like operations: u.diff(2) // 2 (conceptually)."""
        N = chebop([-1, 1])
        N.op = lambda u: np.floor(u.diff(2))
        assert N._detect_order() == 2

    def test_maximum_of_derivatives(self):
        """Max operation: should detect highest."""
        N = chebop([-1, 1])
        N.op = lambda u: np.maximum(u.diff(), u.diff(3))
        assert N._detect_order() == 3

    def test_complex_coefficient_expression(self):
        """Complex: exp(sin(x)) * u.diff(2) + (1+x**2)*u.diff() via x arg."""
        N = chebop([-1, 1])
        N.op = lambda x, u: np.exp(np.sin(x)) * u.diff(2) + (1 + x**2) * u.diff()
        assert N._detect_order() == 2

    def test_deeply_nested_operations(self):
        """Deeply nested: sin(cos(exp(log(1 + u.diff(4)))))."""
        N = chebop([-1, 1])
        N.op = lambda u: np.sin(np.cos(np.exp(np.log(1 + np.abs(u.diff(4))))))
        assert N._detect_order() == 4

    def test_sum_of_many_derivatives(self):
        """Sum: u + u' + u'' + u''' + u'''' + u'''''. Order should be 5."""
        N = chebop([-1, 1])
        N.op = lambda u: u + u.diff() + u.diff(2) + u.diff(3) + u.diff(4) + u.diff(5)
        assert N._detect_order() == 5

    def test_product_of_many_derivatives(self):
        """Product: u' * u'' * u'''. Order should be 3."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff() * u.diff(2) * u.diff(3)
        assert N._detect_order() == 3

    def test_chained_diff_mixed_with_operations(self):
        """Mixed chaining: (u.diff() + 1).diff()."""
        N = chebop([-1, 1])
        N.op = lambda u: (u.diff() + 1).diff()
        # (u' + 1)' = u''
        assert N._detect_order() == 2

    def test_negative_derivative(self):
        """Negation: -u.diff(3)."""
        N = chebop([-1, 1])
        N.op = lambda u: -u.diff(3)
        assert N._detect_order() == 3

    def test_subtraction_of_derivatives(self):
        """Subtraction: u.diff(4) - u.diff(2)."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(4) - u.diff(2)
        assert N._detect_order() == 4

    def test_ratio_of_derivatives(self):
        """Ratio: u.diff(3) / u.diff()."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(3) / u.diff()
        assert N._detect_order() == 3

    def test_high_order_ten(self):
        """Very high order: u.diff(10)."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(10)
        assert N._detect_order() == 10

    def test_nonlinear_with_exponential(self):
        """Exponential nonlinearity: exp(u) * u.diff(2)."""
        N = chebop([-1, 1])
        N.op = lambda u: np.exp(u) * u.diff(2)
        assert N._detect_order() == 2


class TestASTNodeBaseOperators:
    """Tests for ASTNode base class operator implementations."""

    def test_astnode_wrap_operand_with_astnode(self):
        """ASTNode._wrap_operand with ASTNode returns it."""
        var = VarNode("u")
        const = ConstNode(5)
        result = var._wrap_operand(const)
        assert result is const

    def test_astnode_wrap_operand_with_non_astnode(self):
        """ASTNode._wrap_operand with non-ASTNode returns ConstNode."""
        var = VarNode("u")
        result = var._wrap_operand(42)
        assert isinstance(result, ConstNode)

    def test_astnode_add(self):
        """ASTNode.__add__ creates BinOpNode."""
        var = VarNode("u")
        result = var + 5
        assert isinstance(result, BinOpNode)
        assert result.op == "+"

    def test_astnode_radd(self):
        """ASTNode.__radd__ creates BinOpNode."""
        var = VarNode("u")
        result = 5 + var
        assert isinstance(result, BinOpNode)
        assert result.op == "+"

    def test_astnode_sub(self):
        """ASTNode.__sub__ creates BinOpNode."""
        var = VarNode("u")
        result = var - 5
        assert isinstance(result, BinOpNode)
        assert result.op == "-"

    def test_astnode_rsub(self):
        """ASTNode.__rsub__ creates BinOpNode."""
        var = VarNode("u")
        result = 5 - var
        assert isinstance(result, BinOpNode)
        assert result.op == "-"

    def test_astnode_mul(self):
        """ASTNode.__mul__ creates BinOpNode."""
        var = VarNode("u")
        result = var * 5
        assert isinstance(result, BinOpNode)
        assert result.op == "*"

    def test_astnode_rmul(self):
        """ASTNode.__rmul__ creates BinOpNode."""
        var = VarNode("u")
        result = 5 * var
        assert isinstance(result, BinOpNode)
        assert result.op == "*"

    def test_astnode_truediv(self):
        """ASTNode.__truediv__ creates BinOpNode."""
        var = VarNode("u")
        result = var / 5
        assert isinstance(result, BinOpNode)
        assert result.op == "/"

    def test_astnode_rtruediv(self):
        """ASTNode.__rtruediv__ creates BinOpNode."""
        var = VarNode("u")
        result = 5 / var
        assert isinstance(result, BinOpNode)
        assert result.op == "/"

    def test_astnode_pow(self):
        """ASTNode.__pow__ creates BinOpNode."""
        var = VarNode("u")
        result = var**2
        assert isinstance(result, BinOpNode)
        assert result.op == "**"

    def test_astnode_neg(self):
        """ASTNode.__neg__ creates UnaryOpNode."""
        var = VarNode("u")
        result = -var
        assert isinstance(result, UnaryOpNode)
        assert result.op == "-"


class TestASTNodeInternals:
    """Test internal AST node behavior."""

    def test_constnode_call_returns_self(self):
        """ConstNode.__call__ should return self."""
        node = ConstNode(5)
        assert node() is node

    def test_constnode_diff_returns_zero(self):
        """ConstNode.diff() should return a new ConstNode(0)."""
        node = ConstNode(5)
        result = node.diff()
        assert isinstance(result, ConstNode)
        assert result.get_max_order() == 0

    def test_varnode_call_returns_self(self):
        """VarNode.__call__ should return self."""
        node = VarNode("u")
        assert node() is node

    def test_varnode_diff_creates_diffnode(self):
        """VarNode.diff() should create a DiffNode."""
        node = VarNode("u")
        result = node.diff(3)
        assert isinstance(result, DiffNode)
        assert result.get_max_order() == 3

    def test_diffnode_call_returns_self(self):
        """DiffNode.__call__ should return self."""
        var = VarNode("u")
        diff = DiffNode(var, 2)
        assert diff() is diff

    def test_diffnode_diff_chains(self):
        """DiffNode.diff() should chain derivatives."""
        var = VarNode("u")
        diff1 = DiffNode(var, 2)
        diff2 = diff1.diff(3)
        assert isinstance(diff2, DiffNode)
        assert diff2.get_max_order() == 5  # 2 + 3

    def test_unaryop_double_negation(self):
        """UnaryOpNode double negation should return original."""
        var = VarNode("u")
        neg1 = UnaryOpNode("-", var)
        result = -neg1  # Double negation
        assert result is var

    def test_unaryop_non_neg_negation(self):
        """Negating a non-negation UnaryOpNode."""
        var = VarNode("u")
        # Create a unary op that's not negation (hypothetical)
        unary = UnaryOpNode("+", var)  # Not a negation
        result = -unary
        assert isinstance(result, UnaryOpNode)
        assert result.op == "-"

    def test_unaryop_call_returns_functionnode(self):
        """UnaryOpNode.__call__ should return FunctionNode."""
        var = VarNode("u")
        neg = UnaryOpNode("-", var)
        result = neg()
        assert isinstance(result, FunctionNode)

    def test_unaryop_diff(self):
        """UnaryOpNode.diff() should return negation of derivative."""
        var = VarNode("u")
        neg = UnaryOpNode("-", var)
        result = neg.diff(2)
        assert isinstance(result, UnaryOpNode)
        assert result.get_max_order() == 2

    def test_binop_call_returns_functionnode(self):
        """BinOpNode.__call__ should return FunctionNode."""
        var = VarNode("u")
        binop = BinOpNode("+", var, var)
        result = binop()
        assert isinstance(result, FunctionNode)

    def test_binop_diff_addition(self):
        """BinOpNode.diff() for addition."""
        var = VarNode("u")
        binop = BinOpNode("+", var, var)
        result = binop.diff()
        assert isinstance(result, BinOpNode)
        assert result.op == "+"
        assert result.get_max_order() == 1

    def test_binop_diff_subtraction(self):
        """BinOpNode.diff() for subtraction."""
        var = VarNode("u")
        binop = BinOpNode("-", var, var)
        result = binop.diff()
        assert isinstance(result, BinOpNode)
        assert result.op == "-"

    def test_binop_diff_multiplication(self):
        """BinOpNode.diff() for multiplication (product rule)."""
        var = VarNode("u")
        binop = BinOpNode("*", var, var)
        result = binop.diff()
        assert isinstance(result, BinOpNode)
        assert result.op == "+"  # Product rule: f'g + fg'

    def test_binop_diff_division(self):
        """BinOpNode.diff() for division."""
        var = VarNode("u")
        binop = BinOpNode("/", var, var)
        result = binop.diff()
        assert isinstance(result, BinOpNode)
        assert result.op == "/"

    def test_binop_diff_with_constnode(self):
        """BinOpNode.diff() with ConstNode operand."""
        var = VarNode("u")
        const = ConstNode(5)
        binop = BinOpNode("+", var, const)
        result = binop.diff()
        assert result.get_max_order() == 1

    def test_functionnode_call_returns_nested(self):
        """FunctionNode.__call__ should return nested FunctionNode."""
        var = VarNode("u")
        func = FunctionNode("sin", var)
        result = func()
        assert isinstance(result, FunctionNode)
        assert result.func_name == "call"

    def test_functionnode_diff(self):
        """FunctionNode.diff() creates DiffNode inside."""
        var = VarNode("u")
        func = FunctionNode("sin", var)
        result = func.diff(2)
        assert isinstance(result, FunctionNode)
        assert result.get_max_order() == 2

    def test_tracer_domain_property(self):
        """OrderTracerAST.domain should return _domain."""
        tracer = OrderTracerAST("u", domain=[-1, 1])
        assert tracer.domain == [-1, 1]

    def test_tracer_break_returns_list(self):
        """OrderTracerAST._break should return [self]."""
        tracer = OrderTracerAST("u")
        result = tracer._break(None)
        assert result == [tracer]

    def test_tracer_iter(self):
        """OrderTracerAST.__iter__ should iterate over [self]."""
        tracer = OrderTracerAST("u")
        result = list(tracer)
        assert result == [tracer]

    def test_tracer_wrap_operand_with_astnode(self):
        """OrderTracerAST._wrap_operand with ASTNode should return it."""
        tracer = OrderTracerAST("u")
        node = VarNode("x")
        result = tracer._wrap_operand(node)
        assert result is node

    def test_tracer_call(self):
        """OrderTracerAST.__call__ should return FunctionNode wrapper."""
        tracer = OrderTracerAST("u")
        result = tracer()
        assert isinstance(result, OrderTracerAST)
        assert result.get_max_order() == 0

    def test_tracer_array_ufunc_non_call_method(self):
        """OrderTracerAST.__array_ufunc__ with non-__call__ method returns NotImplemented."""
        tracer = OrderTracerAST("u")
        result = tracer.__array_ufunc__(np.sin, "reduce", tracer)
        assert result is NotImplemented

    def test_tracer_array_ufunc_non_tracer_input(self):
        """OrderTracerAST.__array_ufunc__ with non-tracer input returns NotImplemented."""
        tracer = OrderTracerAST("u")
        result = tracer.__array_ufunc__(np.sin, "__call__", 5)
        assert result is NotImplemented

    def test_tracer_array_ufunc_binary(self):
        """OrderTracerAST.__array_ufunc__ with binary ufunc."""
        tracer1 = OrderTracerAST("u")
        tracer2 = OrderTracerAST("v")
        # np.add is a binary ufunc
        result = np.add(tracer1.diff(), tracer2.diff(2))
        assert isinstance(result, OrderTracerAST)
        assert result.get_max_order() == 2

    def test_tracer_array_ufunc_three_inputs(self):
        """OrderTracerAST.__array_ufunc__ with 3+ inputs returns NotImplemented."""
        tracer = OrderTracerAST("u")
        # Manually call with 3 inputs (rare but possible)
        result = tracer.__array_ufunc__(np.sin, "__call__", tracer, tracer, tracer)
        assert result is NotImplemented


class TestOrderDetectionNumericalFallback:
    """Tests that trigger the numerical fallback when AST tracing fails."""

    def test_operator_that_raises_exception(self):
        """Operator that raises exception during AST tracing falls back to numerical."""
        N = chebop([-1, 1])

        # This operator will fail during AST tracing (division by zero on dummy tracers)
        # but numerical probing can handle it
        def op_that_fails(u):
            # Will fail during AST tracing but numerical method handles it
            x = np.linspace(-1, 1, 100)
            return np.sin(x) / (1 + np.sin(x) ** 2) * u.diff(2)

        N.op = op_that_fails
        order = N._detect_order()
        assert order == 2

    def test_operator_with_internal_exception(self):
        """Operator with exception in evaluation still detects order via fallback."""
        N = chebop([-1, 1])

        def problematic_op(u):
            # This will error on certain inputs but still be detectable numerically
            return u.diff(3) + np.log(np.abs(u) + 1e-10)

        N.op = problematic_op
        order = N._detect_order()
        assert order == 3


class TestOrderDetectionWithChebfunCoefficients:
    """Tests using actual Chebfun objects as coefficients (numerical fallback)."""

    def test_chebfun_coefficient_second_order(self):
        """Chebfun coefficient: x_fun * u.diff(2)."""
        x_fun = chebfun(lambda x: x, [-1, 1])
        N = chebop([-1, 1])
        N.op = lambda u: x_fun * u.diff(2)
        assert N._detect_order() == 2

    def test_chebfun_coefficient_mixed_order(self):
        """Mixed: sin(x)*u'' + exp(x)*u' + u (using Chebfun)."""
        sin_x = chebfun(lambda x: np.sin(x), [-1, 1])
        exp_x = chebfun(lambda x: np.exp(x), [-1, 1])
        N = chebop([-1, 1])
        N.op = lambda u: sin_x * u.diff(2) + exp_x * u.diff() + u
        assert N._detect_order() == 2

    def test_chebfun_coefficient_first_order(self):
        """First order with Chebfun: (1+x²)*u'."""
        coeff = chebfun(lambda x: 1 + x**2, [-1, 1])
        N = chebop([-1, 1])
        N.op = lambda u: coeff * u.diff()
        assert N._detect_order() == 1

    def test_chebfun_coefficient_fourth_order(self):
        """Fourth order: x*u''''."""
        x_fun = chebfun(lambda x: x, [0, 1])
        N = chebop([0, 1])
        N.op = lambda u: x_fun * u.diff(4)
        assert N._detect_order() == 4
