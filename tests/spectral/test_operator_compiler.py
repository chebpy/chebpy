"""Tests for operator_compiler.py."""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.operator_compiler import (
    CodeGenerator,
    CoefficientExtractor,
    ExpressionEvaluator,
    OperatorCompiler,
)
from chebpy.order_detection_ast import (
    BinOpNode,
    ConstNode,
    DiffNode,
    FunctionNode,
    UnaryOpNode,
    VarNode,
)
from chebpy.utilities import Domain


class TestCoefficientExtractor:
    """Test CoefficientExtractor class."""

    def test_extract_single_diff(self):
        """Test extracting coefficient from single derivative."""
        extractor = CoefficientExtractor(max_order=2)
        # Create AST for u.diff(2)
        diff_node = DiffNode(VarNode("u"), 2)

        coeff, remaining = extractor.extract(diff_node)

        assert coeff == 1.0
        assert isinstance(remaining, ConstNode)

    def test_extract_sum_with_diff(self):
        """Test extracting from sum containing highest derivative."""
        extractor = CoefficientExtractor(max_order=2)

        # Create AST for u.diff(2) + u.diff(1)
        diff2 = DiffNode(VarNode("u"), 2)
        diff1 = DiffNode(VarNode("u"), 1)
        sum_node = BinOpNode("+", diff2, diff1)

        coeff, remaining = extractor.extract(sum_node)

        assert coeff == 1.0

    def test_extract_multiplication_coeff_left(self):
        """Test extracting coefficient on left of multiplication."""
        extractor = CoefficientExtractor(max_order=2)

        # Create AST for 5 * u.diff(2)
        const = ConstNode(5.0)
        diff = DiffNode(VarNode("u"), 2)
        mult_node = BinOpNode("*", const, diff)

        coeff, remaining = extractor.extract(mult_node)

        # Coefficient is on left
        assert coeff == const

    def test_extract_multiplication_coeff_right(self):
        """Test extracting coefficient on right of multiplication."""
        extractor = CoefficientExtractor(max_order=2)

        # Create AST for u.diff(2) * 5
        diff = DiffNode(VarNode("u"), 2)
        const = ConstNode(5.0)
        mult_node = BinOpNode("*", diff, const)

        coeff, remaining = extractor.extract(mult_node)

        # Coefficient is on right
        assert coeff == const

    def test_split_sum_with_subtraction(self):
        """Test splitting sum that contains subtraction."""
        extractor = CoefficientExtractor(max_order=2)

        # Create AST for u.diff(2) - u
        diff = DiffNode(VarNode("u"), 2)
        var = VarNode("u")
        sub_node = BinOpNode("-", diff, var)

        terms = extractor._split_sum(sub_node)

        # Should split into diff and negated var
        assert len(terms) == 2

    def test_extract_no_highest_deriv(self):
        """Test extracting when no highest derivative present."""
        extractor = CoefficientExtractor(max_order=2)

        # Create AST for just u (no derivatives)
        var_node = VarNode("u")

        coeff, remaining = extractor.extract(var_node)

        # Default coefficient
        assert coeff == 1.0


class TestCodeGenerator:
    """Test CodeGenerator class."""

    def test_preeval_chebfun_coefficient(self):
        """Test pre-evaluating chebfun coefficient."""
        domain = Domain([0, 1])
        codegen = CodeGenerator(domain, order=2)

        # Create a chebfun coefficient
        coef = chebfun(lambda t: np.sin(t), [0, 1])

        # Wrap it in a ConstNode
        const_node = ConstNode(coef)
        const_node.value = coef

        evaluator = codegen.preeval_coefficients(const_node, n_grid=100)

        # Test that evaluator works
        result = evaluator(0.5)
        expected = np.sin(0.5)
        assert abs(result - expected) < 0.1

    def test_preeval_constant_coefficient(self):
        """Test pre-evaluating constant coefficient."""
        domain = Domain([0, 1])
        codegen = CodeGenerator(domain, order=2)

        const_node = ConstNode(3.5)

        evaluator = codegen.preeval_coefficients(const_node)

        result = evaluator(0.5)
        assert result == 3.5

    def test_preeval_callable_coefficient(self):
        """Test pre-evaluating callable coefficient."""
        domain = Domain([0, 1])
        codegen = CodeGenerator(domain, order=2)

        def my_coef(t):
            return t**2

        evaluator = codegen.preeval_coefficients(my_coef)

        result = evaluator(0.5)
        assert result == 0.25

    def test_preeval_unknown_coefficient(self):
        """Test pre-evaluating unknown coefficient type."""
        domain = Domain([0, 1])
        codegen = CodeGenerator(domain, order=2)

        # Pass something that's not a ConstNode, callable, or has value
        class UnknownNode:
            pass

        unknown = UnknownNode()
        evaluator = codegen.preeval_coefficients(unknown)

        # Should return lambda returning 1.0
        result = evaluator(0.5)
        assert result == 1.0

    def test_generate_rhs_function(self):
        """Test generating RHS function."""
        domain = Domain([0, 1])
        codegen = CodeGenerator(domain, order=2)

        # Simple coefficient and expression evaluators
        def coeff_eval(t):
            return 1.0

        def expr_eval(t, u):
            return -u[0]  # -u term

        rhs_func = codegen.generate_rhs_function(coeff_eval, expr_eval, rhs=0.0)

        # Test: u'' = -u, u = [u, u']
        u = np.array([1.0, 0.0])  # u=1, u'=0
        result = rhs_func(0.0, u)

        # result[0] = u' = 0
        # result[1] = u'' = (0 - (-1)) / 1 = 1
        assert result[0] == 0.0
        assert result[1] == 1.0


class TestExpressionEvaluator:
    """Test ExpressionEvaluator class."""

    def test_eval_const_node(self):
        """Test evaluating constant node."""
        evaluator = ExpressionEvaluator({})
        const = ConstNode(5.0)

        eval_func = evaluator.create_evaluator(const)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 5.0

    def test_eval_var_node(self):
        """Test evaluating variable node."""
        evaluator = ExpressionEvaluator({})
        var = VarNode("u")

        eval_func = evaluator.create_evaluator(var)
        result = eval_func(0.0, np.array([3.14, 1.0]))

        assert result == 3.14  # u[0]

    def test_eval_diff_node(self):
        """Test evaluating derivative node."""
        evaluator = ExpressionEvaluator({})
        diff = DiffNode(VarNode("u"), 1)

        eval_func = evaluator.create_evaluator(diff)
        result = eval_func(0.0, np.array([1.0, 2.0, 3.0]))

        assert result == 2.0  # u[1] = u'

    def test_eval_diff_node_order_exceeds(self):
        """Test derivative order exceeds state vector."""
        evaluator = ExpressionEvaluator({})
        diff = DiffNode(VarNode("u"), 5)

        eval_func = evaluator.create_evaluator(diff)

        with pytest.raises(ValueError, match="exceeds state vector"):
            eval_func(0.0, np.array([1.0, 2.0]))

    def test_eval_binop_add(self):
        """Test evaluating addition."""
        evaluator = ExpressionEvaluator({})
        add_node = BinOpNode("+", ConstNode(2.0), ConstNode(3.0))

        eval_func = evaluator.create_evaluator(add_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 5.0

    def test_eval_binop_sub(self):
        """Test evaluating subtraction."""
        evaluator = ExpressionEvaluator({})
        sub_node = BinOpNode("-", ConstNode(5.0), ConstNode(3.0))

        eval_func = evaluator.create_evaluator(sub_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 2.0

    def test_eval_binop_mul(self):
        """Test evaluating multiplication."""
        evaluator = ExpressionEvaluator({})
        mul_node = BinOpNode("*", ConstNode(4.0), ConstNode(3.0))

        eval_func = evaluator.create_evaluator(mul_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 12.0

    def test_eval_binop_div(self):
        """Test evaluating division."""
        evaluator = ExpressionEvaluator({})
        div_node = BinOpNode("/", ConstNode(10.0), ConstNode(2.0))

        eval_func = evaluator.create_evaluator(div_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 5.0

    def test_eval_binop_pow(self):
        """Test evaluating power."""
        evaluator = ExpressionEvaluator({})
        pow_node = BinOpNode("**", ConstNode(2.0), ConstNode(3.0))

        eval_func = evaluator.create_evaluator(pow_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 8.0

    def test_eval_binop_unknown(self):
        """Test evaluating unknown binary operator."""
        evaluator = ExpressionEvaluator({})
        unknown_node = BinOpNode("%", ConstNode(10.0), ConstNode(3.0))

        eval_func = evaluator.create_evaluator(unknown_node)

        with pytest.raises(ValueError, match="Unknown binary operator"):
            eval_func(0.0, np.array([0.0]))

    def test_eval_unary_minus(self):
        """Test evaluating unary minus."""
        evaluator = ExpressionEvaluator({})
        neg_node = UnaryOpNode("-", ConstNode(5.0))

        eval_func = evaluator.create_evaluator(neg_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == -5.0

    def test_eval_unary_plus(self):
        """Test evaluating unary plus (or other)."""
        evaluator = ExpressionEvaluator({})
        pos_node = UnaryOpNode("+", ConstNode(5.0))

        eval_func = evaluator.create_evaluator(pos_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 5.0

    def test_eval_function_sin(self):
        """Test evaluating sin function."""
        evaluator = ExpressionEvaluator({})
        sin_node = FunctionNode("sin", ConstNode(0.0))

        eval_func = evaluator.create_evaluator(sin_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 0.0

    def test_eval_function_cos(self):
        """Test evaluating cos function."""
        evaluator = ExpressionEvaluator({})
        cos_node = FunctionNode("cos", ConstNode(0.0))

        eval_func = evaluator.create_evaluator(cos_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 1.0

    def test_eval_function_exp(self):
        """Test evaluating exp function."""
        evaluator = ExpressionEvaluator({})
        exp_node = FunctionNode("exp", ConstNode(0.0))

        eval_func = evaluator.create_evaluator(exp_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 1.0

    def test_eval_function_log(self):
        """Test evaluating log function."""
        evaluator = ExpressionEvaluator({})
        log_node = FunctionNode("log", ConstNode(1.0))

        eval_func = evaluator.create_evaluator(log_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 0.0

    def test_eval_function_sqrt(self):
        """Test evaluating sqrt function."""
        evaluator = ExpressionEvaluator({})
        sqrt_node = FunctionNode("sqrt", ConstNode(4.0))

        eval_func = evaluator.create_evaluator(sqrt_node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 2.0

    def test_eval_function_unsupported(self):
        """Test evaluating unsupported function."""
        evaluator = ExpressionEvaluator({})
        unsup_node = FunctionNode("arctan", ConstNode(1.0))

        eval_func = evaluator.create_evaluator(unsup_node)

        with pytest.raises(ValueError, match="Unsupported function"):
            eval_func(0.0, np.array([0.0]))

    def test_eval_function_nan_result(self):
        """Test function producing NaN."""
        evaluator = ExpressionEvaluator({})
        # log of negative number produces NaN
        log_node = FunctionNode("log", ConstNode(-1.0))

        eval_func = evaluator.create_evaluator(log_node)

        with pytest.raises(ValueError, match="NaN or Inf"):
            eval_func(0.0, np.array([0.0]))

    def test_eval_chebfun_in_cache(self):
        """Test evaluating chebfun with cache."""
        coef = chebfun(lambda t: t**2, [0, 1])
        t_grid = np.linspace(0, 1, 100)
        values = coef(t_grid)

        cache = {id(coef): (t_grid, values)}
        evaluator = ExpressionEvaluator(cache)

        # Create ConstNode with chebfun
        const = ConstNode(coef)
        const.value = coef

        eval_func = evaluator.create_evaluator(const)
        result = eval_func(0.5, np.array([0.0]))

        assert abs(result - 0.25) < 0.01

    def test_eval_chebfun_not_in_cache(self):
        """Test evaluating chebfun not in cache."""
        coef = chebfun(lambda t: t**2, [0, 1])

        evaluator = ExpressionEvaluator({})

        # Create ConstNode with chebfun
        const = ConstNode(coef)
        const.value = coef

        eval_func = evaluator.create_evaluator(const)
        result = eval_func(0.5, np.array([0.0]))

        assert abs(result - 0.25) < 0.01

    def test_eval_node_with_expr_attr(self):
        """Test evaluating generic node with expr attribute."""
        evaluator = ExpressionEvaluator({})

        # Create a node that has expr attribute
        class GenericNode:
            def __init__(self, expr):
                self.expr = expr

        node = GenericNode(ConstNode(42.0))

        eval_func = evaluator.create_evaluator(node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 42.0

    def test_eval_unknown_node(self):
        """Test evaluating unknown node type returns 0."""
        evaluator = ExpressionEvaluator({})

        class UnknownNode:
            pass

        node = UnknownNode()

        eval_func = evaluator.create_evaluator(node)
        result = eval_func(0.0, np.array([0.0]))

        assert result == 0.0


class TestOperatorCompiler:
    """Test OperatorCompiler class."""

    def test_compile_simple_operator(self):
        """Test compiling simple harmonic oscillator."""
        compiler = OperatorCompiler()
        domain = Domain([0, 2 * np.pi])

        # u'' + u = 0
        def op(y):
            return y.diff(2) + y

        compiled_fn = compiler.compile_ivp_operator(op, domain, max_order=2)

        # Test: with u=[1, 0] (u=1, u'=0), we get u'=0, u''=-1
        u = np.array([1.0, 0.0])
        result = compiled_fn(0.0, u)

        assert result[0] == 0.0  # u' = u[1] = 0
        # u'' = (0 - u) / 1 = -1
        assert abs(result[1] + 1.0) < 1e-10

    def test_compile_with_tracer_root(self):
        """Test compiling operator with _root attribute."""
        compiler = OperatorCompiler()
        domain = Domain([0, 1])

        # u'' = 0
        def op(y):
            return y.diff(2)

        compiled_fn = compiler.compile_ivp_operator(op, domain, max_order=2)

        u = np.array([1.0, 2.0])  # u=1, u'=2
        result = compiled_fn(0.0, u)

        assert result[0] == 2.0  # u' = u[1]
        assert result[1] == 0.0  # u'' = 0/1 = 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
