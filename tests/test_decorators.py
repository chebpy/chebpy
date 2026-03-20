"""Tests for the decorators module.

This module tests the decorator functions in chebpy.decorators:
- cache: method output caching
- self_empty: handling of empty objects
- preandpostprocess: pre/post-processing for bary and clenshaw
- float_argument: consistent input/output types for __call__
- cast_arg_to_chebfun: type conversion for binary operations
- cast_other: generic type casting for binary operators
"""

import numpy as np
import pytest

from chebpy.decorators import (
    cache,
    cast_arg_to_chebfun,
    cast_other,
    float_argument,
    preandpostprocess,
    self_empty,
)

# ---------------------------------------------------------------------------
# cache decorator
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the cache decorator."""

    def test_cache_stores_result(self):
        """Test that a cached method stores its result in _cache."""

        class Obj:
            @cache
            def compute(self):
                return 42

        obj = Obj()
        result = obj.compute()
        assert result == 42
        assert obj._cache["compute"] == 42

    def test_cache_returns_stored_value(self):
        """Test that subsequent calls return the cached value without recomputing."""
        call_count = 0

        class Obj:
            @cache
            def compute(self):
                nonlocal call_count
                call_count += 1
                return 99

        obj = Obj()
        assert obj.compute() == 99
        assert obj.compute() == 99
        assert call_count == 1

    def test_cache_separate_instances(self):
        """Test that different instances have independent caches."""

        class Obj:
            def __init__(self, val):
                self.val = val

            @cache
            def compute(self):
                return self.val

        a = Obj(1)
        b = Obj(2)
        assert a.compute() == 1
        assert b.compute() == 2


# ---------------------------------------------------------------------------
# self_empty decorator
# ---------------------------------------------------------------------------


class TestSelfEmpty:
    """Tests for the self_empty decorator."""

    def test_self_empty_returns_resultif_when_empty(self):
        """Test that an empty object returns the resultif sentinel."""

        class Obj:
            isempty = True

            @self_empty(resultif=-1)
            def method(self):
                return 42

        assert Obj().method() == -1

    def test_self_empty_returns_copy_when_empty_and_resultif_none(self):
        """Test that an empty object returns a copy when resultif is None."""

        class Obj:
            isempty = True

            def copy(self):
                return "copy"

            @self_empty()
            def method(self):
                return 42

        assert Obj().method() == "copy"

    def test_self_empty_calls_method_when_not_empty(self):
        """Test that a non-empty object calls the wrapped method normally."""

        class Obj:
            isempty = False

            @self_empty(resultif=-1)
            def method(self):
                return 42

        assert Obj().method() == 42


# ---------------------------------------------------------------------------
# preandpostprocess decorator
# ---------------------------------------------------------------------------


class TestPreandpostprocess:
    """Tests for the preandpostprocess decorator."""

    def test_empty_first_arg(self):
        """Test that an empty first argument returns an empty array."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx * akfk  # pragma: no cover

        result = fn(np.array([]), np.array([1, 2]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_empty_second_arg(self):
        """Test that an empty second argument returns an empty array."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx * akfk  # pragma: no cover

        result = fn(np.array([1.0]), np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_constant_function_scalar_input(self):
        """Test that a single-element akfk with scalar input returns the constant."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx * akfk  # pragma: no cover

        result = fn(0.5, np.array([3.14]))
        assert result == 3.14

    def test_constant_function_array_input(self):
        """Test that a single-element akfk with array input broadcasts the constant."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx * akfk  # pragma: no cover

        result = fn(np.array([1.0, 2.0, 3.0]), np.array([7.0]))
        np.testing.assert_array_equal(result, np.array([7.0, 7.0, 7.0]))

    def test_nan_coefficients(self):
        """Test that NaN in coefficients produces NaN output."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx * akfk  # pragma: no cover

        result = fn(np.array([1.0, 2.0]), np.array([np.nan, 1.0]))
        assert np.all(np.isnan(result))

    def test_scalar_input_converted_to_array(self):
        """Test that a scalar first argument is converted to array and result unwrapped."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx + akfk[:1]

        result = fn(1.0, np.array([2.0, 3.0]))
        assert np.isscalar(result)
        assert result == pytest.approx(3.0)

    def test_array_input_returns_array(self):
        """Test that an array first argument returns an array result."""

        @preandpostprocess
        def fn(xx, akfk):
            return xx + akfk[:2]

        result = fn(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# float_argument decorator
# ---------------------------------------------------------------------------


class TestFloatArgument:
    """Tests for the float_argument decorator."""

    def test_scalar_input_returns_scalar(self):
        """Test that a scalar input returns a scalar output."""

        class Obj:
            @float_argument
            def __call__(self, x):
                return x * 2

        obj = Obj()
        result = obj(3.0)
        assert np.isscalar(result)
        assert result == pytest.approx(6.0)

    def test_array_input_returns_array(self):
        """Test that an array input returns an array output."""

        class Obj:
            @float_argument
            def __call__(self, x):
                return x * 2

        obj = Obj()
        result = obj(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_0d_array_treated_as_scalar(self):
        """Test that a 0-d array is expanded and result unwrapped like a scalar."""

        class Obj:
            @float_argument
            def __call__(self, x):
                return x + 1

        obj = Obj()
        # np.array(0.5) is a 0-d array with size 1, ndim 0
        result = obj(0.5)
        assert np.isscalar(result)


# ---------------------------------------------------------------------------
# cast_arg_to_chebfun decorator
# ---------------------------------------------------------------------------


class TestCastArgToChebfun:
    """Tests for the cast_arg_to_chebfun decorator."""

    def test_numeric_argument_is_cast(self):
        """Test that a numeric argument is converted via initconst."""

        class FakeChebfun:
            support = (-1, 1)

            @classmethod
            def initconst(cls, val, support):
                obj = cls()
                obj._val = val
                return obj

            @cast_arg_to_chebfun
            def add(self, other):
                return other

        obj = FakeChebfun()
        result = obj.add(5.0)
        assert isinstance(result, FakeChebfun)
        assert result._val == 5.0

    def test_same_type_argument_passes_through(self):
        """Test that an argument of the same class passes through unchanged."""

        class FakeChebfun:
            support = (-1, 1)

            @classmethod
            def initconst(cls, val, support):
                return cls()  # pragma: no cover

            @cast_arg_to_chebfun
            def add(self, other):
                return other

        obj = FakeChebfun()
        other = FakeChebfun()
        result = obj.add(other)
        assert result is other


# ---------------------------------------------------------------------------
# cast_other decorator
# ---------------------------------------------------------------------------


class TestCastOther:
    """Tests for the cast_other decorator."""

    def test_non_matching_type_is_cast(self):
        """Test that a non-matching argument is cast to self's class."""

        class MyClass:
            def __init__(self, val=None):
                self.val = val

            @cast_other
            def op(self, other):
                return other

        obj = MyClass(1)
        result = obj.op(42)
        assert isinstance(result, MyClass)
        assert result.val == 42

    def test_matching_type_passes_through(self):
        """Test that an argument of the same class is not re-cast."""

        class MyClass:
            def __init__(self, val=None):
                self.val = val

            @cast_other
            def op(self, other):
                return other

        obj = MyClass(1)
        other = MyClass(2)
        result = obj.op(other)
        assert result is other
