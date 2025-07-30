"""Generic test functions for algebraic operations.

This module contains test functions for algebraic operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech2). These tests
focus on operations with empty function objects.
"""


def test__pos__empty(emptyfun):
    """Test unary positive operator on an empty fun."""
    assert (+emptyfun).isempty


def test__neg__empty(emptyfun):
    """Test unary negative operator on an empty fun."""
    assert (-emptyfun).isempty


def test_pow_empty(emptyfun):
    """Test power operation with an empty fun."""
    for c in range(10):
        assert (emptyfun**c).isempty


def test_rpow_empty(emptyfun):
    """Test raising a constant to an empty fun power."""
    for c in range(10):
        assert (c**emptyfun).isempty
