"""Generic test functions for class usage.

This module contains test functions for class usage that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech2). These tests
focus on common operations and properties of function objects.
"""


def test_constfun_value(constfun):
    """Test that constfun has the correct value.

    This test verifies that the constant function object created by the constfun
    fixture has the expected constant value of 1.0.

    Args:
        constfun: Fixture providing a constant function object.
    """
    import numpy as np
    # Test at multiple points to ensure it's truly constant
    xx = np.linspace(-1, 1, 10)
    values = constfun(xx)
    # Check that all values are 1.0
    assert np.all(np.abs(values - 1.0) < 1e-14)


def test_restrict__empty(emptyfun) -> None:
    """Test the restrict_ method on an empty fun.

    This test verifies that the restrict_ method on an empty Chebfun
    returns an empty Chebfun.

    Args:
        emptyfun: Fixture providing an empty function object.
    """
    emptyfun.restrict_([-0.5, 0.5])
    assert emptyfun.isempty



def test_translate_empty(emptyfun) -> None:
    """Test the translate method on an empty Chebfun.

    This test verifies that the translate method on an empty Chebfun
    returns an empty Chebfun.

    Args:
        emptyfun: Fixture providing an empty function object.
    """
    g0 = emptyfun.translate(1)
    assert g0.isempty
