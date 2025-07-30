"""Utility functions for testing chebfun."""
def test_cumsum_empty(emptyfun):
    """Test the cumsum method on an empty function object.

    This test verifies that the cumsum method on an empty function object
    returns an empty function object. This is a generic test that works with
    any type of empty function object (Bndfun, Chebfun, or Chebtech2).

    Args:
        emptyfun: Fixture providing an empty function object
    """
    assert emptyfun.cumsum().isempty


def test_sum_empty(emptyfun):
    """Test the sum method on an empty function object.

    This test verifies that the sum method on an empty function object
    returns 0. This is a generic test that works with any type of empty
    function object (Bndfun, Chebfun, or Chebtech2).

    Args:
        emptyfun: Fixture providing an empty function object
    """
    assert emptyfun.sum() == 0


def test_diff_empty(emptyfun):
    """Test the diff method on an empty function object.

    This test verifies that the diff method on an empty function object
    returns an empty function object. This is a generic test that works with
    any type of empty function object (Bndfun, Chebfun, or Chebtech2).

    Args:
        emptyfun: Fixture providing an empty function object
    """
    assert emptyfun.diff().isempty
