"""Generic test functions for root-finding operations.

This module contains test functions for root-finding operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech2). These tests
focus on operations with empty function objects.
"""


def test_empty(emptyfun):
    """Test the roots method on an empty Bndfun."""
    assert emptyfun.roots().size == 0
