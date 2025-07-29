"""Tests for the version."""

import chebpy


def test_version():
    """Test if the version of the chebpy library is defined.

    This function checks that the version constant in the chebpy module is
    not None, ensuring that the version information is properly assigned.

    Raises:
        AssertionError: If the version constant is None.

    """
    assert chebpy.__version__ is not None
