"""Unit-tests for Chebtech roots functionality.

This module contains tests for finding the roots of Chebtech objects,
including empty, constant, and various polynomial and trigonometric functions.
"""

import numpy as np
import pytest

from chebpy.chebtech import Chebtech

from ..generic.roots import rootstestfuns, test_empty  # noqa: F401
from ..utilities import eps


@pytest.mark.parametrize("f, roots", rootstestfuns)
def test_roots(f, roots, tol=eps):
    """Test that the roots of a function are correctly identified.

    This test verifies that the roots() method of a Chebtech object
    correctly identifies the roots of various functions within the
    specified tolerance.

    Args:
        f: Function to find roots of
        roots: Expected roots
        tol: Tolerance for comparison
    """
    ff = Chebtech.initfun_adaptive(f)
    rts = ff.roots()
    assert np.max(np.abs(rts - roots)) <= tol
