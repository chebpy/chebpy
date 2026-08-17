"""Unit tests for chebpy._construction."""

import numpy as np
import pytest

from chebpy._construction import generate_funs
from chebpy.bndfun import Bndfun
from chebpy.compactfun import CompactFun
from chebpy.exceptions import InvalidDomain


def test_generate_funs_infinite_without_compact_constructor():
    """An unbounded piece with no matching CompactFun constructor is rejected."""

    # A constructor whose name is not a CompactFun classmethod yields no compact
    # fallback, so an infinite endpoint has nothing to build the piece with.
    def not_a_real_constructor(**_kwds):
        return None  # pragma: no cover - never reached; the guard fires first

    with pytest.raises(InvalidDomain):
        generate_funs([-np.inf, np.inf], not_a_real_constructor)


def test_generate_funs_finite_pieces_use_the_bndfun_constructor():
    """Every piece of a finite domain is built by the supplied constructor."""
    funs = generate_funs([-1.0, 0.0, 1.0], Bndfun.initconst, {"c": 2.0})
    assert len(funs) == 2
    assert all(isinstance(fun, Bndfun) for fun in funs)


def test_generate_funs_unbounded_piece_dispatches_to_compactfun():
    """An infinite endpoint is built by the matching CompactFun classmethod."""
    funs = generate_funs([-np.inf, 0.0, 1.0], Bndfun.initconst, {"c": 2.0})
    assert len(funs) == 2
    assert isinstance(funs[0], CompactFun)
    assert isinstance(funs[1], Bndfun)


def test_generate_funs_none_domain_falls_back_to_preferences():
    """Passing None uses the default domain from preferences."""
    funs = generate_funs(None, Bndfun.initconst, {"c": 1.0})
    assert len(funs) == 1
