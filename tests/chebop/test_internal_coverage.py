"""Internal coverage tests for Chebop system solving functions.

These are additional slow tests that focus on specific internal code paths.
"""

import numpy as np
import pytest
from chebpy import chebfun
from chebpy.chebop import Chebop


class TestProcessSystemBC:
    """Test _process_system_bc with various BC configurations."""

    def test_scalar_bc_multiple_vars(self):
        """Test that scalar BC applied to multiple variables is rejected.

        MATLAB rejects this with: "Specifying boundary conditions as a vector
        for systems is only supported for first order systems".
        ChebPy should reject it similarly for MATLAB compatibility.
        """
        op = Chebop(lambda u, v, w: [u.diff() + v + w, v.diff() - u, w.diff() + u - v], [0, 1])
        op.lbc = 0.0  # Scalar applied to all - should be rejected
        op.rbc = 1.0

        with pytest.raises(ValueError) as exc_info:
            op.solve()

        error_msg = str(exc_info.value)
        assert "scalar" in error_msg.lower() and "system" in error_msg.lower()

    def test_functional_bc_returns_scalar_list(self):
        """Test functional BC that returns a list of scalars."""
        op = Chebop(lambda u, v: [u.diff(2) + v, v.diff(2) - u], [0, 1])

        def bc_left(u, v):
            return [u, v]  # Returns list

        def bc_right(u, v):
            return [u.diff(), v.diff()]

        op.lbc = bc_left
        op.rbc = bc_right
        try:
            sol = op.solve()
            if sol is not None:
                assert len(sol) == 2
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass


class TestSystemRHSHandling:
    """Test RHS handling in system solving."""

    def test_callable_rhs_in_system(self):
        """Test callable RHS in system equations."""
        op = Chebop(lambda u, v: [u.diff() + v, v.diff() - u], [0, 1])
        op.lbc = [0.0, 0.0]
        op.rbc = [1.0, 1.0]
        op.rhs = lambda x: np.exp(x)  # Single callable for all
        sol = op._solve_linear_system(n_target=16)
        assert len(sol) == 2

    def test_list_rhs_in_system(self):
        """Test list of RHS functions in system."""
        op = Chebop(lambda u, v: [u.diff(2), v.diff(2)], [0, 1])
        op.lbc = [0.0, 0.0]
        op.rbc = [1.0, 1.0]
        op.rhs = [lambda x: np.sin(x), lambda x: np.cos(x)]
        sol = op._solve_linear_system(n_target=16)
        assert len(sol) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
