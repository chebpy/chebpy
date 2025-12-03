import numpy as np
from scipy import sparse

from .spectral import (
    cheb_points_scaled,
    diff_matrix,
    cumsum_matrix,
    mult_matrix,
    identity_matrix,
    zero_matrix
)


class Linop:
    """1D linear operator:
    
    Supports:
        - Zero / Identity operators
        - differential operators
        - multiplication operators
        - integration operators
        - linear combinations of Linops
    """

    def __init__(self, order=0, coeffs=None, domain=(-1.0, 1.0), is_zero=False, is_integral=False, children=None, weights=None):
        self.order       = int(order)
        self.coeffs      = coeffs
        self.domain      = tuple(domain)
        self.is_integral = bool(is_integral)
        self.is_zero     = bool(is_zero)

        # For combination of primitive operators
        if children is not None:
            if weights is None:
                weights = [1.0]*len(children)
            self.children = list(children)
            self.weights  = np.asarray(weights, float)
        else:
            self.children = None
            self.weights  = None

    def _discretize_primitive(self, n, bc):
        dom = self.domain

        x = cheb_points_scaled(n, dom)

        if self.is_integral:
            A = cumsum_matrix(n, dom).toarray()
        else:
            # Build Σ diag(a_k(x)) * D^k
            size = n + 1
            A = np.zeros((size, size))

            for k in range(self.order + 1):
                if k == 0:
                    Dk = identity_matrix(n).toarray()
                else:
                    Dk = diff_matrix(n, dom, order=k).toarray()

                a_k = self.coeffs[k](x)
                A += np.diag(a_k) @ Dk

        # Boundary conditions
        if bc == "dirichlet" and not self.is_zero:
            A[0, :] = 0.0
            A[0, 0] = 1.0
            A[-1, :] = 0.0
            A[-1, -1] = 1.0

        return x, A

    # divides Linop into primitive ops and call _discretize_primitive(), then combine the results
    # TODO: add other border conditions
    def discretize(self, n, bc=None):

        if self.children is not None:
            x = None
            A = None
            for w, Lchild in zip(self.weights, self.children):
                x_child, A_child = Lchild.discretize(n, bc=None)
                if x is None:
                    x = x_child
                    A = w * A_child
                else:
                    if not np.allclose(x, x_child):
                        raise ValueError("Children Linops use mismatched grids.")
                    A += w * A_child

            if bc == "dirichlet" and not self.is_zero:
                A[0, :] = 0.0
                A[0, 0] = 1.0
                A[-1, :] = 0.0
                A[-1, -1] = 1.0

            return x, A

        # self is a primitive operator
        return self._discretize_primitive(n, bc)

    # Linear combination of linear operators
    @classmethod
    def combination(cls, ops, weights=None):
        if weights is None:
            weights = [1.0]*len(ops)
        dom = ops[0].domain
        for op in ops[1:]:
            if op.domain != dom:
                raise ValueError("All operators must share the same domain.")

        return cls(order=0, domain=dom, children=ops, weights=weights)

    def __add__(self, other):
        if not isinstance(other, Linop):
            return NotImplemented
        if self.is_zero:
            return other
        if other.is_zero:
            return self
        return Linop.combination([self, other], [1.0, 1.0])

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Linop):
            return NotImplemented
        if self.is_zero:
            return -other
        if other.is_zero:
            return self
        return Linop.combination([self, other], [1.0, -1.0])

    def __rsub__(self, other):
        if other == 0:
            return -self
        return NotImplemented

    def __neg__(self):
        return (-1.0) * self

    def __mul__(self, scalar):
        if self.is_zero:
            return self
        if isinstance(scalar, (int, float, complex, np.number)):
            return Linop.combination([self], [scalar])
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    # Primitive operators
    @classmethod
    def zero(cls, domain=(-1,1)):
        obj = cls(order=0, coeffs=[lambda x: 0*x], domain=domain, is_zero=True)
        return obj

    @classmethod
    def identity(cls, domain=(-1,1)):
        return cls(order=0, coeffs=[lambda x: np.ones_like(x)], domain=domain)

    @classmethod
    def diff(cls, k=1, domain=(-1,1)):
        coeffs = [lambda x: 0*x] * k + [lambda x: np.ones_like(x)]
        return cls(order=k, coeffs=coeffs, domain=domain)

    @classmethod
    def mul(cls, g, domain=(-1,1)):
        if np.isscalar(g):
            f = lambda x: g * np.ones_like(x)
        else:
            f = lambda x: g(x)
        return cls(order=0, coeffs=[f], domain=domain)

    @classmethod
    def integrate(cls, domain=(-1,1)):
        return cls(order=0, domain=domain, is_integral=True)

