from abc import ABC, abstractmethod


class Onefun(ABC):
    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    @abstractmethod
    def initconst(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initempty(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initidentity(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun_adaptive(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun_fixedlen(cls): # pragma: no cover
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initvalues(cls): # pragma: no cover
        raise NotImplementedError

    # -------------------
    #  "private" methods
    # -------------------
    @abstractmethod
    def __call__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __init__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __repr__(self): # pragma: no cover
        raise NotImplementedError

    # ----------------
    #    algebra
    # ----------------
    @abstractmethod
    def __add__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __mul__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __neg__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __pos__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __pow__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __radd__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def __sub__(self): # pragma: no cover
        raise NotImplementedError

    # ---------------
    #   properties
    # ---------------
    @property
    @abstractmethod
    def coeffs(self): # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def isconst(self): # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def isempty(self): # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self): # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def vscale(self): # pragma: no cover
        raise NotImplementedError

    # ---------------
    #   utilities
    # ---------------
    @abstractmethod
    def copy(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def imag(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def prolong(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def real(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def simplify(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def values(self): # pragma: no cover
        raise NotImplementedError

    # --------------
    #  rootfinding
    # --------------
    @abstractmethod
    def roots(self): # pragma: no cover
        raise NotImplementedError

    # -------------
    #   calculus
    # -------------
    @abstractmethod
    def sum(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def cumsum(self): # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def diff(self): # pragma: no cover
        raise NotImplementedError
