from abc import ABC, abstractmethod, abstractclassmethod


class Onefun(ABC):

    # --------------------------
    #  alternative constructors
    # --------------------------
    @abstractclassmethod
    def initconst(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initempty(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initidentity(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initfun(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initfun_adaptive(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initfun_fixedlen(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initvalues(cls):
        raise NotImplementedError

    # -------------------
    #  "private" methods
    # -------------------
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    # ----------------
    #    algebra
    # ----------------
    @abstractmethod
    def __add__(self):
        raise NotImplementedError

    @abstractmethod
    def __mul__(self):
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError

    @abstractmethod
    def __pos__(self):
        raise NotImplementedError

    @abstractmethod
    def __pow__(self):
        raise NotImplementedError

    @abstractmethod
    def __radd__(self):
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self):
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self):
        raise NotImplementedError

    @abstractmethod
    def __sub__(self):
        raise NotImplementedError

    # ---------------
    #   properties
    # ---------------
    @property
    @abstractmethod
    def coeffs(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def isconst(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def isempty(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def vscale(self):
        raise NotImplementedError

    # ---------------
    #   utilities
    # ---------------
    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def imag(self):
        raise NotImplementedError

    @abstractmethod
    def prolong(self):
        raise NotImplementedError

    @abstractmethod
    def real(self):
        raise NotImplementedError

    @abstractmethod
    def simplify(self):
        raise NotImplementedError

    @abstractmethod
    def values(self):
        raise NotImplementedError

    # --------------
    #  rootfinding
    # --------------
    @abstractmethod
    def roots(self):
        raise NotImplementedError

    # -------------
    #   calculus
    # -------------
    @abstractmethod
    def sum(self):
        raise NotImplementedError

    @abstractmethod
    def cumsum(self):
        raise NotImplementedError

    @abstractmethod
    def diff(self):
        raise NotImplementedError
