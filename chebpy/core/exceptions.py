from abc import ABC, abstractmethod


class ChebpyBaseException(Exception, ABC):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = self.default_message

    def __str__(self):
        return self.message

    @property
    @abstractmethod
    def default_message(self):
        raise NotImplementedError


# ===============================================
#    chebpy.core.utilities.Interval exceptions
# ===============================================


IntervalOverlap = type(
    "IntervalOverlap",
    (ChebpyBaseException,),
    {"default_message": "The supplied Interval objects overlap"},
)

IntervalGap = type(
    "IntervalGap",
    (ChebpyBaseException,),
    {
        "default_message": "The supplied Interval objects do not form a complete "
        "partition of the approximation interval"
    },
)


IntervalMismatch = type(
    "IntervalMismatch",
    (ChebpyBaseException,),
    {
        "default_message": "This operation can only be performed for Fun objects "
        "defined on identical intervals"
    },
)


NotSubinterval = type(
    "NotSubinterval",
    (ChebpyBaseException,),
    {"default_message": "Not a subinterval"},
)


IntervalValues = type(
    "IntervalValues",
    (ChebpyBaseException,),
    {
        "default_message": "The defining values of a Interval object must be "
        "strictly increasing"
    },
)


# ===============================================
#    chebpy.core.utilities.Domain exceptions
# ===============================================


InvalidDomain = type(
    "InvalidDomain",
    (ChebpyBaseException,),
    {
        "default_message": "Domain objects must be initialised from an iterable "
        "collection of at least two monotonically increasing "
        "scalars"
    },
)


NotSubdomain = type(
    "NotSubdomain",
    (ChebpyBaseException,),
    {
        "default_message": "The support of the target Domain object is required "
        "to define a subinterval of the support of the "
        "original"
    },
)


SupportMismatch = type(
    "SupportMismatch",
    (ChebpyBaseException,),
    {
        "default_message": "Both objects are required to be supported on the "
        "same interval"
    },
)


BadFunLengthArgument = type(
    "BadFunLengthArgument",
    (ChebpyBaseException,),
    {
        "default_message": "The 'n' argument must be either a single numeric "
        "value, or iterable thereof posessing one fewer "
        "elements than the size of the domain"
    },
)
