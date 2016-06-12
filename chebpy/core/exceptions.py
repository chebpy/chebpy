# -*- coding: utf-8 -*-

class IntervalOverlap(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The supplied Interval objects overlap"
        super(self.__class__, self).__init__(message)

class IntervalGap(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The supplied Interval objects do not form a "\
            "complete partition of the approximation interval"
        super(self.__class__, self).__init__(message)

class IntervalMismatch(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "This operation can only be performed for Fun "\
            "objects defined on identical intervals"
        super(self.__class__, self).__init__(message)

class NotSubinterval(Exception):
    pass

class IntervalValues(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The defining values of a Interval object must "\
            "be strictly increasing"
        super(self.__class__, self).__init__(message)

# chebpy.core.utilities.Domain
class InvalidDomain(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Domain objects must be initialised from an iterable "\
            "collection of at least two monotonically increasing scalars"
        super(self.__class__, self).__init__(message)

class NotSubdomain(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The support of the target Domain object is required to "\
            "define a subinterval of the support of the original"
        super(self.__class__, self).__init__(message)

# chebpy.core.utilities.Domain
class SupportMismatch(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Both objects are required to be supported "\
            "on the same interval"
        super(self.__class__, self).__init__(message)

class BadDomainArgument(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The \'domain\' argument must be an iterable "\
                "containing two or more elements"
        super(self.__class__, self).__init__(message)

class BadFunLengthArgument(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The \'n\' argument must be either a single "\
                "numeric value, or an iterable thereof containing one "\
                "fewer elements than the size of the domain"
        super(self.__class__, self).__init__(message)

class DomainBreakpoints(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "This operation requires the breakpoints to intersect"
        super(self.__class__, self).__init__(message)
