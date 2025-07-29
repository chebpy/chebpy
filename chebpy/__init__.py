import importlib.metadata

from .api import chebfun, pwc
from .core.settings import ChebPreferences as UserPreferences

__version__ = importlib.metadata.version("chebfun")