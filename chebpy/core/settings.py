import numpy as np


class DefaultPreferences:
    """Default preferences for chebpy."""

    eps = np.finfo(float).eps
    tech = "Chebtech2"
    domain = np.array([-1.0, 1.0])  # TODO: should this be .utilities.Domain?
    N_plot = 2001
    maxpow2 = 16
    maxiter = 10
    sortroots = False
    mergeroots = True

    @classmethod
    def _defaults(cls):
        """Returns all defined class attributes."""
        return {k: v for k, v in cls.__dict__.items() if k[0] != "_"}


class ChebPreferences(DefaultPreferences):
    """Preferences object used in chebpy."""

    def reset(self, *names):
        """Reset default preferences.
        `.reset()` resets all preferences to the DefaultPrefs state
        `.reset(*names)` resets only the selected ones.
        This leaves additional user-added prefs untouched."""
        if len(names) == 0:
            names = DefaultPreferences._defaults()
        for name in names:
            if hasattr(DefaultPreferences, name):
                setattr(self, name, getattr(DefaultPreferences, name))

    # Singleton
    _instance = None  # persistent reference for the singleton object

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DefaultPreferences, cls).__new__(cls)
        return cls._instance

    # Context manager
    _stash = []  # persistent stash for old prefs when entering context(s)

    def __enter__(self):
        self._stash.append(
            {k: getattr(self, k) for k in DefaultPreferences._defaults().keys()}
        )
        return self._instance

    def __exit__(self, exc_type, exc_value, traceback):
        for k, v in self._stash.pop().items():
            setattr(self, k, v)


# create the singleton object for easy import in sister modules
_preferences = ChebPreferences()
