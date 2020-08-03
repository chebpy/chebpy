# -*- coding: utf-8 -*-

import numpy as np


class DefaultPrefs():
    pass


class UserPrefs():
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.tech = "Chebtech2"
        self.domain = np.array([-1., 1.])
        self.N_plot = 2001
    def reset(self, *names):
        """Reset default preferences.
        `.reset()` resets all preferences to the DefaultPrefs state
        `.reset(*names)` resets only the selected ones.
        This leaves additional user-added prefs untouched."""
        if len(names) == 0:
            names = [k for k in DefaultPrefs.__dict__.keys() if k[0] != "_"]
        for name in names:
            self.__setattr__(name, DefaultPrefs.__dict__[name])

userPrefs = UserPrefs()

for name, val in userPrefs.__dict__.items():
    if name[0] != "_":
        setattr(DefaultPrefs, name, val)

defaultPrefs = DefaultPrefs()
