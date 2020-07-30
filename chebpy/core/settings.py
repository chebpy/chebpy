# -*- coding: utf-8 -*-

import numpy as np


def default_names(cls):
    return [n for n in cls.__dict__.keys() if n[0] != "_"]

def default_values(cls):
    return [cls.__dict__[k] for k in default_names(cls)]

def default_prefs(cls):
    prefs = dict()
    for key, val in zip(default_names(cls), default_values(cls)):
        prefs[key] = val
    return prefs

class DefaultPrefs():
    eps = np.finfo(float).eps
    tech = "Chebtech2"
    domain = np.array([-1., 1.])
    N_plot = 2001

class UserPrefs():
    def __init__(self):
        for name, val in default_prefs(DefaultPrefs).items():
            setattr(self, name, val)
    def reset(self, *names):
        # only reset DefaultPrefs, in case user set their own attributes
        names = default_names(DefaultPrefs) if len(names) == 0 else names
        for name in names:
            self.__setattr__(name, DefaultPrefs.__dict__[name])

defaultPrefs = DefaultPrefs()
userPrefs = UserPrefs()