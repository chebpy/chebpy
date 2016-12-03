# -*- coding: utf-8 -*-

import numpy as np

class DefaultPrefs():
    eps  = np.finfo(float).eps
    tech = "Chebtech2"
    domain = np.array([-1., 1.])