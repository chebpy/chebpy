# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:21:22 2016

@author: mark
"""

from numpy import array
from numpy import finfo

class DefaultPrefs():
    eps  = finfo(float).eps
    tech = "ChebTech2"
    domain = array([-1., 1.])