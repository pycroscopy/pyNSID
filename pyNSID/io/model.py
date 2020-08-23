# -*- coding: utf-8 -*-
"""
Utilities for reading and writingNUSID datasets that are highly model-dependent
Depends heavily on sidpy

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somas
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import numpy as np

from sidpy.base.num_utils import contains_integers

if sys.version_info.major == 3:
    unicode = str


