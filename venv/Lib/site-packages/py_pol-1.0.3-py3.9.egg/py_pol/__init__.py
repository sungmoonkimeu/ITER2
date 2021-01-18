# -*- coding: utf-8 -*-
"""Top-level package for Python polarization."""

__author__ = """Jesus del Hoyo Munoz / Luis Miguel Sanchez Brea"""
__email__ = 'jhoyo@ucm.es'
__version__ = '1.0.3'

import numpy as np
import scipy as sp

name = 'py_pol'
um = 1.
mm = 1000 * um
nm = um / 1000.
degrees = np.pi / 180
eta = 376.73

verbose = True

# Angle limit variables
limAlpha = [0, np.pi / 2]
limDelta = [0, 2 * np.pi]
limAz = [0, np.pi]
limEl = [-np.pi / 4, np.pi / 4]
limRet = [0, np.pi]
figsize_default = [5, 5]

eps = 1e-6
num_decimals = 4

number_types = (int, float, complex, np.int32, np.float64)
