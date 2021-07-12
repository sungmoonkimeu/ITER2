# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue JUL 1 15:00:00 2021
@author: prasadarajudandu - modified by SMK
(Circular vessel shape)
Spun fibre model with laming matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan
from numpy.linalg import norm, eig
import concurrent.futures as cf

from py_pol.jones_matrix import create_Jones_matrices
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)

start = pd.Timestamp.now()

cstm_color = ['c', 'm', 'y', 'k', 'r']


# -------------- Using py_pol module -----------------------------------
E = Jones_vector('Input pol')
S = create_Stokes('Output_S')
J = create_Jones_matrices('J')

V_azi = np.hstack((pi/100, arange(0, pi/4, pi/16), pi/4))
V_mod_ret = arange(0, -pi/2, -pi/100)

E.linear_light(azimuth=V_azi)
J.retarder_linear(R=V_mod_ret)
for nn in range(len(V_azi)):
    V_out = J*E[nn]
    S.from_Jones(V_out)
    if nn == 0:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33*pi/180, 53*pi/180], kind='line', color_line='b')
    else:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')

V_ell = np.hstack((arange(0, -pi/4, -pi/16), -pi/4+pi/100))
# ellipticity 0 ~ pi/4 -->
V_mod_rot = arange(0, pi/8, pi/100)

E.general_azimuth_ellipticity(azimuth=0, ellipticity=V_ell)
J.half_waveplate(azimuth=V_mod_rot, length=len(V_mod_rot))
for nn in range(len(V_ell)):
    V_out = J*E[nn]
    S.from_Jones(V_out)
    if nn == 0:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33*pi/180, 53*pi/180], kind='line', color_line='b')
    else:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')

plt.show()
