
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon


"""
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

# _______________________________Parameters#1___________________________________#

# Circulator input matrix

theta = 30 * pi / 180   # random axis of LB
phi = 30* pi / 180  # ellipticity angle change from experiment
theta_e = 0 * pi / 180  # azimuth angle change from experiment

M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_ci = M_rot @ M_theta @ M_phi @ M_theta_T

# Circulator output matrix

theta = 0 * pi / 2  # random axis of LB
phi = 0 * pi / 180  # ellipticity angle change from experiment
theta_e = 0 * pi / 180  # azimuth angle change from experiment

# Mci
M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_co = M_rot @ M_theta @ M_phi @ M_theta_T

# input matrix
V_I = arange(0e6, 300e3 + 1e3, 1e3)

V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
V = 0.54 * 4 * pi * 1e-7

E = Jones_vector('Output')
V_in = np.array([[[1], [0]], [[np.sqrt(0.5)], [np.sqrt(0.5)]], [[np.sqrt(0.5)], [np.sqrt(0.5)*1j]]])
for nn in range(len(V_in)):
    for mm, iter_I in enumerate(V_I):
        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])

        V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]

    E.from_matrix(M=V_out)
    S = create_Stokes('Output_S')
    S.from_Jones(E)
    if nn != 0:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')
    else:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[31 * pi / 180, 164 * pi / 180], kind='line',
                                     color_line='b')
print(V_out)

plt.show()