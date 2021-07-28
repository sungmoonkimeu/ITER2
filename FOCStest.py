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

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format



start = pd.Timestamp.now()

# _______________________________Parameters___________________________________#
# r = 1
L = 0.5  # sensing fiber
# L_lf = [0.153, 0.156, 0.159, 0.162, 0.165]      # lead fiber
# L_lf = L_lf + ones(len(L_lf))*100
L_lf = [1]

LB = 0.132
SP = 0.03
# dz = SP / 1000
dz = 0.00001
q = 0
I = 10e5
V = 0.54 * 4 * pi * 1e-7
# H = I / (2 * pi * r)
# H = I/L
STR = (2 * pi) / SP
A_P = 0
V_in = np.array([[cos(A_P)], [sin(A_P)]])
M_P = mat([[(cos(A_P)) ** 2, (sin(A_P) * cos(A_P))], [(sin(A_P) * cos(A_P)), (sin(A_P)) ** 2]])

cstm_color = ['c', 'm', 'y', 'k', 'r']


def eigen_expm(A):
    """

    Parameters
    ----------
    A : 2 x 2 diagonalizable matrix
        DESCRIPTION.

    Returns
    -------
    expm(A): exponential of the matrix A.

    """
    vals, vects = eig(A)
    return einsum('...ik, ...k, ...kj -> ...ij',
                  vects, np.exp(vals), np.linalg.inv(vects))


n = int(L / dz)
delta = (2 * pi) / LB  # Intrinsic linear birefringence

V_plasmaCurrent = arange(1e5, 10e6, 2e5)
#V_plasmaCurrent = np.append(V_plasmaCurrent, arange(1e6, 18e6, 5e5))

V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

# Requirement specificaion for ITER
absErrorlimit = zeros(len(V_out))
relErrorlimit = zeros(len(V_out))

# Calcuation ITER specification
for nn in range(len(V_plasmaCurrent)):
    if V_plasmaCurrent[nn] < 1e6:
        absErrorlimit[nn] = 10e3
    else:
        absErrorlimit[nn] = V_plasmaCurrent[nn] * 0.01

    if V_plasmaCurrent[nn] == 0:
        relErrorlimit[nn] = 100
    else:
        relErrorlimit[nn] = absErrorlimit[nn] / V_plasmaCurrent[nn]

mm = 0
for iter_I in V_plasmaCurrent:
    H = iter_I / L
    # print(H)
    rho = V * H  # Faraday effect induced birefringence

    V_L = arange(0, L + dz, dz)
    V_theta = V_L * STR
    ang = 0
    MF = np.array([[cos(rho*L), sin(rho*L)], [sin(rho*L), cos(rho*L)]])
    MPDL = np.array([[1, 0], [0, sqrt(10**(-0.2/10))]])
    M45 = np.array([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])
    M45P = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
    V_out[mm] = M45 @ MPDL @ M45P @ MF @ V_in
    #V_out[mm] = MPDL @ MF @ V_in
    mm = mm + 1

# -------------- Using py_pol module -----------------------------------
E = Jones_vector('Output_J')
S = create_Stokes('Output_S')

E.linear_light(azimuth=0 * abs(V_out))
S.linear_light(azimuth=0 * abs(V_out))

# SOP evolution in Lead fiber (Forward)
E.from_matrix(V_out)
S.from_Jones(E)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line', color_line='b')

abs_error = zeros([len(V_out)])
rel_error = zeros([len(V_out)])
Ip = zeros(len(V_out))
V_ang = zeros(len(V_out))

m = 0
for nn in range(len(V_out)):
    if nn > 2 and E[nn].parameters.azimuth() + m * pi - V_ang[nn - 1] < -pi * 0.5:
        m = m + 1
    elif nn > 2 and E[nn].parameters.azimuth() + m * pi - V_ang[nn - 1] > pi * 0.5:
        m = m - 1
    V_ang[nn] = E[nn].parameters.azimuth() + m * pi
    Ip[nn] = (V_ang[nn]) / V
    abs_error[nn] = abs(Ip[nn] - V_plasmaCurrent[nn])
    if V_plasmaCurrent[nn] == 0:
        rel_error[nn] = 100
    else:
        rel_error[nn] = abs_error[nn] / V_plasmaCurrent[nn]

# Ploting graph
fig, ax = plt.subplots(figsize=(6, 3))

ax.plot(V_plasmaCurrent, rel_error, lw='1')
ax.plot(V_plasmaCurrent, relErrorlimit, 'r', label='ITER specification', lw='1')
ax.legend(loc="upper right")

plt.rc('text', usetex=True)
ax.set_xlabel(r'Plasma current $I_{p}(A)$')
ax.set_ylabel(r'Relative error on $I_{P}$')

# plt.title('Output power vs Plasma current')
ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_locator(MaxNLocator(10))

ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
ax.grid(ls='--', lw=0.5)

#fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom= 0.2)
#fig.set_size_inches(6,4)
plt.rc('text', usetex=False)

plt.show()
