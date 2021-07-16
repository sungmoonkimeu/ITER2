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


def eigen_expm(A):
    """

    Parameters
    ----------
    A : 2 x 2 diagonalizable matrix
        DESCRIPTION.

    scify.linalg.expm() is available but only works for a (2,2) matrix.
    This function is for (2,2,n) matrix

    Returns
    -------
    expm(A): exponential of the matrix A.

    """
    vals, vects = eig(A)
    return einsum('...ik, ...k, ...kj -> ...ij',
                  vects, np.exp(vals), np.linalg.inv(vects))


def lamming(LB, SP, DIR, Ip, L, V_theta, M_err):
    """
    :param LB: beatlength
    :param SP: spin period
    :param DIR: direction (+1: forward, -1: backward)
    :param Ip: plasma current
    :param L: fiber length
    :param V_theta: vector of theta (angle of optic axes)
    :param n_div: division number
    :param vib_azi: [0, pi]: Azimuth. Default: 0.
    :param vib_ell: [-pi/4, pi/4]: Ellipticity. Default: 0.
    :return: M matrix calculated from N matrix
    """

    STR = 2 * pi / SP * DIR
    delta = 2 * pi / LB
    # magnetic field in unit length
    # H = Ip / (2 * pi * r)
    H = Ip / L
    V = 0.54 * 4 * pi * 1e-7
    rho = V * H

    ###----------------------Laming parameters---------------------------------###
    n = 0
    m = 0
    # --------Laming: orientation of the local slow axis ------------

    qu = 2 * (STR + rho) / delta
    gma = 0.5 * (delta ** 2 + 4 * ((STR + rho) ** 2)) ** 0.5
    omega = STR * dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) + n * pi

    R_z = 2 * arcsin(sin(gma * dz) / ((1 + qu ** 2) ** 0.5))

    M = np.array([[1, 0], [0, 1]])

    kk = 0  # for counting M_err
    for nn in range(len(V_theta) - 1):
        if DIR == 1:
            phi = ((STR * dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
        elif DIR == -1:
            phi = ((STR * dz) - omega) / 2 + m * (pi / 2) + V_theta[-1 - nn]

        n11 = R_z / 2 * 1j * cos(2 * phi)
        n12 = R_z / 2 * 1j * sin(2 * phi) - omega
        n21 = R_z / 2 * 1j * sin(2 * phi) + omega
        n22 = R_z / 2 * -1j * cos(2 * phi)
        N = np.array([[n11, n12], [n21, n22]])
        N_integral = eigen_expm(N)
        M = N_integral @ M

        nVerr = M_err.shape[2]
        nSet = int(len(V_theta)/(nVerr+1))
        if nVerr > 0:
            if DIR == 1 and (nn + 1) % nSet == 0:
                if kk != nVerr:
                    M = M_err[...,kk] @ M
                    kk = kk + 1
            elif DIR == -1 and int((len(V_theta)-nn) % nSet) == 0:
                if kk != nVerr:
                    M = M_err[...,-1-kk].T @ M
                    kk = kk + 1
    return M
'''
V_theta = arange(10)
nVerr = 2
rem = int(len(V_theta) % (nVerr+1))
tset = int(len(V_theta) / (nVerr+1))
for nn in range(len(V_theta)):
    DIR = 1
    #if (nn + 1 + rem*0.5*(DIR - 1)) % tset == 0:
    if (nn + 1) % tset == 0:
        V_theta[nn] = 99
    DIR = -1
for nn in range(len(V_theta)):
    #if (nn + 1 + rem*0.5*(DIR - 1)) % tset == 0:
    if int((len(V_theta)-(nn)) % tset) == 0:
        print(int((len(V_theta)-(nn)) % tset))
        V_theta[-1-nn] = V_theta[-1-nn]*100
print(V_theta)
'''
# _______________________________Parameters___________________________________#
# r = 1
L = 0.5  # sensing fiber
# L_lf = [0.153, 0.156, 0.159, 0.162, 0.165]      # lead fiber
# L_lf = L_lf + ones(len(L_lf))*100
L_lf = [1]

LB = 1.000
SP = 0.005
# dz = SP / 1000
dz = 0.0001
q = 0
V = 0.54 * 4 * pi * 1e-7
# H = I / (2 * pi * r)
# H = I/L
STR = (2 * pi) / SP  # Spin Twist rate
A_P = 0
V_in = np.array([[cos(A_P)], [sin(A_P)]])
M_P = mat([[(cos(A_P)) ** 2, (sin(A_P) * cos(A_P))], [(sin(A_P) * cos(A_P)), (sin(A_P)) ** 2]])

cstm_color = ['c', 'm', 'y', 'k', 'r']

V_arr = arange(100)
V_out = np.einsum('...i,jk->ijk', ones(len(V_arr)) * 1j, np.mat([[0], [0]]))

mm = 0
for iter_I in V_arr:
    #  Preparing M_err
    n_M_err = 1
    theta = (np.random.rand(n_M_err)-0.5)*2 * pi / 2            # random axis of LB
    phi = (np.random.rand(n_M_err)-0.5)*2 * 0.8 * pi / 180      # ellipticity angle change from experiment
    theta_e = (np.random.rand(n_M_err)-0.5)*2 * 0.8 * pi / 180  # azimuth angle change from experiment

    M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,n_M_err)

    M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,n_M_err)
    M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,n_M_err)

    IB = np.zeros((2, 2, (n_M_err)))
    np.einsum('iij->ij', IB)[:] = 1
    Bexp = np.exp(1j * np.vstack((phi, -phi)))
    M_phi = einsum('ijk, ...ik -> ijk', IB, Bexp)  # Birefringence matrix

    M_err = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)  # matrix calculation

    M_empty = np.array([]).reshape(2, 2, 0)
    V_L_lf = arange(0, L_lf[0] + dz, dz)
    V_theta_lf = V_L_lf * STR

    V_L = arange(0, L + dz, dz)
    V_theta = V_theta_lf[-1] + V_L * STR

    ksi = 45 * pi / 180
    Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
    Jm = np.array([[1, 0], [0, 1]])
    M_FR = Rot @ Jm @ Rot

    M_lf_f = lamming(LB, SP, 1, 0, L, V_theta_lf, M_err)
    M_f = lamming(LB, SP, 1, 0, L, V_theta,  M_empty)
    M_b = lamming(LB, SP, -1, 0, L, V_theta,  M_empty)
    M_lf_b = lamming(LB, SP, -1, 0, L, V_theta_lf, M_err)

    V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in
    #V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
    #V_out[mm] = M_lf_f @ V_in
    mm = mm + 1

# -------------- Using py_pol module -----------------------------------
E = Jones_vector('Output_J')
S = create_Stokes('Output_S')

E.linear_light(azimuth=0 * abs(V_out))
S.linear_light(azimuth=0 * abs(V_out))

# SOP evolution in Lead fiber (Forward)
E.from_matrix(V_out)
S.from_Jones(E)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24*pi/180, 31*pi/180], kind='scatter', color_line='b')

ell_V_out = E.parameters.ellipticity_angle()*180/pi
print("ell=", ell_V_out.max()-ell_V_out.min())
azi_V_out = E.parameters.azimuth()*180/pi

for nn, v in enumerate(azi_V_out):
    if v > 90:
        azi_V_out[nn] = azi_V_out[nn] - 180

print("azi=", azi_V_out.max()-azi_V_out.min())
'''
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
    Ip[nn] = -(V_ang[nn] - pi / 2) / (2 * V)
    abs_error[nn] = abs(Ip[nn] - V_plasmaCurrent[nn])
    if V_plasmaCurrent[nn] == 0:
        rel_error[nn] = 100
    else:
        rel_error[nn] = abs_error[nn] / V_plasmaCurrent[nn]

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

# fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
# fig.set_size_inches(6,4)
'''
plt.show()
