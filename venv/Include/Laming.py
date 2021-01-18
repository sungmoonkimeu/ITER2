
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: prasadarajudandu

(Circular vessel shape)
Spun fibre model with laming matrix

Revised on

@author: sungmoonkim
current output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan
from numpy.linalg import norm, eig
import concurrent.futures as cf

start = pd.Timestamp.now()

# _______________________________Parameters___________________________________#
r = 1
L = 2 * pi * r
LB = 5
SP = 0.5
dz = SP / 100
q = 0

V = 0.54 * 4 * pi * 1e-7
A_P = pi / 2
V_in = mat([[cos(A_P)], [sin(A_P)]])
M_P = mat([[(cos(A_P)) ** 2, (sin(A_P) * cos(A_P))], [(sin(A_P) * cos(A_P)), (sin(A_P)) ** 2]])

STR = (2 * pi) / SP

I = 10e6
V_I = arange(0,1e7+100000,100000)
I_out = ones(len(V_I))

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
dz_rem = L - (n * dz)
if dz_rem == 0:
    V_dz = dz * ones(n)
else:
    V_dz = append(dz * ones(n), dz_rem)

V_L = append(arange(dz, L, dz), L)
n = len(V_L)

# Intrinsic linear birefringence
delta = (2 * pi) / LB
V_delta = delta * ones(n)

###----------------------Laming parameters---------------------------------###
n = 0
m = 0

# --------Laming: orientation of the local slow axis ------------
t_s_f = 0
t_s_b = 0

t_f = STR * V_dz
t_f[0] = 0  # for the first section the azimuth doesn't change, it remains same as that at the entrance.
t_s_f = np.cumsum(t_f)  # initial orientation of the local fast axis in forward direction
'''
t_b = STR * V_dz
t_s_b = np.cumsum(t_b)  # initial orientation of the local fast axis in backward direction
'''
t_s_b = np.flip(t_s_f)

for nn in range(len(V_I)):

    H = V_I[nn] / (2 * pi * r)
    rho = V * H
    # Faraday effect induced birefringence
    V_rho = rho * ones(len(V_dz))

    # -----------------------------------------------------------------------------
    # The following parameters are defined as per Laming (1989) paper
    qu_f = 2 * (STR - V_rho) / V_delta
    qu_b = 2 * (-STR - V_rho) / V_delta

    gma_f = 0.5 * ((V_delta) ** 2 + 4 * ((STR - V_rho) ** 2)) ** 0.5
    gma_b = 0.5 * ((V_delta) ** 2 + 4 * ((-STR - V_rho) ** 2)) ** 0.5
    # The sign of farday rotation is opposite to that of the Laming paper, inorder
    # to be consistant with anti-clockwise (as in Jones paper) orientation for both
    # spin and farday rotation.

    omega_z_f = STR * V_dz + arctan((-qu_f / ((1 + qu_f ** 2) ** 0.5)) * tan(gma_f * V_dz)) + n * pi
    omega_z_b = -STR * V_dz + arctan((-qu_b / ((1 + qu_b ** 2) ** 0.5)) * tan(gma_b * V_dz)) + n * pi

    R_z_f = 2 * arcsin(sin(gma_f * V_dz) / ((1 + qu_f ** 2) ** 0.5))
    R_z_b = 2 * arcsin(sin(gma_b * V_dz) / ((1 + qu_b ** 2) ** 0.5))

    phi_z_f = ((STR * V_dz) - omega_z_f) / 2 + m * (pi / 2) + t_s_f
    phi_z_b = ((-STR * V_dz) - omega_z_b) / 2 + m * (pi / 2) + t_s_b

    # -----------------------------Forward propagation-----------------------------#

    GGG = np.einsum('...i,jk->ijk', R_z_f / 2, np.mat([[1, 0], [0, 1]]))

    #theta_R_laming_f_all = np.reshape((R_z_f / 2), (len(V_dz), 1,1)) * np.array([[1j *
    theta_R_laming_f_all = GGG * np.array([[1j *
                                                                                   cos(2 * phi_z_f), 1j * sin(2 * phi_z_f)],
                                                                                  [1j * sin(2 * phi_z_f),
                                                                                   -1j * cos(2 * phi_z_f)]]).transpose()

    # theta matrix of retarder for each element of the fibre in the forward direction
    # It should be noted that the matrix is arranged as [[a,c],[b,d]] and then
    # transposed to get [[a,b],[c,d]]. In this case b=c.

    theta_omg_laming_f_all = np.einsum('...i,jk->ijk', omega_z_f, np.mat([[0, -1], [1, 0]]))
    # theta matrix of rotator for each element of the fibre in the forward direction

    N_i_f_all = theta_R_laming_f_all + theta_omg_laming_f_all
    # N-matrix of each fibre element considaering the local effects acting along the fibre in forward direction

    # --------------------Backward propagation--------------------------
    theta_R_laming_b_all = np.reshape((R_z_b / 2), (len(V_dz), 1, 1)) * np.array([[1j *
                                                                                   cos(2 * phi_z_b), 1j * sin(2 * phi_z_b)],
                                                                                  [1j * sin(2 * phi_z_b),
                                                                                   -1j * cos(2 * phi_z_b)]]).transpose()
    # theta matrix of retarder for each element of the fibre in the backward direction

    theta_omg_laming_b_all = np.einsum('...i,jk->ijk', omega_z_b, np.mat([[0, -1], [1, 0]]))
    # theta matrix of rotator for each element of the fibre in the backward direction

    N_i_b_all = theta_R_laming_b_all + theta_omg_laming_b_all
    # N-matrix of each fibre element considering the local effects acting along the fibre in backward direction

    M_i_f = eigen_expm(N_i_f_all)  # Matrix exponential of N_i_f
    M_i_b = eigen_expm(N_i_b_all)  # Matrix exponential of N_i_b

    M_f = mat([[1, 0], [0, 1]])
    M_b = mat([[1, 0], [0, 1]])
    PB = zeros(len(V_L))

    for i in range(len(V_L)):
        M_f = M_i_f[i] * M_f  # forward jones matrix
        M_b = M_b * M_i_b[i]  # backward jones matrix

    Vout = M_P * M_b * M_f * V_in  # o/p SOP
    PB[-1] = (norm(Vout)) ** 2

    I_out[nn] = PB[-1]
#plt.plot(V_L, PB)
plt.plot(V_I,I_out)
plt.ylabel('(Ey)^2')
plt.xlabel('Plasma current (A)')
plt.title('Output power vs Plasma current')
plt.axis([0,1e7,0,1])
plt.show()
