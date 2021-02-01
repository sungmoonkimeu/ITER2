
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

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

start = pd.Timestamp.now()

# _______________________________Parameters___________________________________#
#r = 1
L = .5
L_lf = [0.153, 0.156, 0.159, 0.162, 0.165]    #Length of Lead fiber [m]

LB = 0.132
SP = 0.03
#dz = SP / 1000
dz = 0.00001
q = 0
I = 10e5
V = 0.43 * 4 * pi * 1e-7
#H = I / (2 * pi * r)
#H = I/L
STR = (2 * pi) / SP
A_P = 0
V_in = mat([[cos(A_P)], [sin(A_P)]])
M_P = mat([[(cos(A_P)) ** 2, (sin(A_P) * cos(A_P))], [(sin(A_P) * cos(A_P)), (sin(A_P)) ** 2]])

cstm_color = ['c','m','y','k','r']


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

for mm in range(len(L_lf)):
    n = int((L + L_lf[mm])/ dz)
    dz_rem = L + L_lf[mm] - (n * dz)
    if dz_rem == 0:
        V_dz = dz * ones(n)
    else:
        V_dz = append(dz * ones(n), dz_rem)

    V_L = append(arange(dz, L+L_lf[mm], dz), L+L_lf[mm])
    n = len(V_L)

    # Intrinsic linear birefringence
    delta = (2 * pi) / LB
    V_delta = delta * ones(n)

    # Faraday effect induced birefringence
    H = I/L
    rho = V * H
    V_rho = rho * ones(n)
    n_lf = int(L_lf[mm]/dz)

    for nn in range(n_lf):
        V_rho[nn]=0

    ###----------------------Laming parameters---------------------------------###
    n = 0
    m = 0
    # --------Laming: orientation of the local slow axis ------------
    t_s_f = 0
    t_s_b = 0

    t_f = STR * V_dz
    t_f[0] = 0  # for the first section the azimuth doesn't change, it remains same as that at the entrance.
    t_s_f = np.cumsum(t_f)  # initial orientation of the local fast axis in forward direction
    t_b = STR * V_dz
    t_s_b = np.cumsum(t_b)  # initial orientation of the local fast axis in backward direction
    # -----------------------------------------------------------------------------
    # The following parameters are defined as per Laming (1989) paper
    qu_f = 2 * (STR + V_rho) / V_delta
    qu_b = 2 * (-STR + V_rho) / V_delta

    gma_f = 0.5 * ((V_delta) ** 2 + 4 * ((STR + V_rho) ** 2)) ** 0.5
    gma_b = 0.5 * ((V_delta) ** 2 + 4 * ((-STR + V_rho) ** 2)) ** 0.5

    omega_z_f = STR * V_dz + arctan((-qu_f / ((1 + qu_f ** 2) ** 0.5)) * tan(gma_f * V_dz)) + n * pi
    omega_z_b = -STR * V_dz + arctan((-qu_b / ((1 + qu_b ** 2) ** 0.5)) * tan(gma_b * V_dz)) + n * pi

    R_z_f = 2 * arcsin(sin(gma_f * V_dz) / ((1 + qu_f ** 2) ** 0.5))
    R_z_b = 2 * arcsin(sin(gma_b * V_dz) / ((1 + qu_b ** 2) ** 0.5))

    phi_z_f = ((STR * V_dz) - omega_z_f) / 2 + m * (pi / 2) + t_s_f
    phi_z_b = ((-STR * V_dz) - omega_z_b) / 2 + m * (pi / 2) + t_s_b

    # -----------------------------Forward propagation-----------------------------#
    theta_R_laming_f_all = np.reshape((R_z_f / 2), (len(V_dz), 1,1)) * np.array([[1j * cos(2 * phi_z_f), 1j * sin(2 * phi_z_f)],[1j * sin(2 * phi_z_f), -1j * cos(2 * phi_z_f)]]).T

    #theta_R_laming_f_all = GGG * np.array([[1j * cos(2 * phi_z_f), 1j * sin(2 * phi_z_f)],[1j * sin(2 * phi_z_f), -1j * cos(2 * phi_z_f)]]).T
    theta_omg_laming_f_all = np.einsum('...i,jk->ijk', omega_z_f, np.mat([[0, -1], [1, 0]]))

    # theta matrix of rotator for each element of the fibre in the forward direction
    N_i_f_all = theta_R_laming_f_all + theta_omg_laming_f_all
    # N-matrix of each fibre element considaering the local effects acting along the fibre in forward direction

    # --------------------Backward propagation--------------------------
    theta_R_laming_b_all = np.reshape((R_z_b / 2), (len(V_dz), 1, 1)) * np.array([[1j * cos(2 * phi_z_b), 1j * sin(2 * phi_z_b)],
                                                                                  [1j * sin(2 * phi_z_b),-1j * cos(2 * phi_z_b)]]).T
    # (5600,1,1) * (2,2,5600).T = (5600,1,1) * (5600,2,2)
    # theta matrix of retarder for each element of the fibre in the backward direction

    theta_omg_laming_b_all = np.einsum('...i,jk->ijk', omega_z_b, np.mat([[0, -1], [1, 0]]))
    # theta matrix of rotator for each element of the fibre in the backward direction

    N_i_b_all = theta_R_laming_b_all + theta_omg_laming_b_all
    # N-matrix of each fibre element considering the local effects acting along the fibre in backward direction

    M_i_f = eigen_expm(N_i_f_all)  # Matrix exponential of N_i_f
    M_i_b = eigen_expm(N_i_b_all)  # Matrix exponential of N_i_b

    M_f = mat([[1, 0], [0, 1]])
    M_b = mat([[1, 0], [0, 1]])
    M_FR = mat([[0, 1], [-1, 0]])
    #PB = zeros(len(V_L))

    V_prop = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    #V_prop1 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))
    V_prop2 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop3 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    #V_prop4 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))

    for i in range(len(V_L)):
        M_f = M_i_f[i] * M_f  # forward jones matrix
        #M_b = M_b * M_i_b[i]  # backward jones matrix
        V_prop2[i] = M_f *  V_in  # o/p SOP

    for i in range(len(V_L)):
        M_b =  M_i_b[-1-i] * M_b # backward jones matrix
        V_prop3[i] = M_b * M_FR * V_prop2[-1]
    #   PB[i] = (norm(Vout)) ** 2

    # -------------- Using py_pol module -----------------------------------
    E2 = Jones_vector('Output_J')
    S2 = create_Stokes('Output_S')

    E2.linear_light(azimuth=0*(V_L))
    S2.linear_light(azimuth=0*(V_L))

    # SOP evolution in Lead fiber (Forward)
    E2.from_matrix(V_prop2[0:n_lf])
    S2.from_Jones(E2)

    if mm == 0:
        fig, ax = S2.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line',
                                   color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])
    else:
        draw_stokes_points(fig[0], S2, kind='line', color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Sensing fiber (Forward)
    E2.from_matrix(V_prop2[n_lf:-1])
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='r')
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Sensing fiber (Backward)
    E2.from_matrix(V_prop3[0:n-n_lf])
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='g')
    draw_stokes_points(fig[0], S2[0], kind='scatter', color_scatter=cstm_color[mm])
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Lead fiber (Backward)
    E2.from_matrix(V_prop3[n-n_lf:-1])
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='k')
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

plt.show()
