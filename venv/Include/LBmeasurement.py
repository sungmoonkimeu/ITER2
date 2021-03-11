
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon JAN 25 2021
@author: sungmoon
#SOP evolution in spun fiber
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

# _______________________________Parameters#1___________________________________#
LB = [1]    # Linear beatlength [m]
SR = [0.5]     # Spin ratio

V_I = 10e5       # Aplied current 500kA

JF = mat([[0, 1], [-1, 0]]) # Faraday mirror

Len_LF = [2]    #Length of Lead fiber [m]

J0_V = np.einsum('...i,jk->ijk', ones(len(Len_LF)) * 1j, np.mat([[0,0], [0,0]]))
JT0_V = np.einsum('...i,jk->ijk', ones(len(Len_LF)) * 1j, np.mat([[0,0], [0,0]]))

cstm_color = ['c','m','y','k','r']

V_ang = arange(0,2*pi,2*pi/20)

for mm in range(len(V_ang)):
    print("mm = ",mm)
    delta = 2*pi/LB[0]                     # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000             # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 1                        # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.43                         # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.0005                   # delta L [m]
    dq = 2*pi/SR[0]                     # delta q from spin ratio
    q = 0

    #_______________________________ in/out variables preparation ____________________________________
    # ones*1j <-- for type casting (complex number)

    V_L = arange(0, Len_SF + delta_L, delta_L)        #sensing fiber
    V_LF = arange(0, Len_LF[0] + delta_L, delta_L)   #lead fiber
    V_q_LF = V_LF * dq
    V_q_L = V_q_LF[-1] + V_L * dq

    #V_in = mat([[1],[0]])
    V_in = mat([[cos(V_ang[mm])],[1j*sin(V_ang[mm])]])

    V_prop = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop1 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))
    V_prop2 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop3 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop4 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))

    #------------------------------ Variable forward--------------
    rho_1 = rho_C - rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    #------------------------------ Variable backward--------------
    rho_2 = -rho_C - rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2/2*delta_L)
    beta_2 = delta/delta_Beta_2*sin(delta_Beta_2/2*delta_L)
    gamma_2 = 2*rho_2/delta_Beta_2*sin(delta_Beta_2/2*delta_L)

    # ------------------------------ Variable lead fiber --------------------------
    # ------------No Farday effect (rho = 0)--> (forward α, β, γ  = backward α, β, γ) ------

    alpha_lf = cos(delta/2*delta_L)
    beta_lf = sin(delta/2*delta_L)
    gamma_lf = 0

    # -------------- Variables for analysing Jones/Stokes parameter--------
    # -------------- Using py_pol module -----------------------------------
    E2 = Jones_vector('Output_J')
    S2 = create_Stokes('Output_S')

    E2.linear_light(azimuth=0*(V_L))
    S2.linear_light(azimuth=0*(V_L))


    # --------------- Forward propagation in Lead fiber ----------
    #q0 = 0
    J0 = mat([[1, 0], [0, 1]])
    for kk in range(len(V_LF)):
        #q0 = q0 + dq * delta_L
        q0 = V_q_LF[kk]

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        J0 = np.array([[J11,J12],[J21,J22]]) @ J0
        V_prop1[kk] =  J0 @ V_in
        if kk == 0:
            print("q0 =",q0)
    # --------------- Backward propagation in lead fiber ----------
    #q0 = q
    JT0 = mat([[1, 0], [0, 1]])
    for kk in range(len(V_LF)):
        #q0 = q0 - dq * delta_L
        q0 = V_q_LF[-1-kk]

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        JT0 = np.array([[J11,J12],[J21,J22]]) @ JT0 #Not transposed!
        V_prop4[kk] =  JT0 @ J0 @ V_in
        if kk == 0:
            print("q0 =",q0)

    # --------------- Drawing (overlap) ---------------

    # SOP evolution in Lead fiber (Forward)
    E2.from_matrix(V_prop1)
    S2.from_Jones(E2)
    if mm == 0:
        fig, ax = S2.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line',
                                                                 color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[0])
    else:
        #draw_stokes_points(fig[0], S2, kind='line', color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[0])

    '''
    # SOP evolution in Lead fiber (Backward)
    E2.from_matrix(V_prop4)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='k')
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    print("azimuth= ", E2[-1].parameters.azimuth()*180/pi)
    print("ellipticity= ", E2[-1].parameters.ellipticity_angle() * 180/pi)
    '''
plt.show()

