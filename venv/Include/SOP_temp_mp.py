
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
Temp_LF = [0, 50, 100]
#Temp_LF = [0]
Temp_SF = [0, 50, 100]

LB = 0.132
LB_lf = LB* ones(len(Temp_LF)) + Temp_LF*ones(len(Temp_LF))*3e-5     # Linear beatlength of lead fiber [m]
LB_sf = LB* ones(len(Temp_SF)) + Temp_SF*ones(len(Temp_SF))*3e-5     # Linear beatlength of sensing fiber[m]
SR = [0.03]     # Spin ratio

V_I = 10e5       # Aplied current 500kA
V0 = 0.43  # Verdat constant 0.54 but in here 0.43

JF = mat([[0, 1], [-1, 0]]) # Faraday mirror
Len_LF = [0.2]    #Length of Lead fiber [m]

cstm_color = ['c','m','y','k','r']

for mm in range(len(Temp_SF)):
    print("mm = ",mm)
    delta_sf = 2*pi/LB_sf[0]                     # Linear birefringence [rad/m]
    delta_lf = 2*pi/LB_lf[mm]                     # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000             # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 0.5                        # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = V0*(1+8.1e-5*Temp_SF[0])
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.00001                   # delta L [m]
    dq = 2*pi/SR[0]                     # delta q from spin ratio
    q = 0

    #_______________________________ in/out variables preparation ____________________________________
    # ones*1j <-- for type casting (complex number)

    V_L = arange(delta_L, Len_SF + delta_L, delta_L)        #sensing fiber
    V_LF = arange(delta_L, Len_LF[0] + delta_L, delta_L)   #lead fiber

    V_in = mat([[1],[0]])

    V_prop = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop1 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))
    V_prop2 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop3 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop4 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))

    #------------------------------ Variable forward--------------
    rho_1 = rho_C - rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta_sf**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta_sf/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    #------------------------------ Variable backward--------------
    rho_2 = -rho_C - rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta_sf**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2/2*delta_L)
    beta_2 = delta_sf/delta_Beta_2*sin(delta_Beta_2/2*delta_L)
    gamma_2 = 2*rho_2/delta_Beta_2*sin(delta_Beta_2/2*delta_L)

    # ------------------------------ Variable lead fiber --------------------------
    # ------------No Farday effect (rho = 0)--> (forward α, β, γ  = backward α, β, γ) ------

    alpha_lf = cos(delta_lf/2*delta_L)
    beta_lf = sin(delta_lf/2*delta_L)
    gamma_lf = 0

    # -------------- Variables for analysing Jones/Stokes parameter--------
    # -------------- Using py_pol module -----------------------------------
    E2 = Jones_vector('Output_J')
    S2 = create_Stokes('Output_S')

    E2.linear_light(azimuth=0*(V_L))
    S2.linear_light(azimuth=0*(V_L))

    # --------------- Forward propagation in Lead fiber ----------
    q0 = 0
    J0 = mat([[1, 0], [0, 1]])
    for kk in range(len(V_LF)):
        q0 = q0 + dq * delta_L

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        J0 = np.array([[J11,J12],[J21,J22]]) @ J0
        V_prop1[kk] =  J0 @ V_in

    # --------------- Forward propagation in sensing fiber ----------
    q = q0
    J = mat([[1, 0], [0, 1]])
    for kk in range(len(V_L)):
        q = q + dq * delta_L

        J11 = alpha_1 + 1j * beta_1 * cos(2 * q)
        J12 = -gamma_1 + 1j * beta_1 * sin(2 * q)
        J21 = gamma_1 + 1j * beta_1 * sin(2 * q)
        J22 = alpha_1 - 1j * beta_1 * cos(2 * q)

        J = np.array([[J11,J12],[J21,J22]]) @ J
        V_prop2[kk] = J @ J0 @ V_in

    # --------------- Backward propagation in sensing fiber ----------
    JT = mat([[1, 0], [0, 1]])
    for kk in range(len(V_L)):
        q = q - dq * delta_L

        J11 = alpha_2 + 1j * beta_2 * cos(2 * q)
        J12 = -gamma_2 + 1j * beta_2 * sin(2 * q)
        J21 = gamma_2 + 1j * beta_2 * sin(2 * q)
        J22 = alpha_2 - 1j * beta_2 * cos(2 * q)

        JT = np.array([[J11,J12],[J21,J22]]) @ JT #Trasposed
        V_prop3[kk] = JT @ JF @J @ J0 @ V_in

    # --------------- Backward propagation in lead fiber ----------
    q0 = q
    JT0 = mat([[1, 0], [0, 1]])
    for kk in range(len(V_LF)):
        q0 = q0 - dq * delta_L

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        JT0 = np.array([[J11,J12],[J21,J22]]) @ JT0 #Trasposed
        V_prop4[kk] =  JT0 @ JT @ JF @J @ J0 @ V_in

    # --------------- Drawing (overlap) ---------------

    # SOP evolution in Lead fiber (Forward)
    E2.from_matrix(V_prop1)
    S2.from_Jones(E2)
    if mm == 0:
        fig, ax = S2.draw_poincare(figsize=(7, 7), angle_view=[0.2, 1.2], kind='line',
                                                                 color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])
    else:
        draw_stokes_points(fig[0], S2, kind='line', color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Sensing fiber (Forward)
    E2.from_matrix(V_prop2)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0],S2, kind='line',color_line='r' )
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Sensing fiber (Backward)
    E2.from_matrix(V_prop3)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind ='line', color_line='g')
    draw_stokes_points(fig[0], S2[0], kind='scatter', color_scatter=cstm_color[mm])
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    # SOP evolution in Lead fiber (Backward)
    E2.from_matrix(V_prop4)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='k')
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

plt.show()

