
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
LB = [0.132]    # Linear beatlength [m]
SR = [0.03]     # Spin ratio

V_I = 10e5       # Aplied current 500kA

JF = mat([[0, 1], [-1, 0]]) # Faraday mirror
#Len_LF = [0.1, 0.5, 1, 5, 10]    #Length of Lead fiber [m]
#Len_LF = arange(0.1, 10, 0.1)
#Len_LF = [0.153, 0.156, 0.159, 0.162, 0.165]    #Length of Lead fiber [m]
Len_LF = [0.153]    #Length of Lead fiber [m]
#Len_SF = [0.500, 0.503, 0.506, 0.509, 0.512]    #Length of sensing fiber [m]
#Len_LF = np.hstack((arange(0.0001, 0.01, 0.0001), arange(0.01,1, 0.01), arange(1, 100, 1)))
#Len_LF = np.hstack((arange(0.0005, 0.030, 0.001), arange(0.03, 0.9,0.05),  arange(1,1.018,0.003)))
Len_SF = np.hstack((arange(0.0005, 0.030, 0.001), arange(0.03, 0.9,0.05),  arange(1,1.018,0.003)))
Materr = zeros(len(Len_SF))*(1j)
J0_V = np.einsum('...i,jk->ijk', ones(len(Len_SF)) * 1j, np.mat([[0,0], [0,0]]))
JT0_V = np.einsum('...i,jk->ijk', ones(len(Len_SF)) * 1j, np.mat([[0,0], [0,0]]))


cstm_color = ['c','m','y','k','r']

for mm in range(len(Len_SF)):
    print("mm = ",mm)
    delta = 2*pi/LB[0]                     # Linear birefringence [rad/m]
    delta = 0
    LC = 1*2*pi*10000000000000             # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    #Len_SF = 0.5                        # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.43                         # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF[mm]*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.000005                   # delta L [m]
    dq = 2*pi/SR[0]                     # delta q from spin ratio
    q = 0

    #_______________________________ in/out variables preparation ____________________________________
    # ones*1j <-- for type casting (complex number)

    V_L = arange(delta_L, Len_SF[mm] + delta_L, delta_L)        #sensing fiber
    V_LF = arange(delta_L, Len_LF[0] + delta_L, delta_L)   #lead fiber
    V_q_LF = V_LF * dq
    V_q_L = V_q_LF[-1] + V_L * dq

    V_in = mat([[1],[0]])

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
    # q0 = 0
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
            print("q0 [0]=",q0)
        if kk == len(V_LF)-1:
            print("q0 [-1]=", q0)
    # --------------- Forward propagation in sensing fiber ----------
    #q = q0
    J = mat([[1, 0], [0, 1]])
    for kk in range(len(V_L)):
        #q = q + dq * delta_L
        q = V_q_L[kk]
        J11 = alpha_1 + 1j * beta_1 * cos(2 * q)
        J12 = -gamma_1 + 1j * beta_1 * sin(2 * q)
        J21 = gamma_1 + 1j * beta_1 * sin(2 * q)
        J22 = alpha_1 - 1j * beta_1 * cos(2 * q)

        J = np.array([[J11,J12],[J21,J22]]) @ J
        V_prop2[kk] = J @ J0 @ V_in
        if kk == 0:
            print("q [0]=",q)
        if kk == len(V_L)-1:
            print("q [-1]=", q)
    # --------------- Backward propagation in sensing fiber ----------
    JT = mat([[1, 0], [0, 1]])
    for kk in range(len(V_L)):
        #q = q - dq * delta_L
        q = V_q_L[-1-kk]
        J11 = alpha_2 + 1j * beta_2 * cos(2 * q)
        J12 = -gamma_2 + 1j * beta_2 * sin(2 * q)
        J21 = gamma_2 + 1j * beta_2 * sin(2 * q)
        J22 = alpha_2 - 1j * beta_2 * cos(2 * q)

        JT = np.array([[J11,J12],[J21,J22]]) @ JT #Not trasnposed!
        V_prop3[kk] = JT @ JF @J @ J0 @ V_in
        if kk == 0:
            print("qT [0]=",q)
        if kk == len(V_L)-1:
            print("qT [-1]=", q)
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
        V_prop4[kk] =  JT0 @ JT @ JF @J @ J0 @ V_in
        if kk == 0:
            print("qT0 [0]=",q0)
        if kk == len(V_LF)-1:
            print("qT0 [-1]=", q0)

    print("J0 = \n", J0)
    print("JT0 = \n", JT0)

    J0_V[mm] = J
    JT0_V[mm] = JT

AA = zeros(len(Len_SF))

for mm in range(len(AA)):
    AA[mm] = np.linalg.norm(J0_V[mm]-JT0_V[mm])

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(Len_SF,AA)
ax.set_xlabel('Sensing fibre length [m]')
plt.rc('text', usetex=True)
ax.set_ylabel(r'$||\,A\,||$')
fig.suptitle(r'$ A = \,\overrightarrow{J_{SF}} - \overleftarrow{J_{SF}}$')
plt.rc('text', usetex=False)
fig.subplots_adjust(left = 0.17,bottom=0.133)
ax.set(ylim=(0, 0.0000000001))

plt.show()
