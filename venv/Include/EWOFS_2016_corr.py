
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon

#AO 2015 Fig. 16
# Not matched to fig. 16!!!

"""
import time

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import numpy as np
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt
from numpy.linalg import norm, eig, matrix_power
import concurrent.futures as cf
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes

start = pd.Timestamp.now()
start_time = time.time()

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
#LB = 160 + mm*20                    # Linear beatlength [m]
LB = [0.0304]
SR = [0.003]

V_I = arange(0.1e6, 17e6, 0.5e6)
abs_error = zeros([len(V_I)])
rel_error = zeros([len(V_I)])
''
JF = mat([[0, 1], [-1, 0]])
#Temp_SF =  arange(90,110+2,2) # Temperature of sensing fiber
Temp_SF =  arange(90,110+2,30) # Temperature of sensing fiber
#Len_LF =  [0.300, 0.303, 0.306,0.309,0.312]

for mm in range(len(Temp_SF)):
    print("mm = ",mm)
    LBt = LB[0]+0.03e-3*Temp_SF[mm]
    delta = 2*pi/LBt                # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000000                 # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 28                       # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V0 = 0.54                         # Verdat constant 0.54 but in here 0.43
    V = V0 + 0.81 * 1e-4 * V0 * Temp_SF[mm]
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = SR[0]/10                 # delta L [m]
    dq = 2*pi/SR[0]
    q = 0
    #_______________________________Parameters#2____________________________________
    V_in = mat([[1], [0]])
    #V_in = mat([[sqrt(1/(1+tan(chi[mm])**2))],[1j*sqrt(tan(chi[mm])**2/(1+tan(chi[mm])**2))]])
    V_out = np.einsum('...i,jk->ijk', ones(len(V_I))*1j, np.mat([[0], [0]]))
    # ones*1j <-- for type casting

    V_L = arange(0,Len_SF,delta_L)
    #V_L_LF = arange(delta_L_LF,Len_LF[mm]+delta_L_LF,delta_L_LF)
    #print(len(V_L_LF))
    V_q_L = V_L * dq

    n_V_L = len(V_L)
    #n_V_L_LF = len(V_L_LF)
    #------------------------------ Variable forward--------------
    rho_1 = rho_C + rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    #------------------------------ Variable backward--------------
    rho_2 = -rho_C + rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2/2*delta_L)
    beta_2 = delta/delta_Beta_2*sin(delta_Beta_2/2*delta_L)
    gamma_2 = 2*rho_2/delta_Beta_2*sin(delta_Beta_2/2*delta_L)

    #alpha_lf = cos(delta/2*delta_L_LF)
    #beta_lf = sin(delta/2*delta_L_LF)
    #gamma_lf = 0

    #------------------------------ Variable FRM--------------
    #print(J)
    E = Jones_vector('Output')
    Ip = zeros(len(V_I))
    V_ang = zeros(len(V_I))

    J0 = mat([[1, 0], [0, 1]])
    J0T = mat([[1, 0], [0, 1]])
    m = 0
    for nn in range(len(V_I)):
        print("nn = ", nn)
#       q=0
        #q = q0
        J = J0
        JT = J0T

        for kk in range(len(V_L)):
            q = V_q_L[kk]

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J = np.array([[J11, J12],
                          [J21, J22]]) @ J

            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JT = JT @ np.array([[J11, J12],
                                [J21, J22]])

        V_out[nn] =  JT@JF@J @ V_in
        E.from_matrix(M=V_out[nn])

        if nn>2 and E.parameters.azimuth()+ m * pi -V_ang[nn-1] < -pi*0.5:
            m = m+1
        V_ang[nn] = E.parameters.azimuth() + m * pi
        Ip[nn] = (V_ang[nn] - pi/2)/(2*V0*4*pi*1e-7)
        abs_error[nn] = abs(Ip[nn]-V_I[nn])
        rel_error[nn] = abs_error[nn]/V_I[nn]
        #print(error[nn])
        #axis, fig = E.draw_ellipse(draw_arrow=True, figsize=(5,5))
        print("---  %s seconds for 1 time---" % (time.time() - start_time))

Dataout = column_stack((V_I, rel_error[:].T))
savetxt('EWOFS_fig3_saved_corr.dat', Dataout)

fig = plt.figure()
ax = fig.gca(projection='3d')

X = V_I
Y = Temp_SF
X,Y = np.meshgrid(X,Y)
Z = rel_error

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)