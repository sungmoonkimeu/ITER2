
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon

#AO 2015 Fig. 13

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
LB = [0.1]
SR = [0.02]

V_I = arange(0.1e6, 4e6, 0.1e6)
abs_error = zeros([11,len(V_I)])
rel_error = zeros([11,len(V_I)])

JF = mat([[0, 1], [-1, 0]])
chi = [0]
#chi = [0, pi/16, pi/8]
#chi = [0.02713795, 0.09467992, 0.12374727, 0.07893792, 0.01449616]
for mm in range(len(chi)):
    print("mm = ",mm)
    delta = 2*pi/LB[0]                     # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000000                 # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 1                       # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.54                         # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.0001                   # delta L [m]
    dq = 2*pi/SR[0]
    q = 0
    #_______________________________Parameters#2____________________________________

    #V_in = mat([[1], [0]])
    V_in = mat([[sqrt(1/(1+tan(chi[mm])**2))],[1j*sqrt(tan(chi[mm])**2/(1+tan(chi[mm])**2))]])


    V_out = np.einsum('...i,jk->ijk', ones(len(V_I))*1j, np.mat([[0], [0]]))
    # ones*1j <-- for type casting

    V_L = arange(delta_L,Len_SF+delta_L,delta_L)
    #print(V_L)
    n_V_L = len(V_L)


    #------------------------------ Variable forward--------------
    rho_1 = rho_C + rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    '''
    J11 = alpha_1 + 1j * beta_1 * cos(2 * q)
    J12 = -gamma_1 + 1j * beta_1 * sin(2 * q)
    J21 = gamma_1 + 1j * beta_1 * sin(2 * q)
    J22 = alpha_1 - 1j * beta_1 * cos(2 * q)

    J = np.vstack((J11,J12,J21,J22)).T.reshape(len(V_I),2,2)
    '''
    #------------------------------ Variable backward--------------
    rho_2 = rho_C - rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2 / 2 * delta_L)
    beta_2 = delta / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)
    gamma_2 = 2 * rho_2 / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)
    '''
    J11 = alpha_2 + 1j * beta_2 * cos(2 * q)
    J12 = -gamma_2 + 1j * beta_2 * sin(2 * q)
    J21 = gamma_2 + 1j * beta_2 * sin(2 * q)
    J22 = alpha_2 - 1j * beta_2 * cos(2 * q)

    JT = np.vstack((J11,J21,J12,J22)).T.reshape(len(V_I),2,2)
    '''
    #------------------------------ Variable FRM--------------

    #print(J)

    E = Jones_vector('Output')

    Ip = zeros(len(V_I))
    V_ang = zeros(len(V_I))

    m = 0
    for nn in range(len(V_I)):
        print("mm = ",mm, " nn = ", nn)
        q=0
        J = mat([[1, 0], [0, 1]])
        JT = mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q+dq*delta_L

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)

            J =  np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J

            #J = J*mat([[J11, J12],[J21,J22]])
            #J = np.vstack((J11, J12, J21, J22)).T.reshape(len(V_I), 2, 2)

            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)

            JT = JT @ np.vstack((J11, J21, J12, J22)).T.reshape(2, 2)
            # JT = JT*mat([[J11, J21],[J12,J22]])
            # JT = np.vstack((J11, J21, J12, J22)).T.reshape(len(V_I), 2, 2)
        #print(q)
        V_out[nn] =  JT@JF@J @ V_in
        E.from_matrix(M=V_out[nn])

        if nn>2 and E.parameters.azimuth()+ m * pi -V_ang[nn-1] < -pi*0.9:
            m = m+1
        V_ang[nn] = E.parameters.azimuth() + m * pi
        Ip[nn] = (V_ang[nn] - pi/2)/(2*V*4*pi*1e-7)
        abs_error[mm,nn] = abs(Ip[nn]-V_I[nn])
        rel_error[mm,nn] = abs_error[mm,nn]/V_I[nn]
        #print(error[nn])
        #axis, fig = E.draw_ellipse(draw_arrow=True, figsize=(5,5))


## Requirement specificaion for ITER
absErrorlimit = zeros(len(V_I))
relErrorlimit = zeros(len(V_I))

#Calcuation ITER specification
for nn in range(len(V_I)):
    if V_I[nn] < 1e6:
        absErrorlimit[nn] = 10e3
    else:
        absErrorlimit[nn] = V_I[nn]*0.01
    relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]


Dataout = column_stack((V_I, rel_error[0:3,:].T))
savetxt('Stackingmethod.dat', Dataout)


## Ploting graph

fig, ax = plt.subplots(figsize=(6, 3))

for i in range(len(chi)):
   ax.plot(V_I,rel_error[i,:],lw='1')
ax.plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
ax.legend(loc="upper right")

ax.set_xlabel('Plasma current (A)')
ax.set_ylabel('Relative error on Ip(A)')
#ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#plt.title('Output power vs Plasma current')
ax.set(xlim = (0,18e6), ylim = (0,0.1))
ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax.ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax.grid(ls='--',lw=0.5)

fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95)
plt.show()


