
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon

#AO 2015 Fig. 16
# input Polarization from Fig. 15 (AO2015_7.py output)

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

from py_pol.jones_vector import Jones_vector, degrees, draw_ellipse
from py_pol.stokes import Stokes
from py_pol.drawings import draw_stokes_points


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
LB = [0.132]
SR = [0.03]

DataIN = loadtxt('fig15output.dat',unpack=True, usecols=[0,1,2,3,4])
S_in = Stokes('Original')
S_in.from_matrix(DataIN)
V_in_J = Jones_vector("Source 1")
V_in_J.from_Stokes(S_in)
#V_in_J.general_azimuth_ellipticity(azimuth=0,ellipticity=S_in.parameters.ellipticity_angle())
V_in = np.array(V_in_J.parameters.components()).T
Ang_V_in =  S_in.parameters.azimuth()
for nn in range(len(Ang_V_in)):
    if Ang_V_in[nn] > pi/2:
        Ang_V_in[nn] = Ang_V_in[nn]-pi

print(S_in.parameters.ellipticity_angle())
print(V_in_J.parameters.ellipticity_angle())
#print(S_in.parameters.azimuth()*180/pi)
print(V_in)
chi = S_in.parameters.ellipticity_angle()
print(np.array([[sqrt(1 / (1 + tan(chi) ** 2))], [1j * sqrt(tan(chi) ** 2 / (1 + tan(chi) ** 2))]]))

#draw_ellipse(V_in_J)
'''
fig, ax = S_in.draw_poincare(figsize=(10, 10))
S_tmp = Stokes('test')
S_tmp.from_Jones(V_in_J)
draw_stokes_points(fig[0], S_tmp)
'''

V_I = np.hstack((arange(0.1e6, 1e6, 0.1e6),arange(1e6,17e6,0.2e6)))
abs_error = zeros([11,len(V_I)])
rel_error = zeros([11,len(V_I)])
JF = mat([[0, 1], [-1, 0]])
Len_LF =  [0.3003]
#Len_LF =  [0.300, 0.303, 0.306,0.309,0.312]

for mm in range(len(Ang_V_in)):

    print("mm = ",mm)
    delta = 2*pi/LB[0]                     # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000000                 # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 28                       # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.43                         # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.0001                   # delta L [m]
    dq = 2*pi/SR[0]
    q = 0
    #_______________________________Parameters#2____________________________________
    #V_in = mat([[1], [0]])
    #V_in = mat([[sqrt(1/(1+tan(chi[mm])**2))],[1j*sqrt(tan(chi[mm])**2/(1+tan(chi[mm])**2))]])
    V_out = np.einsum('...i,jk->ijk', ones(len(V_I))*1j, np.mat([[0], [0]]))
    # ones*1j <-- for type casting

    V_L = arange(delta_L,Len_SF+delta_L,delta_L)
    n_V_L = len(V_L)
    #------------------------------ Variable forward--------------
    rho_1 = rho_C + rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    #------------------------------ Variable backward--------------
    rho_2 = rho_C - rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2/2*delta_L)
    beta_2 = delta/delta_Beta_2*sin(delta_Beta_2/2*delta_L)
    gamma_2 = 2*rho_2/delta_Beta_2*sin(delta_Beta_2/2*delta_L)

    #------------------------------ Variable FRM--------------
    #print(J)
    E = Jones_vector('Output')
    Ip = zeros(len(V_I))
    V_ang = zeros(len(V_I))

    J0 = mat([[1, 0], [0, 1]])
    J0T = mat([[1, 0], [0, 1]])
    q0 = 0

    m = 0
    print("J0=", J0)
    print("J0T=",J0T)
    print("V_in[",mm,"] = ", V_in[mm])
    for nn in range(len(V_I)):
        print("nn = ", nn)
#       q=0
        q = q0
        J = J0
        JT = J0T

        for kk in range(len(V_L)):
            q = q+dq*delta_L

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J =  np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J

            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JT = JT @ np.vstack((J11, J21, J12, J22)).T.reshape(2, 2)
            #JT = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ JT

        #JT= np.array([[1, 0], [0, -1]]) @ JT.T @ np.array([[1, 0], [0, -1]])
        #print(q)
        V_out[nn] =  JT@JF@J @ (V_in[mm].reshape(2,1))
        #V_out[nn] = JT @ JF @ J @ V_in[mm]

        E.from_matrix(M=V_out[nn])
        print(E.parameters.azimuth()+ m * pi -V_ang[nn-1])
        if nn>2 and E.parameters.azimuth()+ m * pi -V_ang[nn-1]< -pi*0.6:
            m = m+1
            print("here", nn)
        elif E.parameters.azimuth() + m* pi - V_ang[nn-1] > pi*0.9:
            m= m-1
            print("here2", nn)

        V_ang[nn] = E.parameters.azimuth() + m * pi - Ang_V_in[mm]
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

Dataout = column_stack((V_I, rel_error[0:5,:].T))
savetxt('fig16_saved.dat', Dataout)

## Ploting graph

fig, ax = plt.subplots(2,1)
for i in range(len(Ang_V_in)):
    ax[0].plot(V_I,abs_error[i,:],lw='1')

ax[0].plot(V_I,absErrorlimit,'r', label='ITER specification',lw='1')
ax[0].legend(loc="upper right")

ax[0].set_xlabel('Plasma current (A)')
ax[0].set_ylabel('Absolute error on Ip(A)')
ax[0].set(xlim = (0,18e6), ylim = (0,5e5))
ax[0].yaxis.set_major_formatter(OOMFormatter(5, "%1.0f"))
ax[0].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax[0].ticklabel_format(axis='both', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax[0].grid(ls='--',lw=0.5)
#ax = plt.axes()

for i in range(len(Ang_V_in)):
   ax[1].plot(V_I,rel_error[i,:],lw='1')
ax[1].plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
ax[1].legend(loc="upper right")

ax[1].set_xlabel('Plasma current (A)')
ax[1].set_ylabel('Relative error on Ip(A)')
#ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#plt.title('Output power vs Plasma current')
ax[1].set(xlim = (0,18e6), ylim = (0,0.12))
ax[1].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax[1].ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax[1].grid(ls='--',lw=0.5)

fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95)
plt.show()


