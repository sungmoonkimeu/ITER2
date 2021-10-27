
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon


"""
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt

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
LB = arange(160,380,20)

V_I = arange(0.1e6, 17e6, 0.1e6)
abs_error = zeros([11,len(V_I)])
rel_error = zeros([11,len(V_I)])


# Circulator input matrix

theta = 30 * pi / 2  # random axis of LB
phi = 30 * pi / 180  # ellipticity angle change from experiment
theta_e = 30 * pi / 180  # azimuth angle change from experiment

M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(1j*phi)]])

M_ci = M_rot @ M_theta @ M_phi @ M_theta_T

# Circulator output matrix

theta = 30 * pi / 2  # random axis of LB
phi = 30 * pi / 180  # ellipticity angle change from experiment
theta_e = 30 * pi / 180  # azimuth angle change from experiment

# Mci
M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(1j*phi)]])

M_co = M_rot @ M_theta @ M_phi @ M_theta_T

# Faraday rotation matirx
th_FR = pi/2
M_FR = np.array([[cos(th_FR), sin(th_FR)],[sin(th_FR), cos(th_FR)]])

# input matrix

V_I = np.array([[0], [1]])


for mm in range(len(LB)):

    delta = 2*pi/LB[mm]                     # Linear birefringence [rad/m]
    LC = 10000000000                    # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 28                       # length of sensing fiber 28 m
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.43                     # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.001                     # delta L [m]
    q = 0
    #_______________________________Parameters#2____________________________________

    V_in = mat([[1], [0]])

    V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
    # ones*1j <-- for type casting

    V_out = M_co @ M_FR @ M_ci @ V_I

    #print(J)

    E = Jones_vector('Output')

    Ip = zeros(len(V_I))
    V_ang = zeros(len(V_I))

    m = 0
    for nn in range(len(V_I)):
        V_out[nn] = M_co @ M_FR[nn] @ M_ci @ V_I
        E.from_matrix(M=V_out[nn])
        if nn>2 and E.parameters.azimuth()+ m * pi -V_ang[nn-1] < -pi*0.8:
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

for nn in range(len(V_I)):
    if V_I[nn] < 1e6:
        absErrorlimit[nn] = 10e3
    else:
        absErrorlimit[nn] = V_I[nn]*0.01
    relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

fig, ax = plt.subplots(2,1)
for i in range(len(LB)):
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
for i in range(len(LB)):
   ax[1].plot(V_I,rel_error[i,:],lw='1')
ax[1].plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
ax[1].legend(loc="upper right")

ax[1].set_xlabel('Plasma current (A)')
ax[1].set_ylabel('Relative error on Ip(A)')
#ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#plt.title('Output power vs Plasma current')
ax[1].set(xlim = (0,18e6), ylim = (0,0.08))
ax[1].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax[1].ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax[1].grid(ls='--',lw=0.5)

fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95)
plt.show()