
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon

#AO 2015 Fig. 14,15

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
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

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

V_I = arange(2e5, 4e5, 2e5)

JF = mat([[0, 1], [-1, 0]])
#JF = mat([[1, 0], [0, -1]])

Len_LF = [0.153, 0.156, 0.159, 0.162, 0.165]

E3 = Jones_vector('Output_SAVEDATA')

E3.linear_light(azimuth= zeros(len(Len_LF)))
S3 = create_Stokes('output3')
S3.from_Jones(E3)

cstm_color = ['c','m','y','k','r']
for mm in range(len(Len_LF)):
    print("mm = ",mm)
    delta = 2*pi/LB[0]                     # Linear birefringence [rad/m]
    LC = 1*2*pi*10000000000000             # Reciprocal circular beatlength [m]
    rho_C = 2*pi/LC                     # Reciprocal circular birefringence [rad/m]
    Len_SF = 0.5                   # length of sensing fiber 28 m
    #Len_LF = 0.165
    I = 1                               # Applied plasma current 1A for normalization
    V = 0.43                         # Verdat constant 0.54 but in here 0.43
    rho_F = V*4*pi*1e-7/(Len_SF*I)   # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
    delta_L = 0.00001                   # delta L [m]
    dq = 2*pi/SR[0]
    q = 0
    #_______________________________Parameters#2____________________________________

    #V_in = mat([[0.707], [-0.707]])
    V_in1 = mat([[0.707], [0.707]])
    V_in2 = mat([[0], [1]])
    V_in3 = mat([[1], [0]])
    V_in = mat([[1],[0]])

    #[+0.924+0.000j][+0.383+0.000j]
    V_out = np.einsum('...i,jk->ijk', ones(len(V_I))*1j, np.mat([[0], [0]]))

    # ones*1j <-- for type casting
    V_J = [1j*ones(len(V_I)),1j*ones(len(V_I))]
    V_L = arange(delta_L,Len_SF+delta_L,delta_L)
    V_LF = arange(delta_L, Len_LF[mm] + delta_L, delta_L)

    V_prop = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop1 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))
    V_prop2 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop3 = np.einsum('...i,jk->ijk', ones(len(V_L)) * 1j, np.mat([[0], [0]]))
    V_prop4 = np.einsum('...i,jk->ijk', ones(len(V_LF)) * 1j, np.mat([[0], [0]]))

    #print(V_L)
    n_V_L = len(V_L)

    #------------------------------ Variable forward--------------
    rho_1 = rho_C - rho_F*V_I
    delta_Beta_1 = 2*(rho_1**2 + (delta**2)/4)**0.5

    alpha_1 = cos(delta_Beta_1/2*delta_L)
    beta_1 = delta/delta_Beta_1*sin(delta_Beta_1/2*delta_L)
    gamma_1 = 2*rho_1/delta_Beta_1*sin(delta_Beta_1/2*delta_L)

    #------------------------------ Variable backward--------------
    rho_2 = rho_C + rho_F*V_I
    delta_Beta_2 = 2*(rho_2**2 + (delta**2)/4)**0.5

    alpha_2 = cos(delta_Beta_2/2*delta_L)
    beta_2 = delta/delta_Beta_2*sin(delta_Beta_2/2*delta_L)
    gamma_2 = 2*rho_2/delta_Beta_2*sin(delta_Beta_2/2*delta_L)

    alpha_lf = cos(delta/2*delta_L)
    beta_lf = sin(delta/2*delta_L)
    gamma_lf = 0


    #------------------------------ Variable FRM--------------
    #print(J)

    E = Jones_vector('Output')
    E2 = Jones_vector('Output')

    S = create_Stokes(N=len(V_I))
    S2 = create_Stokes('output2')

    E2.linear_light(azimuth=0*(V_L))
    S2.linear_light(azimuth=0*(V_L))

    J0 = mat([[1, 0], [0, 1]])
    J0T = mat([[1, 0], [0, 1]])
    q0 = 0

    for kk in range(len(V_LF)):
        q0 = q0 + dq * delta_L

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        J0 = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J0
        #J0T = J0T @ np.vstack((J11, J21, J12, J22)).T.reshape(2, 2)
        V_prop1[kk] =  J0 @ V_in
        '''
        V_prop1[kk] =  J @ V_in1
        V_prop2[kk] =  J @ V_in2
        V_prop3[kk] = J @ V_in3
        '''
    for nn in range(len(V_I)):
        print("mm = ", mm, ", nn = ", nn)
        #       q=0
        q = q0
        J = mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q + dq * delta_L

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J
            V_prop2[kk] = J @ J0 @ V_in

        JT = mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q - dq * delta_L
            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JT = np.vstack((J11, J21, J12, J22)).T.reshape(2, 2) @ JT
            # JT = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ JT
            V_prop3[kk] = JT @ JF @J @ J0 @ V_in
        # JT= np.array([[1, 0], [0, -1]]) @ JT.T @ np.array([[1, 0], [0, -1]])
        # print(q)
        #V_out[nn] = J0T @ JT @ JF @ J @ J0 @ V_in
    q0 = q
    for kk in range(len(V_LF)):
        q0 = q0 - dq * delta_L

        J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
        J12 = 1j * beta_lf * sin(2 * q0)
        J21 = 1j * beta_lf * sin(2 * q0)
        J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)

        J0T = np.vstack((J11, J21, J12, J22)).T.reshape(2, 2) @ J0T
        V_prop4[kk] =  J0T @ JT @ JF @J @ J0 @ V_in


    #print(S2)
    #fig, ax = S2[arange(0,len(V_LF),10)].draw_poincare(figsize=(10,10),angle_view=[0,0],kind='line',color_line='b')
    #draw_stokes_points(fig[0],S2[arange(10090,10390,60)])
    '''
    fig, ax = S2[arange(-1-300,-1,10)].draw_poincare(figsize=(10, 10), angle_view=[0, 0], kind='line',
                                                                 color_line='b')
    
Sout = S2[arange(-1-300,-1,60)].parameters.matrix()
print(V_L[arange(-1-300,-1,60)])
'''
    E2.from_matrix(V_prop1)
    S2.from_Jones(E2)
    if mm == 0:
        fig, ax = S2.draw_poincare(figsize=(7, 7), angle_view=[0.5, 0.2], kind='line',
                                                                 color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])
    else:
        draw_stokes_points(fig[0], S2, kind='line', color_line='b')
        draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])
    #draw_stokes_points(fig[0], S2[arange(-1 -300, -1, 60)])

    #fig2, ax2 = S2[arange(-1 -300, -1, 60)].draw_ellipse(figsize=(7,7))

    E2.from_matrix(V_prop2)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0],S2, kind='line',color_line='r' )
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

    E2.from_matrix(V_prop3)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind ='line', color_line='g')
    draw_stokes_points(fig[0], S2[0], kind='scatter', color_scatter=cstm_color[mm])
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])


    E2.from_matrix(V_prop4)
    S2.from_Jones(E2)
    draw_stokes_points(fig[0], S2, kind='line', color_line='k')
    draw_stokes_points(fig[0], S2[-1], kind='scatter', color_scatter=cstm_color[mm])

#   E3[mm].from_matrix(J0T @ JT @ JF @J @ J0 @ V_in)
#   print(E3[mm])

#draw_stokes_points(fig[0], E3, kind='scatter')

    #draw_stokes_points(fig[0],S2[arange(-1-300,-1,60)])
#Sout = S2[arange(0,300,60)].parameters.matrix()
#print(V_L[arange(0,300,60)])

#print(Sout)
#E2.from_Stokes(S2[arange(5090,5390,60)])
#print(E2)

#Dataout = Sout
#savetxt('fig15output.dat', Dataout)
#Dataout = column_stack((V_I, rel_error[0:6,:].T))
#savetxt('fig10_saved.dat', Dataout)

## Ploting graph
'''
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
ax[1].set(xlim = (0,18e6), ylim = (0.002,0.018))
ax[1].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax[1].ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax[1].grid(ls='--',lw=0.5)

fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95)
'''
plt.show()

