
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
import matplotlib.pylab as pl
from matplotlib.colors import rgb2hex


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import matplotlib as mpl
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd

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

def cal_arclength(S):
    L = 0
    for nn in range(len(S)-1):
        c = pi/2 - S.parameters.ellipticity_angle()[nn]*2
        b = pi/2 - S.parameters.ellipticity_angle()[nn+1]*2

        A0 = S.parameters.azimuth()[nn]*2
        A1 = S.parameters.azimuth()[nn+1]*2
        A = A1 - A0
        if A == np.nan:
            A = 0

        L = L + arccos(cos(b) * cos(c) + sin(b) * sin(c) * cos(A))
        #print("c",c,"b",b,"A0",A0,"A1",A1, "L",L)

    return L

# _______________________________Parameters#1___________________________________#
# Circulator input matrix
theta = 50* pi / 180   # birefringence axis of LB
phi = 5 * pi / 180  # ellipticity angle change from experiment
theta_e = 20 * pi / 180  # azimuth angle change from experiment

M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_ci = M_rot @ M_theta @ M_phi @ M_theta_T

# Circulator output matrix
theta = 10* pi / 180  # random axis of LB
phi = 10 * pi / 180  # ellipticity angle change from experiment
theta_e = 50 * pi / 180  # azimuth angle change from experiment

M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_co = M_rot @ M_theta @ M_phi @ M_theta_T

# input matrix
'''
strfile0 = 'Filteredsignal.csv'
data = pd.read_csv(strfile0)
V_I = np.array(data)
V_I = V_I.reshape(V_I.size,)
'''
V_I = arange(0e6, 40e3 + 1e3, 5e3)

V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
V = 0.7 * 4 * pi * 1e-7

E0 = Jones_vector('input')
E = Jones_vector('Output')

azi = np.linspace(0,180,20)*pi/180
ell = np.linspace(-45,45,27)*pi/180
aziell = np.meshgrid(azi,ell)

colors = pl.cm.hsv(np.linspace(0, 1, len(aziell[0].reshape(aziell[0].shape[0] * aziell[0].shape[1]))))

E0.general_azimuth_ellipticity(azimuth=aziell[0].reshape(aziell[0].shape[0] * aziell[0].shape[1],),
                               ellipticity=aziell[1].reshape(aziell[1].shape[0] * aziell[1].shape[1],))

OV = np.array([])
midpnt = int(len(V_I)/2)
length_S = []
for nn in range(len(E0)):
    for mm, iter_I in enumerate(V_I):
        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])
        #V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]
        V_out[mm] = M_co @ M_FR @ M_ci @ E0[nn].parameters.matrix()

    E.from_matrix(M=V_out)
    S = create_Stokes('Output_S')
    S.from_Jones(E)

    length_S.append(cal_arclength(S)[0])


    if nn != 0:
        draw_stokes_points(fig[0], S, kind='line', color_line=rgb2hex(colors[nn]))
        draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
    else:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[23 * pi / 180, 32 * pi / 180], kind='line',
                                     color_line=rgb2hex(colors[nn]))
        draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
    print(S[-1])

print(length_S)
#labelTups = [('LP0', 0), ('LP45', 1), ('RCP', 2)]
#colors = color_code
#custom_lines = [plt.Line2D([0], [0], ls="", marker='.',mec='k', mfc=c, mew=.1, ms=20) for c in colors]
#ax.legend(custom_lines, [lt[0] for lt in labelTups],loc='center left', bbox_to_anchor=(0.7, .8))

#print(V_out)

#fig = plt.figure(figsize=(14,9))
#ax = plt.axes(projection='3d')

fig, ax = plt.subplots(1,1)
B = np.array(length_S).reshape(np.array(aziell).shape[1],np.array(aziell).shape[2])

#surf = ax.plot_surface(aziell[0], aziell[1],B)
contour = ax.contourf(aziell[0]*180/pi,aziell[1]*180/pi*2,B/4/V, levels = np.linspace(0,40000,41),cmap='Reds')
fig.colorbar(contour) # Add a colorbar to a plot

ax.set_xlabel('azimuth angle [deg]')
ax.set_ylabel('elevation angle [deg]')

plt.show()
