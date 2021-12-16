
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import matplotlib as mpl
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd


def cart2sph(x, y, z):
    # r, theta, phi cooridinate
    # theta is inclination angle between the zenith direction and the line segment (0,0,0) - (x,y,z)
    # phi is the azimuthal angle between the azimuth reference direction and
    # the orthogonal projection of the line segment OP on the reference plane.

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def sph2cart(r, theta, phi):

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


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

# Circulator input matrix

theta = 0 * pi / 180   # birefringence axis of LB
phi = 0 * pi / 180  # ellipticity angle change from experiment
theta_e = 0 * pi / 180  # azimuth angle change from experiment

M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_ci = M_rot @ M_theta @ M_phi @ M_theta_T

# Circulator output matrix

theta = 0 * pi / 180  # random axis of LB
phi = 0 * pi / 180  # ellipticity angle change from experiment
theta_e = 0 * pi / 180  # azimuth angle change from experiment

# Mci
M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

M_co = M_rot @ M_theta @ M_phi @ M_theta_T

# input matrix
V_I = arange(0e6, 200e3 + 1e3, 50e3)

V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
V = 0.7 * 4 * pi * 1e-7

E0 = Jones_vector('input')
E = Jones_vector('Output')
E0.general_azimuth_ellipticity(azimuth=[0,pi/4, pi/4], ellipticity=[0,0, pi/4])
#V_in = np.array([[[1], [0]], [[np.sqrt(0.5)], [np.sqrt(0.5)]], [[np.sqrt(0.5)], [np.sqrt(0.5)*1j]]])
#V_in = np.array([[[1], [0]], [[-cos(pi/7)*1j], [sin(pi/7)]], [[0.707], [0.707*1j]]])
#V_in = np.array([[[1], [0]], [[-cos(pi/7)*1j], [sin(pi/7)]]])

color_code = ['b', 'k', 'r']
OV = np.array([])
#color_code = ['r', 'r', 'r']
midpnt = int(len(V_I)/2)

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

    #print(S[0].parameters.azimuth())
    '''
    # Vector drawing
    P0_s = np.array([1, S[0].parameters.azimuth(), S[0].parameters.ellipticity_angle()])
    P1_s = np.array([1, S[midpnt].parameters.azimuth(), S[midpnt].parameters.ellipticity_angle()])
    P2_s = np.array([1, S[-1].parameters.azimuth(), S[-1].parameters.ellipticity_angle()])

    P0 = sph2cart(1, P0_s[1], P0_s[2])
    P1 = sph2cart(1, P1_s[1], P1_s[2])
    P2 = sph2cart(1, P2_s[1], P2_s[2])
    print(P0, P1, P2)
    '''

    if nn != 0:
        draw_stokes_points(fig[0], S, kind='line', color_line=color_code[nn])
        draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=color_code[nn])
    else:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[23 * pi / 180, 32 * pi / 180], kind='line',
                                     color_line=color_code[nn])
        draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=color_code[nn])
    print(S[-1])
labelTups = [('LP0', 0), ('LP45', 1), ('RCP', 2)]
colors = color_code
custom_lines = [plt.Line2D([0], [0], ls="", marker='.',
                           mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],loc='center left', bbox_to_anchor=(0.7, .8))

#print(V_out)


plt.show()