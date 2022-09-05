# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 05 Sep 10:05:11 2022
@author: sungmoon

Remark: To read excel file using pandas module, install openpyxl module (pip install openpyxl)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)



    #strfile = "Magnetic field info/Field_around_VV_cyl_remove_fileheader.xlsx"
strfile = "Field_around_VV_cyl_remove_fileheader.xlsx"
data = pd.read_excel(strfile, engine='openpyxl')

nsel = 25
XG, YG, ZG = data['XG'][::nsel], data['YG'][::nsel], data['ZG'][::nsel]
Bx, By, Bz = data['B_X'][::nsel], data['B_Y'][::nsel], data['B_Z'][::nsel]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.quiver(XG,ZG,YG, Bx, Bz, By, color='r')

ax.set(xlim = [-2,13], ylim = [-9,6], zlim = [-7.5, 7.5])
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.plot(XG,ZG,YG,'k',lw='3')


XG2, YG2, ZG2 = np.roll(XG,-1), np.roll(YG,-1), np.roll(ZG,-1)
dX, dY, dZ = XG2-XG, YG2-YG, ZG2-ZG
V_f = np.vstack((dX,dZ,dY)).T
nV_f = normalize(V_f)
V_B = np.vstack((Bx,Bz,By)).T
V_Bdotf = np.sum(V_B*V_f, axis=1)

arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                nV_f[i][0], nV_f[i][1], nV_f[i][2], **arrow_prop_dict)
    ax.add_artist(a)
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                Bx[i * nsel], Bz[i * nsel], By[i * nsel], **arrow_prop_dict)
    ax.add_artist(a)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='c', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][0],
                V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][1],
                V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][2],
                **arrow_prop_dict)
    ax2.add_artist(a)

# # ax.quiver(XG,ZG,YG, nV_f.T[0], nV_f.T[1], nV_f.T[2])
ax2.set(xlim = [-2,13], ylim = [-9,6], zlim = [-7.5, 7.5])
ax2.set_xlabel('x[m]')
ax2.set_ylabel('y[m]')
ax2.set_zlabel('z[m]')

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.plot(data['S'][::nsel], V_Bdotf)
plt.show()




