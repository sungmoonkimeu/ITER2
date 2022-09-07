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

nsel = 1
XG, YG, ZG = data['XG'][::nsel], data['YG'][::nsel], data['ZG'][::nsel]
Bx, By, Bz = data['B_X'][::nsel], data['B_Y'][::nsel], data['B_Z'][::nsel]


XG2, YG2, ZG2 = np.roll(XG,-1), np.roll(YG,-1), np.roll(ZG,-1)
dX, dY, dZ = XG2-XG, YG2-YG, ZG2-ZG
XG2, YG2, ZG2 = np.roll(XG,1), np.roll(YG,1), np.roll(ZG,1)
dX1, dY1, dZ1 = XG-XG2, YG-YG2, ZG-ZG2
dX, dY, dZ = (dX+dX1)/2, (dY+dY1)/2, (dZ+dZ1)/2

# Vector of light propagation along fiber
V_f = np.vstack((dX,dZ,dY)).T
# Vector of light propagation along fiber -> normalize
nV_f = normalize(V_f)
# Vector of magnetic field
V_B = np.vstack((Bx,Bz,By)).T
# calculate the tangential vector of magnetic field parallel to the light propagation vector
V_Bdotf = np.sum(nV_f*V_B, axis=1)

plt.rcParams['mathtext.fontset']='cm'

# 3D graph --
# 1. Draw D-shaped fiber installed around the VV
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.quiver(XG,ZG,YG, Bx, Bz, By, color='r')

ax.set(xlim = [-2,13], ylim = [-9,6], zlim = [-7.5, 7.5])
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.plot(XG,ZG,YG,'k',lw='3')

# 2. Draw the normalized light propagating vector
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    if i == len(XG)-1:
        arrow_prop_dict['label']= r'$\vec{u}_l$'
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                nV_f[i][0], nV_f[i][1], nV_f[i][2], **arrow_prop_dict)
    ax.add_artist(a)

# 3. Draw the magnetic field vector
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    if i == len(XG) - 1:
        arrow_prop_dict['label'] = r'$\vec{B}$'
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                Bx[i * nsel], Bz[i * nsel], By[i * nsel], **arrow_prop_dict)
    ax.add_artist(a)
ax.legend()

# 4. Draw the tangential component of magnetic field vector parallel to the light propagating vector
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
for i in range(len(XG)):
    if i == len(XG)-1:
        arrow_prop_dict['label']= r'$\vec{B}\cdot\vec{u}_l$'
    a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
                # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][0],
                # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][1],
                # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][2],
                V_Bdotf[i] * nV_f[i][0],
                V_Bdotf[i] * nV_f[i][1],
                V_Bdotf[i] * nV_f[i][2],
                **arrow_prop_dict)

    ax2.add_artist(a)

ax2.set(xlim=[-2,13], ylim=[-9,6], zlim=[-7.5, 7.5])
ax2.set_xlabel('x[m]')
ax2.set_ylabel('y[m]')
ax2.set_zlabel('z[m]')
ax2.legend()
# Draw the magnitude of the tangential component of the magentic field vector along fiber loop
fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.plot(data['S'][::nsel], V_Bdotf, label=r'$\vec{B}\cdot\vec{u}_l$')

#ax3.plot(data['S'][::nsel], abs(V_Bdotf), label='|B dot f|')
# Draw the magnetic field vector along fiber loop
# ax3.plot(data['S'][::nsel], np.linalg.norm(V_B,axis=1), label='|Bxyz|')
# ax3.plot(data['S'][::nsel], V_B[:,0], label='Bx')
# ax3.plot(data['S'][::nsel], V_B[:,1], label='By')
# ax3.plot(data['S'][::nsel], V_B[:,2], label='Bz')

ax3.set_xlabel("FOCS fiber position [m]")
ax3.set_ylabel("Magnetic induction [T]")
int_V_Bdotf = np.trapz(V_Bdotf,x=data['S'][::nsel])
ax3.text(0,0.8, r'$\int\vec{B}\cdot\vec{u}_{l}dl$='+'%5.3f' % int_V_Bdotf + r'm$\cdot$T', fontsize=12)
ax3.legend()
print("integral of V_Bdotf = ", int_V_Bdotf)

savedata = np.column_stack((V_Bdotf,data['S'][::nsel]))
np.savetxt("B-field_around_VV.txt", savedata.T, fmt='%10.5f', delimiter=",", newline=' ')



fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
i = 1100
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0)
a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
            nV_f[i][0], nV_f[i][1], nV_f[i][2], **arrow_prop_dict)
ax4.add_artist(a)

arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0)
a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
            Bx[i * nsel], Bz[i * nsel], By[i * nsel], **arrow_prop_dict)
ax4.add_artist(a)

arrow_prop_dict = dict(mutation_scale=12, arrowstyle='-|>', color='g', shrinkA=0, shrinkB=0)
a = Arrow3D(XG[i * nsel], ZG[i * nsel], YG[i * nsel],
            # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][0],
            # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][1],
            # V_Bdotf[i]/np.sqrt(np.abs(V_Bdotf[i]))*nV_f[i][2],
            V_Bdotf[i] * nV_f[i][0],
            V_Bdotf[i] * nV_f[i][1],
            V_Bdotf[i] * nV_f[i][2],
            **arrow_prop_dict)
ax4.add_artist(a)
ax4.set(xlim = [5,8], ylim = [-2.5,0.5], zlim = [-6, -3])
ax4.set_xlabel('x[m]')
ax4.set_ylabel('y[m]')
ax4.set_zlabel('z[m]')
ax4.plot(XG,ZG,YG,'k',lw='2')


S = np.array(data['S'][::nsel])
data = np.column_stack([S,V_Bdotf])
np.savetxt("B-field_around_VV.txt", data, fmt=['%f','%f'])

plt.show(block=True)
# plt.show()