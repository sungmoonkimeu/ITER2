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


#strfile = "Magnetic field info/Field_around_VV_cyl_remove_fileheader.xlsx"
strfile = "Field_around_VV_cyl_remove_fileheader.xlsx"
data = pd.read_excel(strfile, engine='openpyxl')

nsel = 15
XG, YG, ZG = data['XG'][::nsel], data['YG'][::nsel], data['ZG'][::nsel]
Bx, By, Bz = data['B_X'][::nsel], data['B_Y'][::nsel], data['B_Z'][::nsel]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(XG,ZG,YG, Bx, Bz, By, color='r')

ax.set(xlim = [-2,13], ylim = [-9,6], zlim = [-7.5, 7.5])
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_zlabel('z[m]')
ax.plot(XG,ZG,YG,'k',lw='3')
plt.show()




