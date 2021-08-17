import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, \
    arccos, sqrt, arcsin, arctan,tan, random, column_stack,savetxt,loadtxt
from numpy.linalg import norm, eig, matrix_power
import concurrent.futures as cf
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.jones_matrix import *
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

JM = Jones_matrix("QWP_112.5")
JM.quarter_waveplate(azimuth=(45)*degrees)
th = 30 * 180/pi
R = np.array([[cos(th), sin(th)],[-sin(th),cos(th)]])
JR = Jones_matrix("Rotator")
JR.from_matrix(R)
JV = Jones_vector("input")
JV.linear_light(azimuth = 22.5*degrees)

Jout = JM*JV

# starting point
S = create_Stokes('Output_S')
S.from_Jones(Jout)
print("azimuth =",S.parameters.azimuth()*180/pi)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='scatter')

# end point
S1 = create_Stokes('Output_S1')
S1.from_Jones(JV.linear_light(azimuth=S.parameters.azimuth()))
draw_stokes_points(fig[0], S1, kind='scatter', color_line='b')

# middile points 0
ell0 = np.arange(0, S.parameters.ellipticity_angle()+0.05, 0.05)
JV.general_azimuth_ellipticity(azimuth=S.parameters.azimuth(),ellipticity=ell0)
S2 = create_Stokes('Output_S2')
S2.from_Jones(JV)
draw_stokes_points(fig[0], S2, kind='line', color_line='b')

JM.retarder_azimuth_ellipticity()

# calculate trace length

c = pi/2 - S.parameters.ellipticity_angle()
b = pi/2 - S1.parameters.ellipticity_angle()
A = S.parameters.azimuth() - S1.parameters.azimuth()

L_pc= arccos(cos(b)*cos(c) + sin(b)*sin(c)*cos(A))
print(c *180/pi , b*180/pi, A*180/pi)
print(L_pc *180/pi)


# case 1 rotating ideal HWP
th = arange(0, 90*pi/180, 0.05)
JR = Jones_matrix("Rotator")
JR.half_waveplate(azimuth=th)

JV = Jones_vector("input")
JV.linear_light(azimuth = 0*degrees)
JV.general_azimuth_ellipticity(azimuth=pi/4, ellipticity=pi/16)
Jout = JR*JV
S.from_Jones(Jout)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='scatter')

# case 2 ratating non-ideal HWP (80deg)
th = arange(0, pi, 0.05)
JR = Jones_matrix("Rotator")
JR.retarder_azimuth_ellipticity(R = 20*pi/180, azimuth=45*pi/180)

JV = Jones_vector("input")
JV.linear_light(azimuth = th)
Jout = JR*JV
S.from_Jones(JV)
S1.from_Jones(Jout)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')
draw_stokes_points(fig[0], S1, kind='line', color_line='b')

plt.show()