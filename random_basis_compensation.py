# Compensate the randomly rotated basis by adjusting the SOP rotation is aligned to the parallel to the equator.
# 1. Calculate the normal vector of SOP rotation
# 2. Basis rotation based on the axis 'R (S3)'
# 3. Another basis rotation base don the axis '+(S2)'


import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt, exp

from py_pol.jones_matrix import Jones_matrix
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.mueller import Mueller
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
import matplotlib.pyplot as plt

E = Jones_vector('Input')
S = create_Stokes('Output')
S2 = create_Stokes('cal')
J1 = Jones_matrix('Random element')
J2 = Jones_matrix('Random element')
M = Mueller('cal')

azi = np.arange(0, pi/2, 0.1)
#ell = np.arange(0, pi/6, 0.1)
E.general_azimuth_ellipticity(azimuth=azi, ellipticity=pi/12)
S.from_Jones(E)
fig, ax = S.draw_poincare(kind='line', color_line='k')

phi0 = -pi/12
Mp = np.array([[exp(1j*phi0/2), 0], [0, exp(-1j*phi0/2)]])
phi1 = pi/12
Mr = np.array([[cos(phi1), -sin(phi1)], [sin(phi1), cos(phi1)]])

J1.from_matrix(Mr)
J2.from_matrix(Mp)
Out = J2*J1*E
S.from_Jones(Out)
draw_stokes_points(fig[0], S, kind='line', color_line='r')

a = S.parameters.matrix()[1:]
for nn in range(3):
    b0 = a[:, 0] - a[:, nn+1]
    b1 = a[:, 0] - a[:, nn+2]
    b = np.cross(b0, b1)
    c = b/(b[0]**2 + b[1]**2 + b[2]**2)**0.5
    print(c)

fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1,)

z = [0, 0, 1]
y = [0, 1, 0]
x = [1, 0, 0]

th_x = np.arccos(np.dot(x,c))
th_y = np.arccos(np.dot(y,c))
th_z = np.arccos(np.dot(z,c))
print("x=", th_x*180/pi, "y=", th_y*180/pi, "z=", th_z*180/pi)

Rx = np.array([[cos(th_x), -sin(th_x), 0], [sin(th_x), cos(th_x), 0], [0, 0, 1]])
Ry = np.array([[1, 0, 0], [0, cos(th_y), -sin(th_y)], [0, sin(th_y), cos(th_y)]])
Rz = np.array([[cos(th_z), 0, sin(th_z)], [0,1,0], [-sin(th_z), 0, cos(th_z)]])


th = th_x
if th_y > pi/2:
    th = -th_x
Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

th = th_z
R45 = np.array([[cos(th), 0, sin(th)], [0,1,0], [-sin(th), 0, cos(th)]])     # S2, + 기준 rotation

th = 0
Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])   # S1, H 기준 rotation

TT = R45.T@Rh.T@Rr.T@a
zT = ones(np.shape(TT)[1])

Sp = np.vstack((zT,TT))
S.from_matrix(Sp)

draw_stokes_points(fig[0], S, kind='line', color_line='b')

a = S.parameters.matrix()[1:]
for nn in range(3):
    b0 = a[:,0] - a[:,nn+1]
    b1 = a[:, 0]-a[:, nn+2]
    b = np.cross(b0,b1)
    c = b/(b[0]**2 + b[1]**2 + b[2]**2)**0.5
    print(c)
fig[0].plot([0, c[0]],[0, c[1]],[0,c[2]], 'b-', lw=1,)

'''
azi, ell = S2.parameters.azimuth_ellipticity()
print("azi=",azi*180/pi, "ell=", ell*180/pi)

azi = 0
ell= ell
Mp_c = np.array([[exp(1j*-ell/2), 0], [0, exp(-1j*-ell/2)]])
Mr_c = np.array([[cos(-azi), -sin(-azi)], [sin(-azi), cos(-azi)]])
J1.from_matrix(Mr_c)
J2.from_matrix(Mp_c)

K = J1*J2*Out
S.from_Jones(K).draw_poincare()
'''
plt.show()