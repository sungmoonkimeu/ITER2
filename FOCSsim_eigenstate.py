# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue JUL 1 15:00:00 2021
@author: prasadarajudandu - modified by SMK
(Circular vessel shape)
Spun fibre model with laming matrix
"""
import cmath
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import cos, pi, zeros, sin, einsum, arange
from numpy.linalg import norm, eig
from py_pol.drawings import draw_stokes_points
from py_pol.jones_matrix import create_Jones_matrices
from py_pol.jones_vector import Jones_vector
from py_pol.stokes import create_Stokes


def eigen_expm(A):
    """

    Parameters
    ----------
    A : 2 x 2 diagonalizable matrix
        DESCRIPTION.

    Returns
    -------
    expm(A): exponential of the matrix A.

    """
    vals, vects = eig(A)
    return einsum('...ik, ...k, ...kj -> ...ij',
                  vects, np.exp(vals), np.linalg.inv(vects))

start = pd.Timestamp.now()

cstm_color = ['c', 'm', 'y', 'k', 'r']


# -------------- Using py_pol module -----------------------------------
E = Jones_vector('Input pol')
S = create_Stokes('Output_S')
J = create_Jones_matrices('J')

# Case 1
V_azi = np.hstack((pi/100, arange(0, pi/4, pi/16), pi/4))
V_mod_ret = arange(0, -pi/2, -pi/100)
E.linear_light(azimuth=V_azi)
J.retarder_linear(R=V_mod_ret)
for nn in range(len(V_azi)):
    V_out = J*E[nn]
    S.from_Jones(V_out)
    if nn == 0:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33*pi/180, 53*pi/180], kind='line', color_line='b')
    else:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')

# Case 2
V_ell = np.hstack((arange(0, -pi/4, -pi/16), -pi/4+pi/100))
# ellipticity 0 ~ pi/4 --> linear to circular
V_mod_rot = arange(0, pi/8, pi/100)
E.general_azimuth_ellipticity(azimuth=0, ellipticity=V_ell)
J.half_waveplate(azimuth=V_mod_rot)
for nn in range(len(V_ell)):
    V_out = J*E[nn]
    S.from_Jones(V_out)
    if nn == 0:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33*pi/180, 53*pi/180], kind='line', color_line='b')
    else:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')

# Case 3
J2 = create_Jones_matrices('Test matrix (Rotation)')

theta = arange(pi/4, pi/2, pi/8)
A = np.array([[cos(theta), -sin(theta)],
              [sin(theta), cos(theta)]])
J2.from_matrix(A)

[lambda1, lambda2, V1, V2] = J2.parameters.eig()
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

# lambda1, lambda2 : eigen value
# V1, V2 : eigen vector (state)
MQ = np.dstack((V1.T, V2.T))
MQ_I = np.linalg.inv(MQ)
ML = np.dstack((lambda1,lambda2))

MR = einsum('...ik, ...k, ...kj -> ...ij', MQ, ML, MQ_I)
print(MR)
print(A)

for nn in range(len(c)):
    phi_a = phase_a[nn] * pi / 180
    m_a = np.array([[cos(phi_a), -sin(phi_a)], [sin(phi_a), cos(phi_a)]])
    phi_b = phase_b[nn]*pi/180
    m_b = np.array([[cos(phi_b), -sin(phi_b)], [sin(phi_b), cos(phi_b)]])

    print(m_b@e[nn]@m_a)

E.linear_light(azimuth=-45*pi/180)
J.half_waveplate(azimuth=30*pi/180)

V_mod_ret = arange(0, -pi/2, -pi/100)


J2.retarder_linear(R=V_mod_ret)
V_out = J2*J*E
S.from_Jones(V_out)
fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33 * pi / 180, 53 * pi / 180], kind='line', color_line='b')

plt.show()

