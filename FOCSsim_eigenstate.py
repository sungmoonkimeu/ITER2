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
from numpy import cos, pi, zeros, sin, einsum, arange, exp
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
# Constant retardaion with different azimuth angle
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
# Constant rotation with different ellipticity angle
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
# finding intial basis from eigen value, eigen vector calculation
# Eigen value, vector --> eig(M) --> [v1, v2] [[λ1, 0], [v1,v2]^-1
#                                              [0, λ2]]
# eig(Jones Matrix) --> V M V-1
# V: eigen vector --> Chainging basis
# M: eigen value [[λ1, 0],
#                 [0, λ2]]  --> birefringence or rotation or both

J2 = create_Jones_matrices('Test matrix (Rotation)')

# Case 3-1
# test for Rotation matrix A (no retardation)
theta = arange(pi/6, pi/2, pi/6)
A = np.array([[cos(theta), -sin(theta)],
              [sin(theta), cos(theta)]])

J2.from_matrix(A)
[lambda1, lambda2, V1, V2] = J2.parameters.eig()
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

MQ = np.dstack((V1.T, V2.T))  # create (len(theta),2,2) array to use np.linalg.inv
MQ_I = np.linalg.inv(MQ)      # np.linalg.inv works only in (N,M,M) form
ML = np.dstack((lambda1, lambda2))

MR = einsum('...ik, ...k, ...kj -> ...ij', MQ, ML, MQ_I).reshape(len(theta), 2, 2)
# (len(theta),2,2) (1, len(theta), 2), len(theta),2,2) --> len(theta),2,2
print(MR[0])        # (len(theta),2,2)
print(A[..., 0])    # (2,2,len(theta))

# Case 3-2
# test for Retadation matrix A (with rotation)
theta = arange(pi/6, pi/2, pi/6)
A = np.array([[cos(theta), -sin(theta)],
              [sin(theta), cos(theta)]])
AT = np.array([[cos(theta), sin(theta)],
              [-sin(theta), cos(theta)]])
# create (2,2, len(theta)) array
# create Birefringence matrix
IB = np.zeros((2, 2, (len(theta))))
np.einsum('iij->ij', IB)[:] = 1
Bexp = exp(1j*np.vstack((theta, -theta)))
B = einsum('ijk, ...ik -> ijk', IB, Bexp)   # Birefringence matrix
C = einsum('ij..., jk..., kl...-> il...', A, B, AT)    # matrix calculation

J2.from_matrix(C)
[lambda1, lambda2, V1, V2] = J2.parameters.eig()
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

MQ = np.dstack((V1.T, V2.T))  # create (len(theta),2,2) array to use np.linalg.inv
MQ_I = np.linalg.inv(MQ)      # np.linalg.inv works only in (N,M,M) form
ML = np.dstack((lambda1, lambda2))

MR = einsum('...ik, ...k, ...kj -> ...ij', MQ, ML, MQ_I).reshape(len(theta), 2, 2)
# (len(theta),2,2) (1, len(theta), 2), len(theta),2,2) --> len(theta),2,2
print(MR[0])        # (len(theta),2,2)
print(C[..., 0])    # (2,2,len(theta))

plt.show()


# lambda1, lambda2 : eigen value
# V1, V2 : eigen vector (state) [2,len(theta)] --> reshape [2,1,len(theta)] + Hstack
Vects = np.hstack((V1.reshape(2, 1, len(theta)), V2.reshape(2, 1, len(theta))))

# Case4-1
# To estimate the equivalent rotation in the presence of retardation
# Rotation after induction retardation
# Rotation matrix A (no retardation)
theta = arange(0, pi/2, pi/50)
MA = np.array([[cos(theta), -sin(theta)],
               [sin(theta), cos(theta)]])
J3 = create_Jones_matrices('Test matrix (Rotation)')
J3.from_matrix(MA)

phi = arange(0, pi/4, pi/12)
# create Birefringence matrix
IB = np.zeros((2, 2, (len(phi))))
np.einsum('iij->ij', IB)[:] = 1
Bexp = exp(1j*np.vstack((phi, -phi)))
MB = einsum('ijk, ...ik -> ijk', IB, Bexp)   # Birefringence matrix

J4 = create_Jones_matrices('Changing basis')
J4.from_matrix(MB)

[lambda1, lambda2, V1, V2] = J4.parameters.eig()
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

MQ = np.dstack((V1.T, V2.T))  # create (len(theta),2,2) array to use np.linalg.inv
MQ_I = np.linalg.inv(MQ)      # np.linalg.inv works only in (N,M,M) form
ML = np.dstack((lambda1, lambda2))

MR = einsum('...ik, ...k, ...kj -> ...ij', MQ, ML, MQ_I).reshape(len(phi), 2, 2)
# (len(theta),2,2) (1, len(theta), 2), len(theta),2,2) --> len(theta),2,2
print(MR[0])        # (len(theta),2,2)
print(MB[..., 0])    # (2,2,len(theta))

J5 = create_Jones_matrices('Basis')
J6 = create_Jones_matrices('Basis')
E.linear_light(azimuth=0)

J5.from_matrix(MQ[2])
J6.from_matrix(MQ_I[2])

# Vout = J6 * J3 * J5 * J4[nn] * E[0]

for nn, vv in enumerate(phi):

    J5.from_matrix(MQ[nn])
    J6.from_matrix(MQ_I[nn])
    J5.from_matrix(np.array([[cos(pi/4), sin(pi/4)], [-sin(pi/4), cos(pi/4)]]))
    V_out = J3 * J4[nn] * J5 * E[0]
    S.from_Jones(V_out)

    if nn == 0:
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[33*pi/180, 53*pi/180], kind='line', color_line='b')
    else:
        draw_stokes_points(fig[0], S, kind='line', color_line='b')

# Case4-2
# inducing retadation in the initial basis
