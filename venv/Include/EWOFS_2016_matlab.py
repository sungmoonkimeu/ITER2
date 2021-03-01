import time

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import numpy as np
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt, arccos
from numpy.linalg import norm, eig, matrix_power
import concurrent.futures as cf
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes

from multiprocessing import Process, Queue, Manager,Lock
import os

IP_debut        = 0e3
IP_pas          = 5e5
IP_fin          = 17e6
division        = 1  # precision from 0->1MA
IP_0_to_1       = arange(IP_debut,1e6+(IP_pas/division),(IP_pas/division))
IP_1_to_17      = arange(1e6+IP_pas,IP_fin+IP_pas, IP_pas)
IP_vector       = np.hstack((IP_0_to_1,IP_1_to_17))

l = 28 # the lenght of fiber
R = l / 2 / pi # the radius
lb = 0.0304 # the beat length[m]
sp = 0.003 # spinning period[m]
f = lb / sp # the ratio
tautwist = 0 # it variate between[0, 209.3 rad / m]

# declaring matrix vin elements
a = 0 # a is the azimuth
chhi = 0 # chi is the ellipcity

# matrix input vin
Vin = np.array([[cos(a) * cos(chhi) - 1j * sin(a) * sin(chhi)],
                [sin(a) * cos(chhi) + 1j * cos(a) * sin(chhi)]])

# lr=min(lb,sp)/10

if (lb<sp):
    lr=lb/10
else:
    lr=sp/10  #in this case lr=0,00304

deltai=(2*pi)/lb
tau=1/sp

# declarating the elemets and operation for rhoi: rhoi=rhoci+rhofi
g=0.15
rhoci=g*tautwist #reciprocal circular birefringence
V0=0.54 #the Verdet constant
mu0=4*pi*1e-7

m=1
_lambda=1550
Lb0=lb
deltan0=_lambda/lb
delta0=(2*pi)/_lambda*deltan0

deltaT = 90
Lb = Lb0 + 0.03 * 1e-3 * deltaT
deltaB = (2 * pi) / Lb
V = V0 + 0.81 * 1e-4 * V0 * deltaT
K = 0

phi_calc1= zeros(len(IP_vector))


for c in range (len(IP_vector)):
    IP = IP_vector[c]
    B = (mu0 * IP) / (2 * pi * R)
    rhofi = V * B # nonreciprocal Faraday induced circular birefringence
    rhoi = rhoci + rhofi #  total circular birefringence

    deltai = deltaB
    DELTAi = sqrt(rhoi** 2 + deltai** 2 / 4) # the phase delay
    #q = 0:tau * lr * 2 * pi: tau * 2 * pi * l # q is the parameter which is modified in matrix jones so it will be modified in for loop
    q = arange(0,tau * 2 * pi * l,tau * lr * 2 * pi)
    alpha = cos(DELTAi * lr)
    beta = (deltai / 2) * (sin(DELTAi * lr) / DELTAi)
    gamma = rhoi * (sin(DELTAi * lr) / DELTAi)
    Jt = np.array([[alpha + 1j * beta * cos(2 * q[0]), -gamma + 1j * beta * sin(2 * q[0])],
                   [gamma + 1j * beta * sin(2 * q[0]), alpha - 1j * beta * cos(2 * q[0])]])

    # implementation of optical fiber
    # the first loop is for the case when we are going in the right direction

    # original code in matalb - for i=2:l / lr.
    # In matlab, if (l/lr = 10), i = 2:10
    # code in python for i in range(2:10) --> i = 2:9

    for i in range (1,np.int(l / lr)+1):
        Ji = np.array([[alpha + 1j * beta * cos(2 * q[i]), -gamma + 1j * beta * sin(2 * q[i])],
                       [gamma + 1j * beta * sin(2 * q[i]), alpha - 1j * beta * cos(2 * q[i])]])
        Jt = Ji @ Jt

    # calculation the last fiber part
    n = np.int(l / lr) # n is the number without commas
    ln = l - n * lr # The length for the smallest part
    qn = q[n] + tau * lr * 2 * pi

    # parameters for the smallest part
    alpha = cos(DELTAi * ln)
    beta = (deltai / 2) * (sin(DELTAi * ln) / DELTAi)
    gamma = rhoi * (sin(DELTAi * ln) / DELTAi)

    Jn = np.array([[alpha + 1j * beta * cos(2 * qn), -gamma + 1j * beta * sin(2 * qn)],
                  [gamma + 1j * beta * sin(2 * qn), alpha - 1j * beta * cos(2 * qn)]])
    Jt = Jn @ Jt

    # Jones matrix for Faraday mirror
    Jf = np.array([[0, -1],[1, 0]])
    Jt = Jf @ Jt

    rhoj = -rhoci + rhofi
    DELTAj = sqrt(rhoj**2 + (deltai** 2) / 4)

    alpha = cos(DELTAj * ln)
    beta = (deltai / 2) * (sin(DELTAj * ln) / DELTAj)
    gamma = rhoj * (sin(DELTAj * ln) / DELTAj)

    Jn = np.array([[alpha + 1j * beta * cos(2 * qn), -gamma + 1j * beta * sin(2 * qn)],
                   [gamma + 1j * beta * sin(2 * qn), alpha - 1j * beta * cos(2 * qn)]])
    Jt = Jn @ Jt

    # the second loop is for the case when we are going in the left direction

    for i in range(np.int(l / lr), -1, -1):
        # declaring matrix Jones elements
        alphaj = cos(DELTAj * lr)
        betaj = (deltai / 2) * (sin(DELTAj * lr) / DELTAj)
        gammaj = rhoj * (sin(DELTAj * lr) / DELTAj)

        Ji = np.array([[alphaj + 1j * betaj * cos(2 * q[i]), -gammaj + 1j * betaj * sin(2 * q[i])],
                       [gammaj + 1j * betaj * sin(2 * q[i]), alphaj - 1j * betaj * cos(2 * q[i])]])
        Jt = Ji @ Jt

    Vout = Jt @ Vin

    # the second methot with Stokes
    S0 = abs(Vout[0]) * abs(Vout[0]) + abs(Vout[1]) * abs(Vout[1])
    S1 = abs(Vout[0]) * abs(Vout[0]) - abs(Vout[1]) * abs(Vout[1])
    phix = np.angle(Vout[0])
    phiy = np.angle(Vout[1])
    deltaphi = phiy - phix
    S2 = 2 * abs(Vout[0]) * abs(Vout[1]) * cos(deltaphi)
    S3 = 2 * abs(Vout[0]) * abs(Vout[1]) * sin(deltaphi)
    S1 = S1 / S0
    S2 = S2 / S0
    S3 = S3 / S0
    cos2chi = sqrt(S1 ** 2 + S2 ** 2)
    cos2phi = S1 / cos2chi
    sin2phi = S2 / cos2chi
    phi2c = arccos(cos2phi)

    if sin2phi < 0:
        phi2 = -phi2c
    else:
        phi2 = phi2c

    phi_calc1[K] = phi2
    K = K + 1
    print(K,"/",len(IP_vector), " done ")

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(IP_vector,phi_calc1)

V_I = loadtxt('result_fromMat2.txt',unpack=True)
Data = loadtxt('result_fromMat.txt',unpack=True)
#DataIN = loadtxt('result_fromMat.txt',unpack=True, usecols=[1,2,3,4,5])

## Ploting graph
ax.plot(V_I,Data,lw='1')

plt.show()
