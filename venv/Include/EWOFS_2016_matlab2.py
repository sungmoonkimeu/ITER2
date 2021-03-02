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

Data_pyt = loadtxt('EWOFS_fig3comp.txt',unpack=True)

IP_vector = Data_pyt.T[0,:]
phi_calc1 = Data_pyt.T[1,:]

V_I_mat = loadtxt('result_fromMat2.txt',unpack=True)
Data_mat = loadtxt('result_fromMat.txt',unpack=True)
#DataIN = loadtxt('result_fromMat.txt',unpack=True, usecols=[1,2,3,4,5])

## Ploting graph


#ax.plot(V_I_mat,Data_mat,lw='1')
fig, ax = plt.subplots(figsize=(6,3))
V0=0.54 #the Verdet constant
mu0=4*pi*1e-7

phi_unwrap = np.unwrap(phi_calc1)
phi_uns = phi_unwrap/2
if phi_uns[0] <0 :
    phi_uns = phi_uns+pi
thetas = ((phi_uns-pi/2)/2)
Ims = thetas/(V0*mu0)
Ers = (abs(Ims-IP_vector))/IP_vector*100
#ax.plot(IP_vector,phi_unwrap)
#ax.plot(IP_vector,thetas)
ax.plot(IP_vector, Ers, label='python result (Matlab code)')

phi_unwrap_mat = np.unwrap(Data_mat)
phi_uns_mat = phi_unwrap_mat/2
if phi_uns_mat[0] <0 :
    phi_uns_mat = phi_uns_mat+pi
thetas_mat = ((phi_uns_mat-pi/2)/2)
Ims_mat = thetas_mat/(V0*mu0)
Ers_mat = (abs(Ims_mat-V_I_mat))/V_I_mat*100
print(V_I_mat)
#ax.plot(IP_vector,phi_unwrap)
#ax.plot(IP_vector,thetas)
ax.plot(V_I_mat, Ers_mat, label='Matlab result')

DataIN = loadtxt('EWOFS_fig3_saved5.txt',unpack=True)
V_I = DataIN.T[0,:]
ax.plot(V_I, DataIN[:, 1] * 100, lw='1', label='previous python result(SM)')

DataIN2 = loadtxt('EWOFS_fig3_saved_corr.dat',unpack=True)
V_I = DataIN2[0,:]
ax.plot(V_I, DataIN2[1,:] * 100, lw='1', label='python result2(SM)')
ax.legend(loc="upper right")

plt.show()

