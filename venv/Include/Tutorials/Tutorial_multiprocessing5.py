import time

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import numpy as np
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt
from numpy.linalg import norm, eig, matrix_power
import concurrent.futures as cf
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes

from multiprocessing import Process, Queue, Manager,Lock
import os

#Jdic.clear()
#시작 시간


# Matrix multiplication & result sharing with dictionary
def Mat_MUL_JD(alpha, beta, gamma, q0, dq, n_Vq, num, Jdic, JTdic):
    J = np.array([[1,0],[0,1]])
    JT = np.array([[1,0],[0,1]])
    q=q0
    qt = q0+dq*(n_Vq-1)
    for kk in range(n_Vq):
        q = q+dq*kk
        '''
        J11 = alpha + 1j * beta * cos(2 * q)
        J12 = -gamma + 1j * beta * sin(2 * q)
        J21 = gamma + 1j * beta * sin(2 * q)
        J22 = alpha - 1j * beta * cos(2 * q)
        J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J
        '''
        J = np.array([[alpha[0] + 1j * beta[0] * cos(2 * q), -gamma[0] + 1j * beta[0] * sin(2 * q)],
                      [gamma[0] + 1j * beta[0] * sin(2 * q), alpha[0] - 1j * beta[0] * cos(2 * q)]]) @ J


        qt = qt-dq*kk
        JT = np.array([[alpha[1] + 1j * beta[1] * cos(2 * qt), -gamma[1] + 1j * beta[1] * sin(2 * qt)],
                      [gamma[1] + 1j * beta[1] * sin(2 * qt), alpha[1] - 1j * beta[1] * cos(2 * qt)]]) @ JT

    Jdic[num] = J
    JTdic[num] = JT
    proc = os.getpid()
    #print(num,"J=",J, "by process id: ",proc, ", ", len(V_q), "times calcuation")
    #print(num, "JT=", JT, "by process id: ", proc, ", ", len(V_q), "times calcuation")

# 멀티쓰레드(멀티프로세싱) 사용

LB = [0.03042]
SR = [0.003]

#V_I = arange(0.1e6, 17e6, 0.1e6)
V_I = 0.1e6

delta = 2 * pi / LB[0]  # Linear birefringence [rad/m]
LC = 1 * 2 * pi * 10000000000000000  # Reciprocal circular beatlength [m]
rho_C = 2 * pi / LC  # Reciprocal circular birefringence [rad/m]
Len_SF = 28  # length of sensing fiber 28 m
I = 1  # Applied plasma current 1A for normalization
V = 0.54  # Verdat constant 0.54 but in here 0.43
rho_F = V * 4 * pi * 1e-7 / (Len_SF * I)  # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]
delta_L = 0.00003  # delta L [m]
dq = 2 * pi / SR[0]
q = 0
#_______________________________Parameters#2____________________________________



rho = rho_C + rho_F * V_I
delta_Beta = 2 * (rho ** 2 + (delta ** 2) / 4) ** 0.5
alpha = [0,0]
beta = [0,0]
gamma = [0,0]

alpha[0] = cos(delta_Beta / 2 * delta_L)
beta[0]= delta / delta_Beta * sin(delta_Beta / 2 * delta_L)
gamma[0] = 2 * rho / delta_Beta * sin(delta_Beta / 2 * delta_L)

rho = rho_C - rho_F * V_I
delta_Beta = 2 * (rho ** 2 + (delta ** 2) / 4) ** 0.5

alpha[1] = cos(delta_Beta / 2 * delta_L)
beta[1] = delta / delta_Beta * sin(delta_Beta / 2 * delta_L)
gamma[1] = 2 * rho / delta_Beta * sin(delta_Beta / 2 * delta_L)

num_processor = 8
num_list = arange(0,num_processor,1)

V_L = arange(delta_L, Len_SF + delta_L, delta_L)
V_q = dq * V_L
spl_V_q = np.array_split(V_q,num_processor)
start_time = time.time()

if __name__ == '__main__':
    procs = []
    manager = Manager()
    Jdic = manager.dict()
    JTdic = manager.dict()

    for num in range(num_processor):
        proc = Process(target=Mat_MUL_JD,
                       args=(alpha, beta, gamma, spl_V_q[num][0], dq, len(spl_V_q[num]), num, Jdic, JTdic,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    J = np.array([[1,0],[0,1]])
    JT = np.array([[1,0],[0,1]])
    for kk in range(num_processor):
        J = Jdic[kk] @ J
        JT = JTdic[kk] @ JT
    #print(Jdic)
    print("--- %s seconds for multiprocessing---" % (time.time() - start_time))

    start_time = time.time()
    J2 = np.array([[1,0],[0,1]])
    for kk in range(len(V_q)):
        q = V_q[kk]

        J11 = alpha[0] + 1j * beta[0] * cos(2 * q)
        J12 = -gamma[0] + 1j * beta[0] * sin(2 * q)
        J21 = gamma[0] + 1j * beta[0] * sin(2 * q)
        J22 = alpha[0] - 1j * beta[0] * cos(2 * q)
        #J2 = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J2

        J2 = np.array([[J11, J12],
                       [J21, J22]]) @ J2

        #J2 = np.array([[alpha + 1j * beta * cos(2 * q), -gamma + 1j * beta * sin(2 * q)], [gamma + 1j * beta * sin(2 * q), alpha - 1j * beta * cos(2 * q)]]) @ J2
    print("--- %s seconds for singleprocessing---" % (time.time() - start_time))
    print("J = ", J)
    print("J2 = ", J2)

'''
p4 s= 1250025000
--- 0.27277445793151855 seconds ---
'''