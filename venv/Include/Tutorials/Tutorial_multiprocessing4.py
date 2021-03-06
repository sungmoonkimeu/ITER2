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
def Mat_MUL_JD(alpha, beta, gamma, q0, dq, n_Vq, num, Jdic):
    J = np.array([[1,0],[0,1]])
    q=q0
    for kk in range(n_Vq):
        q = q+dq*kk
        '''
        J11 = alpha + 1j * beta * cos(2 * q)
        J12 = -gamma + 1j * beta * sin(2 * q)
        J21 = gamma + 1j * beta * sin(2 * q)
        J22 = alpha - 1j * beta * cos(2 * q)
        J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J
        '''
        J = np.array([[alpha + 1j * beta * cos(2 * q), -gamma + 1j * beta * sin(2 * q)], [gamma + 1j * beta * sin(2 * q), alpha - 1j * beta * cos(2 * q)]]) @J

    Jdic[num] = J
    proc = os.getpid()
    #print(num,"J=",J, "by process id: ",proc, ", ", len(V_q), "times calcuation")

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

alpha_1 = cos(delta_Beta / 2 * delta_L)
beta_1= delta / delta_Beta * sin(delta_Beta / 2 * delta_L)
gamma_1 = 2 * rho / delta_Beta * sin(delta_Beta / 2 * delta_L)

rho = rho_C - rho_F * V_I
delta_Beta = 2 * (rho ** 2 + (delta ** 2) / 4) ** 0.5

alpha_2 = cos(delta_Beta / 2 * delta_L)
beta_2 = delta / delta_Beta * sin(delta_Beta / 2 * delta_L)
gamma_2 = 2 * rho / delta_Beta * sin(delta_Beta / 2 * delta_L)

num_processor = 4
num_list = arange(0,num_processor,1)

V_L = arange(delta_L, Len_SF + delta_L, delta_L)
V_q = dq * V_L
spl_V_q = np.array_split(V_q,num_processor)
start_time = time.time()

if __name__ == '__main__':
    procs1 = []
    procs2 = []
    manager = Manager()
    Jdic = manager.dict()
    JTdic = manager.dict()

    for num in range(num_processor):
        proc1 = Process(target=Mat_MUL_JD, args=(alpha_1, beta_1, gamma_1, spl_V_q[num][0],dq,len(spl_V_q[num]),num,Jdic,))
        procs1.append(proc1)
        proc1.start()
        proc2 = Process(target=Mat_MUL_JD, args=(alpha_2, beta_2, gamma_2, spl_V_q[num][-1], -dq, len(spl_V_q[num]), num, JTdic,))
        procs2.append(proc2)
        proc2.start()

    for proc1 in procs1:
        proc1.join()

    for proc2 in procs2:
        proc2.join()



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

        J11 = alpha_1 + 1j * beta_1 * cos(2 * q)
        J12 = -gamma_1 + 1j * beta_1 * sin(2 * q)
        J21 = gamma_1 + 1j * beta_1 * sin(2 * q)
        J22 = alpha_1 - 1j * beta_1 * cos(2 * q)
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