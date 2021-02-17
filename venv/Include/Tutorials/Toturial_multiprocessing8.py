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
import parmap
from tqdm import tqdm, tqdm_gui
#Jdic.clear()
#시작 시간
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

# Matrix multiplication & result sharing with dictionary
def Cal_Rotation(LB, LC, SR, V, Len_SF, dL, I, num, Vout_dic):
    start_time = time.time()

    delta = 2*pi/LB
    rho_C = 2*pi/LC
    #rho_F = V * (1 + 8.1e-5 * Temp_SF[mm]) * 4 * pi * 1e-7 / (Len_SF * I)
    rho_F = V * 4 * pi * 1e-7 / (Len_SF)
    dq = 2*pi/SR

    V_in = np.array([[1],[0]])
    V_L = arange(dL,Len_SF+dL,dL)
    V_out = np.einsum('...i,jk->ijk', ones(len(I))*1j, np.array([[0], [0]]))
    # ones*1j <-- for type casting

    # ------------------------------ Variable forward--------------

    rho_1 = rho_C + rho_F * I
    delta_Beta_1 = 2 * (rho_1 ** 2 + (delta ** 2) / 4) ** 0.5

    alpha_1 = cos(delta_Beta_1 / 2 * dL)
    beta_1 = delta / delta_Beta_1 * sin(delta_Beta_1 / 2 * dL)
    gamma_1 = 2 * rho_1 / delta_Beta_1 * sin(delta_Beta_1 / 2 * dL)

    # ------------------------------ Variable backward--------------
    rho_2 = rho_C - rho_F * I
    delta_Beta_2 = 2 * (rho_2 ** 2 + (delta ** 2) / 4) ** 0.5

    alpha_2 = cos(delta_Beta_2 / 2 * dL)
    beta_2 = delta / delta_Beta_2 * sin(delta_Beta_2 / 2 * dL)
    gamma_2 = 2 * rho_2 / delta_Beta_2 * sin(delta_Beta_2 / 2 * dL)

    J0 = np.array([[1, 0], [0, 1]])
    JT0 = np.array([[1, 0], [0, 1]])
    JF = np.array([[0, 1], [-1, 0]])

    q0 = 0
    for nn in tqdm(range(len(I)),"ALL", ncols = 80, position = 0):
        q= q0
        J = J0
        JT= JT0
        for kk in tqdm(range(len(V_L)),"sub", ncols= 80, position = 1, leave = False):
            q = q + dq * dL

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J = np.array([[J11, J12],[J21, J22]]) @ J

            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JT = JT @ np.array([[J11, J21],[J12, J22]])

        V_out[nn] = JT @ JF @ J @ V_in
        #print("---  %s seconds for %s A ---" % (time.time() - start_time, I[nn]))
    #print("---  %s seconds for total time---" % (time.time() - start_time))

    Vout_dic[num] = V_out

    #proc = os.getpid()
    #print(num,"J=",J, "by process id: ",proc, ", ", len(V_q), "times calcuation")
    #print(num, "JT=", JT, "by process id: ", proc, ", ", len(V_q), "times calcuation")

#start_time = time.time()

if __name__ == '__main__':
    num_processor = 8
    LB = [0.03042]
    SR = [0.003]
    LC = 1*2*pi* 1000000000000

    #V = 0.54*(1+8.1e-5*Temp_SF[mm])
    V = 0.54
    Len_SF = 28
    dL = 0.00003
    V_I = arange(0.1e6, 2e6, 0.1e6)
    # V_I = 0.1e6

    spl_I = np.array_split(V_I, num_processor)

    procs = []
    manager = Manager()
    Vout_dic = manager.dict()

    #abs_error = zeros([11, len(V_I)])
    #rel_error = zeros([11, len(V_I)])

    abs_error = zeros(len(V_I))
    rel_error = zeros(len(V_I))

    #start_time = time.time()
    for num in range(num_processor):
        proc = Process(target=Cal_Rotation,
                       args=(LB[0], LC, SR[0], V, Len_SF, dL, spl_I[num], num, Vout_dic))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    Vout = Vout_dic[0]
    #print(Vout_dic)
    for kk in range(num_processor-1):
        Vout = np.vstack((Vout, Vout_dic[kk+1]))
    #print(Vout)

    E = Jones_vector('Output')
    E.from_matrix(M=Vout)
    V_ang = zeros(len(V_I))
    Ip = zeros(len(V_I))
    m = 0
    for kk in range(len(V_I)):
        if kk>2 and E[kk].parameters.azimuth()+m*pi - V_ang[kk-1] < -pi*0.8:
            m= m+1
        V_ang[kk] = E[kk].parameters.azimuth()+m*pi
        Ip[kk] = (V_ang[kk] - pi/2)/(2*V*4*pi*1e-7)
        #abs_error[mm,nn] = abs(Ip[nn]-V_I[nn])
        #rel_error[mm,nn] = abs_error[mm,nn]/V_I[nn]
        abs_error[kk] = abs(Ip[kk] - V_I[kk])
        rel_error[kk] = abs_error[kk] / V_I[kk]

    ## Requirement specificaion for ITER
    absErrorlimit = zeros(len(V_I))
    relErrorlimit = zeros(len(V_I))

    # Calcuation ITER specification
    for nn in range(len(V_I)):
        if V_I[nn] < 1e6:
            absErrorlimit[nn] = 10e3
        else:
            absErrorlimit[nn] = V_I[nn] * 0.01
        relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

    ## Ploting graph

    fig, ax = plt.subplots(2, 1)
    #for i in range(len(Len_LF)):
    ax[0].plot(V_I, abs_error, lw='1')
    ax[0].plot(V_I, absErrorlimit, 'r', label='ITER specification', lw='1')
    ax[0].legend(loc="upper right")

    ax[0].set_xlabel('Plasma current (A)')
    ax[0].set_ylabel('Absolute error on Ip(A)')
    ax[0].set(xlim=(0, 18e6), ylim=(0, 5e5))
    ax[0].yaxis.set_major_formatter(OOMFormatter(5, "%1.0f"))
    ax[0].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    ax[0].ticklabel_format(axis='both', style='sci', useMathText=True, scilimits=(-3, 5))
    ax[0].grid(ls='--', lw=0.5)
    # ax = plt.axes()
    #for i in range(len(Len_LF)):
    ax[1].plot(V_I, rel_error, lw='1')
    ax[1].plot(V_I, relErrorlimit, 'r', label='ITER specification', lw='1')
    ax[1].legend(loc="upper right")

    ax[1].set_xlabel('Plasma current (A)')
    ax[1].set_ylabel('Relative error on Ip(A)')
    # ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # plt.title('Output power vs Plasma current')
    ax[1].set(xlim=(0, 18e6), ylim=(0, 0.12))
    ax[1].xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    ax[1].ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    ax[1].grid(ls='--', lw=0.5)

    fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0.4, right=0.95)
    plt.show()
