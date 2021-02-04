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
def Cal_Rotation(LB_lf, LB_sf, LC, SR, V, Len_LF, Len_SF,  dL, I, num, Vout_dic):
    start_time = time.time()

    delta_lf = 2*pi/LB_lf

    delta = 2*pi/LB_sf
    rho_C = 2*pi/LC
    #rho_F = V * (1 + 8.1e-5 * Temp_SF[mm]) * 4 * pi * 1e-7 / (Len_SF * I)
    rho_F = V * 4 * pi * 1e-7 / (Len_SF)
    dq = 2*pi/SR

    V_in = np.array([[1],[0]])
    V_L = arange(dL,Len_SF+dL,dL)
    V_LF = arange(dL,Len_LF+dL,dL)
    V_out = np.einsum('...i,jk->ijk', ones(len(I))*1j, np.array([[0], [0]]))
    # ones*1j <-- for type casting

    # ------------------------------ Variable lead fiber --------------------------
    # ------------No Farday effect (rho = 0)--> (forward α, β, γ  = backward α, β, γ) ------

    alpha_lf = cos(delta_lf/2*dL)
    beta_lf = sin(delta_lf/2*dL)
    gamma_lf = 0

    # ------------------------------ Variable forward--------------
    rho_1 = rho_C + rho_F * I
    delta_Beta_1 = 2 * (rho_1 ** 2 + (delta ** 2) / 4) ** 0.5

    alpha_1 = cos(delta_Beta_1 / 2 * dL)
    beta_1 = delta / delta_Beta_1 * sin(delta_Beta_1 / 2 * dL)
    gamma_1 = 2 * rho_1 / delta_Beta_1 * sin(delta_Beta_1 / 2 * dL)

    # ------------------------------ Variable backward--------------
    rho_2 = -rho_C + rho_F * I
    delta_Beta_2 = 2 * (rho_2 ** 2 + (delta ** 2) / 4) ** 0.5

    alpha_2 = cos(delta_Beta_2 / 2 * dL)
    beta_2 = delta / delta_Beta_2 * sin(delta_Beta_2 / 2 * dL)
    gamma_2 = 2 * rho_2 / delta_Beta_2 * sin(delta_Beta_2 / 2 * dL)

    JF = np.array([[0, 1], [-1, 0]])
    for nn in range(len(I)):
        q0 = 0
        J0 = np.array([[1, 0], [0, 1]])
        for kk in range(len(V_LF)):
            q0= q0 + dq *dL

            J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
            J12 = 1j * beta_lf * sin(2 * q0)
            J21 = 1j * beta_lf * sin(2 * q0)
            J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)
            J0 = np.array([[J11, J12], [J21, J22]]) @ J0

        q = q0
        J = mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q + dq * dL

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J = np.array([[J11, J12],[J21, J22]]) @ J

        JB = mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q - dq * dL
            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JB = np.array([[J11, J12],[J21, J22]]) @    JB

        q0 = q
        JB0 = np.array([[1, 0], [0, 1]])
        for kk in range(len(V_LF)):
            q0 = q0 - dq * dL

            J11 = alpha_lf + 1j * beta_lf * cos(2 * q0)
            J12 = 1j * beta_lf * sin(2 * q0)
            J21 = 1j * beta_lf * sin(2 * q0)
            J22 = alpha_lf - 1j * beta_lf * cos(2 * q0)
            JB0 = np.array([[J11, J12], [J21, J22]]) @ JB0

        V_out[nn] = JB0 @ JB @ JF @ J @ J0 @ V_in
        print("---  %s seconds for %s A ---" % (time.time() - start_time, I[nn]))

    print("---  %s seconds for 1 process---" % (time.time() - start_time))

    Vout_dic[num] = V_out

    #proc = os.getpid()
    #print(num,"J=",J, "by process id: ",proc, ", ", len(V_q), "times calcuation")
    #print(num, "JT=", JT, "by process id: ", proc, ", ", len(V_q), "times calcuation")

#start_time = time.time()

if __name__ == '__main__':
    num_processor = 8

    LB_lf= [0.132]
    LB_sf= [0.132]
    SR = [0.03]
    LC = 1*2*pi* 1000000000000
    #Temp_SF = arange(90,110+5,5)

    V = 0.43

    Len_SF = 28
    #Len_LF = [100,100.003,100.006,100.009,100.012]
    Len_LF = [0.003, 0.006, 0.009, 0.012, 0.015]
    dL = 0.0001
    V_I = arange(0.2e6, 2e6+0.2e6, 0.2e6)
    # V_I = 0.1e6

    spl_I = np.array_split(V_I, num_processor)

    f = open('AO2015_fig16_mp.txt', 'w')
    savetxt(f, V_I, newline="\t")
    f.write("\n")
    f.close()

    procs = []
    manager = Manager()
    Vout_dic = manager.dict()

    abs_error = zeros([len(Len_LF), len(V_I)])
    rel_error = zeros([len(Len_LF), len(V_I)])

    #start_time = time.time()
    f = open('AO2015_fig16_mp.txt', 'a')

    for mm in range(len(Len_LF)):
        for num in range(num_processor):
            proc = Process(target=Cal_Rotation,
                           args=(LB_lf[0], LB_sf[0], LC, SR[0], V, Len_LF[mm], Len_SF, dL, spl_I[num], num, Vout_dic))
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
            abs_error[mm,kk] = abs(Ip[kk]-V_I[kk])
            rel_error[mm,kk] = abs_error[mm,kk]/V_I[kk]
            #abs_error[kk] = abs(Ip[kk] - V_I[kk])
            #rel_error[kk] = abs_error[kk] / V_I[kk]

        print(" %s mm was calcualated, (%s / %s)" % (Len_LF[mm], mm+1,len(Len_LF)))
        savetxt(f, rel_error[mm], newline="\t")
        f.write("\n")

    f.close()
    #Dataout = column_stack((V_I, rel_error[0:-1, :].T))
    #savetxt('EWOFS_fig3_saved4.dat', Dataout)

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
    fig, ax = plt.subplots(figsize=(6, 3))

    for i in range(len(Len_LF)):
        str_legend = str(Len_LF[i])
        print(str_legend)
        ax.plot(V_I, rel_error[i], lw='1', label=str_legend)

    ax.plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
    ax.legend(loc="upper right", prop={'family': 'monospace'})
    ax.legend(loc="upper right")

    plt.rc('text', usetex=True)
    ax.set_xlabel(r'Plasma current $I_{p}(A)$')
    ax.set_ylabel(r'Relative error on $I_{P}$')

    # plt.title('Output power vs Plasma current')
    ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

    ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    ax.grid(ls='--', lw=0.5)

    # fig.align_ylabels(ax)
    fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
    # fig.set_size_inches(6,4)
    plt.rc('text', usetex=False)


    ## Ploting graph
    fig2, ax2 = plt.subplots(figsize=(6, 3))

    for i in range(len(Len_LF)):
        str_legend = str(Len_LF[i])
        #print(str_legend)
        ax2.plot(V_I, abs_error[i], lw='1', label=str_legend)

    ax2.plot(V_I,absErrorlimit,'r', label='ITER specification',lw='1')
    # ax.legend(loc="upper right", prop={'family': 'monospace'})
    ax2.legend(loc="upper right")

    plt.rc('text', usetex=True)
    ax2.set_xlabel(r'Plasma current $I_{p}(A)$')
    ax2.set_ylabel(r'abs error on $I_{P}$')

    # plt.title('Output power vs Plasma current')
    #ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
    ax2.set(xlim=(0, 18e6))
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_locator(MaxNLocator(10))

    ax2.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    #ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

    ax2.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    ax2.grid(ls='--', lw=0.5)

    # fig.align_ylabels(ax)
    fig2.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
    # fig.set_size_inches(6,4)
    plt.rc('text', usetex=False)

    plt.show()

