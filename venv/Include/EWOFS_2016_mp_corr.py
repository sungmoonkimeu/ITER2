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
def Cal_Rotation(LB, LC, SR, V, Len_SF, dL, I, num, Vout_dic):
    start_time = time.time()

    delta = 2*pi/LB
    rho_C = 2*pi/LC
    #rho_F = V * (1 + 8.1e-5 * Temp_SF[mm]) * 4 * pi * 1e-7 / (Len_SF * I)
    rho_F = V * 4 * pi * 1e-7 / (Len_SF)
    dq = 2*pi/SR

    V_in = np.array([[1],[0]])
    V_L = arange(dL,Len_SF+dL,dL)
    V_q_L = V_L * dq
    V_out = np.einsum('...i,jk->ijk', ones(len(I))*1j, np.array([[0], [0]]))
    # ones*1j <-- for type casting

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

    J0 = np.array([[1, 0], [0, 1]])
    JT0 = np.array([[1, 0], [0, 1]])
    JF = np.array([[0, -1], [1, 0]])

    #q0 = 0
    for nn in range(len(I)):
        #q= q0
        J = J0
        JT= JT0
        for kk in range(len(V_L)):
            #q = q + dq * dL
            q = V_q_L[kk]

            J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
            J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
            J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)
            J = np.array([[J11, J12],[J21, J22]]) @ J

            J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
            J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
            J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)
            JT = JT @ np.array([[J11, J12],[J21, J22]])

        V_out[nn] = JT @ JF @ J @ V_in
        #print("---  %s seconds for %s A ---" % (time.time() - start_time, I[nn]))
    print("---  %s seconds for 1 process---" % (time.time() - start_time))

    Vout_dic[num] = V_out

    #proc = os.getpid()
    #print(num,"J=",J, "by process id: ",proc, ", ", len(V_q), "times calcuation")
    #print(num, "JT=", JT, "by process id: ", proc, ", ", len(V_q), "times calcuation")

#start_time = time.time()

if __name__ == '__main__':
    num_processor = 16
    LB = [0.0304]
    SR = [0.003]
    LC = 1*2*pi* 1000000000000
    Temp_SF = arange(90,110+5,10)
    #Temp_SF = arange(90,91,10)
    V = 0.54*(1+8.1e-5*Temp_SF)
    V0 = 0.54
    #V = 0.54
    Len_SF = 28
    dL = SR[0]/10
    V_I = arange(0.1e6, 17e6+0.2e6, 0.2e6)
    # V_I = 0.1e6

    spl_I = np.array_split(V_I, num_processor)

    f = open('EWOFS_fig3_saved_corr.txt', 'w')
    savetxt(f, V_I, newline="\t")
    f.write("\n")
    f.close()

    procs = []
    manager = Manager()
    Vout_dic = manager.dict()

    abs_error = zeros([len(Temp_SF), len(V_I)])
    rel_error = zeros([len(Temp_SF), len(V_I)])

    #start_time = time.time()
    f = open('EWOFS_fig3_saved_corr.txt', 'a')

    for mm in range(len(Temp_SF)):
        for num in range(num_processor):
            LB_t = LB[0]+0.03e-3*Temp_SF[mm]
            proc = Process(target=Cal_Rotation,
                           args=(LB_t, LC, SR[0], V[mm], Len_SF, dL, spl_I[num], num, Vout_dic))
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
        '''
        for kk in range(len(V_I)):
            V_ang[kk] = E[kk].parameters.azimuth()
        V_ang = np.unwrap(V_ang)
        '''
        for kk in range(len(V_I)):

            if kk>2 and E[kk].parameters.azimuth()+m*pi - V_ang[kk-1] < -pi*0.8:
                m= m+1
            V_ang[kk] = E[kk].parameters.azimuth()+m*pi
            Ip[kk] = (V_ang[kk] - pi/2)/(2*V0*4*pi*1e-7)
            abs_error[mm,kk] = abs(Ip[kk]-V_I[kk])
            rel_error[mm,kk] = abs_error[mm,kk]/V_I[kk]
            #abs_error[kk] = abs(Ip[kk] - V_I[kk])
            #rel_error[kk] = abs_error[kk] / V_I[kk]

        print(" %s degC was calcualated, (%s / %s)" % (Temp_SF[mm], mm,len(Temp_SF)))
        savetxt(f, rel_error[mm], newline="\t")
        f.write("\n")

    f.close()
    #Dataout = column_stack((V_I, rel_error[0:-1, :].T))
    #savetxt('EWOFS_fig3_saved4.dat', Dataout)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = V_I
    Y = Temp_SF
    X, Y = np.meshgrid(X, Y)
    Z = rel_error

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
