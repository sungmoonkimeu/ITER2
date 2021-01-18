import time

import numpy as np
from numpy import cos, sin
from multiprocessing import Process, Queue, Manager,Lock
import os

#Jdic.clear()
#시작 시간

start_time = time.time()

# Matrix multiplication & result sharing with dictionary
def Mat_MUL_JD(alpha, beta, gamma, dq, delta_L, num, q0, n_L, Jdic):
    q = q0
    J = np.array([[1,0],[0,1]])

    for kk in range(n_L):
        q = q + dq * delta_L

        J11 = alpha + 1j * beta * cos(2 * q)
        J12 = -gamma + 1j * beta * sin(2 * q)
        J21 = gamma + 1j * beta * sin(2 * q)
        J22 = alpha - 1j * beta * cos(2 * q)
        J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J

    Jdic[num] = J
    proc = os.getpid()
    print(num,"J=",J, "by process id: ",proc)

# 멀티쓰레드(멀티프로세싱) 사용

if __name__ == '__main__':

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

    procs = []
    manager = Manager()
    Jdic = manager.dict( )

    rho = rho_C + rho_F * V_I
    delta_Beta = 2 * (rho ** 2 + (delta ** 2) / 4) ** 0.5

    alpha = cos(delta_Beta / 2 * delta_L)
    beta_= delta / delta_Beta * sin(delta_Beta / 2 * delta_L)
    gamma = 2 * rho / delta_Beta * sin(delta_Beta / 2 * delta_L)

    num_processor = 4
    calnum_processor = n_L / num_processor

    num_list = arange(0,num_processor,1)

    n_L_list = zeros(num_processor)
    q0_list = zeros(num_processor)
    for nn in range(num_processor):
        n_L_list[nn] = int(n_L / num_processor)
        q0_list[nn] = q + dq*n_L_list[0]*nn
        if nn == num_processor - 1:
            n_L_list[nn] = n_L - int(n_L / num_processor) * num_processor-1


    for index, num in enumerate(num_list):

        #proc = Process(target=Mat_MUL, args=(num,qN,qV))
        proc = Process(target=Mat_MUL_JD, args=(alpha, beta, gamma, dq,delta_L,num,q0,n_L,Jdic,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    #print(qV.get(),qV.get(),qV.get(),qV.get())
    #print(qN.get(), qN.get(), qN.get(), qN.get())
    for index, num in enumerate(Jdic):
        print(Jdic[index])

    print(Jdic)
print("--- %s seconds ---" % (time.time() - start_time))

'''
p4 s= 1250025000
--- 0.27277445793151855 seconds ---
'''