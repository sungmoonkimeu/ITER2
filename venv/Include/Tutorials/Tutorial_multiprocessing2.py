import time

import numpy as np
from multiprocessing import Process, Queue, Manager,Lock
import os

#Jdic.clear()
#시작 시간

start_time = time.time()
def Mat_MUL(num, qN,qV):
    s = 0
    J = np.array([[1,0],[0,num]])
    for i in range(1,5):
        #print(name,":", i)
        J = np.vstack((1,0,0,num)).T.reshape(2,2)@J
    proc = os.getpid()
    qN.put(num)
    qV.put(J)
    print(num,"J=",J, "by process id: ",proc)

# Matrix multiplication & result sharing with dictionary
def Mat_MUL_d(num, Jdic):
    s = 0
    J = np.array([[1,0],[0,num]])
    for i in range(1,5):
        #print(name,":", i)
        J = np.vstack((1,0,0,num)).T.reshape(2,2)@J
    proc = os.getpid()
    Jdic[num] = J
    print(num,"J=",J, "by process id: ",proc)

def Mat_MUL_JD(alpha, beta, gamma, dq, delta_L, num, q0, n_L, Jdic):
    q = q0
    for kk in range(n_L):
        q = q + dq * delta_L

        J11 = alpha + 1j * beta * cos(2 * q)
        J12 = -gamma + 1j * beta * sin(2 * q)
        J21 = gamma + 1j * beta * sin(2 * q)
        J22 = alpha - 1j * beta * cos(2 * q)
        J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J

    Jdic[num] = J
    #proc = os.getpid()
    print(num,"J=",J, "by process id: ",proc)

# 멀티쓰레드(멀티프로세싱) 사용

if __name__ == '__main__':

    num_list = [1, 2, 3, 4]
    procs = []
    #qV = Queue()
    #qN = Queue()
    manager = Manager()
    Jdic = manager.dict( )
    print(Jdic)

    for index, num in enumerate(num_list):

        #proc = Process(target=Mat_MUL, args=(num,qN,qV))
        proc = Process(target=Mat_MUL_d, args=(num, Jdic,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    #print(qV.get(),qV.get(),qV.get(),qV.get())
    #print(qN.get(), qN.get(), qN.get(), qN.get())
    for index, num in enumerate(Jdic):
        print(Jdic[index+1])

    print(Jdic)
print("--- %s seconds ---" % (time.time() - start_time))

'''
p4 s= 1250025000
--- 0.27277445793151855 seconds ---
'''