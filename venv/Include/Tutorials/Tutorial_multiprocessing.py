import time

import numpy as np
from multiprocessing import Process
import os

#시작 시간


def count(name):
    s = 0
    for i in range(1,100000010):
        #print(name,":", i)
        s = s+i
    proc = os.getpid()
    print(name,"s=",s, "by process id: ",proc)


# 멀티쓰레드(멀티프로세싱) 사용
start_time = time.time()

if __name__ == '__main__':
    num_list = ['p1', 'p2', 'p3', 'p4']
    procs = []

    for index, name in enumerate(num_list):
        proc = Process(target=count, args=(name,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    s = 0
    for i in range(1, 4 * 100000010):
        # print(name,":", i)
        s = s + i

    print("s=", s)
    print("--- %s seconds ---" % (time.time() - start_time))



'''
p4 s= 1250025000
--- 0.27277445793151855 seconds ---
'''