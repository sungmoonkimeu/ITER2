import numpy as np
from multiprocessing import Process, Queue, Manager,Lock

import cupy as cp
import time
from numba import jit


@jit(nopython=True)
def cal_mat_nb(mm):
    mat_1 = np.array([[1, 0], [0, 1]])

    for nn in range(mm):
        mat_1 = mat_1 @ mat_1

class Test:
    def __init__(self, n_iter):
        self.nn = n_iter

    def cal_mat(self):
        mat_1 = np.array([[1, 0], [0, 1]])

        for nn in range(self.nn):
            mat_1 = mat_1 @ mat_1

    def calc_mp(self, num_processor):
        start = time.time()

        procs = []

        for num in range(num_processor):
            proc = Process(target=self.cal_mat)
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        print(time.time()-start)

    def cal_mat_cp(self, mat_1):
        for nn in range(self.nn):
            cp.matmul(mat_1, mat_1, out=None)

    def calc_mp2(self, num_processor, mm):
        start = time.time()

        procs = []

        for num in range(num_processor):
            proc = Process(target=self.cal_mat, args=mm)
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        print(time.time()-start)

    def calc_cp(self):
        start = time.time()
        mat_1 = cp.array([[1, 0], [0, 1]])
        self.cal_mat_cp(mat_1)
        cp.cuda.Device(0).synchronize()
        print(time.time() - start)

if __name__ == '__main__':
    n_iter = int(1000)
    test = Test(n_iter)
    test.calc_mp(10)
    cal_mat_nb(1)
    start = time.time()
    cal_mat_nb(n_iter)
    print(time.time() - start)

    test.calc_cp()

