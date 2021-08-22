import numpy as np
from multiprocessing import Process, Queue, Manager,Lock

import cupy as cp
import time
from numba import jit


def test(mm):
    N = np.array([[1,0],[0,1]])
    for nn in range(mm):
        n11 = 1*mm
        n12 = 0
        n21 = 0
        n22 = 1
        N = N @ np.array([[n11, n12], [n21, n22]])

def test2(mm):
    A = np.arange(mm)
    G = np.einsum('...i,jk->ijk', A, np.mat([[1, 0], [0, 1]]))
    N = np.array([[1,0],[0,1]])

    for nn in range(mm):
       N =N@G[nn]

if __name__ == '__main__':
    n_iter = int(1000000)
    start = time.time()
    test(n_iter)
    print(time.time() - start)

    start = time.time()
    test2(n_iter)
    print(time.time() - start)

