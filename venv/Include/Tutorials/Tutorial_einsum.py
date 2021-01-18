import numpy as np
from numpy import pi,mat, arange, append, zeros, flip, cos, sin, sqrt, concatenate, arctan, arccos, array, sum
from math import atan, tan
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

A = arange(-5,5,0.1)
M1 = mat([[1,2],[3,4]])
M2 = mat([[5,6],[7,8]])
M3 = mat([[1,0],[0,1]])
M4 = mat([1,2])
print(M1)
Out = np.einsum('ij,jk',M1,M2) #matrix multiplication
print(np.einsum('ij,...j',M1,M4)) #2x2 * 2x1 multiplication

"""
print(Out)
Out2 = np.einsum('ji',M1) #transpose
print(Out2)
Out3 = np.einsum('ij,jh',M1,M2) #matrix multiplication & transpose
print(Out3)
#Column sum --> the output is 1XN row matrix
print(np.einsum('ij->j',M1))
#Row sum --> the output is 1XN row matrix
print(np.einsum('ij->i',M1))
#test
print("test",np.einsum('jk->k',M1))
#test
A = np.ones(5)*3
print("Lamingeinsum 1",np.einsum('...i,jk->ijk',A,M1))

A = arange(1,6,1)*5
B = arange(5,0,-1)/5
print(A, B)
print("simple multiplication", A*B)
print("multi using einsum i,j->ij",np.einsum('i,j->ij',A,B))
print("multi using einsum i,j->ji",np.einsum('i,j->ji',A,B))
"""