from scipy import optimize
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

def f(x):
    return abs(x[0] + x[1] + x[0]*x[1] - 50)

minimum = optimize.fmin(f, [1,2], initial_simplex=np.array([[10, 2],[-1,4],[0,-5]]), retall=True)
print(minimum[1])