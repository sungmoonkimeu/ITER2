import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt
from numpy.linalg import norm, eig, matrix_power
import concurrent.futures as cf
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes
from py_pol.jones_matrix import *

JM = Jones_matrix("QWP_112.5")
JM.quarter_waveplate(azimuth=(112.5)*degrees)
JV = Jones_vector("input")
JV.linear_light(azimuth = 22.5*degrees)

Jout = JM*JV
Jout.parameters