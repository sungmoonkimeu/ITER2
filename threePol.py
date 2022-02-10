import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt

th0 = 0:0.1: pi;
phi0 = 0:0.1: pi;

Mcir_i = [cos(th0), -sin(th0); sin(th0), cos(th0)]
Mbi_i = [exp(1i * phi0) 0; 0 exp(-1i * phi0)]

th1 = 0.1
phi1 = 0.1

Mcir = [cos(th1) - sin(th1); sin(th1) cos(th1)]
Mbi = [exp(1i * phi1) 0; 0 exp(-1i * phi1)]

for nn = 1:length(th0)
ang_th = th0(nn);
ang_phi = phi0(nn);
Mcir_i = [cos(ang) - sin(ang); sin(ang) cos(ang)];
Mbi_i = [exp(1i * ang_phi) 0;0 exp(-1i * ang_phi)];

LP0_i = Mbi_i * Mcir_i * [1;0]
LP45_i = Mbi_i * Mcir_i * [1; 1]
RCP_i = Mbi_i * Mcir_i * [1; 1i]

LP0_o = Mbi * Mcir * Mbi_i * Mcir_i * [1;0]
LP45_o = Mbi * Mcir * Mbi_i * Mcir_i * [1;1]
RCP_o = Mbi * Mcir * Mbi_i * Mcir_i * [1;1i]

end