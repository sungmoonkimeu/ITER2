import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, conj
from numpy.linalg import norm

dBeta = 1e-20 #1234*pi       # delta Beta
xi = 1e-20 #1219*pi          # twist rate
I = 0           # current
D = 0.1         # coil diameter assume 10 cm
wl = 780e-9    # wavelength
f = 5.79e-19/(wl**2)*I/D    # Faraday-induced rotation angle per unit length
theta_1s = 0    # initial orientation of the local slow axis fo the fiber
Nturn = 100
N= 1000

#I = 100 # 100A
I = arange(.1,1.1,.1)
z = pi*D*Nturn
dz = z/N
Out = zeros(len(I))

M_in = sqrt(1 / 2) * mat([1, 1])
M_pol0 = mat([[0, 0], [0, 1]])
M_pol90 = mat([[1, 0], [0, 0]])
M_m = mat([[1,0],[0,-1]])


# forward calculation
f = 5.79e-19 / (wl ** 2) * I[nn] / D  # Faraday-induced rotation angle per unit length
q = 2*(xi + f)/dBeta
gamma = 1/2*(dBeta**2+4*(xi+f)**2)**0.5

Rz = 2*arcsin(1/sqrt(1+q**2)*sin(gamma*dz))
Omegaz = xi*dz+arctan(-q/sqrt(1+q**2)*tan(gamma*dz))
Phiz = (xi*dz-Omegaz)/2 + pi/2 + theta_1s

m11 = 1j * Rz / 2  * cos(2 * Phiz)
m12 = 1j * Rz / 2  * sin(2 * Phiz) - Omegaz
m21 = -1j * Rz / 2  * sin(2 * Phiz) + Omegaz
m22 = 1j * Rz / 2  * cos(2 * Phiz)

#GGG = np.einsum('...i,jk->ijk', R_z / 2, np.mat([[1, 0], [0, 1]]))
m11 = np.einsum('...i,jk->ijk',m11,mat([[1,0],[0,1]]))


# backward calculation f = -f, M_R' --> conj(M_R), M_Omega' --> conj(M_Omega)
'''
f = -f
q = 2 * (xi + f) / dBeta
gamma = 1 / 2 * (dBeta ** 2 + 4 * (xi + f) ** 2) ** 0.5

Rz = 2 * arcsin(1 / sqrt(1 + q ** 2) * sin(gamma * z))
Omegaz = xi * z + arctan(-q / sqrt(1 + q ** 2) * tan(gamma * z))
Phiz = (xi * z - Omegaz) / 2 + pi / 2 + theta_1s

m11 = 1j * Rz / 2  * cos(2 * Phiz)
m12 = 1j * Rz / 2  * sin(2 * Phiz) - Omegaz
m21 = -1j * Rz / 2  * sin(2 * Phiz) + Omegaz
m22 = 1j * Rz / 2  * cos(2 * Phiz)
'''

M_cal1 = M_pol0*M_Rp*M_Omegap*M_m*M_Omega*M_R
M_out1 = np.einsum('ij,...j',M_cal1,M_in)
M_cal2 = M_pol90*M_Rp*M_Omegap*M_m*M_Omega*M_R
M_out2 = np.einsum('ij,...j', M_cal2, M_in)
Out[nn] = (norm(M_out1)**2-norm(M_out2)**2)/(norm(M_out1)**2+norm(M_out2)**2)

print(Out)
plt.figure(1)
plt.plot(I,Out)
plt.show()