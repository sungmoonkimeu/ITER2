import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, conj
from numpy.linalg import norm

#verifying 1. The rotation matrix only

dBeta = 1e-10 #1234*pi       # delta Beta
xi = 1e-10 #1219*pi          # twist rate
I = 1           # current
D = 1         # coil diameter assume 10 cm
wl = 1e-6       # wavelength
f = 1e-12/(wl**2)*I/D    # Faraday-induced rotation angle per unit length
theta_1s = 0    # initial orientation of the local slow axis fo the fiber
Nturn = 1

maxI = 1
dI = maxI/100


#I = 100 # 100A
I = arange(-maxI,maxI,dI)
z = pi*D*Nturn
Out = zeros(len(I))

M_in = sqrt(1 / 2) * mat([1, 1])
M_pol0 = mat([[1, 0], [0, 0]])
M_pol90 = mat([[0, 0], [0, 1]])
M_m = mat([[-1,0],[0,1]])

for nn in range(len(I)):
    # forward calculation
    f = 1e-12 / (wl ** 2) * I[nn] / D  # Faraday-induced rotation angle per unit length
    q = 2*(xi + f)/dBeta
    gamma = 1/2*(dBeta**2+4*(xi+f)**2)**0.5

    Rz = 2*arcsin(1/sqrt(1+q**2)*sin(gamma*z))
    Omegaz = xi*z+arctan(-q/sqrt(1+q**2)*tan(gamma*z))
    Phiz = (xi*z-Omegaz)/2 + pi/2 + theta_1s

    r11 = cos(Rz/2)+1j*sin(Rz/2)*cos(2*Phiz)
    r12 = 1j*sin(Rz/2)*sin(2*Phiz)
    r21 = r12
    r22 = cos(Rz/2)-1j*sin(Rz/2)*cos(2*Phiz)
    M_R = mat([[r11, r12], [r21, r22]])
    M_Omega = mat([[cos(Omegaz), -sin(Omegaz)], [sin(Omegaz), cos(Omegaz)]])

    # backward calculation f = -f, M_R' --> conj(M_R), M_Omega' --> conj(M_Omega)
    f = -f
    q = 2 * (xi + f) / dBeta
    gamma = 1 / 2 * (dBeta ** 2 + 4 * (xi + f) ** 2) ** 0.5

    Rz = 2 * arcsin(1 / sqrt(1 + q ** 2) * sin(gamma * z))
    Omegaz = xi * z + arctan(-q / sqrt(1 + q ** 2) * tan(gamma * z))
    Phiz = (xi * z - Omegaz) / 2 + pi / 2 + theta_1s

    r11 = cos(Rz / 2) + 1j * sin(Rz / 2) * cos(2 * Phiz)
    r12 = 1j * sin(Rz / 2) * sin(2 * Phiz)
    r21 = r12
    r22 = cos(Rz / 2) - 1j * sin(Rz / 2) * cos(2 * Phiz)
    M_Rp = conj(mat([[r11, r12], [r21, r22]]))
    M_Omegap = conj(mat([[cos(Omegaz), -sin(Omegaz)], [sin(Omegaz), cos(Omegaz)]]))

    M_cal1 = M_pol0*M_m*M_Rp*M_Omegap*M_m*M_Omega*M_R
    M_out1 = np.einsum('ij,...j',M_cal1,M_in)
    M_cal2 = M_pol90*M_m*M_Rp*M_Omegap*M_m*M_Omega*M_R
    M_out2 = np.einsum('ij,...j', M_cal2, M_in)
    Out[nn] = (norm(M_out1)**2-norm(M_out2)**2)/(norm(M_out1)**2+norm(M_out2)**2)

print(Out)
plt.figure(1)
plt.plot(I,Out)
plt.show()