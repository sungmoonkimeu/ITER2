import matplotlib.pyplot as plt
import numpy as np

from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan

x = arange(0.25,1000,0.01)

Lp_p = x/((4*x**2+1)**0.5-2*x)

S = 4*(x**2) /(1+4*(x**2))*100

plt.figure(1)
plt.subplot(211)
plt.plot(1/x,Lp_p)
plt.axis([0, 4, 0, 10])
plt.ylabel('Lp_p/Lp')
plt.xlabel('Lt/Lp')
plt.title('TEST!!')
plt.legend('Lp_p/Lp')

plt.subplot(212)
plt.plot(1/x, S)
plt.axis([0, 4, 0, 120])
plt.ylabel('some numbers')
plt.xlabel('x(t)')
plt.title('TEST!!')
plt.legend('S')

plt.show()

