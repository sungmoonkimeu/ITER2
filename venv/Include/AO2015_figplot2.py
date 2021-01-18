'''Drawing figuer_10 of AO2015

'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, mat, ones, zeros, sin, einsum, append, arange, array, cumsum, argmin, sqrt, arcsin, arctan, \
    tan, random, column_stack,savetxt,loadtxt
from numpy.linalg import norm, eig, matrix_power
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter,ScalarFormatter)

#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'Computer Modern'
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
#plt.rcParams['font.family']='sans-serif'
#plt.rcParams['font.sans-serif']='Comic Sans MS'


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

V_I = loadtxt('fig10_saved.dat',unpack=True, usecols=[0])
DataIN = loadtxt('fig10_saved.dat',unpack=True, usecols=[1,2,3,4])

LB = [4,  5, 6, 7]
SR = [1, 1, 1, 1, ]

## Requirement specificaion for ITER
absErrorlimit = zeros(len(V_I))
relErrorlimit = zeros(len(V_I))

#Calcuation ITER specification
for nn in range(len(V_I)):
    if V_I[nn] < 1e6:
        absErrorlimit[nn] = 10e3
    else:
        absErrorlimit[nn] = V_I[nn]*0.01
    relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]


## Ploting graph
fig, ax = plt.subplots(figsize=(7,4))

for i in range(len(DataIN)):
    str_legend = r'$L_{B}/S_{P}$='+'{:1.0f}'.format(LB[i]/SR[i])
    print(str_legend)
    ax.plot(V_I,DataIN[i,:],lw='1', label=str_legend)
ax.plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
#ax.legend(loc="upper right", prop={'family': 'monospace'})
ax.legend(loc="upper right")

plt.rc('text',usetex = True)
ax.set_xlabel(r'Plasma current $I_{p}(A)$')
ax.set_ylabel(r'Relative error $\epsilon_{rel}$')


#plt.title('Output power vs Plasma current')
ax.set(xlim = (0,18e6), ylim = (0,0.02))
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

ax.ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax.grid(ls='--',lw=0.5)

#fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom= 0.13)
#fig.set_size_inches(6,4)
plt.show()


