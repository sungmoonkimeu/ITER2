'''Drawing figuer_16 of AO2015
'''

#os.chdir('C:/Users/Iter/PycharmProjects/pythonProject/venv/Include')

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

V_I = loadtxt('BeforeLF_AfterLF_0.dat',unpack=True, usecols=[0])
DataIN = loadtxt('BeforeLF_AfterLF_0.dat',unpack=True, usecols=[1,2,3,4,5, 12,13,14,15,16])

LB = [0.0132]
SR = [0.003]
chi = [0.000, 0.003, 0.006, 0.009, 0.012]
strr = ['0,000 mm','0,003 mm','0,006 mm','0,009 mm','0,012 mm']
#strr = ['pi/16','pi/8 mm','0 mm']

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
fig, ax = plt.subplots(figsize=(6,3))

for i in range(len(chi)):
    str_legend = strr[i]
    print(str_legend)
    ax.plot(V_I,DataIN[i,:],lw='1', label=str_legend)
ax.plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
#ax.legend(loc="upper right", prop={'family': 'monospace'})
ax.legend(loc="upper right")

plt.rc('text',usetex = True)
ax.set_xlabel(r'Plasma current $I_{p}(A)$')
ax.set_ylabel(r'Relative error on $I_{P}$')


#plt.title('Output power vs Plasma current')
ax.set(xlim = (0,18e6), ylim = (0,0.1))
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_locator(MaxNLocator(10))

ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

ax.ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax.grid(ls='--',lw=0.5)

#fig.align_ylabels(ax)
fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom= 0.2)
#fig.set_size_inches(6,4)
plt.rc('text',usetex = False)

plt.show()

## Ploting graph
fig2, ax2 = plt.subplots(figsize=(6,3))

for i in range(len(chi)):
    str_legend = strr[i]
    print(str_legend)
    ax2.plot(V_I,DataIN[i+5,:],lw='1', label=str_legend)
ax2.plot(V_I,relErrorlimit,'r', label='ITER specification',lw='1')
#ax.legend(loc="upper right", prop={'family': 'monospace'})
ax2.legend(loc="upper right")

plt.rc('text',usetex = True)
ax2.set_xlabel(r'Plasma current $I_{p}(A)$')
ax2.set_ylabel(r'Relative error on $I_{P}$')


#plt.title('Output power vs Plasma current')
ax2.set(xlim = (0,18e6), ylim = (0,0.1))
ax2.yaxis.set_major_locator(MaxNLocator(4))
ax2.xaxis.set_major_locator(MaxNLocator(10))

ax2.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
ax2.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

ax2.ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))
ax2.grid(ls='--',lw=0.5)

#fig.align_ylabels(ax)
fig2.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom= 0.2)
#fig.set_size_inches(6,4)
plt.rc('text',usetex = False)

plt.show()
