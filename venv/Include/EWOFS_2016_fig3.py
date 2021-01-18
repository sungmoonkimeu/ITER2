'''Drawing figuer_16 of AO2015
'''

#os.chdir('C:/Users/Iter/PycharmProjects/pythonProject/venv/Include')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

#V_I = loadtxt('EWOFS_fig3_saved.dat',unpack=True, usecols=[0])
DataIN = loadtxt('EWOFS_fig3_saved.dat',unpack=True)
V_I = DataIN[0,:]


fig = plt.figure()
ax = fig.gca(projection='3d')

X = V_I
Y = arange(90,110,2) # Temperature of sensing fiber
#Y =  arange(100,110+5,5) # Temperature of sensing fiber


X,Y = np.meshgrid(X,Y)
Z = DataIN[1:,:]

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

'''
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
'''
plt.show()
