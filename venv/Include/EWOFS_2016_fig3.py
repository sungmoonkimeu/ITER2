'''Drawing figuer_16 of AO2015
'''

import os
#os.chdir('C:/Users/Iter/PycharmProjects/pythonProject/venv/Include')
os.chdir('C:/Users/SMK/PycharmProjects/ITER2/venv/Include')
import time
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
from scipy.interpolate import interp1d
import matplotlib.transforms

#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'Computer Modern'
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
#matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
#plt.rcParams['font.family']='sans-serif'
#plt.rcParams['font.sans-serif']='Comic Sans MS'

###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


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
DataIN = loadtxt('EWOFS_fig3_saved5.txt',unpack=True)
V_I = DataIN.T[0,:]

fig = plt.figure()
ax = fig.gca(projection='3d')
# Set axis pane color to white
ax.xaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax.yaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax.zaxis.set_pane_color((1.0,1.0,1.0,1.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0.2)
X = V_I
#Y = arange(90,110,2) # Temperature of sensing fiber
Y =  arange(100,110+5,5) # Temperature of sensing fiber

print(X)
print(Y)
X,Y = np.meshgrid(X,Y)
Z = DataIN.T[1:,:]*100


#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
surf = ax.plot_surface(X,Y, Z, cmap = 'viridis', rstride=1, cstride=1, alpha = None, edgecolor=(0,0,0,0.5))
fig.colorbar(surf, ax=ax, shrink=0.8, aspect=25)
ax.view_init(elev=15, azim= -125)

#plt.rc('text',usetex = True)

ax.ticklabel_format(axis='x', style= 'sci' ,useMathText=True, scilimits=(-3,5))


ax.set(xlim = [0,20e6], ylim = [90,110], zlim = [0.4, 0.9])
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))

ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
#ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

#ax.tick_params(axis='z', width=10, labelsize=10, pad=0)

ax.set_xlabel('Plasma current IP(A)', labelpad=5)
ax.set_ylabel('Temperature[oC]',labelpad=5)
ax.set_zlabel('Relative error [%]',labelpad=5)

#plt.tight_layout()
'''
## Ploting graph
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