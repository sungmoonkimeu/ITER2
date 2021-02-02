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

import pandas as pd

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

data = pd.read_csv('SOPcomparison.csv')

## Ploting graph
fig, ax = plt.subplots(figsize=(6,3))

#head_array = np.array([])
ax.plot(data[data.head(0).columns[0]],data[data.head(0).columns[1]],lw='1', label='AO2015',color='r')
ax.plot(data[data.head(0).columns[0]],data[data.head(0).columns[3]],lw='1', label='Laming',color='b')

ax2 = ax.twinx()
ax2.set_ylabel('sin',color='tab:blue')
ax2.plot(data[data.head(0).columns[0]],data[data.head(0).columns[2]],lw='1', label='AO2015',color='r')
ax2.plot(data[data.head(0).columns[0]],data[data.head(0).columns[4]],lw='1', label='Laming',color='b')
ax2.tick_params(axis='y')

#ax.legend(loc="upper right", prop={'family': 'monospace'})
ax.legend(loc="upper right")
#plt.rc('text',usetex = True)

ax.set_xlabel('Lead fiber length [mm]')
ax.set_ylabel('Azimuth angle [deg]')

#plt.title('Output power vs Plasma current')
ax.set(xlim = (0.152,0.166), ylim = (28.86,28.88))

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax2.yaxis.set_major_locator(MaxNLocator(6))
ax2.set(ylim = (4.9,5.1))

ax.xaxis.set_major_formatter(OOMFormatter(0, "%1.3f"))
ax.yaxis.set_major_formatter(OOMFormatter(0, "%2.2f"))

fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom= 0.2)
#fig.set_size_inches(6,4)
#plt.rc('text',usetex = False)

plt.show()

plt.show()
