"""
Created on Mon May 02 15:14:00 2022
@author: SMK

functions to draw figures from files
"""

import numpy as np
from numpy import pi, zeros
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)

from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes

import pandas as pd

# from .basis_calibration_lib import calib_basis3

import os
import sys
#print(os.getcwd())
#print(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '\My_library')

from draw_poincare_plotly import *

from cycler import cycler
cc = (cycler(color=list('rkbgcmy')))
# cc = (cycler(color=list('rkbgcmy')) * cycler(linestyle=['-', '--', '-.']))


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def cm_to_rgba_tuple(colors,alpha=1):
    for nn in range(len(colors)):
        r = int(colors[nn].split(",")[0].split("(")[1])/256
        g = int(colors[nn].split(",")[1])/256
        b = int(colors[nn].split(",")[2].split(")")[0])/256
        if nn == 0:
            tmp = np.array([r, g, b, alpha])
        else:
            tmp = np.vstack((tmp, np.array([r, g, b, alpha])))
    return tmp


# just plotting error (*.csv)
def plot_error_byfile(filename, fig=None, ax=None, lines=None):
    data = pd.read_csv(filename)
    if data['Ip'][0] == 0:
        data.drop(0, inplace=True)
        data.index -= 1
    V_I = data['Ip']

    # Calcuation ITER specification
    absErrorlimit = zeros(len(V_I))
    relErrorlimit = zeros(len(V_I))

    for nn in range(len(V_I)):
        absErrorlimit[nn] = 10e3 if V_I[nn] < 1e6 else V_I[nn] * 0.01
        relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

    if fig is None or ax is None or lines is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        ax.set_prop_cycle(cc)

    for col_name in data:
        if col_name != 'Ip':
            lines += ax.plot(V_I, abs((data[col_name] - V_I) / V_I))
            # lines += ax.plot(V_I, abs((data[col_name]-V_I)/V_I), label=col_name)

    #lines += ax.plot(V_I, relErrorlimit, 'r', label='ITER specification')
    ax.legend(loc="upper right")

    ax.set_xlabel(r'Plasma current $I_{p}(A)$')
    ax.set_ylabel(r'Relative error on $I_{P}$')

    ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

    ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    ax.grid(ls='--', lw=0.5)
    fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)

    return fig, ax, lines


# plotting error from Stokes parameters
def plot_error_byfile2(filename, fig=None, ax=None, lines=None, v_calc_init=None, V_custom=None):
    data = pd.read_csv(filename)
    if data['Ip'][0] == 0:
        data.drop(0, inplace=True)
        data.index -= 1
    V_I = data['Ip']
    E = Jones_vector('Output')
    S = create_Stokes('Output_S')
    V_ang = zeros(len(V_I))
    Ip = zeros([int((data.shape[1] - 1) / 2), len(V_I)])

    V = 0.54 * 4 * pi * 1e-7 if V_custom is None else V_custom

    # Calcuation of ITER specification
    absErrorlimit = zeros(len(V_I))
    for nn in range(len(V_I)):
        absErrorlimit[nn] = 10e3 if V_I[nn] < 1e6 else V_I[nn] * 0.01
    relErrorlimit = absErrorlimit / V_I

    if fig is None or ax is None or lines is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        ax.set_prop_cycle(cc)

    for nn in range(int((data.shape[1] - 1) / 2)):
        str_Ex = str(nn) + ' Ex'
        str_Ey = str(nn) + ' Ey'
        Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                         [complex(y) for y in data[str_Ey].to_numpy()]])
        E.from_matrix(Vout)
        S.from_Jones(E)

        m = 0
        for kk in range(len(V_I)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi

            c = V_ang[0] if v_calc_init is None else v_calc_init
            Ip[nn][kk] = (V_ang[kk] - c) / V

        lines += ax.plot(V_I, abs((Ip[nn, :] - V_I) / V_I), label=str(nn))

    lines += ax.plot(V_I, relErrorlimit, 'r--', label='ITER specification')
    ax.legend(loc="upper right")

    ax.set_xlabel(r'Plasma current $I_{p}(A)$')
    ax.set_ylabel(r'Relative error on $I_{P}$')

    ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
    ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

    ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    ax.grid(ls='--', lw=0.5)

    fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
    return fig, ax, lines


def plot_error_byStokes(V_I, S, fig=None, ax=None, lines=None, v_calc_init=None, V_custom=None):

    V_ang = zeros(len(V_I))
    Ip = zeros(len(V_I))
    V = 0.54 * 4 * pi * 1e-7 if V_custom is None else V_custom

    # Calcuation of ITER specification
    absErrorlimit = zeros(len(V_I))
    for nn in range(len(V_I)):
        absErrorlimit[nn] = 10e3 if V_I[nn] < 1e6 else V_I[nn] * 0.01
    relErrorlimit = absErrorlimit / V_I

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        ax.set_prop_cycle(cc)

    m = 0
    for kk in range(len(V_I)):
        if kk > 2 and S[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
            m = m + 1
        elif kk > 2 and S[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
            m = m - 1
        V_ang[kk] = S[kk].parameters.azimuth() + m * pi

        c = V_ang[0] if v_calc_init is None else v_calc_init
        Ip[kk] = (V_ang[kk] - c) / V

    lines += ax.plot(V_I, abs((Ip - V_I) / V_I), label='added')
    ax.legend(loc="upper right")

    if fig is None:
        lines += ax.plot(V_I, relErrorlimit, 'r--', label='ITER specification')

        ax.set_xlabel(r'Plasma current $I_{p}(A)$')
        ax.set_ylabel(r'Relative error on $I_{P}$')

        ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(10))

        ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

        ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
        ax.grid(ls='--', lw=0.5)

        fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
    return fig, ax, lines


def plot_Stokes_on_Poincare_file(filename, fig=None, lines=None, opacity=1):
    data = pd.read_csv(filename)
    if data['Ip'][0] == 0:
        data.drop(0, inplace=True)
        data.index -= 1
    V_I = data['Ip']
    E = Jones_vector('Output')
    S = create_Stokes('Output_S')
    V_ang = zeros(len(V_I))
    Ip = zeros([int((data.shape[1] - 1) / 2), len(V_I)])

    if fig is None or lines is None:
        fig = PS5(opacity)

    for nn in range(int((data.shape[1] - 1) / 2)):
        if nn >1:
            pass
        str_Ex = str(nn) + ' Ex'
        str_Ey = str(nn) + ' Ey'
        Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                         [complex(y) for y in data[str_Ey].to_numpy()]])
        E.from_matrix(Vout)
        S.from_Jones(E)

        S1 = S.parameters.matrix()[1]
        S2 = S.parameters.matrix()[2]
        S3 = S.parameters.matrix()[3]

        tick_vals = np.linspace(0, V_I.max(), 5)
        tick_text = [str(int(nn / 1e6)) + "MA" for nn in tick_vals]
        colorbar_param = dict(lenmode='fraction', len=0.75, thickness=10, tickfont=dict(size=20),
                              tickvals=tick_vals,
                              ticktext=tick_text,
                              # title='Azimuth angle',
                              outlinewidth=1,
                              x=0.8)

        fig.add_scatter3d(x=S1, y=S2, z=S3, mode="lines+markers",
                          marker=dict(size=2.5,
                                      opacity=1,
                                      color=V_I,
                                      colorscale='Viridis'),
                          line=dict(width=8,
                                    color=V_I,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=colorbar_param),
                          name='F1')

    return fig, lines


def draw_Stokes(Ip, S, fig=None, lines=None, opacity=1):

    if fig is None:
        fig = PS5(opacity)
        colorscale = 'Viridis'
    else:
        colorscale = 'Inferno'

    S1 = S.parameters.matrix()[1]
    S2 = S.parameters.matrix()[2]
    S3 = S.parameters.matrix()[3]

    tick_vals = np.linspace(0, Ip.max(), 5)
    tick_text = [str(int(nn / 1e6)) + "MA" for nn in tick_vals]
    colorbar_param = dict(lenmode='fraction', len=0.75, thickness=10, tickfont=dict(size=20),
                          tickvals=tick_vals,
                          ticktext=tick_text,
                          # title='Azimuth angle',
                          outlinewidth=1,
                          x=0.8)

    fig.add_scatter3d(x=S1, y=S2, z=S3, mode="lines+markers",
                      marker=dict(size=2.5,
                                  opacity=1,
                                  color=Ip,
                                  colorscale=colorscale),
                      line=dict(width=8,
                                color=Ip,
                                colorscale=colorscale,
                                showscale=True,
                                colorbar=colorbar_param),
                      name='F1')

    return fig, lines



if (__name__ == "__main__"):

    fig = PS5()
    # inp = np.arange(0, np.pi, 0.01)
    # S = create_Stokes('Output_S')
    # S.linear_light(azimuth=inp)
    #
    # S1 = S.parameters.matrix()[1]
    # S2 = S.parameters.matrix()[2]
    # S3 = S.parameters.matrix()[3]
    #
    # fig.add_scatter3d(x=S1, y=S2, z=S3, mode="markers",
    #                   marker=dict(size=3,
    #                               opacity=1,
    #                               color=S.parameters.azimuth(),
    #                               colorscale='Viridis'),
    #                   name='F1')

    plot_Stokes_on_Poincare
    fig.show()
    #main()
    #plt.show()
