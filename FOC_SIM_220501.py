
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt, log10
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager, Lock
import pandas as pd
import matplotlib.pyplot as plt
import os

# print(os.getcwd())
# print(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
import time
start = time.process_time()
# from My_Library import SPUNFIBRE_lib

from My_Library.SPUNFIBRE_lib import *
from My_Library.draw_figures_FOCS import *

from My_Library.draw_poincare_plotly import *
from My_Library.basis_calibration_lib import *

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


if __name__ == '__main__':
    mode = 3
    # Crystal Techno lobi spun fiber
    LB = 0.009
    SP = 0.005
    # dz = SP / 1000
    dz = 0.0001
    len_lf = 1  # lead fiber
    len_ls = 1  # sensing fiber
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)

    strfile1 = 'Test1_hibi.csv'

    if mode == 0:

        num_iter = 3
        num_processor = 8
        V_I = arange(0e6, 6e6 + 0.1e6, 0.1e6)
        # V_I = 1e6
        out_dict = {'Ip': V_I}
        out_dict2 = {'Ip': V_I}
        nM_vib = 0
        start = pd.Timestamp.now()
        ang_FM = 45
        #Vin = np.array([[1/np.sqrt(0.5)], [1/np.sqrt(0.5)]])
        #Vin = np.array([[1], [0]])

        E = Jones_vector('input')
        azi = np.array([0, pi/6, pi/4])
        E.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)
        # print(E)
        #Vin = E.parameters.matrix()
        # Vin = np.array([[0], [1]])

        fig1, ax1 = spunfiber.init_plot_SOP()
        S = create_Stokes('O')
        for nn in range(num_iter):
            Vin = E[nn].parameters.matrix()
            M_vib = spunfiber.create_Mvib(nM_vib, 20, 0)
            Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
            save_Jones(strfile1,V_I,Ip,Vout)

            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        fig2, ax2, lines = spunfiber.plot_error(strfile1)

        labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
                     ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]

        # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
        ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right')
        ax2.set(xlim=(0, 4e6), ylim=(0, 0.002))
        ax2.xaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-3, "%1.3f"))

        fig3, ax3, lines3 = plot_error_byfile2(strfile1+"_S")

    elif mode == 1:
        #strfile1 = 'Hibi_test.csv'
        #strfile1 = 'Hibi_IdealFM_Errdeg1x5.csv'
        opacity = 0.8
        V_I, S = load_stokes_fromfile(strfile1+"_S", 1)

        #fig3, ax = plot_Stokes_byfile(strfile1+"_S", opacity=opacity)
        #fig3, lines = plot_Stokes(V_I, S, opacity=opacity)

        V2 = 0.54 * 4 * pi * 1e-7*2/1.0375
        fig, ax, lines3 = plot_error_byfile2(strfile1 + "_S", V_custom=V2/1.0365)
        #fig, ax, lines3 = plot_error_byStokes(V_I, S, V_custom=V2)
        #fig, ax, lines3 = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines3, V_custom=V2)
        # #
        S2, c = calib_basis1(S)
        #fig, ax, lines3 = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines3, V_custom=V2)
        #
        # fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
        #                     mode='lines',
        #                     line=dict(width=8))
        # fig3, lines = plot_Stokes(V_I, S2, fig=fig3, opacity=opacity)
        # fig3.show()
    elif mode == 2:
        ### effect of basis correction
        #strfile1 = "IdealFM_Hibi_Errdeg1x5_0.csv"
        #strfile1 = "Hibi_44FM_errdeg1x5.csv"
        #strfile1 = "IdealFM_Errdeg1x5_1.csv"
        # load whole Jones and convert to measured Ip
        #V2 = 0.54 * 4 * pi * 1e-7 * 2 *(0.9700180483394489)
        V2 = 0.54 * 4 * pi * 1e-7 * 2

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        c = np.array([None, None, None])

        dic_err = {}
        while isEOF is False:
            V_I, S, isEOF = load_stokes_fromfile(strfile1+"_S", nn)
            #before cal.
            fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='Hibi spun fiber (LB/SP=1.875)')
            #fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            #fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)

            S2, c = calib_basis1(S, c)
            # fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
            #                    mode='lines',line=dict(width=8))
            # fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
            #                                       label=str(nn)+"calibrated")
            # fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)
            c = np.array([None, None, None])
            S2, c = calib_basis1(S2, c)
            fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
                               mode='lines',line=dict(width=8))
            fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='After basis correction')
            fig2, ax2, lines2 = plot_error_byStokes(V_I, S, V_custom=V2*(0.9700180483394489),label='After calibration')


            if nn == 0:
                dic_err['V_I'] = V_I
            dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=V2)
            c = np.array([None, None, None])
            nn += 1
            if nn > 0:
                break
        # fig10, ax10 = plot_errorbar_byDic(dic_err)
        fig3.show()

    elif mode ==3:
        strfile1 = "Hibi_42FM_errdeg1x5.csv"
        # strfile1 = "IdealFM_Hibi_Errdeg1x5_0.csv"
        # strfile1 = "Hibi_44FM_errdeg1x5.csv"
        # strfile1 = "IdealFM_Errdeg1x5_1.csv"
        # load whole Jones and convert to measured Ip
        V2 = 0.54 * 4 * pi * 1e-7 * 2 *(0.9700180483394489)
        #V2 = 0.54 * 4 * pi * 1e-7 * 2

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        c = np.array([None, None, None])

        dic_err = {}
        while isEOF is False:
            V_I, S, isEOF = load_stokes_fromfile(strfile1 + "_S", nn)
            # before cal.
            fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='Hibi spun fiber (LB/SP=1.875)')
            # fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            # fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)

            # S2, c = calib_basis1(S, c)
            # fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
            #                    mode='lines',line=dict(width=8))
            # fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
            #                                       label=str(nn)+"calibrated")
            # fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)

            if nn == 0:
                dic_err['V_I'] = V_I
            dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=V2)
            c = np.array([None, None, None])
            nn += 1
            # if nn > 0:
            #     break
        fig10, ax10 = plot_errorbar_byDic(dic_err)
    plt.show()

