'''
Created 2022.06.02
for analysing JET fiber's behavior

'''

# print(os.getcwd())
# print(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '\My_library')
import time

import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import rgb2hex

start = time.process_time()
# from My_Library import SPUNFIBRE_lib
import plotly.offline

from My_Library.SPUNFIBER2_lib import *
from My_Library.draw_figures_FOCS import *

from My_Library.basis_correction_lib import *

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

if __name__ == '__main__':
    mode =4
    # Crystal Techno lobi spun fiber
    LB = 0.3
    SP = 0.005
    # dz = SP / 1000
    dz = 0.0001
    len_lf = 1  # lead fiber
    len_ls = 5  # sensing fiber
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)

    #strfile1 = 'Hibi_46FM_errdeg1x5_220506.csv'
    #strfile1 = 'Lobi_46FM_errdeg1x5_220506.csv'
    strfile1 = 'CrystalTechno.csv'

    if mode == 0:

        num_iter = 1
        num_processor = 16
        V_I = arange(0e6, 20e6 + 0.1e6, 0.2e6)
        # V_I = 1e6
        out_dict = {'Ip': V_I}
        out_dict2 = {'Ip': V_I}
        nM_vib = 1
        start = pd.Timestamp.now()
        ang_FM = 45

        E = Jones_vector('input')
        azi = np.array([0, pi/6, pi/4])
        E.general_azimuth_ellipticity(azimuth=azi, ellipticity=pi/18)
        fig1, ax1 = spunfiber.init_plot_SOP()
        S = create_Stokes('O')
        S2 = create_Stokes('1')

        for nn in range(num_iter):
            Vin = E[1].parameters.matrix()
            #M_vib = spunfiber.create_Mvib(nM_vib, 10, 10)
            M_vib = spunfiber.create_Mvib2(nM_vib, 10*pi/180, 0*pi/180, 10*pi/180)
            Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
            save_Jones(strfile1,V_I,Ip,Vout)

            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        fig2, ax2, lines = spunfiber.plot_error(strfile1)

        # labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
        #              ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]
        # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))


        fig3, ax3, lines3 = plot_error_byfile2(strfile1+"_S")

    elif mode == 1:
        #strfile1 = 'Hibi_test.csv'
        #strfile1 = 'Hibi_IdealFM_Errdeg1x5.csv'
        opacity = 0.8
        V2 = 0.54 * 4 * pi * 1e-7 * 2 / 1.0375

        # 1, plot error directly from file
        fig, ax, lines, V2 = None, None, None, None
        fig, ax, lines3 = plot_error_byfile2(strfile1 + "_S", V_custom=V2)

        # 1-1, plot Stokes on the Poincare directly from file
        opacity = 1
        fig3, ax = plot_Stokes_byfile(strfile1 + "_S", opacity=opacity)

        # 2, Load Stokes from file data
        ncol = 0
        V_I, S, isEOF = load_stokes_fromfile(strfile1+"_S", ncol)

        # 2-1, plot error from Stokes
        fig, ax, line3, V2 = None, None, None, None
        fig, ax, lines3 = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines3, V_custom=V2)

        # 2-2, plot Stokes from loaded Stokes
        fig, opacity = None, None
        fig, lines = plot_Stokes(V_I, S, fig=fig, opacity=opacity)

        # 3, Basis correction
        c = [None, None, None]
        S2, c = basis_correction1(S, c=c)

        # 3-1, draw Farday roataion plane vector
        fig3, opacity = None, 0.5
        fig3, lines = plot_Stokes(V_I, S2, fig=fig3, opacity=opacity)
        fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
                            mode='lines',
                            line=dict(width=8))

        # These three functions need fig.show() to show figure in new browser
        # add_scatter3d
        # plot_Stokes
        # plot_Stokes_byfile
        fig3.show()

    elif mode == 2:
        ### effect of basis correction
        #strfile1 = "IdealFM_Hibi_Errdeg1x5_0.csv"
        #strfile1 = "Hibi_44FM_errdeg1x5.csv"
        #strfile1 = "IdealFM_Errdeg1x5_1.csv"
        # load whole Jones and convert to measured Ip
        #V2 = 0.54 * 4 * pi * 1e-7 * 2 *(0.9700180483394489)
        V2 = 0.54 * 4 * pi * 1e-7

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        c = np.array([None, None, None])

        dic_err = {}
        while isEOF is False:
            V_I, S, isEOF = load_stokes_fromfile(strfile1+"_S", nn)
            #before cal.
            fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='Small LB/SP (60)')
            fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            #fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)

            # S2, c = basis_correction1(S[:14], c)
            # # fig3.add_scatter3d(x=(0, c[0]*1.2), y=(0, c[1]*1.2), z=(0,c[2]*1.2),
            # #                    mode='lines',line=dict(width=8))
            #
            # S2, c = basis_correction1(S, c)
            # # fig3.add_scatter3d(x=(0, c[0] * 1.2), y=(0, c[1] * 1.2), z=(0, c[2] * 1.2),
            # #                    mode='lines', line=dict(width=8))
            # # fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
            # #                                       label=str(nn)+"calibrated")
            # # fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            # fig3, lines3 = plot_Stokes(V_I, S2, fig=fig3, lines=lines3, opacity=opacity)
            # # c = np.array([None, None, None])
            #
            # fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
            #                                       label='After basis correction')
            # fig2, ax2, lines2 = plot_error_byStokes(V_I, S, V_custom=V2,label='After calibration')

            if nn == 0:
                dic_err['V_I'] = V_I
            dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=V2)
            c = np.array([None, None, None])
            nn += 1
            if nn > 2:
                break
        # fig10, ax10 = plot_errorbar_byDic(dic_err)
        fig3.show()

    elif mode == 3:
        ### effect of basis correction
        # strfile1 = "IdealFM_Hibi_Errdeg1x5_0.csv"
        # strfile1 = "Hibi_44FM_errdeg1x5.csv"
        # strfile1 = "IdealFM_Errdeg1x5_1.csv"
        # load whole Jones and convert to measured Ip
        # V2 = 0.54 * 4 * pi * 1e-7 * 2 *(0.9700180483394489)
        V2 = 0.54 * 4 * pi * 1e-7

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        c = np.array([None, None, None])

        dic_err = {}
        while isEOF is False:
            V_I, S, isEOF = load_stokes_fromfile(strfile1 + "_S", nn)
            # before cal.
            fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='Small LB/SP (60)')
            fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            # fig3, lines3 = plot_Stokes(V_I[:25], S[:25], fig=fig3, lines=lines3, opacity=opacity)

            # cal. total
            S2, c = basis_correction1(S, c)
            fig3.add_scatter3d(x=(0, c[0] * 1.2), y=(0, c[1] * 1.2), z=(0, c[2] * 1.2),
                               mode='lines', line=dict(width=8))
            c = np.array([None, None, None])
            # cal. few
            nn = 0
            np_S = None
            while nn < len(S):
                if nn+5 > len(S):
                    Stmp, c = basis_correction1(S[nn:], c)
                else:
                    Stmp, c = basis_correction1(S[nn:nn+5], c)
                if np_S is None:
                    np_S = Stmp.parameters.matrix()
                else:
                    np_S = np.hstack((np_S, Stmp.parameters.matrix()))
                fig3.add_scatter3d(x=(0, c[0] * 1.2), y=(0, c[1] * 1.2), z=(0, c[2] * 1.2),
                                   mode='lines', line=dict(width=8))
                c = np.array([None, None, None])

                nn += 5
            S2.from_matrix(np_S)
            #S2, c = basis_correction1(S, c)

            # fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
            #                                       label=str(nn)+"calibrated")
            # fig3, lines3 = plot_Stokes(V_I, S, fig=fig3, lines=lines3, opacity=opacity)
            # fig3, lines3 = plot_Stokes(V_I[:], S2[:], fig=fig3, lines=lines3, opacity=opacity)
            # c = np.array([None, None, None])
            # S2, c = basis_correction1(S2, c)
            # fig3.add_scatter3d(x=(0, c[0] * 1.2), y=(0, c[1] * 1.2), z=(0, c[2] * 1.2),
            #                    mode='lines', line=dict(width=8))
            fig3, lines3 = plot_Stokes(V_I, S2[:], fig=fig3, lines=lines3, opacity=opacity)

            fig, ax, lines = plot_error_byStokes(V_I, S2, fig=fig, ax=ax, lines=lines, V_custom=V2,
                                                 label='After basis correction')
            # fig2, ax2, lines2 = plot_error_byStokes(V_I, S, V_custom=V2,label='After calibration')

            if nn == 0:
                dic_err['V_I'] = V_I
            dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=V2)
            c = np.array([None, None, None])
            nn += 1
            if nn > 0:
                break
        # fig10, ax10 = plot_errorbar_byDic(dic_err)
        fig3.update_scenes(camera_projection_type='orthographic')
        fig3.show()
        #plotly.offline.plot(fig3, filename='xx.html')

    elif mode == 4:
        fig = None
        V_I = arange(0e6, 10e6 + 0.1e6, 0.2e6)
        Ip = zeros(len(V_I))

        V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))

        # V_I = 1e6
        out_dict = {'Ip': V_I}
        out_dict2 = {'Ip': V_I}
        start = pd.Timestamp.now()
        ang_FM = 45

        E = Jones_vector('input')
        azi = np.array([0, pi/6, pi/4])
        E.general_azimuth_ellipticity(azimuth=azi, ellipticity=pi/12)
        fig1, ax1 = spunfiber.init_plot_SOP()
        S = create_Stokes('O')
        S2 = create_Stokes('1')


        Vin = E[1].parameters.matrix()

        s_t_r = 2 * pi / spunfiber.SP
        spunfiber.dz = spunfiber.L / 1000
        V_L = arange(0, spunfiber.L + spunfiber.dz, spunfiber.dz)
        nV_L = V_L/spunfiber.L

        V_theta= V_L * s_t_r

        V_delta = 2*pi/(LB * (1- 1*cos(2*pi*0.5*nV_L))+0.000001)
        print(V_delta)

        for mm, iter_I in enumerate(V_I):
            M_f = spunfiber.lamming_JET(iter_I, 1, V_delta, V_theta, V_L)
            V_out[mm] = M_f @ Vin

        E = Jones_vector('Output')
        E.from_matrix(M=V_out)
        V_ang = zeros(len(V_I))

        # SOP evolution in Lead fiber (Forward)
        S = create_Stokes('Output_S')
        S.from_Jones(E)

        if fig is not None:
            draw_stokes_points(fig[0], S, kind='line', color_line='b')
        else:
            fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line',
                                      color_line='b')

        m = 0
        for kk in range(len(V_I)):
            if kk > 0 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 0 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi
            Ip[kk] = (V_ang[kk] - pi / 2) / (spunfiber.V)

        # save_Jones(strfile1,V_I,Ip,Vout)
        #
        #
        #
        # fig2, ax2, lines = spunfiber.plot_error(strfile1)
        #
        # # labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
        # #              ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]
        # # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
        #
        #
        # fig3, ax3, lines3 = plot_error_byfile2(strfile1+"_S")


    plt.show()