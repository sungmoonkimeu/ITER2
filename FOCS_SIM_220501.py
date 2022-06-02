'''
Created 2022.06.02
for journal paper

FOCS accuracy Simulation with FM imperfection.

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
    mode = 3
    LB = 1.000
    SP = 0.005
    # dz = SP / 1000
    dz = 0.0001
    len_lf = 6  # lead fiber
    len_ls = 28  # sensing fiber
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)

    # strfile1 = 'Lobi_45FM_errdeg1x5_220601.csv'
    # strfile2 = 'Lobi_46FM_errdeg1x5_220601.csv'
    # strfile3 = 'Lobi_65FM_errdeg1x5_220601.csv'
    # strfile4 = 'Lobi_0FM_errdeg1x5_220601.csv'

    strfile1 = 'Hibi_45FM_errdeg1x5_220601.csv'
    strfile2 = 'Hibi_46FM_errdeg1x5_220601.csv'
    strfile3 = 'Hibi_65FM_errdeg1x5_220601.csv'
    strfile4 = 'Hibi_0FM_errdeg1x5_220601.csv'

    V_strfile = [strfile1, strfile2, strfile3, strfile4]
    V_angFM= [45, 46, 65, 0]
    # strfile1 = 'Hibi_test.csv'

    if mode == 0:

        num_iter = 100
        num_processor = 16
        V_I = np.hstack((np.zeros(1),np.logspace(0,5, 20), np.arange(0.1e6, 18e6, 0.2e6)))
        # V_I = arange(0e6, 18e6 + 0.1e6, 0.1e6)
        # V_I = np.hstack((np.arange(0e6, 0.1e6, 0.005e6), np.arange(0.1e6, 18e6, 0.2e6)))
        # V_I = arange(0e6, 4e6 + 0.1e6, 0.1e6)
        out_dict = {'Ip': V_I}
        out_dict2 = {'Ip': V_I}
        nM_vib = 5
        start = pd.Timestamp.now()

        xx = 0
        for xx, strfile1 in enumerate(V_strfile):

            # ang_FM = 45
            ang_FM = V_angFM[xx]

            E = Jones_vector('input')
            azi = np.array([0, pi/6, pi/4])
            E.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)
            fig1, ax1 = spunfiber.init_plot_SOP()
            S = create_Stokes('O')
            for nn in range(num_iter):
                Vin = E[0].parameters.matrix()
                M_vib = spunfiber.create_Mvib(nM_vib, 1, 1)
                Ip, Vout = spunfiber.calc_mp3(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
                save_Jones(strfile1,V_I,Ip,Vout)

                checktime = pd.Timestamp.now() - start
                print(nn, "/", num_iter, checktime)
                start = pd.Timestamp.now()

            # fig2, ax2, lines = spunfiber.plot_error(strfile1)

            # labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
            #              ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]
            # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))


            fig3, ax3, lines3 = plot_error_byfile2(strfile1+"_S", V_custom=0.54 * 4 * pi * 1e-7*2)

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
        # strfile1 = 'Lobi_45FM_errdeg1x5_220529_2.csv'
        # strfile1 = "IdealFM_Hibi_Errdeg1x5_0.csv"
        # strfile1 = "Hibi_44FM_errdeg1x5.csv"
        # strfile1 = "IdealFM_Errdeg1x5_1.csv"
        # load whole Jones and convert to measured Ip
        # V2 = 0.54 * 4 * pi * 1e-7 * 2 *(0.9700180483394489)
        # V_strfile = ['Lobi_45FM_errdeg1x5_220531.csv',
        #              'Lobi_46FM_errdeg1x5_220531.csv',
        #              'Lobi_65FM_errdeg1x5_220531.csv',
        #              'Lobi_0FM_errdeg1x5_220531.csv']
        #
        # V_strfile = ['Hibi_45FM_errdeg1x5_220531.csv',
        #              'Hibi_46FM_errdeg1x5_220531.csv',
        #              'Hibi_65FM_errdeg1x5_220531.csv',
        #              'Hibi_0FM_errdeg1x5_220531.csv']
        #V_strfile = ['Lobi_45FM_errdeg1x5_220531.csv']
        # V_strfile = ['Hibi_0FM_errdeg1x5_220531.csv']
        V_label = ['Ideal FM', 'Nonideal ']
        V2 = 0.54 * 4 * pi * 1e-7 * 2

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        fig10, ax10, lines10 = None, None, []
        c = np.array([None, None, None])


        for strfile1 in V_strfile:
            dic_err = {}
            nn = 0
            while isEOF is False:
                V_I, S, isEOF = load_stokes_fromfile(strfile1 + "_S", nn)

                fig, ax, lines = plot_error_byStokes(V_I, S, fig=fig, ax=ax, lines=lines, V_custom=V2*0.965,label=str(nn))
                fig3, lines3 = plot_Stokes(V_I[20:35:2], S[20:35:2], fig=fig3, lines=lines3, opacity=opacity)

                if nn == 0:
                    dic_err['V_I'] = V_I
                dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=V2*0.965)
                c = np.array([None, None, None])
                nn += 1
                # if nn > 0:
                #     break


            fig10, ax10, lines10 = plot_errorbar_byDic(dic_err, fig=fig10, ax=ax10, lines=lines10, init_index=21)
            isEOF = False
        # ax10.legend(lines10, ['ITER specification', '',
        #                                  r'Ideal FM $\theta_{err}$=0$\degree$',
        #                                  r'$\theta_{err}$=1$\degree$', '', '','', ''])

        ax10.legend( [lines10[0], lines10[4], lines10[7], lines10[10], lines10[13]],
                     ['ITER specification', 'Ideal FM',
                      r'$\theta_{err}$=1$\degree$',
                      r'$\theta_{err}$=20$\degree$',
                      r'$\theta_{err}$=45$\degree$'],
                     bbox_to_anchor=(1.01, 1.01), loc="upper left")
        fig10.savefig("fig9.(b).jpg", dpi=330)

        # print(legend)
        # r'$\theta_{err}$=20$\degree$'])
        # r'$\theta_{err}$=45$\degree$'])

        #ax.legend()
    # elif mode ==5:
    #
    #     strfile1 = 'FMerror.csv'
    #     num_iter = 500
    #
    #     nM_vib = 5
    #     #ang_FM = np.arange(0,46,1)
    #     ang_FM = np.arange(0, 46, 1)
    #
    #     E = Jones_vector('input')
    #     E1 = Jones_vector('output')
    #     azi = np.array([0, pi / 6, pi / 4])
    #     E.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)
    #     #fig1, ax1 = spunfiber.init_plot_SOP()
    #     S = create_Stokes('O')
    #
    #     # outdict = {}
    #     # for nn in ang_FM:
    #     #     Vin = E[0].parameters.matrix()
    #     #     print(nn)
    #     #     Vout = spunfiber.cal_2ndBridge(45+nn/10, num_iter, Vin=Vin)
    #     #     outdict[str(int(nn)) + ' Ex'] = Vout[:, 0, 0]
    #     #     outdict[str(int(nn)) + ' Ey'] = Vout[:, 1, 0]
    #     # df = pd.DataFrame(outdict)
    #     # df.to_csv(strfile1, index=False)
    #
    #
    #     ######## FM error############
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     lines = []
    #     # ax.set_prop_cycle(cc)
    #
    #     ax.set_xlabel(r'FM angle error $\theta_{err}(\degree)$')
    #     ax.set_ylabel(r'SOP deviation ($\degree$)')
    #
    #     # ax.set(xlim=(0, 45), ylim=(0, 12))
    #     ax.set(xlim=(0, 45), ylim=(0, 12))
    #
    #     ax.yaxis.set_major_locator(MaxNLocator(6))
    #     ax.xaxis.set_major_locator(MaxNLocator(10))
    #
    #     ax.xaxis.set_major_formatter(OOMFormatter(0, "%1.0f"))
    #     ax.yaxis.set_major_formatter(OOMFormatter(0, "%2.1f"))
    #
    #     ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    #     ax.grid(ls='--', lw=0.5)
    #
    #     fig.subplots_adjust(hspace=0.4,left =0.17, right=0.95, top=0.93, bottom=0.2)
    #
    #     ######################
    #
    #
    #     data = pd.read_csv(strfile1)
    #     fig3, lines3, opacity = None, None, 1
    #
    #     # color palette prepared for each input SOPs with plotly library
    #     colors_Viridis_tmp = px.colors.sample_colorscale("Viridis", [n / (len(ang_FM) - 1) for n in range(len(ang_FM))])
    #     colors_Viridis = cm_to_rgba_tuple(colors_Viridis_tmp)
    #     out_ellip = zeros(len(ang_FM))
    #
    #     for nn in range(int(data.shape[1] / 2)):
    #
    #         str_Ex = str(nn) + ' Ex'
    #         str_Ey = str(nn) + ' Ey'
    #         Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
    #                          [complex(y) for y in data[str_Ey].to_numpy()]])
    #         E.from_matrix(Vout)
    #         S.from_Jones(E)
    #         #draw_stokes_points(fig1[0], S, kind='scatter', color_scatter='r')
    #         fig3, lines3 = plot_Stokes_pnt2(S, fig=fig3, lines=lines3, opacity=opacity, color_pnt=rgb2hex(colors_Viridis[nn]))
    #         ellip = S.parameters.ellipticity_angle()
    #         out_ellip[-1-nn] = ellip.max() - ellip.min()
    #
    #     ax.plot(ang_FM, out_ellip*180/pi, 'k')
    #     #fig3.update_traces(marker_size=3)
    #     colorbar_param = dict(lenmode='fraction', len=0.75, thickness=10, tickfont=dict(size=20),
    #                           tickvals=np.linspace(0, len(ang_FM), 4),
    #                           ticktext=['0', '15', '30', '45'],
    #                           # title='Azimuth angle',
    #                           outlinewidth=1,
    #                           x=0.2)
    #     colorbar_trace = go.Scatter(x=[None], y=[None],
    #                                 mode='markers',
    #                                 marker=dict(
    #                                     colorscale='Viridis',
    #                                     showscale=True,
    #                                     cmin=0,
    #                                     cmax=len(ang_FM),
    #                                     colorbar=colorbar_param
    #                                 ),
    #                                 hoverinfo='none'
    #                                 )
    #     fig3.add_trace(colorbar_trace)
    #     fig3['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
    #     fig3['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
    #     fig3.update_yaxes(showticklabels=False, showgrid=False, visible=False)
    #     fig3.update_xaxes(showticklabels=False, showgrid=False, visible=False)
    #
    #
    #     #fig2, ax2, lines = spunfiber.plot_error(strfile1)
    #
    #     # labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
    #     #              ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]
    #     # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
    #
    #     #fig3, ax3, lines3 = plot_error_byfile2(strfile1 + "_S")
        fig3.show()

    elif mode == 3:

        V_strfile = ['Lobi_45FM_errdeg1x5_220601.csv',
                     'Hibi_45FM_errdeg1x5_220601.csv']

        V_label = ['Ideal FM', 'Nonideal ']
        V2 = 0.54 * 4 * pi * 1e-7 * 2

        fig, ax, lines, isEOF, nn = None, None, None, False, 0
        fig3, lines3, opacity = None, None, 0.5
        fig10, ax10, lines10 = None, None, []
        c = np.array([None, None, None])


        for strfile1 in V_strfile:
            dic_err = {}
            nn = 0
            while isEOF is False:
                V_I, S, isEOF = load_stokes_fromfile(strfile1 + "_S", nn)

                Vtmp  = V2 if strfile1 == 'Lobi_45FM_errdeg1x5_220531.csv' else V2 * 0.962
                if nn == 0:
                    dic_err['V_I'] = V_I
                dic_err[str(nn)] = cal_error_fromStocks(V_I, S, V_custom=Vtmp)
                c = np.array([None, None, None])
                nn += 1
                # if nn > 0:
                #     break

            fig10, ax10, lines10 = plot_errorbar_byDic(dic_err, fig=fig10, ax=ax10, lines=lines10, init_index=21)
            isEOF = False
        # ax10.legend(lines10, ['ITER specification', '',
        #                                  r'Ideal FM $\theta_{err}$=0$\degree$',
        #                                  r'$\theta_{err}$=1$\degree$', '', '','', ''])

        ax10.legend( [lines10[0], lines10[4], lines10[7]],
                     ['ITER specification',
                      'lo-bi spun fiber',
                      'hi-bi spun fiber'],
                     bbox_to_anchor=(1.01, 1.01), loc="upper left")
        #fig10.savefig("fig9.(b).jpg", dpi=330)

        # print(legend)
        # r'$\theta_{err}$=20$\degree$'])
        # r'$\theta_{err}$=45$\degree$'])

        #ax.legend()
    # elif mode ==5:
    #
    #     strfile1 = 'FMerror.csv'
    #     num_iter = 500
    #
    #     nM_vib = 5
    #     #ang_FM = np.arange(0,46,1)
    #     ang_FM = np.arange(0, 46, 1)
    #
    #     E = Jones_vector('input')
    #     E1 = Jones_vector('output')
    #     azi = np.array([0, pi / 6, pi / 4])
    #     E.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)
    #     #fig1, ax1 = spunfiber.init_plot_SOP()
    #     S = create_Stokes('O')
    #
    #     # outdict = {}
    #     # for nn in ang_FM:
    #     #     Vin = E[0].parameters.matrix()
    #     #     print(nn)
    #     #     Vout = spunfiber.cal_2ndBridge(45+nn/10, num_iter, Vin=Vin)
    #     #     outdict[str(int(nn)) + ' Ex'] = Vout[:, 0, 0]
    #     #     outdict[str(int(nn)) + ' Ey'] = Vout[:, 1, 0]
    #     # df = pd.DataFrame(outdict)
    #     # df.to_csv(strfile1, index=False)
    #
    #
    #     ######## FM error############
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     lines = []
    #     # ax.set_prop_cycle(cc)
    #
    #     ax.set_xlabel(r'FM angle error $\theta_{err}(\degree)$')
    #     ax.set_ylabel(r'SOP deviation ($\degree$)')
    #
    #     # ax.set(xlim=(0, 45), ylim=(0, 12))
    #     ax.set(xlim=(0, 45), ylim=(0, 12))
    #
    #     ax.yaxis.set_major_locator(MaxNLocator(6))
    #     ax.xaxis.set_major_locator(MaxNLocator(10))
    #
    #     ax.xaxis.set_major_formatter(OOMFormatter(0, "%1.0f"))
    #     ax.yaxis.set_major_formatter(OOMFormatter(0, "%2.1f"))
    #
    #     ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
    #     ax.grid(ls='--', lw=0.5)
    #
    #     fig.subplots_adjust(hspace=0.4,left =0.17, right=0.95, top=0.93, bottom=0.2)
    #
    #     ######################
    #
    #
    #     data = pd.read_csv(strfile1)
    #     fig3, lines3, opacity = None, None, 1
    #
    #     # color palette prepared for each input SOPs with plotly library
    #     colors_Viridis_tmp = px.colors.sample_colorscale("Viridis", [n / (len(ang_FM) - 1) for n in range(len(ang_FM))])
    #     colors_Viridis = cm_to_rgba_tuple(colors_Viridis_tmp)
    #     out_ellip = zeros(len(ang_FM))
    #
    #     for nn in range(int(data.shape[1] / 2)):
    #
    #         str_Ex = str(nn) + ' Ex'
    #         str_Ey = str(nn) + ' Ey'
    #         Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
    #                          [complex(y) for y in data[str_Ey].to_numpy()]])
    #         E.from_matrix(Vout)
    #         S.from_Jones(E)
    #         #draw_stokes_points(fig1[0], S, kind='scatter', color_scatter='r')
    #         fig3, lines3 = plot_Stokes_pnt2(S, fig=fig3, lines=lines3, opacity=opacity, color_pnt=rgb2hex(colors_Viridis[nn]))
    #         ellip = S.parameters.ellipticity_angle()
    #         out_ellip[-1-nn] = ellip.max() - ellip.min()
    #
    #     ax.plot(ang_FM, out_ellip*180/pi, 'k')
    #     #fig3.update_traces(marker_size=3)
    #     colorbar_param = dict(lenmode='fraction', len=0.75, thickness=10, tickfont=dict(size=20),
    #                           tickvals=np.linspace(0, len(ang_FM), 4),
    #                           ticktext=['0', '15', '30', '45'],
    #                           # title='Azimuth angle',
    #                           outlinewidth=1,
    #                           x=0.2)
    #     colorbar_trace = go.Scatter(x=[None], y=[None],
    #                                 mode='markers',
    #                                 marker=dict(
    #                                     colorscale='Viridis',
    #                                     showscale=True,
    #                                     cmin=0,
    #                                     cmax=len(ang_FM),
    #                                     colorbar=colorbar_param
    #                                 ),
    #                                 hoverinfo='none'
    #                                 )
    #     fig3.add_trace(colorbar_trace)
    #     fig3['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
    #     fig3['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
    #     fig3.update_yaxes(showticklabels=False, showgrid=False, visible=False)
    #     fig3.update_xaxes(showticklabels=False, showgrid=False, visible=False)
    #
    #
    #     #fig2, ax2, lines = spunfiber.plot_error(strfile1)
    #
    #     # labelTups = [('Stacking matrix (dz = SP/25)', 0), ('Lamming method with small step (dz = SP/25)', 1),
    #     #              ('Lamming method for whole fiber (dz = L)', 2), ('Iter specification', 3)]
    #     # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
    #
    #     #fig3, ax3, lines3 = plot_error_byfile2(strfile1 + "_S")
    #     fig3.show()

    plt_fmt, plt_res = '.png', 330  # 330 is max in Word'16
    plt.rcParams["axes.titlepad"] = 5  # offset for the fig title
    # plt.rcParams["figure.autolayout"] = True  # tight_layout
    #  plt.rcParams['figure.constrained_layout.use'] = True  # fit legends in fig window
    fsize = 9
    plt.rc('font', size=fsize)  # controls default text sizes
    plt.rc('axes', labelsize=fsize)  # f-size of the x and y labels
    plt.rc('xtick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('ytick', labelsize=fsize)  # f-size of the tick labels
    plt.rc('legend', fontsize=fsize - 1)  # f-size legend
    plt.rc('axes', titlesize=11)  # f-size of the axes title (??)
    plt.rc('figure', titlesize=11)  # f-size of the figure title
    plt.show()


