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
from brokenaxes import brokenaxes
from matplotlib.colors import rgb2hex

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy.interpolate import CubicSpline

start = time.process_time()
# from My_Library import SPUNFIBRE_lib

from My_Library.SPUNFIBER3_lib import *
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


# try:
#     sys.path.append(os.getcwd() + '\My_library')
#     os.chdir(os.getcwd() + '\My_library')
#     print("os path is changed")
# except:
#     print("ggg")


if __name__ == '__main__':
    sys.path.append(os.getcwd() + '\My_library')
    print(os.getcwd())
    os.chdir(os.getcwd() + '\My_library')
    print(os.getcwd())

    mode = 0
    LB = 1.000
    SP = 0.005
    # dz = SP / 1000
    dz = 0.0001
    len_lf = 6  # lead fiber
    len_ls = 28  # sensing fiber
    angle_FM = 45
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls, angle_FM)

    # strfile1 = 'lobi_45FM_errdeg1x5_220622_LF_1000_500.csv'
    # strfile2 = 'lobi_45FM_errdeg1x5_220622_LF_1000_1000.csv'

    strfile1 = 'test.csv'

    V_strfile = [strfile1]
    V_iter = [3]
    # V_strfile = [strfile1, strfile2, strfile3, strfile4]
    V_angFM = [45]
    # strfile1 = 'Hibi_test.csv'
    V_temp = [20+ 273.15, 92+ 273.15, None]
    if mode == 0:

        num_iter = 1
        num_processor = 8
        V_I = np.hstack((np.zeros(1),np.logspace(0,5, 20), np.arange(0.1e6, 18e6, 0.2e6)))
        # V_I = arange(0e6, 18e6 + 0.1e6, 0.1e6)
        # V_I = np.hstack((np.arange(0e6, 0.1e6, 0.005e6), np.arange(0.1e6, 18e6, 0.2e6)))
        # V_I = arange(0e6, 4e6 + 0.1e6, 0.1e6)

        nM_vib = 0
        start = pd.Timestamp.now()

        # Load the temperature distribution along the VV (20 cm with a clamp)
        data = pd.read_csv('VVtemp.csv', delimiter=';')
        l_vv = data['L'].to_numpy() / 100
        temp_vv = data['TEMP'].to_numpy()
        F_temp_interp = CubicSpline(l_vv, temp_vv)

        # Spunfiber model initialization
        spunfiber.set_Vectors()
        # Spunfiber model with temperature information
        spunfiber.set_tempVV(l_vv[0], l_vv[-1], F_temp_interp)
        # No_vib matrix
        spunfiber.create_Mvib(nM_vib, 1, 1)

        xx = 0
        for xx, strfile1 in enumerate(V_strfile):

            # ang_FM = 45
            ang_FM = V_angFM[xx]
            num_iter = V_iter[xx]

            azi = np.array([0, pi/6, pi/4])
            spunfiber.set_Vin([0], 0)
            for nn in range(num_iter):
                #V_out = spunfiber.calc_mp(num_processor, V_I)
                V_out = spunfiber.calc_sp(V_I, V_temp[nn])
                save_Jones2(strfile1,V_I,V_out)

                checktime = pd.Timestamp.now() - start
                print(nn, "/", num_iter, checktime)
                start = pd.Timestamp.now()

            # 1, plot error directly from file
            fig, ax, lines = None, None, None
            fig, ax, lines = plot_error_byfile2(strfile1 + "_S")

            labelTups = [('20 degC', 0), ('100 degC', 1),
                         ('Temp. distribution', 2), ('Iter specification', 3)]
            ax.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))

            # fig3, ax3, lines3 = plot_error_byfile2(strfile1+"_S", V_custom=0.54 * 4 * pi * 1e-7*2)

        # plot the temperature effect
        fig_temp, ax_temp = plt.subplots(3, 1, figsize=(8, 5))
        fig_temp.subplots_adjust(hspace=0.32, left=0.24)
        ax_temp[0].plot(spunfiber.V_L, spunfiber.V_temp-273.15)
        ax_temp[0].plot(spunfiber.V_L, 20+ (spunfiber.V_temp - 273.15)*0)
        ax_temp[0].plot(spunfiber.V_L, 100 + (spunfiber.V_temp - 273.15) * 0)

        ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB*spunfiber.V_delta_temp))
        ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB + 0*spunfiber.V_delta_temp))
        ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB *(1 + 3e-5 * (373.15 - 273.15 - 20)) + 0 * spunfiber.V_delta_temp))

        r = spunfiber.L / (2 * pi)
        V_H = V_I[-1] / (2 * pi * r) * ones(len(spunfiber.V_temp))
        ax_temp[2].plot(spunfiber.V_L, spunfiber.V * spunfiber.V_f_temp)
        ax_temp[2].plot(spunfiber.V_L, spunfiber.V + 0* spunfiber.V_f_temp)
        ax_temp[2].plot(spunfiber.V_L, spunfiber.V*(1 + 8.1e-5 * (373.15 - 273.15 - 20)) + 0 * spunfiber.V_f_temp)

        print('avg V :', spunfiber.V * spunfiber.V_f_temp.mean())
        print('avg T :', spunfiber.V_temp.mean())
        ax_temp[0].set(xlim=(0, 5.5), ylim=(18,110))
        ax_temp[1].set(xlim=(0, 5.5), ylim=(0.9995, 1.003))
        ax_temp[2].set(xlim=(0, 5.5), ylim=(0.6780e-6, 0.6835e-6))
        #ax.yaxis.set_major_formatter(OOMFormatter(0, "%3.2f"))
        ax_temp[1].yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))
        ax_temp[2].yaxis.set_major_formatter(OOMFormatter(-6, "%5.4f"))
        ax_temp[0].set_ylabel('Temperature \n(degC)')
        ax_temp[1].set_ylabel('Beatlength \n(m)')
        ax_temp[2].set_ylabel('Verdet constant  \n(rad/A)')
        ax_temp[2].set_xlabel('Fiber position (m)')
        fig_temp.align_ylabels(ax_temp)

        for nn in range(3):
            ax_temp[nn].spines['right'].set_visible(False)

        # # hide the spines between ax and ax2
        # ax.spines['right'].set_visible(False)
        # ax2.spines['left'].set_visible(False)
        # ax.yaxis.tick_left()
        # ax.tick_params(labelright='off')
        # ax2.yaxis.tick_right()
        #
        # d = .015  # how big to make the diagonal lines in axes coordinates
        # # arguments to pass plot, just so we don't keep repeating them
        # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        # ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        #
        # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        # ax2.plot((-d, +d), (-d, +d), **kwargs)

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
        # 1, plot error directly from file
        fig, ax, lines = None, None, None
        fig, ax, lines = plot_error_byfile2(strfile1 + "_S")

        labelTups = [('Uniform dist. 20 degC', 0), ('Uniform dist. 100 degC', 1),
                     ('From temp. simulation data', 2), ('Iter specification', 3)]
        ax.legend(lines, [lt[0] for lt in labelTups], loc='upper right')
        ax.set(ylim=(0, 0.012))

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


