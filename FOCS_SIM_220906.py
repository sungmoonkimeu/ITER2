'''
Created 2022.09.06
for temperature effect with non-uniform magnetic field

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

if __name__ == '__main__':
    sys.path.append(os.getcwd() + '\My_library')
    print(os.getcwd())
    os.chdir(os.getcwd() + '\My_library')
    print(os.getcwd())

    mode = 1
    LB = 1.000
    SP = 0.005
    # dz = SP / 1000
    dz = 0.001
    len_lf = 6  # lead fiber
    len_ls = 29.975 # sensing fiber
    angle_FM = 45
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls, angle_FM)

    strfile1 = 'FOCS_uniform_T92_uniform_B.csv'
    strfile2 = 'FOCS_uniform_T20_uniform_B.csv'
    strfile3 = 'FOCS_nonuniform_T_uniform_B.csv'
    strfile4 = 'FOCS_nonuniform_T_nonuniform_B.csv'

    # V_strfile = [strfile1, strfile2, strfile3, strfile4]
    # V_iter = [1, 1, 1, 1]
    # V_angFM = [45, 45, 45, 1]
    # V_temp = [92.052056 + 273.15, 20 + 273.15, None, None]
    # V_nonuniform= [None, None, 'T', 'MT']

    V_strfile = [strfile4]
    V_iter = [1]
    V_angFM = [45]
    V_temp = [None]
    V_nonuniform= ['MT']

    if mode == 0:

        num_iter = 1
        num_processor = 8
        V_I = np.hstack((np.zeros(1),np.logspace(0,5, 5), np.arange(0.1e6, 18e6, 0.2e6)))

        nM_vib = 0
        start = pd.Timestamp.now()

        # Load the temperature distribution along the VV (20 cm with a clamp)
        data = pd.read_csv('VVtemp.csv', delimiter=';')
        l_vv = data['L'].to_numpy() / 100
        temp_vv = data['TEMP'].to_numpy()
        F_temp_interp = CubicSpline(l_vv, temp_vv)

        # Load the magnetic field distribution along the VV (30 m)
        strfile_B = 'B-field_around_VV.txt'
        data_B = np.loadtxt(strfile_B)
        F_B_interp = scipy.interpolate.interp1d(data_B[:,0], data_B[:,1], kind='cubic')

        # Spunfiber model initialization
        spunfiber.set_Vectors()
        # Spunfiber model with temperature information
        spunfiber.set_tempVV(l_vv[0], l_vv[-1], F_temp_interp)
        print(spunfiber.f_temp_avg)

        # No_vib matrix
        spunfiber.create_Mvib(nM_vib, 1, 1)
        # nonunifrom Magnetic field
        spunfiber.set_B(F_B_interp)
        print(((spunfiber.int_V_B / (4 * pi * 1e-7)) / 15000000))

        xx = 0
        for xx, strfile in enumerate(V_strfile):

            # ang_FM = 45
            ang_FM = V_angFM[xx]
            num_iter = V_iter[xx]

            azi = np.array([0, pi/6, pi/4])
            spunfiber.set_Vin([0], 0)

            V_out = spunfiber.calc_sp(V_I, V_temp[xx], V_nonuniform[xx])

            if V_nonuniform[xx] is None:
                save_Jones2(strfile, V_I, V_out)
            elif V_nonuniform[xx] == 'T':
                save_Jones2(strfile, V_I, V_out)
            elif V_nonuniform[xx]== 'MT' or 'M':
                V_Itotal = V_I * ((-spunfiber.int_V_B/(4*pi*1e-7))/15000000)
                save_Jones2(strfile, V_Itotal, V_out)

            if V_nonuniform[xx] == 'MT' or 'T':
                # V_custom = spunfiber.V*spunfiber.f_temp_avg*2
                V_custom = None

            print("temperature effect on V: ", V_custom)
            checktime = pd.Timestamp.now() - start
            print(xx, "/", num_iter, checktime)
            start = pd.Timestamp.now()

            # 1, plot error directly from file
            if xx == 0:
                fig, ax, lines = None, None, None
            fig, ax, lines = plot_error_byfile3(strfile + "_S", fig, ax, lines,
                                                V_custom=V_custom,
                                                I_custom=None
                                                )

        labelTups = [('Iter specification', 0),
                     ('Uniform 92degC', 1),
                     ('Uniform 20degC', 2),
                     ('Nonuniform_T', 3),
                     ('Nonuniform_BT', 4)]

        # labelTups = [('Nonuniform Temp. & Nonuniform B-field', 1), ('Iter specification', 2)]
        # ax.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
        ax.legend(lines, [lt[0] for lt in labelTups], loc='lower right')


        # plot the temperature effect
        fig_temp, ax_temp = plt.subplots(4, 1, figsize=(8, 5))
        fig_temp.subplots_adjust(hspace=0.32, left=0.24)
        ax_temp[0].plot(spunfiber.V_L, spunfiber.V_temp-273.15)
        # ax_temp[0].plot(spunfiber.V_L, 20+ (spunfiber.V_temp - 273.15)*0)
        # ax_temp[0].plot(spunfiber.V_L, 100 + (spunfiber.V_temp - 273.15) * 0)

        ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB*spunfiber.V_delta_temp))
        # ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB + 0*spunfiber.V_delta_temp))
        # ax_temp[1].plot(spunfiber.V_L, (spunfiber.LB *(1 + 3e-5 * (373.15 - 273.15 - 20)) + 0 * spunfiber.V_delta_temp))

        r = spunfiber.L / (2 * pi)
        V_H = V_I[-1] / (2 * pi * r) * ones(len(spunfiber.V_temp))
        ax_temp[2].plot(spunfiber.V_L, spunfiber.V * spunfiber.V_f_temp)
        # ax_temp[2].plot(spunfiber.V_L, spunfiber.V + 0* spunfiber.V_f_temp)
        # ax_temp[2].plot(spunfiber.V_L, spunfiber.V*(1 + 8.1e-5 * (373.15 - 273.15 - 20)) + 0 * spunfiber.V_f_temp)

        ax_temp[3].plot(spunfiber.V_L, spunfiber.V_B /5, label='3MA')
        ax_temp[3].plot(spunfiber.V_L, spunfiber.V_B/3, label='5MA')
        ax_temp[3].plot(spunfiber.V_L, spunfiber.V_B/1, label='15MA')

        print('avg V :', spunfiber.V * spunfiber.V_f_temp.mean())
        print('avg T :', spunfiber.V_temp.mean())
        xmax = 29
        ax_temp[0].set(xlim=(0, xmax), ylim=(18, 110))
        ax_temp[1].set(xlim=(0, xmax), ylim=(0.9995, 1.003))
        ax_temp[2].set(xlim=(0, xmax), ylim=(0.6780e-6, 0.6835e-6))
        ax_temp[3].set(xlim=(0, xmax), ylim=(-2, 2))
        #ax.yaxis.set_major_formatter(OOMFormatter(0, "%3.2f"))
        ax_temp[1].yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))
        ax_temp[2].yaxis.set_major_formatter(OOMFormatter(-6, "%5.4f"))
        ax_temp[0].set_ylabel('Temperature \n(degC)')
        ax_temp[1].set_ylabel('Beatlength \n(m)')
        ax_temp[2].set_ylabel('Verdet constant  \n(rad/A)')
        ax_temp[3].set_ylabel('B-field  \n(T)')
        ax_temp[3].set_xlabel('Fiber position (m)')
        ax_temp[3].legend()
        fig_temp.align_ylabels(ax_temp)

    elif mode == 2:

        V_strfile = [strfile1, strfile2, strfile3]
        xx = 0
        markers = ['o', 'v', '>']
        for xx, strfile in enumerate(V_strfile):
            if xx == 0:
                fig, ax, lines = None, None, None
            fig, ax, lines = plot_error_byfile3(strfile + "_S", fig, ax, lines,
                                                V_custom=None,
                                                I_custom=None,
                                                markers=markers[xx])
        labelTups = [('Iter specification', 0),
                     ('Uniform 92degC', 1),
                     ('Uniform 20degC', 2),
                     ('Nonuniform', 3)]
        ax.legend(lines, [lt[0] for lt in labelTups], loc='lower right')
        ax.set(ylim=(0, 0.012))



        V_strfile2 = [strfile1, strfile3]
        xx = 0
        markers = ['o', '>']
        for xx, strfile in enumerate(V_strfile2):

            V_custom = spunfiber.V*1.0058362165940955*2

            if xx == 0:
                fig, ax, lines = None, None, None
            fig, ax, lines = plot_error_byfile3(strfile + "_S", fig, ax, lines,
                                                V_custom=V_custom,
                                                I_custom=None,
                                                markers=markers[xx])

        labelTups = [('Iter specification', 0),
                     ('Uniform 92degC (V='+r'V$_{92^\circ C})$', 1),
                     ('Nonuniform (V='+r'V$_{92^\circ C})$', 2)]
        ax.legend(lines, [lt[0] for lt in labelTups], loc='lower right')
        ax.set(ylim=(0, 0.012))


        V_I = np.hstack((np.zeros(1),np.logspace(0,5, 5), np.arange(0.1e6, 18e6, 0.2e6)))
        xx = 0
        markers = ['o', 'x']
        fig, ax, lines = None, None, None
        V_custom = spunfiber.V * 1.0058362165940955 * 2

        fig, ax, lines = plot_error_byfile3(strfile4 + "_S", fig, ax, lines,
                                            V_custom=V_custom,
                                            I_custom=V_I[1:],
                                            markers=markers[0]
                                            )
        fig, ax, lines = plot_error_byfile3(strfile4 + "_S", fig, ax, lines,
                                            V_custom=V_custom,
                                            I_custom=None,
                                            markers=markers[1]
                                            )

        labelTups = [('Iter specification', 0),
                     ('Nonuniform T & Nonuniform B' + r' (I=I$_p$)', 1),
                     ('Nonuniform T & Nonuniform B' + r' (I=I$_{total}$)', 2)]
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



