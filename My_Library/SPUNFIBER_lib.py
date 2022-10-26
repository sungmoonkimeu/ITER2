# -*- coding: utf-8 -*-
"""
Created on Mon May 02 15:14:00 2022
@author: SMK

FOCS simulation with optical modeling of spun fiber

Note)
py_pol module is used for conversion between numpy array and Jones matrix and Stokes vector.
used py_pol version is 1.0.3
pip install py-pol==1.0.3 (Conda install does not work for this module)
"""
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector
from py_pol.stokes import create_Stokes
from py_pol.drawings import draw_stokes_points

import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
from multiprocessing import Process, Manager
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import os
import csv

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    """
    A patch of a function in matplotlib
    Control the number of digit with scientific expression of ticks in x or y axis in Matplotlib

    # See link below
    # Set scientific notation with fixed exponent and significant digits for multiple subplots
    # https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    """

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


def periodicf(li, lf, f, x):
    """ Create a new periodic function based on a given function [f]
    The value of the new function is determined by comparing the source function's length [lf-li] with
    the current position [x] of the new function.

    :param li: initial position of source function
    :param lf: finial position of source function
    :param f: callable source function f(x)
    :param x: accessing parameter to create the new function value
    :return the value of a new periodic fuction

    example:
    def func(x):
        return x*x
    x = np.array([0,1,2,3])

    new_x = np.array([0,1,2,3,4,5,6,7,8,9,10])
    new_y = np.array([periodicf(0, len(x), func, x2) for x2 in new_x])
    print(new_y)
    """

    if x >= li and x <= lf:
        return f(x)
    elif x > lf:
        x_new = x - (lf - li)
        return periodicf(li, lf, f, x_new)
    elif x < (li):
        x_new = x + (lf - li)
        return periodicf(li, lf, f, x_new)


class SPUNFIBER:
    ''' This is a CLASS to evaluate FOCS through optical modeling and simulation using Jones matrix formalism.
    The spun fiber is optically modeled taking into account various environmental conditions such as
    vibration, magnetic field, temperature.

    '''

    def __init__(self, beat_length, spin_pitch, delta_l, len_bf, len_sf, angle_FM=45):
        ''' To initialize the class with given parameters
        :param beat_length: intrinsic beatlengh of spun fiber (LB)
        :param spin_pitch: spun period of spun fiber (SP)
        :param delta_l: length of the piece of spun fiber (e.g. SP/100)
        :param len_lf: length of the fiber in Bridge section
        :param len_sf: length of the fiber in VV section (sensing fiber)
        :param angle_FM: angle of the Faraday mirror (ideally 45deg)
        '''
        self.LB = beat_length
        self.SP = spin_pitch
        self.dz = delta_l
        self.V = 0.54 * 4 * pi * 1e-7  # μV [rad/A]                 # Verdet constant
        # V=0.54 [rad/mT] --> μV = 4*pi*1e-7 * 0.54  [rad/A]
        self.len_bf = len_bf
        self.len_sf = len_sf
        self.V_SF = []                   # Vector of sensing fiber
        self.V_BF1 = []                  # Vector of 1st Brige fiber
        self.V_BF2 = []                  # Vector of 2nd Brige fiber
        self.V_theta = []                # vector of theta in sensing fiber
        self.V_theta_BF1 = []            # vector of theta in 1st bridge fiber
        self.V_theta_BF2 = []            # vector of theta in 2st bridge fiber
        self.V_temp = []                 # vector of temperautre along sensing fiber
        self.V_B = []                    # vector of magnetic field along sensing fiber
        self.ang_FM = angle_FM
        self.Vin = np.array([[1], [0]])
        self.Mvib = []
        self.is_spunfiber_set_with_Mvib_in_forward = False
        self.is_spunfiber_set_with_Mvib_in_backward = False
        self.int_V_B = 0

        self.V_SF = arange(0, self.len_sf + self.dz, self.dz)
        self.V_BF1 = arange(0, self.len_bf + self.dz, self.dz)
        self.V_BF2 = self.V_BF1

        s_t_r = 2 * pi / self.SP
        self.V_theta_BF1 = self.V_BF1 * s_t_r
        self.V_theta = self.V_theta_BF1[-1] + self.V_SF * s_t_r
        self.V_theta_BF2 = self.V_theta[-1] + self.V_BF1 * s_t_r

        self.V_H = 1 / self.len_sf * ones(len(self.V_theta))
        self.V_h1_f = zeros(len(self.V_theta_BF1))
        self.V_h2_f = zeros(len(self.V_theta_BF2))
        self.V_h2_b = zeros(len(self.V_theta_BF2))
        self.V_h1_b = zeros(len(self.V_theta_BF1))

        print('Initialized spun fiber model: '
              '\n-LB = ', self.LB,
              '\n-SP = ', self.SP,
              '\n-dz = ', self.dz,
              '\n-length of spun fiber in the Bridge section= ', self.len_bf,
              '\n-length of spun fiber around the VV = ', self.len_sf,
              '\n-structure : [1st_Bridge]--[VV]--[2nd_Bridge]')

        ''' 
        self.V_theta = self.V_SF * s_t_r
        print('Spun fiber model: \n\n[VV]\n')
        '''

        # Faraday mirror
        ksi = self.ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        self.M_FR = Rot @ Jm @ Rot
        print('-rotation angle of Faraday mirror : ', self.ang_FM, 'deg')

        # Plasma current (Ip)
        self.V_Ip = arange(0e6, 4e6 + 0.1e6, 0.2e6)
        print('-Plasma current values (Ip) to be used for simulation: V_Ip = ',
              self.V_Ip[0:5],'...',self.V_Ip[-5:-1])

        print('---------------------------------------------------------------\n')

    def set_input_current(self,I0,Imax,step):
        self.V_Ip = arange(I0,Imax+step, step)
        print("-input current (V_Ip) has been reset")
        print("V_Ip = ",
              self.V_Ip[0:5],'...', self.V_Ip[-5:-1])
        print('---------------------------------------------------------------\n')

    @staticmethod
    def _eigen_expm(A):
        """Calculate the exponential of 2x2 matices(A) (i.e. exp^(A))
        same with scify.linalg.expm() but it only works for single (2,2) matrix.
        This function is designed for (2,2,n) matrix

        :param A: 2 x 2 diagonalizable matrices
        :return: expm(A): calculate the exponential for each matrices of A
        """
        vals, vects = eig(A)
        return einsum('...ik, ...k, ...kj -> ...ij',
                      vects, np.exp(vals), np.linalg.inv(vects))

    def laming0(self, Ip, DIR, V_theta):
        """Calculation of Jones matrix of spun fiber when plasma current (Ip) flows.
        Spun fiber model was designed following Laming's paper
        Use of the infinisimal approximation of Jones matrix in Kapron's paper to calculate whole spun fiber
        1972, IEEE J. of Quantum Electronics,"Birefringence in dielectric optical wavegudies"

        :param Ip: plasma current
        :param DIR: direction of light propagation  (+1: forward, -1: backward)
        :param V_theta: vector of theta (angle of oprientation of optic axes for each sliced fiber section)
        :return: M matrix calculated from N matrix

        example: self.cal_Jout0

        """

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        H = Ip / self.len_sf
        rho = self.V * H

        # ----------------------Laming parameters--------------------------------- #
        n = 0
        m = 0
        # --------Laming: orientation of the local slow axis ------------

        qu = 2 * (s_t_r - rho) / delta
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper,
        # inorder to be consistant with anti-clockwise orientation (as in Jones paper)
        # for both spin and faraday rotation.

        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5
        omega = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) + n * pi
        R_z = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))

        M = np.array([[1, 0], [0, 1]])

        if DIR == 1:
            V_phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta

        elif DIR == -1:
            V_phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + np.flip(V_theta)

        n11 = R_z / 2 * 1j * cos(2 * V_phi)
        n12 = R_z / 2 * 1j * sin(2 * V_phi) - omega
        n21 = R_z / 2 * 1j * sin(2 * V_phi) + omega
        n22 = R_z / 2 * -1j * cos(2 * V_phi)

        N = np.array([[n11, n21], [n12, n22]]).T  # See Note/Note 2(building array and reshape).jpg
        N_integral = self._eigen_expm(N)

        for nn in range(len(V_theta) - 1):
            M = N_integral[nn] @ M

        return M

    def laming1(self, Ip, DIR, V_theta):
        """Calculation of Jones matrix of spun fiber when plasma current (Ip) flows.
        Spun fiber model was designed following Laming's paper
        --NOT USING THE INFINISIMAL MODEL--

        :param Ip: plasma current
        :param DIR: direction of light propagation  (+1: forward, -1: backward)
        :param V_theta: vector of theta (angle of oprientation of optic axes for each sliced fiber section)
        :return: Single 2x2 Jones matrix

        example: self.cal_Jout1
        """

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB
        V_delta = delta * ones(len(V_theta))
        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        # H = Ip / self.len_sf
        # rho = self.V * H

        m = 0
        # magnetic field in unit length
        r = self.len_sf / (2 * pi)
        V_H = Ip / (2 * pi * r) * ones(len(V_theta))
        V_rho = self.V * V_H  # <<- Vector

        # --------Laming: orientation of the local slow axis ------------
        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector


        V_n = zeros(len(V_rho))  # for compensating the 2pi ambiguity in arctan calculation of V_omega
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) - 0.5 * (1 + DIR)
            else:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) + 0.5 * (1 - DIR)

        V_omega = s_t_r * self.dz + \
                  arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector

        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector

        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += V_theta if DIR == 1 else np.flip(V_theta)

        # Not using N-matrix technique
        m11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        m12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[m11, m21], [m12, m22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that the result of [[n11,n21],[n21,n22]].T is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.
        # See Note2 in Note folder

        M = np.array([[1, 0], [0, 1]])
        for nn in range(len(self.V_theta) - 1):
            M = M_omega[nn] @ M_R[nn] @ M
        return M

    def cal_Jout0(self, num = 0, dic_Jout=None, V_Ip=None, Jin=None, model=1):
        """ Calcuate the output Jones vector for each Ip

        :param num: index of dictionary of Vout_dic (default: num = 0 --> Not using the multiprocessing)
        :param dic_Jout: output Jones vector   (default: Vout_dic = None --> Not using the multiprocessing)
        :param V_Ip: Plasma current (Ip) vector (default: None --> using the initialized vector)
        :param Jin: Input Jones vector (default: None) --> using LHP (np.array([[1],[0]]))
        :param model: Selection of the Laming model
                      (0: laming0 (use of N-technique), 1: laming1 (not using N-technique))
                      default = 1 (laming1)

        :return:
        Case 1) normal calculation --> output Jones vectors (see example #1)
        Case 2) multiprocssing calculation --> No return    (see example #2)

        """

        if V_Ip is None:
            V_Ip = self.V_Ip

        if model == 0:
            laming = self.laming0
        else:
            laming = self.laming1

        V_Jout = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        if Jin is None:
            Jin = np.array([[1],[0]])
        mm = 0
        for Ip in V_Ip:

            M_lf_f = laming(0, 1, self.V_theta_BF1)
            M_f = laming(Ip, 1, self.V_theta)
            M_lf_f2 = laming(0, 1, self.V_theta_BF2)
            M_lf_b2 = laming(0, -1, self.V_theta_BF2)
            M_b = laming(Ip, -1, self.V_theta)
            M_lf_b = laming(0, -1, self.V_theta_BF1)

            if num == 0 and Ip == V_Ip[0]:
                print("Verification of Matrix Calculation bewteen the foward and backward propagation")
                print("1) When Ip = 0A, the two matrices must be transposed to each other.")
                print("--> Calcuate the difference of M_f and M_b when Ip = 0A")
                print("dz = ",self.dz)
                print("--> Norm (M_f - M_b) = ", norm(M_lf_f - M_lf_b.T))

            if Ip == V_Ip[-1] and self.LB > 10000000:
                print("2) When LB is the infinite, the two matrices must be the same")
                print("--> Calcuate the difference of M_f and M_b when LB > 10000000")
                print("--> Norm (M_f - M_b) = ", norm(M_f - M_b))

            V_Jout[mm] = M_lf_b @ M_b @ M_lf_b2 @ self.M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Jin

            mm = mm + 1
            print("process [",num,"], ",mm,"/",len(V_Ip))

        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout

    def eval_FOCS_fromJones(self, V_Jout, V_Itotal=None, V_custom=None, angle_init=None, FOCSTYPE=2):
        """ evaluation of FOCS for given paramters
        Calculate the measured current from the SOP rotation that FOCS will be measured
        Convert the SOP rotation to Plasma current by using the Verdet constant

        :param V_Jout: output Jones vectors calculated from FOCS simulation
        :param V_Itotal: Total current (default: None --> using the Plasma current, self.V_Ip)
        :param V_custom: Customized Verdet constant (default: None --> using initial value, self.V)
        :param angle_init: Customized initial rotation angle
                           (default=None --> using the rotation angle of 0A plasma current)
        :param FOCSTYPE: Choose the FOCS type (default=2, 1: Transimission type, 2: reflection type)
        :return: V_IFOCS, V_err
                V_IFOCS: FOCS output (measured plasma current by converting the SOP rotation)
                V_err: relative erorr (V_IFOCS- V_Iref)/V_Iref
        """
        V_Iref = self.V_Ip if V_Itotal is None else V_Itotal
        if V_Iref[0] != 0:
            print("V_Iref[0] must be 0A for referencing the initial point")
        V_ang, V_IFOCS = zeros(len(V_Iref)), zeros(len(V_Iref))

        J = Jones_vector('Output')
        J.from_matrix(V_Jout)

        V = self.V if V_custom is None else V_custom
        m = 0
        for kk in range(len(V_ang)):
            if kk > 0 and J[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 0 and J[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = J[kk].parameters.azimuth() + m * pi

            c = V_ang[0] if angle_init is None else angle_init
            V_IFOCS[kk] = (V_ang[kk] - c) / (V * FOCSTYPE)

        V_err = (V_IFOCS[1:]-V_Iref[1:])/V_Iref[1:]

        return V_IFOCS, V_err

    def plot_error(self, V_err, V_Iref=None, fig=None, ax=None, lines=None, label=None):
        """ Plot the calculated relative error

        :param V_err: Calculated error
        :param V_Iref: Reference current (Ip or Itotal) self.V_Ip will be used if None
        :param fig: figure of Matplotlib. New variable will be created if None
        :param ax: axis of Matplotlib. New variable will be created if None
        :param lines: lines to access the label. New variable will be created if None
        :param label: legend of error data
        :return: fig, ax, lines to overlab the graph
        """
        V_Iref = self.V_Ip if V_Iref is None else V_Iref
        if len(V_err) != len(V_Iref)-1:
            print("Error!!: Check array size of V_err and V_Iref")
            print("len(V_err) = len(V_Iref) -1")

        V_Iref = V_Iref[1:]

        if fig is None and ax is None and lines is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            lines = []

            # Requirement specificaion for ITER
            absErrorlimit = zeros(len(V_Iref))
            relErrorlimit = zeros(len(V_Iref))

            # Calcuation ITER specification
            for nn in range(len(V_Iref)):
                if V_Iref[nn] < 1e6:
                    absErrorlimit[nn] = 10e3
                else:
                    absErrorlimit[nn] = V_Iref[nn] * 0.01
                relErrorlimit[nn] = absErrorlimit[nn] / V_Iref[nn]

            lines += ax.plot(V_Iref, relErrorlimit, 'r', label='ITER specification')

        lines += ax.plot(V_Iref, V_err, label=label)

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

    def save_Jones(self, filename, V_Jout, V_Iref=None, append=True):
        """ save the output Jones vectors from FOCS simulation
            If there is the same file,
            the data will be saved in the last column of data file (when append=Ture)

        :param filename: file name
        :param V_Jout: numpy array of output Jones vector
        :param V_Iref: reference current (default=None : self.V_Ip)
        :param append: selecte save mode (defualt=True)
        """
        V_I = self.V_Ip if V_Iref is None else V_Iref
        if os.path.exists(filename) and append is True:
            df2 = pd.read_csv(filename)
            ncol2 = int((df2.shape[1] - 1) / 2)
            df2[str(ncol2) + ' Ex'] = V_Jout[:, 0, 0]
            df2[str(ncol2) + ' Ey'] = V_Jout[:, 1, 0]
        else:
            out_dict2 = {'Iref': V_I}
            out_dict2['0 Ex'] = V_Jout[:, 0, 0]
            out_dict2['0 Ey'] = V_Jout[:, 1, 0]
            df2 = pd.DataFrame(out_dict2)
        df2.to_csv(filename, index=False)

    def load_Jones(self, filename, ncol=0):
        """ load Jones vector from file

        :param filename: file name
        :param ncol: column number to load (default: 0)

        :return: V_Iref, V_J, isEOF
                 V_Iref: reference current
                 V_J: loaded Jones vector
                 isEOF: returns TRUE after accessing the last column of file
        """
        data = pd.read_csv(filename)

        V_Iref = data['Iref']
        for nn in range(int((data.shape[1] - 1) / 2)):
            if nn == ncol + 1:
                break
            str_Ex = str(nn) + ' Ex'
            str_Ey = str(nn) + ' Ey'
            V_Jout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                             [complex(y) for y in data[str_Ey].to_numpy()]])

        isEOF = True if ncol >= int((data.shape[1] - 1) / 2) - 1 else False

        return V_Iref, V_Jout, isEOF

    def cal_Jout0_mp(self, num_processor):
        """ FOCS simulation using multiprocessing technique

        :param num_processor: number of processor to use in multiprocess
        :return: Calculated output Jones vector for each input Plasma current
        """
        spl_I = np.array_split(self.V_Ip, num_processor)

        procs = []
        manager = Manager()
        dic_Jout = manager.dict()

        # print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            # proc = Process(target=self.cal_rotation,
            proc = Process(target=self.cal_Jout0,
                           args=(num, dic_Jout, spl_I[num], None, 1))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        V_Jout = dic_Jout[0]
        for kk in range(num_processor - 1):
            V_Jout = np.vstack((V_Jout, dic_Jout[kk + 1]))

        return V_Jout

    def draw_empty_poincare(self, title=None):
        """ plot empty Poincare Sphere, ver. 20/03/2020
        Created on Fri Jan 14 11:50:03 2022
        @author: agoussar
        :param title: figure title
        :return: ax of figure
        """


        fig = plt.figure(figsize=(6, 6))
        #    plt.figure(constrained_layout=True)
        ax = fig.add_subplot(projection='3d')

        # white panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # no ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # no panes
        ax.set_axis_off()

        # plot greed
        u = np.linspace(0, 2 * np.pi, 61)  # azimuth
        v = np.linspace(0, np.pi, 31)  # elevation
        sprad = 1
        x = sprad * np.outer(np.cos(u), np.sin(v))
        y = sprad * np.outer(np.sin(u), np.sin(v))
        z = sprad * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z,
                        color='w',  # (0.5, 0.5, 0.5, 0.0),
                        edgecolor='k',
                        linestyle=(0, (5, 5)),
                        rstride=3, cstride=3,
                        linewidth=.5, alpha=0.0, shade=0, zorder=1)

        # main circles
        ax.plot(np.sin(u), np.cos(u), np.zeros_like(u), 'g-.', linewidth=0.75, zorder=0)  # equator
        #    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), 'b-', linewidth=0.5)
        #    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), 'b-', linewidth=0.5)

        # axes and captions
        amp = 1.3 * sprad
        ax.plot([-amp, amp], [0, 0], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=0)
        ax.plot([0, 0], [-amp, amp], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=0)
        ax.plot([0, 0], [0, 0], [-amp, amp], 'k-.', lw=1, alpha=0.5, zorder=0)

        distance = 1.3 * sprad
        ax.text(distance, 0, 0, '$S_1$', fontsize=18)
        ax.text(0, distance, 0, '$S_2$', fontsize=18)
        ax.text(0, 0, distance, '$S_3$', fontsize=18)

        # points
        px = [1, -1, 0, 0, 0, 0]
        py = [0, 0, 1, -1, 0, 0]
        pz = [0, 0, 0, 0, 1, -1]

        ax.plot(px, py, pz,
                color='black', marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=22)
        #

        max_size = 1.05 * sprad
        ax.set_xlim(-max_size, max_size)
        ax.set_ylim(-max_size, max_size)
        ax.set_zlim(-max_size, max_size)

        #    plt.tight_layout()            #not compatible
        ax.set_box_aspect([1, 1, 1])  # for the same aspect ratio

        ax.view_init(elev=90 / np.pi, azim=90 / np.pi)
        #    ax.view_init(elev=0/np.pi, azim=0/np.pi)

        ax.set_title(label=title, pad=-10, fontsize=8)

        #    ax.legend()

        return fig, ax

    def draw_Stokes(self, V_Jout, ax=None, label=None):
        """ draw stokes points on the poincare sphere
        Use py-pol library to convert numpy array to Stokes vector array

        :param V_Jout: output Jones vector
        :param ax: axis of figure
        :param label: legend
        :return: fig, ax of matplotlib figure
        """
        if ax is None:
            fig, ax = self.draw_empty_poincare()

        J= Jones_vector("output")
        J.from_matrix(V_Jout)
        S = create_Stokes("Output")
        S.from_Jones(J)
        SS = S.parameters.matrix()

        cm = np.linspace(0, 1, len(SS[0]))  # color map
        cm[-1] = 1.3
        ax.scatter3D(SS[1], SS[2], SS[3],
                     zdir='z',
                     marker = 'o',
                     s=10,
                     c=cm,
                     alpha = 1.0,
                     label = label,
                     cmap="brg")

        if label is not None:
            ax.legend()
        return fig, ax

    # advanced version

    # Bridge vibration 1st
    #   - Unifrom B-field only around VV
    #   - No B-field along Bridge
    #   - See example 4
    # todo : multi processing

    def create_Mvib(self, nM_vib, max_phi, max_theta_e):
        """ Create vibration matrices (pertubation matrices)

        retardation (phi) values are randomly set within -max_phi < phi < max_phi
        angles of the birefringence axis (theta) will be randomly set within -2pi < theta < 2pi
        rotation (theta_e) values are randomly set within -max_theta_e < theta_e < max_theta_e

        :param nM_vib: number of vibration matrices
        :param max_phi: maximum value of retardation
        :param max_theta_e: maximum value of rotation
        """
        theta = (np.random.rand(nM_vib) - 0.5) * 2 * pi / 2  # random axis of LB
        phi = (np.random.rand(nM_vib) - 0.5) * 2 * max_phi * pi / 180  # ellipticity angle change from experiment
        theta_e = (np.random.rand(nM_vib) - 0.5) * 2 * max_theta_e * pi / 180  # azimuth angle change from experiment

        print("angle of Retarder's optic axis:", theta *180/pi, "deg")
        print("retardation of Retarder:", phi * 180 / pi, "deg")
        print("rotation angle of Rotator :", theta_e * 180 / pi, "deg")

        M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
        M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
        M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
        # print(theta)
        # Create (2,2,n_M_vib) Birefringence matrix
        IB = np.zeros((2, 2, nM_vib))
        np.einsum('iij->ij', IB)[:] = 1
        Bexp = np.exp(1j * np.vstack((phi, -phi)))
        M_phi = einsum('ijk, ...ik -> ijk', IB, Bexp)

        # Flag to show how the vibration matrices inserted
        # see
        self.is_spunfiber_set_with_Mvib_in_forward = False
        self.is_spunfiber_set_with_Mvib_in_backward = False

        # Random birefringence(circular + linear), random optic axis matrix calculation
        self.M_vib = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)

    def laming2(self, Ip, DIR, V_theta, Vib = False):
        """Calculation of Jones matrix of spun fiber when plasma current (Ip) flows.
        Spun fiber model was designed following Laming's paper
        --NOT USING THE INFINISIMAL MODEL--

        :param Ip: plasma current
        :param DIR: direction of light propagation  (+1: forward, -1: backward)
        :param V_theta: vector of theta (angle of oprientation of optic axes for each sliced fiber section)
        :param Vib: False (default) : No vbiration
                    Ture: including vibration matrices self.Merr
       :return: Single 2x2 Jones matrix

        example: self.cal_Jout1


        """
        # todo: including matrix calculation log in spun fiber class

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB
        V_delta = delta * ones(len(V_theta))

        m = 0
        # magnetic field in unit length
        r = self.len_sf / (2 * pi)
        V_H = Ip / (2 * pi * r) * ones(len(V_theta))
        V_rho = self.V * V_H  # <<- Vector


        # --------Laming: orientation of the local slow axis ------------
        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector

        V_n = zeros(len(V_rho))  # for compensating the 2pi ambiguity in arctan calculation of V_omega
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) - 0.5 * (1 + DIR)
            else:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) + 0.5 * (1 - DIR)

        V_omega = s_t_r * self.dz + \
                  arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector

        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector

        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += V_theta if DIR == 1 else np.flip(V_theta)

        # Not using N-matrix technique
        m11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        m12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[m11, m21], [m12, m22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that the result of [[n11,n21],[n21,n22]].T is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.
        # See Note2 in Note folder

        M = np.array([[1, 0], [0, 1]])

        if Vib is True:
            tmp = np.array([])  # Logging matrix caculation
            tmp2 = np.array([])  # logging the position of Merr
            nM_vib = self.M_vib.shape[2]
            nSet = int((len(V_theta) - 1) / (nM_vib + 1))
            rem = (len(V_theta) - 1) % nSet

            kk = 0

        for nn in range(len(self.V_theta) - 1):
            M = M_omega[nn] @ M_R[nn] @ M

            if Vib is True:
                # When vibration mode is selected (Vib = True), vibration matrices (self.Merr) are inserted.
                # For example, if self.Merr.shape[2] == 2, two Merr matrices will be inserted at the 1/3 and 2/3 point of spun fiber

                if DIR == 1:
                    tmp = np.append(tmp, "M" + str(nn))  # For indexing matrix to indicate the position of Merr
                    if (nn + 1) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = self.M_vib[..., kk] @ M
                            # print('Merr has been added at ',
                            #       nn+1,
                            #       'th position of spun fiber model')
                            tmp2 = np.append(tmp2, nn + 1)
                            tmp = np.append(tmp, "Merr" + str(kk))

                            kk = kk + 1

                elif DIR == -1:
                    tmp = np.append(tmp, "M" + str(
                        len(V_theta) - 1 - nn))  # For indexing matrix to indicate the position of Merr
                    if (nn + 1 - rem) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = self.M_vib[..., -1 - kk].T @ M

                            # print('Merr has been added at ',
                            #       len(V_theta) - 1 - nn -1,
                            #       'th position of spun fiber model (backward)')
                            tmp2 = np.append(tmp2, len(V_theta) - 1 - nn - 1)
                            tmp = np.append(tmp, "Merr" + str(nM_vib - kk - 1))

                            kk = kk + 1

        if Vib is True:
            # The configurations of forward and backward matrices are saved
            if DIR == 1 and self.is_spunfiber_set_with_Mvib_in_forward is False:
                self.is_spunfiber_set_with_Mvib_in_forward = True
                np.savetxt("MatrixCalculationlog_forward.txt", tmp, fmt='%s')
                print("Merr matrices have been added at", end='')
                for nn in tmp2:
                    print(int(nn), "th, ", end='')
                print("postions of spun fiber model")
                print("Entire spun fiber model has been saved in MatrixCalculationlog_forward.txt")
            elif DIR == -1 and self.is_spunfiber_set_with_Mvib_in_backward is False:
                self.is_spunfiber_set_with_Mvib_in_backward = True
                np.savetxt("MatrixCalculationlog_backward.txt", tmp, fmt='%s')
                print("Merr matrices have been added at ", end='')
                for nn in tmp2:
                    print(int(nn), "th, ", end='')
                print("postions of spun fiber model")
                print("Entire spun fiber model has been saved in MatrixCalculationlog_backward.txt")

        return M

    def cal_Jout1(self, num = 0, dic_Jout=None, V_Ip=None, Jin=None):
        """ Calcuate the output Jones vector for each Ip

        :param num: index of dictionary of Vout_dic (default: num = 0 --> Not using the multiprocessing)
        :param dic_Jout: output Jones vector   (default: Vout_dic = None --> Not using the multiprocessing)
        :param V_Ip: Plasma current (Ip) vector (default: None --> using the initialized vector)
        :param Jin: Input Jones vector (default: None) --> using LHP (np.array([[1],[0]]))
        :param perturbations: Perturbations
                              'V': Vibration

        :return:
        Case 1) normal calculation --> output Jones vectors (see example #3)
        Case 2) multiprocssing calculation --> No return    (see example #4)

        """

        if V_Ip is None:
            V_Ip = self.V_Ip

        laming = self.laming2

        V_Jout = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        if Jin is None:
            Jin = np.array([[1],[0]])
        mm = 0
        for Ip in V_Ip:

            M_lf_f = laming(0, 1, self.V_theta_BF1, Vib=True)
            M_f = laming(Ip, 1, self.V_theta)
            M_lf_f2 = laming(0, 1, self.V_theta_BF2, Vib=True)
            M_lf_b2 = laming(0, -1, self.V_theta_BF2, Vib=True)
            M_b = laming(Ip, -1, self.V_theta)
            M_lf_b = laming(0, -1, self.V_theta_BF1, Vib=True)

            if num == 0 and Ip == V_Ip[0]:
                print("Verification of Matrix Calculation bewteen the foward and backward propagation")
                print("1) When Ip = 0A, the two matrices must be transposed to each other.")
                print("--> Calcuate the difference of M_f and M_b when Ip = 0A")
                print("dz = ",self.dz)
                print("--> Norm (M_f - M_b) = ", norm(M_lf_f - M_lf_b.T))

            if Ip == V_Ip[-1] and self.LB > 10000000:
                print("2) When LB is the infinite, the two matrices must be the same")
                print("--> Calcuate the difference of M_f and M_b when LB > 10000000")
                print("--> Norm (M_f - M_b) = ", norm(M_f - M_b))

            V_Jout[mm] = M_lf_b @ M_b @ M_lf_b2 @ self.M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Jin

            print("process [",num,"], ",mm,"/",len(V_Ip), V_Ip[mm]/1000, "kA")
            mm = mm + 1


        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout

    def cal_Jout1_mp(self, num_processor):
        """ FOCS simulation using multiprocessing technique

        :param num_processor: number of processor to use in multiprocess
        :return: Calculated output Jones vector for each input Plasma current
        """
        spl_I = np.array_split(self.V_Ip, num_processor)

        procs = []
        manager = Manager()
        dic_Jout = manager.dict()

        # print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            # proc = Process(target=self.cal_rotation,
            proc = Process(target=self.cal_Jout1,
                           args=(num, dic_Jout, spl_I[num], None))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        V_Jout = dic_Jout[0]
        for kk in range(num_processor - 1):
            V_Jout = np.vstack((V_Jout, dic_Jout[kk + 1]))

        return V_Jout

    def save_error(self, filename, V_err, V_Iref=None, append=True):
        """ save the calculated error by self.eval_FOCS_fromJones
            If there is the same file,
            the data will be saved in the last column of data file (when append=Ture)

        :param filename: file name
        :param V_Jout: numpy array of output Jones vector
        :param V_Iref: reference current (default=None : self.V_Ip)
        :param append: selecte save mode (defualt=True)
        """
        V_I = self.V_Ip if V_Iref is None else V_Iref
        if os.path.exists(filename) and append is True:
            df2 = pd.read_csv(filename)
            ncol2 = df2.shape[1] - 1
            df2[str(ncol2)] = V_err
        else:
            out_dict2 = {'Iref': V_I[1:]}
            out_dict2['0'] = V_err
            df2 = pd.DataFrame(out_dict2)
        df2.to_csv(filename, index=False)

    def plot_errorbar(self, filename):
        """ Plot the calculated relative error

        :param filename: data file
        :return: fig, ax, lines to overlab the graph
        """

        data = pd.read_csv(filename)
        V_Iref = data['Iref']

        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []

        # Draw ITER specifications
        V_Iref = V_Iref[:]
        absErrorlimit = V_Iref * 0.01
        absErrorlimit[V_Iref < 1e6] = 10e3
        relErrorlimit = absErrorlimit/V_Iref

        lines += ax.plot(V_Iref, relErrorlimit, 'r', label='ITER specification')
        lines += ax.plot(V_Iref, -relErrorlimit, 'r')

        # Calculate mean and std
        # (V_Ip - V_Iref) / V_Iref
        df_mean = data.drop(['Iref'], axis=1).mean(axis=1)
        df_std = data.drop(['Iref'], axis=1).std(axis=1)

        # Draw mean error
        lines += ax.plot(V_Iref[:], df_mean[:], label="mean value")
        # Draw std as error bar
        lines += ax.errorbar(V_Iref[::2], df_mean[::2], yerr=df_std[::2], label="std", ls='None', c='black', ecolor='g', capsize=4)

        ax.legend(loc="upper right")
        ax.set_xlabel(r'Plasma current $I_{p}(A)$')
        ax.set_ylabel(r'Relative error on $I_{P}$')
        ax.set(xlim=(0, 18e6), ylim=(-0.012, 0.012))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))
        ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
        ax.grid(ls='--', lw=0.5)
        fig.subplots_adjust(left = 0.17, hspace=0.4, right=0.95, top=0.93, bottom=0.2)

        return fig, ax, lines

    # Bridge vibration 2nd
    #   - Uniform B-field only around VV
    #   - Nonuniform  B-field along Bridge (using Biot-Savart's law)
    #   - See example 5
    # todo : multi processing

    def set_nonuniform_B_along_bridge(self):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
        """
        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        r = self.len_sf /(2*pi)

        self.V_h1_f = r / (2*pi*(r**2+ (self.len_bf - self.V_BF1)**2))
        self.V_h2_f = r / (2*pi*(r**2+ self.V_BF2**2))
        self.V_h2_b = r / (2*pi*(r**2+ (self.len_bf - self.V_BF2)**2))
        self.V_h1_b = r / (2*pi*(r**2+ self.V_BF1**2))

        print("----------------------------")
        print("Non uniform magnetic field in bridge has been calculated")
        print("----------------------------")
        # V_H =  V_h * Ip
        # V_rho = self.V * V_H

    def laming2_nonuniform(self, Ip, DIR, V_theta, V_temp=None, V_H=None, Vib=False):
        """Calculation of Jones matrix of spun fiber when plasma current (Ip) flows.
        Spun fiber model was designed following Laming's paper
        --NOT USING THE INFINISIMAL MODEL--

        :param Ip: plasma current
        :param DIR: direction of light propagation  (+1: forward, -1: backward)
        :param V_theta: vector of theta (angle of oprientation of optic axes for each sliced fiber section)
        :param V_temp: temperature vector (kelvin)
                        None (default) : uniform temperature (20degC)

        :param V_H: magnetic field vector
                    None (default)--> no magenetic field

                    self.V_H: uniform magnetic field around VV (Circular shape of VV)

                    # predefined Magnetic field vector used in OE paper (see cal_Jout2)
                    self.V_h1_f : 1st bridge forward direction (Straight Bridge)
                    self.V_h2_f : 2nd bridge forward direction (Straight Bridge)
                    self.V_h2_b : 2nd bridge backward direction (Straight Bridge)
                    self.V_h1_b : 1st bridge backward direction (Straight Bridge)

        :param Vib: False (default) : no vibration
                    True : including the Vibration matrices (self.Merr)
                            self.Merr can be defined by self.create_Mvib
        :return: Single 2x2 Jones matrix

        example:

        self.cal_Jout2 : B field on Circular shape of VV and straight line of Bridge
        self.cal_Jout3 : B field on Circular shape, no B field along bridge, Nonuniform temperature around VV
        self.cal_Jout4 : Nonuniform B field arond VV, no B field along Bridge, Nonuniform temperature around VV

        """
        # todo: including matrix calculation log in spun fiber class

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio

        if V_H is None:
            V_H = zeros(len(V_theta))
        V_rho = self.V * Ip * V_H

        delta = 2 * pi / self.LB
        if V_temp is None:
            V_delta = delta * ones(len(V_theta))
        else:
            V_delta = delta * (1 + 3e-5 * (V_temp - 273.15 - 20))
            V_rho = V_rho * (1 + 8.1e-5 * (V_temp - 273.15 - 20))

        # ------------------- Below is just copy of self.laming2 no change --------------------
        # ------------------- Below is just copy of self.laming2 no change --------------------
        # ------------------- Below is just copy of self.laming2 no change --------------------

        m = 0
        # --------Laming: orientation of the local slow axis ------------
        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector

        V_n = zeros(len(V_rho))  # for compensating the 2pi ambiguity in arctan calculation of V_omega
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) - 0.5 * (1 + DIR)
            else:
                V_n[nn] = -DIR * int(V_gma[nn] * self.dz / pi) + 0.5 * (1 - DIR)

        V_omega = s_t_r * self.dz + \
                  arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector

        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector

        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += V_theta if DIR == 1 else np.flip(V_theta)

        # Not using N-matrix technique
        m11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        m12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        m22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[m11, m21], [m12, m22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that the result of [[n11,n21],[n21,n22]].T is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.
        # See Note2 in Note folder

        M = np.array([[1, 0], [0, 1]])

        if Vib is True:
            tmp = np.array([])  # Logging matrix caculation
            tmp2 = np.array([])  # logging the position of Merr
            nM_vib = self.M_vib.shape[2]
            nSet = int((len(V_theta) - 1) / (nM_vib + 1))
            rem = (len(V_theta) - 1) % nSet

            kk = 0

        for nn in range(len(V_theta) - 1):
            M = M_omega[nn] @ M_R[nn] @ M

            if Vib is True:
                # When vibration mode is selected (Vib = True), vibration matrices (self.Merr) are inserted.
                # For example, if self.Merr.shape[2] == 2, two Merr matrices will be inserted at the 1/3 and 2/3 point of spun fiber

                if DIR == 1:
                    tmp = np.append(tmp, "M" + str(nn))  # For indexing matrix to indicate the position of Merr
                    if (nn + 1) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = self.M_vib[..., kk] @ M
                            # print('Merr has been added at ',
                            #       nn+1,
                            #       'th position of spun fiber model')
                            tmp2 = np.append(tmp2, nn + 1)
                            tmp = np.append(tmp, "Merr" + str(kk))

                            kk = kk + 1

                elif DIR == -1:
                    tmp = np.append(tmp, "M" + str(
                        len(V_theta) - 1 - nn))  # For indexing matrix to indicate the position of Merr
                    if (nn + 1 - rem) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = self.M_vib[..., -1 - kk].T @ M

                            # print('Merr has been added at ',
                            #       len(V_theta) - 1 - nn -1,
                            #       'th position of spun fiber model (backward)')
                            tmp2 = np.append(tmp2, len(V_theta) - 1 - nn - 1)
                            tmp = np.append(tmp, "Merr" + str(nM_vib - kk - 1))

                            kk = kk + 1

        if Vib is True:
            if DIR == 1 and self.is_spunfiber_set_with_Mvib_in_forward is False:
                self.is_spunfiber_set_with_Mvib_in_forward = True
                np.savetxt("MatrixCalculationlog_forward.txt", tmp, fmt='%s')
                print("Merr matrices have been added at", end='')
                for nn in tmp2:
                    print(int(nn), "th, ", end='')
                print("postions of spun fiber model")
                print("Entire spun fiber model has been saved in MatrixCalculationlog_forward.txt")
            elif DIR == -1 and self.is_spunfiber_set_with_Mvib_in_backward is False:
                self.is_spunfiber_set_with_Mvib_in_backward = True
                np.savetxt("MatrixCalculationlog_backward.txt", tmp, fmt='%s')
                print("Merr matrices have been added at ", end='')
                for nn in tmp2:
                    print(int(nn), "th, ", end='')
                print("postions of spun fiber model")
                print("Entire spun fiber model has been saved in MatrixCalculationlog_backward.txt")

        return M

    def cal_Jout2(self, num=0, dic_Jout=None, V_Ip=None, Jin=None):
        """ Calcuate the output Jones vector for each Ip

        :param num: index of dictionary of Vout_dic (default: num = 0 --> Not using the multiprocessing)
        :param dic_Jout: output Jones vector   (default: Vout_dic = None --> Not using the multiprocessing)
        :param V_Ip: Plasma current (Ip) vector (default: None --> using the initialized vector)
        :param Jin: Input Jones vector (default: None) --> using LHP (np.array([[1],[0]]))
        :param Vib: Vibration
                    True (default) : Vibration in Bridge section
                    False : No vibration
        :return:
        Case 1) normal calculation --> output Jones vectors (see example #3)
        Case 2) multiprocssing calculation --> No return    (see example #4)

        """

        if V_Ip is None:
            V_Ip = self.V_Ip

        laming = self.laming2_nonuniform

        V_Jout = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        if Jin is None:
            Jin = np.array([[1], [0]])
        mm = 0
        for Ip in V_Ip:

            M_lf_f = laming(0, 1, self.V_theta_BF1, V_H=self.V_h1_f, Vib=True)
            M_f = laming(Ip, 1, self.V_theta, V_H=self.V_H)
            M_lf_f2 = laming(0, 1, self.V_theta_BF2, V_H=self.V_h2_f, Vib=True)
            M_lf_b2 = laming(0, -1, self.V_theta_BF2, V_H=self.V_h2_b, Vib=True)
            M_b = laming(Ip, -1, self.V_theta, V_H=self.V_H)
            M_lf_b = laming(0, -1, self.V_theta_BF1, V_H=self.V_h1_b, Vib=True)

            if num == 0 and Ip == V_Ip[0]:
                print("Verification of Matrix Calculation bewteen the foward and backward propagation")
                print("1) When Ip = 0A, the two matrices must be transposed to each other.")
                print("--> Calcuate the difference of M_f and M_b when Ip = 0A")
                print("dz = ", self.dz)
                print("--> Norm (M_f - M_b) = ", norm(M_lf_f - M_lf_b.T))

            if Ip == V_Ip[-1] and self.LB > 10000000:
                print("2) When LB is the infinite, the two matrices must be the same")
                print("--> Calcuate the difference of M_f and M_b when LB > 10000000")
                print("--> Norm (M_f - M_b) = ", norm(M_f - M_b))

            V_Jout[mm] = M_lf_b @ M_b @ M_lf_b2 @ self.M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Jin

            print("process [", num, "], ", mm, "/", len(V_Ip), V_Ip[mm] / 1000, "kA")
            mm = mm + 1

        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout

    def cal_Jout2_mp(self, num_processor):
        """ FOCS simulation using multiprocessing technique

        :param num_processor: number of processor to use in multiprocess
        :return: Calculated output Jones vector for each input Plasma current
        """
        spl_I = np.array_split(self.V_Ip, num_processor)

        procs = []
        manager = Manager()
        dic_Jout = manager.dict()

        # print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            # proc = Process(target=self.cal_rotation,
            proc = Process(target=self.cal_Jout2,
                           args=(num, dic_Jout, spl_I[num], None))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        V_Jout = dic_Jout[0]
        for kk in range(num_processor - 1):
            V_Jout = np.vstack((V_Jout, dic_Jout[kk + 1]))

        return V_Jout

    # Nonuniform Temperature effect
    #    - No vibration
    #    - Uniform magnetic field around VV
    #    - No magnetic field along Bridge
    #    - Non uniform temperature around VV given by ITER report  (VVtemp.csv)
    #    - See example 6
    # todo : multi processing

    def set_tempVV(self, li, lf, func_temp_interp):
        self.V_temp_VV = np.array([periodicf(li, lf, func_temp_interp, xi) for xi in self.V_SF])
        tmp = self.V_temp_VV.mean()
        print("average temperature of VV is ", tmp)
        print("corresponding calibration factor is", (1 + 8.1e-5 * (tmp - 273.15 - 20)))
        print("Use V_custom=",
              self.V*(1 + 8.1e-5 * (tmp - 273.15 - 20)),
              " when calculate the FOCS accuracy")
        return self.V*(1 + 8.1e-5 * (tmp - 273.15 - 20))

    def cal_Jout3(self, num=0, dic_Jout=None, V_Ip=None, Jin=None, Vib=True):
        """ Calcuate the output Jones vector for each Ip

        :param num: index of dictionary of Vout_dic (default: num = 0 --> Not using the multiprocessing)
        :param dic_Jout: output Jones vector   (default: Vout_dic = None --> Not using the multiprocessing)
        :param V_Ip: Plasma current (Ip) vector (default: None --> using the initialized vector)
        :param Jin: Input Jones vector (default: None) --> using LHP (np.array([[1],[0]]))
        :param Vib: Vibration
                    True (default) : Vibration in Bridge section
                    False : No vibration
        :return:
        Case 1) normal calculation --> output Jones vectors (see example #3)
        Case 2) multiprocssing calculation --> No return    (see example #4)

        """

        if V_Ip is None:
            V_Ip = self.V_Ip

        laming = self.laming2_nonuniform

        V_Jout = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        if Jin is None:
            Jin = np.array([[1], [0]])
        mm = 0
        for Ip in V_Ip:

            M_lf_f = laming(0, 1, self.V_theta_BF1)
            M_f = laming(Ip, 1, self.V_theta, V_temp=self.V_temp_VV, V_H=self.V_H)
            M_lf_f2 = laming(0, 1, self.V_theta_BF2)
            M_lf_b2 = laming(0, -1, self.V_theta_BF2)
            M_b = laming(Ip, -1, self.V_theta, V_temp=self.V_temp_VV, V_H=self.V_H)
            M_lf_b = laming(0, -1, self.V_theta_BF1)

            if num == 0 and Ip == V_Ip[0]:
                print("Verification of Matrix Calculation bewteen the foward and backward propagation")
                print("1) When Ip = 0A, the two matrices must be transposed to each other.")
                print("--> Calcuate the difference of M_f and M_b when Ip = 0A")
                print("dz = ", self.dz)
                print("--> Norm (M_f - M_b) = ", norm(M_lf_f - M_lf_b.T))

            if Ip == V_Ip[-1] and self.LB > 10000000:
                print("2) When LB is the infinite, the two matrices must be the same")
                print("--> Calcuate the difference of M_f and M_b when LB > 10000000")
                print("--> Norm (M_f - M_b) = ", norm(M_f - M_b))

            V_Jout[mm] = M_lf_b @ M_b @ M_lf_b2 @ self.M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Jin

            print("process [", num, "], ", mm, "/", len(V_Ip), V_Ip[mm] / 1000, "kA")
            mm = mm + 1

        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout

    # Nonuniform Temperature + magnetic effect
    #    - No vibration
    #    - non Uniform magnetic field around VV given by ITER report ('B-field_around_VV.txt')
    #    - No magnetic field along Bridge
    #    - Non uniform temperature around VV given by ITER report  (VVtemp.csv)
    #    - See example 7
    # todo : multi processing

    def set_nonuniform_B_around_VV(self, func_B_interp):
        # the magnetic field profile given by ITER is only 15MA scenario.
        # However,
        self.V_H = -func_B_interp(self.V_SF) * 1 / (4 * pi * 1e-7) / 15e6
        tmp = np.trapz(self.V_H, x=self.V_SF) # integral calculation of B-field along VV
        c = tmp
        print('Use total current Itotal = Ip x', c)
        return c

    def cal_Jout4(self, num=0, dic_Jout=None, V_Ip=None, Jin=None, Vib=True):
        """ Calcuate the output Jones vector for each Ip

        :param num: index of dictionary of Vout_dic (default: num = 0 --> Not using the multiprocessing)
        :param dic_Jout: output Jones vector   (default: Vout_dic = None --> Not using the multiprocessing)
        :param V_Ip: Plasma current (Ip) vector (default: None --> using the initialized vector)
        :param Jin: Input Jones vector (default: None) --> using LHP (np.array([[1],[0]]))
        :param Vib: Vibration
                    True (default) : Vibration in Bridge section
                    False : No vibration
        :return:
        Case 1) normal calculation --> output Jones vectors (see example #3)
        Case 2) multiprocssing calculation --> No return    (see example #4)

        """

        if V_Ip is None:
            V_Ip = self.V_Ip

        laming = self.laming2_nonuniform

        V_Jout = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        if Jin is None:
            Jin = np.array([[1], [0]])
        mm = 0
        for Ip in V_Ip:

            M_lf_f = laming(0, 1, self.V_theta_BF1)
            M_f = laming(Ip, 1, self.V_theta, V_temp=self.V_temp_VV, V_H=self.V_H)
            M_lf_f2 = laming(0, 1, self.V_theta_BF2)
            M_lf_b2 = laming(0, -1, self.V_theta_BF2)
            M_b = laming(Ip, -1, self.V_theta, V_temp=self.V_temp_VV, V_H=self.V_H)
            M_lf_b = laming(0, -1, self.V_theta_BF1)

            if num == 0 and Ip == V_Ip[0]:
                print("Verification of Matrix Calculation bewteen the foward and backward propagation")
                print("1) When Ip = 0A, the two matrices must be transposed to each other.")
                print("--> Calcuate the difference of M_f and M_b when Ip = 0A")
                print("dz = ", self.dz)
                print("--> Norm (M_f - M_b) = ", norm(M_lf_f - M_lf_b.T))

            if Ip == V_Ip[-1] and self.LB > 10000000:
                print("2) When LB is the infinite, the two matrices must be the same")
                print("--> Calcuate the difference of M_f and M_b when LB > 10000000")
                print("--> Norm (M_f - M_b) = ", norm(M_f - M_b))

            V_Jout[mm] = M_lf_b @ M_b @ M_lf_b2 @ self.M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Jin

            print("process [", num, "], ", mm, "/", len(V_Ip), V_Ip[mm] / 1000, "kA")
            mm = mm + 1

        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout


if __name__ == '__main__':
    mode = 0
    if mode == 0:
        LB = 1.000
        SP = 0.005
        dz = 0.0005
        len_lf = 1  # lead fiber
        len_ls = 29  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        str_file1 ='xx.csv'
        str_file2 ='xx_err.csv'


        # example 1
        # V_Jout= spunfiber.cal_Jout0()
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        # spunfiber.save_Jones(str_file1, V_Jout)

        # example 2
        # num_processor = 8
        # spunfiber.set_input_current(0,18e6,1e5)
        # V_Jout= spunfiber.cal_Jout0_mp(num_processor)
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        # spunfiber.save_Jones(str_file1, V_Jout)

        # example 3
        # V_Iref, V_Jout, isEOF = spunfiber.load_Jones(str_file1)
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        # fig2, ax2 = spunfiber.draw_Stokes(V_Jout)

        # -------------------------------------------------------------------------------- #
        # example 4
        # Vibration
        # -------------------------------------------------------------------------------- #

        # num_iter = 3 # number of iterations
        # nM_vib = 5
        # spunfiber.set_input_current(0,1e6,2e5)
        #
        # for nn in range(num_iter):
        #     spunfiber.create_Mvib(nM_vib, 1, 1)
        #     V_Jout= spunfiber.cal_Jout1()
        #     V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        #     if nn == 0:
        #         spunfiber.save_Jones(str_file1, V_Jout, append=False)
        #         spunfiber.save_error(str_file2, V_err, append = False)
        #         fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        #     else:
        #         spunfiber.save_Jones(str_file1, V_Jout, append=True)
        #         spunfiber.save_error(str_file2, V_err, append=True)
        #         fig, ax, lines = spunfiber.plot_error(V_err,
        #                                               fig=fig, ax=ax, lines=lines,
        #                                               label='LB/SP=200')
        # fig, ax, lines = spunfiber.plot_errorbar(str_file2)

        # -------------------------------------------------------------------------------- #
        # example 5
        # Vibration with nonunifrom magnetic field along Bridge
        # -------------------------------------------------------------------------------- #

        num_iter = 10  # number of iterations
        nM_vib = 5
        num_processor = 16
        spunfiber.set_input_current(0, 18e6, 2e5)
        spunfiber.set_nonuniform_B_along_bridge()

        for nn in range(num_iter):
            spunfiber.create_Mvib(nM_vib, 1, 1)
            # V_Jout = spunfiber.cal_Jout2()
            V_Jout = spunfiber.cal_Jout2_mp(num_processor)
            V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
            if nn == 0:
                spunfiber.save_Jones(str_file1, V_Jout, append=False)
                spunfiber.save_error(str_file2, V_err, append=False)
                fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
            else:
                spunfiber.save_Jones(str_file1, V_Jout, append=True)
                spunfiber.save_error(str_file2, V_err, append=True)
                fig, ax, lines = spunfiber.plot_error(V_err,
                                                      fig=fig, ax=ax, lines=lines,
                                                      label='LB/SP=200')
        fig, ax, lines = spunfiber.plot_errorbar(str_file2)

        # -------------------------------------------------------------------------------- #
        # example 6
        # no vibration
        # nonuniform magnetic field along Bridge
        # uniform magnetic field around VV
        # Non uniform temperature given by ITER report (VVtemp.csv)
        # todo: multiprocessing
        # -------------------------------------------------------------------------------- #

        # spunfiber.set_input_current(0, 1e6, 2e5)
        #
        # # Load a part of temperature distribution along the VV (20 cm, a clamp in the middle)
        # data = pd.read_csv('VVtemp.csv', delimiter=';')
        # V_l = data['L'].to_numpy() / 100
        # V_temp = data['TEMP'].to_numpy()
        # func_temp_interp = CubicSpline(V_l, V_temp)
        # V_custom = spunfiber.set_tempVV(V_l[0], V_l[-1], func_temp_interp)
        #
        # V_Jout = spunfiber.cal_Jout3()
        # # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)          # without considering temp dependence of Verdet constant in VV section
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout, V_custom=V_custom)
        #
        # spunfiber.save_Jones(str_file1, V_Jout, append=False)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')

        # -------------------------------------------------------------------------------- #
        # example 7
        # Non unifrom temperature along VV
        # Non uniform magnetic field along VV
        # No magnetic field in Bridge section
        # No Vibration
        # -------------------------------------------------------------------------------- #

        # spunfiber.set_input_current(0, 18e6, 2e5)
        # #spunfiber.set_nonuniform_B_along_bridge()
        #
        # # Load a part of temperature distribution along the VV (20 cm, a clamp in the middle)
        # data = pd.read_csv('VVtemp.csv', delimiter=';')
        # V_l = data['L'].to_numpy() / 100
        # V_temp = data['TEMP'].to_numpy()
        # func_temp_interp = CubicSpline(V_l, V_temp)
        # V_custom = spunfiber.set_tempVV(V_l[0], V_l[-1], func_temp_interp)
        #
        # # Load the magnetic field distribution along the VV (30 m)
        # strfile_B = 'B-field_around_VV.txt'
        # data_B = np.loadtxt(strfile_B)
        # func_B_interp = interp1d(data_B[:, 0], data_B[:, 1], kind='cubic')
        # c = spunfiber.set_nonuniform_B_around_VV(func_B_interp)
        #
        # V_Jout = spunfiber.cal_Jout4()
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout, V_Itotal=spunfiber.V_Ip*c, V_custom=V_custom)
        #
        # spunfiber.save_Jones(str_file1, V_Jout, append=False)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        #
        # # plot the temperature effect
        # fig_temp, ax_temp = plt.subplots(4, 1, figsize=(8, 5))
        # fig_temp.subplots_adjust(hspace=0.32, left=0.24)
        # ax_temp[0].plot(spunfiber.V_SF, spunfiber.V_temp_VV-273.15)
        #
        # ax_temp[1].plot(spunfiber.V_SF, (spunfiber.LB*(1 + 3e-5 * (spunfiber.V_temp_VV - 273.15 - 20))))
        #
        # ax_temp[2].plot(spunfiber.V_SF, spunfiber.V * (1 + 8.1e-5 * (spunfiber.V_temp_VV - 273.15 - 20)))
        #
        # ax_temp[3].plot(spunfiber.V_SF, spunfiber.V_H*3e6*4*pi*1e-7, label='3MA')
        # ax_temp[3].plot(spunfiber.V_SF, spunfiber.V_H*5e6*4*pi*1e-7, label='5MA')
        # ax_temp[3].plot(spunfiber.V_SF, spunfiber.V_H*15e6*4*pi*1e-7, label='15MA')
        #
        # xmax = 29
        # ax_temp[0].set(xlim=(0, xmax), ylim=(18, 110))
        # ax_temp[1].set(xlim=(0, xmax), ylim=(0.9995, 1.003))
        # ax_temp[2].set(xlim=(0, xmax), ylim=(0.6780e-6, 0.6835e-6))
        # ax_temp[3].set(xlim=(0, xmax), ylim=(-2, 2))
        # #ax.yaxis.set_major_formatter(OOMFormatter(0, "%3.2f"))
        # ax_temp[1].yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))
        # ax_temp[2].yaxis.set_major_formatter(OOMFormatter(-6, "%5.4f"))
        # ax_temp[0].set_ylabel('Temperature \n(degC)')
        # ax_temp[1].set_ylabel('Beatlength \n(m)')
        # ax_temp[2].set_ylabel('Verdet constant  \n(rad/A)')
        # ax_temp[3].set_ylabel('B-field  \n(T)')
        # ax_temp[3].set_xlabel('Fiber position (m)')
        # ax_temp[3].legend()
        # fig_temp.align_ylabels(ax_temp)


plt.show()
