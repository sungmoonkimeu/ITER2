# -*- coding: utf-8 -*-
"""
Created on Mon May 02 15:14:00 2022
@author: SMK

functions to investigate Spun fiber's behavior
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
        self.int_V_B = 0

        self.V_SF = arange(0, self.len_sf + self.dz, self.dz)
        self.V_BF1 = arange(0, self.len_bf + self.dz, self.dz)
        self.V_BF2 = self.V_BF1

        s_t_r = 2 * pi / self.SP
        self.V_theta_BF1 = self.V_BF1 * s_t_r
        self.V_theta = self.V_theta_BF1[-1] + self.V_SF * s_t_r
        self.V_theta_BF2 = self.V_theta[-1] + self.V_BF1 * s_t_r
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

        print('---------------------------------------------------------------\n\n')

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
        :return: M matrix calculated from N matrix

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
        Case 1) normal calculation --> output Jones vectors
        Case 2) multiprocssing calculation --> No return

        example:

        Case 1) normal calculation

        LB = 0.009
        SP = 0.005
        dz = 0.0001
        len_bf = 1  # lead fiber
        len_sf = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_sf, len_bf)
        V_Jout = spunfiber.cal_rotation0()

        Case 2) Using the multiprocessing
        No return!!!
        - The calcuated vectors for each Ip in V_Ip will be saved in Vout_dic[num] variable
        - Dictionary is required to regroup the results randomly seperated by multiprocessing

        LB = 0.009
        SP = 0.005
        dz = 0.0001
        len_bf = 1  # lead fiber
        len_sf = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_sf, len_bf)

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
            print("[",num,"], ",mm,"/",len(V_Ip))

        if dic_Jout is None:
            return V_Jout
        else:
            dic_Jout[num] = V_Jout

    def eval_FOCS_fromJones(self, V_Jout, V_Itotal=None, V_custom=None, angle_init=None, FOCSTYPE=2):
        """

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
            print("V_Iref[0] must be 0A for referencing the intiial point")
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
        """ Plot calculated relative error

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

    def save_Jones(self, filename, V_Jout, V_Iref=None):
        """ save the output Jones vectors from FOCS simulation
            If there is the same file,
            the data will be saved in the last column of data file (appended)

        :param filename: file name
        :param V_Jout: numpy array of output Jones vector
        :param V_Iref: reference current (default=None : self.V_Ip)
        """
        V_I = self.V_Ip if V_Iref is None else V_Iref
        if os.path.exists(filename):
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
        # for temperature distribution simulation
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
        """
        Created on Fri Jan 14 11:50:03 2022
        @author: agoussar
        """
        '''
            plot Poincare Sphere, ver. 20/03/2020
            return:
            ax, fig
            '''
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

if __name__ == '__main__':
    mode = 0
    if mode == 0:
        LB = 1.000
        SP = 0.005
        dz = 0.00005
        len_lf = 1  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        str_file1 ='xx.csv'

        # example 1
        # V_Jout= spunfiber.cal_Jout0()
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        # spunfiber.save_Jones(str_file1, V_Jout)

        # example 2
        V_Iref, V_Jout, isEOF = spunfiber.load_Jones(str_file1)
        V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        fig2, ax2 = spunfiber.draw_Stokes(V_Jout)

        # example 3
        # num_processor = 2
        # V_Jout= spunfiber.cal_Jout0_mp(num_processor)
        # V_IFOCS, V_err = spunfiber.eval_FOCS_fromJones(V_Jout)
        # fig, ax, lines = spunfiber.plot_error(V_err, label='LB/SP=200')
        # spunfiber.save_Jones(str_file1, V_Jout)


plt.show()
