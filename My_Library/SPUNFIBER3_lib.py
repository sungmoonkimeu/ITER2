# -*- coding: utf-8 -*-
"""
Created on Mon May 02 15:14:00 2022
@author: SMK

functions to investigate Spun fiber's behavior
"""
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt, log10
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

from scipy.interpolate import CubicSpline

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
# from .basis_correction_lib import calib_basis3
# from .draw_poincare_plotly import *

# import parmap
# import tqdm


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

# Function that will convert any given function 'f' defined in a given range '[li,lf]' to a periodic function of period 'lf-li'
def periodicf(li, lf, f, x):
    if x >= li and x <= lf:
        return f(x)
    elif x > lf:
        x_new = x - (lf - li)
        return periodicf(li, lf, f, x_new)
    elif x < (li):
        x_new = x + (lf - li)
        return periodicf(li, lf, f, x_new)

class SPUNFIBER:
    def __init__(self, beat_length, spin_pitch, delta_l, len_lf, len_sf, angle_FM):
        self.LB = beat_length
        self.SP = spin_pitch
        self.dz = delta_l
        self.V = 0.54 * 4* pi * 1e-7 # μV [rad/A]
        # Verdet constant
        # V=0.54 [rad/mT] --> μV = 0.54 * 4*pi*1e-7 [rad/A]
        self.LF = len_lf
        self.L = len_sf
        self.V_L = []
        self.V_LF = []
        self.V_theta = []
        self.V_theta_LF = []
        self.V_temp = []
        self.V_B =[]
        self.ang_FM = angle_FM
        self.Vin = np.array([[1],[0]])
        self.Mvib = []

    def set_Vin(self, azi, ell):
        E = Jones_vector('input')
        E.general_azimuth_ellipticity(azimuth=azi, ellipticity=ell)
        print("input pol: ", E)
        self.Vin = E.parameters.matrix()

    def set_Vectors(self):
        self.dz = self.L / 10000
        self.V_L = arange(0, self.L + self.dz, self.dz)
        self.V_LF = arange(0, self.LF + self.dz, self.dz)

        s_t_r = 2 * pi / self.SP
        self.V_theta = self.V_L * s_t_r

        print('Vectors are set!')
    def set_B(self, F_B_interp):
        self.V_B = F_B_interp(self.V_L)
        print('Nonunifrom magnetic vector has set!')

    def set_tempVV(self, li, lf, F_temp_interp):
        # self.V_temp = + 273.15 + 20 + 0 * np.array([periodicf(li, lf, F_temp_interp, xi) for xi in self.V_L])
        self.V_temp = np.array([periodicf(li, lf, F_temp_interp, xi) for xi in self.V_L])
        # birefringence (delta) distribution due to the temperature distribution
        self.V_delta_temp = 1 + 3e-5 * (self.V_temp - 273.15 - 20)
        # faraday effect distibution due to the temperature distribution
        self.V_f_temp = 1 + 8.1e-5 * (self.V_temp - 273.15 - 20)

        print('Temperature Vector is set!')

    def create_Mvib(self, nM_vib, max_phi, max_theta_e):
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

        # Random birefringence(circular + linear), random optic axis matrix calculation
        self.M_vib = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)

    @staticmethod
    def _eigen_expm(A):
        """

        Parameters
        ----------
        A : 2 x 2 diagonalizable matrix
            DESCRIPTION.

        scify.linalg.expm() is available but only works for a (2,2) matrix.
        This function is for (2,2,n) matrix

        Returns
        -------
        expm(A): exponential of the matrix A.

        """
        vals, vects = eig(A)
        return einsum('...ik, ...k, ...kj -> ...ij',
                      vects, np.exp(vals), np.linalg.inv(vects))

    def lamming(self, Ip, DIR, V_theta, M_vib=None):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
        Only valid dz < SP/100
        """

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        H = Ip / self.L
        rho = self.V * H

        # ----------------------Laming parameters--------------------------------- #
        n = 0
        m = 0
        # --------Laming: orientation of the local slow axis ------------

        qu = 2 * (s_t_r - rho) / delta
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5
        omega = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) + n * pi
        R_z = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))

        M = np.array([[1, 0], [0, 1]])

        kk = 0  # for counting M_vib
        if M_vib is not None:
            nM_vib = M_vib.shape[2]
            nSet = int((len(V_theta)-1) / (nM_vib + 1))
            rem = (len(V_theta)-1) % nSet

        if DIR == 1:
            V_phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta

        elif DIR == -1:
            V_phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + np.flip(V_theta)

        n11 = R_z / 2 * 1j * cos(2 * V_phi)
        n12 = R_z / 2 * 1j * sin(2 * V_phi) - omega
        n21 = R_z / 2 * 1j * sin(2 * V_phi) + omega
        n22 = R_z / 2 * -1j * cos(2 * V_phi)

        N = np.array([[n11, n21], [n12, n22]]).T
        # Note that [[n11,n21],[n21,n22]].T calculation is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, N array should be defined as transposed matrix to have correct matrix after it.

        N_integral = self._eigen_expm(N)
        tmp = np.array([])  # for test

        for nn in range(len(V_theta)-1):
            M = N_integral[nn] @ M

            # If vibration matrix (Merr) is presence, it will be inserted automatically.
            # For example, if Merr.shape[2] == 2, two Merr will be inserted
            # in the 1/3, 2/3 position of L

            if M_vib is not None:
                if DIR == 1:
                    # strM = "M" + str(nn)        # For indexing matrix to indicate the position of Merr
                    # tmp = np.append(tmp, strM)
                    if (nn + 1) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = M_vib[..., kk] @ M
                            '''
                            print(nn+1, "번째에 에러 매트릭스 추가")
                            strM = "Merr" + str(kk)
                            tmp = np.append(tmp, strM)  # for test
                            '''
                            kk = kk + 1

                elif DIR == -1:
                    strM = "M" + str(len(V_theta) - 1 - nn)
                    # tmp = np.append(tmp, strM)  # for test
                    if (nn + 1 - rem) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = M_vib[..., -1 - kk].T @ M
                            '''
                            print(len(V_theta) - 1 - nn, "번째에 에러 매트릭스 추가 (-backward)")
                            strM = "Merr" + str(nM_vib - kk - 1)
                            tmp = np.append(tmp, strM)  # for test
                            '''
                            kk = kk + 1

        # print("rem=", rem, "nVerr=", nVerr, "nSet = ", nSet) # To show current spun fiber's info.
        # print(tmp) # To show where is the position of Merr
        return M

    def lamming_VV_temp(self, Ip, DIR):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :return: 2x2 Jones matrix of spun fiber in VV with temperature effect
        """
        m = 0
        # Fiber parameter
        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio

        #V_delta = 2*pi / (self.LB *(1 + 3e-5*(self.V_temp-273.15-20)))
        V_delta = 2 * pi / (self.LB * self.V_delta_temp)
        #delta = 2 * pi / self.LB

        # magnetic field in unit length
        # unifrom magnetic field
        # H = Ip / (2 * pi * r)
        # unifrom magnetic field array
        # r = self.L / (2 * pi)
        # V_H = Ip / (2 * pi * r) * ones(len(self.V_temp))
        # V_rho = self.V * V_H
        # V_rho = self.V * V_H * (1 + 8.1e-5*(self.V_temp-273.15-20))
        # V_rho = self.V * V_H * self.V_f_temp

        # V_rho = self.V/(4*pi*1e-7) * -self.V_B *self.V_f_temp * Ip/15e6
        shiftV_B = int(np.random.rand(1)*(0.2/self.dz))
        V_rho = self.V / (4 * pi * 1e-7) * -np.roll(self.V_B,shiftV_B) * self.V_f_temp * Ip / 15e6
        #print(V_rho)
        # --------Laming: orientation of the local slow axis ------------

        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector
        '''
        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nf = -int(gma * self.dz / pi) - 1
        else:
            nf = -int(gma * self.dz / pi)

        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nb = int(gma * self.dz / pi)
        else:
            nb = int(gma * self.dz / pi) + 1
        '''
        V_n = zeros(len(V_rho))
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) - 0.5*(1+DIR)
            else:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) + 0.5*(1-DIR)

        V_omega = s_t_r * self.dz + arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector
        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector
        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += self.V_theta if DIR == 1 else np.flip(self.V_theta)

        n11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        n12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[n11, n21], [n12, n22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that [[n11,n21],[n21,n22]].T calculation is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.

        M = np.array([[1, 0], [0, 1]])
        for nn in range(len(self.V_theta)-1):
            M = M_omega[nn] @ M_R[nn] @ M

        return M

    def lamming_VV_const_temp(self, Ip, DIR, temp):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :return: 2x2 Jones matrix of spun fiber in VV with constant temperature
        """
        m = 0
        # Fiber parameter
        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio

        # V_delta = 2*pi / (self.LB *(1 + 3e-5*(self.V_temp-273.15-20)))
        # temp = 293.15  # kelvin

        LB_temp = 1 + 3e-5 * (temp - 273.15 - 20)
        V_delta = 2 * pi / (self.LB * LB_temp) * ones(len(self.V_L))


        f_temp = 1 + 8.1e-5 * (temp - 273.15 - 20)
        # magnetic field in unit length
        r = self.L / (2 * pi)       # H = Ip / (2 * pi * r)
        V_H = Ip / (2 * pi * r) * ones(len(self.V_L))
        V_rho = self.V * V_H * f_temp


        # --------Laming: orientation of the local slow axis ------------
        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector
        '''
        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nf = -int(gma * self.dz / pi) - 1
        else:
            nf = -int(gma * self.dz / pi)

        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nb = int(gma * self.dz / pi)
        else:
            nb = int(gma * self.dz / pi) + 1
        '''
        V_n = zeros(len(V_rho))
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) - 0.5*(1+DIR)
            else:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) + 0.5*(1-DIR)

        V_omega = s_t_r * self.dz + arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector
        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector
        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += self.V_theta if DIR == 1 else np.flip(self.V_theta)

        n11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        n12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[n11, n21], [n12, n22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that [[n11,n21],[n21,n22]].T calculation is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.

        M = np.array([[1, 0], [0, 1]])
        for nn in range(len(self.V_theta)-1):
            M = M_omega[nn] @ M_R[nn] @ M

        return M

    def lamming_bridge(self, Ip, DIR, n_Bridge, V_theta, V_LF, M_vib=None ):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
        """
        m = 0
        # Fiber parameter
        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        r = self.L /(2*pi)
        if DIR == 1 and n_Bridge == 1:
            V_H = Ip * r / (2*pi*(r**2+ (self.LF - V_LF)**2))
        elif DIR ==1 and n_Bridge == 2:
            V_H = Ip * r / (2*pi*(r**2+ V_LF**2))
        elif DIR == -1 and n_Bridge == 2:
            V_H = Ip * r / (2*pi*(r**2+ (self.LF - V_LF)**2))
        elif DIR == -1 and n_Bridge == 1:
            V_H = Ip * r / (2*pi*(r**2+ V_LF**2))
        V_rho = self.V * V_H    # <<- Vector

        # --------Laming: orientation of the local slow axis ------------

        V_qu = 2 * (s_t_r - V_rho) / delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector
        '''
        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nf = -int(gma * self.dz / pi) - 1
        else:
            nf = -int(gma * self.dz / pi)

        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nb = int(gma * self.dz / pi)
        else:
            nb = int(gma * self.dz / pi) + 1
        '''
        V_n = zeros(len(V_rho))
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) - 0.5*(1+DIR)
            else:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) + 0.5*(1-DIR)

        V_omega = s_t_r * self.dz + arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector
        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector
        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += V_theta if DIR == 1 else np.flip(V_theta)

        n11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        n12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[n11, n21], [n12, n22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that [[n11,n21],[n21,n22]].T calculation is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.

        kk = 0  # for counting M_vib
        if M_vib is not None:
            nM_vib = M_vib.shape[2]
            nSet = int((len(V_theta)-1) / (nM_vib + 1))
            rem = (len(V_theta)-1) % nSet

        tmp = np.array([])  # for test
        M = np.array([[1, 0], [0, 1]])
        for nn in range(len(V_theta)-1):
            M = M_omega[nn] @ M_R[nn] @ M

            # If vibration matrix (Merr) is presence, it will be inserted automatically.
            # For example, if Merr.shape[2] == 2, two Merr will be inserted
            # in the 1/3, 2/3 position of L

            if M_vib is not None:
                if DIR == 1:
                    # strM = "M" + str(nn)        # For indexing matrix to indicate the position of Merr
                    # tmp = np.append(tmp, strM)
                    if (nn + 1) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = M_vib[..., kk] @ M
                            '''
                            print(nn+1, "번째에 에러 매트릭스 추가")
                            strM = "Merr" + str(kk)
                            tmp = np.append(tmp, strM)  # for test
                            '''
                            kk = kk + 1

                elif DIR == -1:
                    strM = "M" + str(len(V_theta) - 1 - nn)
                    # tmp = np.append(tmp, strM)  # for test
                    if (nn + 1 - rem) % nSet == 0:
                        if kk != nM_vib and (nn + 1 - rem) != 0:
                            M = M_vib[..., -1 - kk].T @ M
                            '''
                            print(len(V_theta) - 1 - nn, "번째에 에러 매트릭스 추가 (-backward)")
                            strM = "Merr" + str(nM_vib - kk - 1)
                            tmp = np.append(tmp, strM)  # for test
                            '''
                            kk = kk + 1

        # print("rem=", rem, "nVerr=", nVerr, "nSet = ", nSet) # To show current spun fiber's info.
        # print(tmp) # To show where is the position of Merr
        return M

    def lamming_JET(self, Ip, DIR, V_delta, V_theta, V_L):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
        """
        m = 0
        # Fiber parameter
        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB


        r = self.L /(2*pi)
        V_H = Ip / (2*pi*r) * ones(len(V_L))
        V_rho = self.V * V_H    # <<- Vector

        # --------Laming: orientation of the local slow axis ------------

        V_qu = 2 * (s_t_r - V_rho) / V_delta  # <<- Vector
        # See Note/Note 1 (sign of Farday effect in Laming's method).jpg
        # The sign of farday rotation (rho) is opposite to that of the Laming paper, inorder
        # to be consistant with anti-clockwise (as in Jones paper) orientation for both
        # spin and faraday rotation.

        V_gma = 0.5 * (V_delta ** 2 + 4 * ((s_t_r - V_rho) ** 2)) ** 0.5  # <<- Vector
        '''
        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nf = -int(gma * self.dz / pi) - 1
        else:
            nf = -int(gma * self.dz / pi)

        if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) > 0:
            nb = int(gma * self.dz / pi)
        else:
            nb = int(gma * self.dz / pi) + 1
        '''
        V_n = zeros(len(V_rho))
        for nn in range(len(V_rho)):
            if arctan((-V_qu[nn] / ((1 + V_qu[nn] ** 2) ** 0.5)) * tan(V_gma[nn] * self.dz)) > 0:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) - 0.5*(1+DIR)
            else:
                V_n[nn] = -DIR*int(V_gma[nn] * self.dz / pi) + 0.5*(1-DIR)

        V_omega = s_t_r * self.dz + arctan((-V_qu / ((1 + V_qu ** 2) ** 0.5)) * tan(V_gma * self.dz)) + V_n * pi  # <<- Vector
        V_R = 2 * arcsin(sin(V_gma * self.dz) / ((1 + V_qu ** 2) ** 0.5))  # <<- Vector
        V_phi = ((s_t_r * self.dz) - V_omega) / 2 + m * (pi / 2)
        V_phi += V_theta if DIR == 1 else np.flip(V_theta)

        n11 = cos(V_R / 2) + 1j * sin(V_R / 2) * cos(2 * V_phi)
        n12 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n21 = 1j * sin(V_R / 2) * sin(2 * V_phi)
        n22 = cos(V_R / 2) - 1j * sin(V_R / 2) * cos(2 * V_phi)

        M_R = np.array([[n11, n21], [n12, n22]]).T
        M_omega = np.array([[cos(V_omega), sin(V_omega)], [-sin(V_omega), cos(V_omega)]]).T

        # Note that [[n11,n21],[n21,n22]].T calculation is [[n11[0], n12[0]],[n21[0],n22[0]], ...
        # Therefore, M_R, M_omega array should be defined as transposed matrix to have correct matrix.

        M = np.array([[1, 0], [0, 1]])
        for nn in range(len(V_theta)-1):
            M = M_omega[nn] @ M_R[nn] @ M

        return M

    def f_in_bridge(self, Ip, DIR, n_Bridge, V_theta, V_LF, M_vib=None ):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
        """
        m = 0
        # Fiber parameter
        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        r = self.L /(2*pi)
        if DIR == 1 and n_Bridge == 1:
            V_H = Ip * r / (2*pi*(r**2+ (self.LF - V_LF)**2))
        elif DIR ==1 and n_Bridge == 2:
            V_H = Ip * r / (2*pi*(r**2+ V_LF**2))
        elif DIR == -1 and n_Bridge == 2:
            V_H = Ip * r / (2*pi*(r**2+ (self.LF - V_LF)**2))
        elif DIR == -1 and n_Bridge == 1:
            V_H = Ip * r / (2*pi*(r**2+ V_LF**2))
        V_rho = self.V * V_H    # <<- Vector

        return V_rho

    def cal_rotation_ori(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None, Vin=None):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        s_t_r = 2 * pi / self.SP
        #Vin = np.array([[1], [0]])

        mm = 0
        for iter_I in V_plasmaCurrent:
            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, self.LF+self.dz , self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, self.L+self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Another lead fiber vector with V_theta_lf2
            V_L_lf2 = arange(0, self.LF+self.dz , self.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf2 * s_t_r

            # Faraday mirror
            ksi = ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming(0, 1, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, V_theta)
            M_lf_f2 = self.lamming(0, 1, V_theta_lf2, M_vib)
            M_lf_b2 = self.lamming(0, -1, V_theta_lf2, M_vib)
            M_b = self.lamming(iter_I, -1, V_theta)
            M_lf_b = self.lamming(0, -1, V_theta_lf, M_vib)


            # M_lf_f = self.lamming(iter_I, 1, V_theta_lf, M_vib)
            # M_lf_b = self.lamming(iter_I, -1, V_theta_lf, M_vib)
            # M_f = self.lamming(iter_I, 1, L, V_theta)
            # M_b = self.lamming(iter_I, -1, L, V_theta)

            if num == 0 and iter_I == V_plasmaCurrent[0]:
                #print("M_lf_f = ", M_lf_f[0, 1], M_lf_f[1, 0])
                #print("M_lf_b = ", M_lf_b[0, 1], M_lf_b[1, 0])
                #print("abs() = ", abs(M_lf_f[0, 1])-abs(M_lf_b[1, 0]))
                print("Norm (MLead_f - MLead_b.T) = ", norm(M_lf_f - M_lf_b.T))
                # print("M_f = ", M_f[0, 1], M_f[1, 0])
                # print("M_b = ", M_b[0, 1], M_b[1, 0])
                #print("Norm (Msens_f - Msens_b) = ", norm(M_f - M_b))

            V_out[mm] = M_lf_b @ M_b @ M_lf_b2 @ M_FR @ M_lf_f2 @ M_f @ M_lf_f @ Vin
            #V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ Vin
            #V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ Vin
            #V_out[mm] = M_f @ M_lf_f @ Vin
            # V_out[mm] =  M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
            print("mm")

        Vout_dic[num] = V_out

    def cal_rotation3(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None, Vin=None):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        s_t_r = 2 * pi / self.SP
        # Vin = np.array([[1], [0]])

        mm = 0
        for iter_I in V_plasmaCurrent:
            # Lead fiber vector with V_theta_lf
            self.dz = self.LF/100
            V_L_lf = arange(0, self.LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r
            # Sensing fiber vector with V_theta
            #self.dz = self.SP / 100
            self.dz = self.L / 1
            V_L = arange(0, self.L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Another lead fiber vector with V_theta_lf2
            self.dz = self.LF/100
            V_L_lf2 = arange(0, self.LF + self.dz, self.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf2 * s_t_r

            # Faraday mirror
            ksi = ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            # Fiber bundle
            phi = pi/4
            Ret = np.array([[np.exp(1j*phi/2), 0], [0, np.exp(-1j*phi/2)]])
            M_FB = Ret

            self.dz = self.LF/100
            M_lf_f = self.lamming_bridge(-iter_I, 1, 1, V_theta_lf, V_L_lf, M_vib)
            self.dz = self.L/1
            M_f = self.lamming_VV(iter_I, 1, V_theta, V_L)
            self.dz = self.LF /100
            M_lf_f2 = self.lamming_bridge(iter_I, 1, 2, V_theta_lf2, V_L_lf2, M_vib)
            M_lf_b2 = self.lamming_bridge(iter_I, -1, 2, V_theta_lf2, V_L_lf2, M_vib)
            self.dz = self.L / 1
            M_b = self.lamming_VV(iter_I, -1, V_theta, V_L)
            self.dz = self.LF / 100
            M_lf_b = self.lamming_bridge(-iter_I, -1, 1, V_theta_lf, V_L_lf, M_vib)

            if num == 0 and iter_I == V_plasmaCurrent[0]:
                # print("M_lf_f = ", M_lf_f[0, 1], M_lf_f[1, 0])
                # print("M_lf_b = ", M_lf_b[0, 1], M_lf_b[1, 0])
                # print("abs() = ", abs(M_lf_f[0, 1])-abs(M_lf_b[1, 0]))
                print("Norm (MLead_f - MLead_b.T) = ", norm(M_lf_f - M_lf_b.T))
                # print("M_f = ", M_f[0, 1], M_f[1, 0])
                # print("M_b = ", M_b[0, 1], M_b[1, 0])
                # print("Norm (Msens_f - Msens_b) = ", norm(M_f - M_b))

            V_out[mm] = M_lf_b @ M_b @ M_lf_b2 @ M_FB @ M_FR @ M_FB @ M_lf_f2 @ M_f @ M_lf_f @ Vin

            mm = mm + 1
        # print("done")

        Vout_dic[num] = V_out

    def cal_rotation(self, V_Ip, num, Vout_dic, temp = None):
        # for temperature distribution simulation
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        mm = 0
        for iter_I in V_plasmaCurrent:

            # Faraday mirror
            ksi = self.ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot
            if temp == None:
                M_f = self.lamming_VV_temp(iter_I, 1)
                M_b = self.lamming_VV_temp(iter_I, -1)
            else:
                M_f = self.lamming_VV_const_temp(iter_I, 1, temp)
                M_b = self.lamming_VV_const_temp(iter_I, -1, temp)

            V_out[mm] = M_b @ M_FR @ M_f @ self.Vin

            mm = mm + 1

        Vout_dic[num] = V_out

    def calc_mp(self, num_processor, V_I):
        # for temperature distribution simulation
        spl_I = np.array_split(V_I, num_processor)

        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        #print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            # proc = Process(target=self.cal_rotation,
            proc = Process(target=self.cal_rotation,
                           args=(spl_I[num], num, Vout_dic))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        Vout = Vout_dic[0]
        for kk in range(num_processor - 1):
            Vout = np.vstack((Vout, Vout_dic[kk + 1]))

        return Vout

    def calc_sp(self, V_I, temp = None):

        Vout = [[]]
        self.cal_rotation(V_I, 0, Vout, temp)

        return Vout[0]

    def plot_error(self, filename):

        data = pd.read_csv(filename)

        V_I = data['Ip']

        # Requirement specificaion for ITER
        absErrorlimit = zeros(len(V_I))
        relErrorlimit = zeros(len(V_I))

        # Calcuation ITER specification
        for nn in range(len(V_I)):
            if V_I[nn] < 1e6:
                absErrorlimit[nn] = 10e3
            else:
                absErrorlimit[nn] = V_I[nn] * 0.01
            if V_I[nn] == 0:
                pass
            else:
                relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        for col_name in data:
            if col_name != 'Ip':
                if V_I[0] == 0:
                    lines += ax.plot(V_I[1:], abs((data[col_name][1:] - V_I[1:]) / V_I[1:]))
                # lines += ax.plot(V_I, abs((data[col_name]-V_I)/V_I), label=col_name)
                else:
                    lines += ax.plot(V_I, abs((data[col_name] - V_I) / V_I))

        if V_I[0] == 0:
            lines += ax.plot(V_I[1:], relErrorlimit[1:], 'r', label='ITER specification')
        else:
            lines += ax.plot(V_I, relErrorlimit, 'r', label='ITER specification')
        ax.legend(loc="upper right")
        plt.rc('text', usetex=True)
        ax.set_xlabel(r'Plasma current $I_{p}(A)$')
        ax.set_ylabel(r'Relative error on $I_{P}$')
        ax.set_title("Spun fiber.plot_error")
        # plt.title('Output power vs Plasma current')
        ax.set(xlim=(0, 18e6), ylim=(0, 0.1))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(10))

        ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

        ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
        ax.grid(ls='--', lw=0.5)

        # fig.align_ylabels(ax)
        fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.2)
        # fig.set_size_inches(6,4)
        plt.rc('text', usetex=False)

        return fig, ax, lines
        # plt.show()

    def plot_errorbar(self, filename):

        data = pd.read_csv(filename)

        V_I = data['Ip']

        ## Requirement specificaion for ITER
        absErrorlimit = zeros(len(V_I))
        relErrorlimit = zeros(len(V_I))

        # Calcuation ITER specification
        for nn in range(len(V_I)):
            if V_I[nn] < 1e6:
                absErrorlimit[nn] = 10e3
            else:
                absErrorlimit[nn] = V_I[nn] * 0.01
            if V_I[nn] == 0:
                pass
            else:
                relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]
        if V_I[0] == 0:
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1).drop(0, axis=0)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1).drop(0, axis=0)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(V_I[1:], df_mean[:], label="mean value")
            ax.errorbar(V_I[1::3], df_mean[::3], yerr=df_std[::3], label="std", ls='None', c='black', ecolor='g', capsize=4)
        else:
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(V_I, df_mean, label="mean value")
            ax.errorbar(V_I[::2], df_mean[::2], yerr=df_std[::2], label="std", ls='None', c='black', ecolor='g', capsize=4)


        lines = []

        if V_I[0] == 0:
            ax.plot(V_I[1:], relErrorlimit[1:], 'r', label='ITER specification')
            ax.plot(V_I[1:], -relErrorlimit[1:], 'r')
        else:
            ax.plot(V_I, relErrorlimit, 'r', label='ITER specification')
            ax.plot(V_I, -relErrorlimit, 'r')

        ax.legend(loc="upper right")

        plt.rc('text', usetex=True)
        ax.set_xlabel(r'Plasma current $I_{p}(A)$')
        ax.set_ylabel(r'Relative error on $I_{P}$')

        # plt.title('Output power vs Plasma current')
        ax.set(xlim=(0, 18e6), ylim=(-0.012, 0.012))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(10))

        ax.xaxis.set_major_formatter(OOMFormatter(6, "%1.0f"))
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%4.3f"))

        ax.ticklabel_format(axis='x', style='sci', useMathText=True, scilimits=(-3, 5))
        ax.grid(ls='--', lw=0.5)

        # fig.align_ylabels(ax)
        fig.subplots_adjust(left = 0.17, hspace=0.4, right=0.95, top=0.93, bottom=0.2)
        # fig.set_size_inches(6,4)
        plt.rc('text', usetex=False)

        return fig, ax, lines
        #plt.show()

    def add_plot(self, filename, ax, str_label):

        data = pd.read_csv(filename)

        V_I = data['Ip']

        ## Requirement specificaion for ITER
        absErrorlimit = zeros(len(V_I))
        relErrorlimit = zeros(len(V_I))

        for col_name in data:
            if col_name != 'Ip':
                ax.plot(V_I, abs((data[col_name] - V_I) / V_I), label=str_label)
        ax.legend(loc="upper right")

    def init_plot_SOP(self):
        S = create_Stokes('Output_S')
        fig, ax= S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')
        return fig, ax


def save_Jones2(filename, V_I, V_out):
    if os.path.exists(filename + "_S"):
        df2 = pd.read_csv(filename + "_S")
        ncol2 = int((df2.shape[1] - 1) / 2)
        df2[str(ncol2) + ' Ex'] = V_out[:, 0, 0]
        df2[str(ncol2) + ' Ey'] = V_out[:, 1, 0]
    else:
        out_dict2 = {'Ip': V_I}
        out_dict2['0 Ex'] = V_out[:, 0, 0]
        out_dict2['0 Ey'] = V_out[:, 1, 0]
        df2 = pd.DataFrame(out_dict2)

    df2.to_csv(filename + "_S", index=False)

def save_Jones(filename, Vin, Ip_m, Vout):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        ncol1 = int((df.shape[1] - 1))
        df[str(ncol1)] = Ip_m

        df2 = pd.read_csv(filename + "_S")
        ncol2 = int((df2.shape[1] - 1) / 2)
        df2[str(ncol2) + ' Ex'] = Vout[:, 0, 0]
        df2[str(ncol2) + ' Ey'] = Vout[:, 1, 0]
    else:
        out_dict = {'Ip': Vin}
        out_dict['0'] = Ip_m
        df = pd.DataFrame(out_dict)

        out_dict2 = {'Ip': Vin}
        out_dict2['0 Ex'] = Vout[:, 0, 0]
        out_dict2['0 Ey'] = Vout[:, 1, 0]
        df2 = pd.DataFrame(out_dict2)

    df.to_csv(filename, index=False)
    df2.to_csv(filename + "_S", index=False)

def load_stokes_fromfile(filename, ncol=0):
    data = pd.read_csv(filename)
    # if data['Ip'][0] == 0:
    #     data.drop(0, inplace=True)
    #     data.index -= 1
    V_I = data['Ip']
    E = Jones_vector('Output')
    S = create_Stokes('Output_S')
    for nn in range(int((data.shape[1] - 1) / 2)):
        if nn == ncol+1:
            break
        str_Ex = str(nn) + ' Ex'
        str_Ey = str(nn) + ' Ey'
        Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                         [complex(y) for y in data[str_Ey].to_numpy()]])
        E.from_matrix(Vout)
        S.from_Jones(E)
    isEOF = True if ncol >= int((data.shape[1] - 1) / 2)-1 else False

    return V_I, S, isEOF

def cal_error_fromStocks(V_I, S, V_custom=None, v_calc_init=None):
    V_ang = zeros(len(V_I))
    Ip = zeros(len(V_I))
    V = 0.54 * 4 * pi * 1e-7 if V_custom is None else V_custom

    m = 0
    for kk in range(len(V_I)):
        if kk > 0 and S[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
            m = m + 1
        elif kk > 0 and S[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
            m = m - 1
        V_ang[kk] = S[kk].parameters.azimuth() + m * pi

        c = V_ang[0] if v_calc_init is None else v_calc_init
        Ip[kk] = (V_ang[kk] - c) / V
    print("init angle? = ",V_ang[0])
    return (Ip[1:] - V_I[1:]) / V_I[1:]

# Progress bar is not easy/
# Todo comparison between transmission and reflection
# Todo FM effect
# Todo Ip calculation method change --> azimuth angle --> arc length

if __name__ == '__main__':
    mode =6
    if mode == 0:
        LB = 0.009
        SP = 0.005
        # dz = SP / 1000
        dz = 0.0001
        len_lf = 1  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        # 44FM_Errdeg1x5_0 : length of leadfiber 10 m
        # 44FM_Errdeg1x5_1 : length of leadfiber 10->20 m

        num_iter = 4
        strfile1 = 'AAAA1.csv'
        strfile2 = 'AAAA2.csv'
        num_processor = 16
        V_I = arange(0e6, 4e6 + 0.1e6, 0.2e6)
        #V_I = 1e6
        outdict = {'Ip': V_I}
        outdict2 = {'Ip': V_I}
        nM_vib = 0
        start = pd.Timestamp.now()
        ang_FM = 45
        Vin = np.array([[1], [0]])

        fig1, ax1 = spunfiber.init_plot_SOP()
        for nn in range(num_iter):
            M_vib = spunfiber.create_Mvib(nM_vib, 0, 0)
            #Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
            if nn == 0:
                Ip, Vout = spunfiber.stacking_matrix_rotation(V_I, Vin)
                outdict[str(nn)] = Ip
            elif nn ==1:
                Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
                outdict[str(nn)] = Ip
            elif nn==2:
                Ip, Vout = spunfiber.total_rotation(V_I, fig1, Vin)
                outdict[str(nn)] = -Ip
            else:
                Ip, Vout = spunfiber.stacking_laming(V_I, fig1, Vin)
                outdict[str(nn)] = -Ip

            outdict2[str(nn) + ' Ex'] = Vout[:, 0, 0]
            outdict2[str(nn) + ' Ey'] = Vout[:, 1, 0]
            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        df = pd.DataFrame(outdict)
        df.to_csv(strfile1, index=False)
        df2 = pd.DataFrame(outdict2)
        df2.to_csv(strfile1 + "_S", index=False)
        fig2, ax2, lines = spunfiber.plot_error(strfile1)

        labelTups = [('Stacking matrix (dz = SP/25)', 0),
                     ('Lamming method with small step (dz = SP/25)', 1),
                     ('Lamming method for whole fiber (dz = L)', 2),
                     ('Lamming method with small step2 (dz = SP/25)', 3),
                     ('Iter specification', 4)]

        #ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
        ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right')
        ax2.set(xlim=(0, 4e6), ylim=(0, 0.2))
        ax2.xaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-1, "%1.1f"))
    if mode == 1:
        LB = 1
        SP = 0.005
        # dz = SP / 1000
        dz = 0.0002
        len_lf = 0  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        strfile1 = 'b1.csv'
        strfile2 = 'b2.csv'

        fig1, ax1 = spunfiber.init_plot_SOP()

        vV_I = [1e6]

        nM_vib = 0
        ang_FM = 45
        Vin = np.array([[1], [0]])

        E = Jones_vector('Output')
        S_dL = create_Stokes('Laming_dL')
        S_L = create_Stokes('Laming_L')
        S_S = create_Stokes('Stacking')
        S_STL = create_Stokes('Stacking_lamming')

        for V_I in vV_I:

            V_dL = np.array([])
            V_St = np.array([])

            var_dL = SP*10**(-np.arange(0, 5, 1, dtype=float))

            for nn, var in enumerate(var_dL):
                spunfiber.dz = var
                Vout = spunfiber.single_rotation2(V_I, Vin)         # cal rotation angle using lamming method (variable dL)
                V_dL = np.append(V_dL, S_dL.from_Jones(E.from_matrix(Vout)).parameters.matrix())
                draw_stokes_points(fig1[0], S_dL, kind='scatter', color_scatter='b')

                # Vout = spunfiber.single_rotation3(V_I, Vin)         # cal rotation angle using stacking method (dL=variable)
                # V_St = np.append(V_St, S_S.from_Jones(E.from_matrix(Vout)).parameters.matrix())
                # draw_stokes_points(fig1[0], S_S, kind='scatter', color_scatter='k')

        V_dL = V_dL.reshape(len(var_dL), 4)
        # V_St = V_St.reshape(len(var_dL), 4)
        # V_StL = V_StL.reshape(len(var_dL), 4)

        # print(V_dL)
        # print(V_St)
        # V_L = np.ones(len(var_dL))*V_L
        figure, ax = plt.subplots(3, figsize=(5, 8))
        figure.subplots_adjust(left=0.179, bottom=0.15, right=0.94, hspace=0.226, top=0.938)

        ax[0].plot(var_dL, V_dL[...,1], 'r', label='Laming')
        # ax[0].plot(var_dL, V_St[...,1], 'b', label='Stacking')
        # ax[0].plot(var_dL, V_L[1,...], 'k--', label='Laming(w/o slicing)')
        # ax[0].plot(var_dL, V_StL[...,1], 'm--', label='Laming(w/o slicing)')

        ax[0].set_xscale('log')
        ax[0].set_ylabel('S1')
        ax[0].legend(loc='upper left')
        # ax[0].set_title('S1')
        ax[0].set_xticklabels('')

        ax[1].plot(var_dL, -V_dL[..., 2], 'r',  label='Laming')
        # ax[1].plot(var_dL, V_St[..., 2], 'b', label='Stacking')
        # ax[1].plot(var_dL, V_L[2,...], 'k--', label='Laming(w/o slicing)')
        # ax[1].plot(var_dL, V_StL[..., 2], 'm--', label='Laming(w/o slicing)')
        ax[1].set_ylabel('S2')
        ax[1].set_xscale('log')
        # ax[1].legend(loc='upper left')
        ax[1].set_title('S2')
        ax[1].set_xticklabels('')
        #
        ax[2].plot(var_dL, V_dL[..., 3], 'r', label='Laming')
        # ax[2].plot(var_dL, V_St[..., 3], 'b', label='Stacking')
        # ax[2].plot(var_dL, V_L[3,...], 'k--', label='Laming(w/o slicing)')
        # #ax[2].plot(var_dL, V_StL[..., 3], 'm--', label='Laming(w/o slicing)')
        ax[2].set_xscale('log')
        ax[2].set_xlabel('dL [m]')
        ax[2].set_ylabel('S3')
        # ax[2].legend(loc='lower left')
        # #ax[2].set_title('S3')
        ax[2].set_xticks(var_dL)
        str_xtick = ['SP/1', 'SP/10', 'SP/100', 'SP/1000', 'SP/10000']
        ax[2].set_xticklabels(str_xtick, minor=False, rotation=-45)

    if mode == 2:
        LB = 0.009
        SP = 0.0048
        # dz = SP / 1000
        dz = 0.0002
        len_lf = 0  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)

        num_iter = 50
        num_processor = 8
        V_I = arange(0e6, 18e6 + 0.1e6, 0.1e6)
        # V_I = 1e6

        nM_vib = 5
        start = pd.Timestamp.now()
        ang_FM = 20

        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        ksi2 = 45 * pi/ 180
        M_Ip = np.array([[cos(ksi2), -sin(ksi2)], [sin(ksi2), cos(ksi2)]])

        E = Jones_vector('input')
        Eo = Jones_vector('output')
        So = create_Stokes('1')

        azi = np.array([0, pi / 6, pi / 4])
        E.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)
        fig1, ax1 = spunfiber.init_plot_SOP()
        S = create_Stokes('O')
        Vin = E[0].parameters.matrix()

        fig1, ax1 = spunfiber.init_plot_SOP()
        tmp, SOPchange_mean, SOPchange_std, SOPchange_max = np.array([]), np.array([]), np.array([]), np.array([])
        tmp2 = np.array([])
        V_ang = np.arange(0,45,5)
        for ang_FM in V_ang:
            for nn in range(50):
                ksi = (45-ang_FM) * pi / 180
                Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
                Jm = np.array([[1, 0], [0, 1]])
                M_FR = Rot @ Jm @ Rot

                M_vib = spunfiber.create_Mvib(nM_vib, 1, 1)
                # Vout = M_vib[..., nn].T @ M_Ip @ M_vib2[..., nn].T @ M_vib2[..., nn] @ M_Ip @ M_vib[..., nn] @ Vin
                # Vout  = M_vib[..., nn].T@M_Ip@M_vib2[..., nn].T @ M_FR @ M_vib2[..., nn] @M_Ip@M_vib[..., nn]@ Vin
                # Vout = M_vib[..., nn].T @ M_Ip @M_FR @ M_Ip @ M_vib[..., nn] @ Vin

                M_v = M_vib[...,4]@M_vib[...,3]@M_vib[...,2]@M_vib[...,1]@M_vib[...,0]

                Vout = M_v.T@ M_FR @M_v @ Vin
                Eo.from_matrix(Vout)
                tmp = np.hstack((tmp, Eo.parameters.ellipticity_angle()*180/pi))
                # tmp2 = np.hstack((tmp2, Eo.parameters.azimuth()*180/pi + ang_FM*2))
                So.from_Jones(Eo)
                draw_stokes_points(fig1[0], So, kind='scatter', color_line='r')

            SOPchange_mean = np.hstack((SOPchange_mean, tmp.mean()))
            SOPchange_std = np.hstack((SOPchange_std, tmp.std()))
            SOPchange_max = np.hstack((SOPchange_max, tmp.max()))

        fig, ax = plt.subplots()
        ax.plot(V_ang, SOPchange_mean)
        ax.plot(V_ang, SOPchange_std)
        ax.plot(V_ang, SOPchange_max)
    if mode == 3:
        LB = 1
        SP = 0.005
        # dz = SP / 1000
        dz = 0.00001
        len_lf = 6  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        # 44FM_Errdeg1x5_0 : length of leadfiber 10 m
        # 44FM_Errdeg1x5_1 : length of leadfiber 10->20 m

        num_iter = 2
        strfile1 = 'AAAA1.csv'
        strfile2 = 'AAAA2.csv'
        num_processor = 16
        V_I = arange(0e6, 4e6 + 0.1e6, 0.1e6)
        # V_I = 1e6
        outdict = {'Ip': V_I}
        outdict2 = {'Ip': V_I}
        nM_vib = 5
        start = pd.Timestamp.now()
        ang_FM = 46
        Vin = np.array([[1], [0]])

        fig1, ax1 = spunfiber.init_plot_SOP()
        M_vib = spunfiber.create_Mvib(nM_vib, 1, 1)
        for nn in range(num_iter):

            # Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
            if nn == 0:
                Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
                outdict[str(nn)] = Ip
            elif nn == 1:
                Ip, Vout = spunfiber.calc_mp3(num_processor, V_I, ang_FM, M_vib, fig1, Vin)
                outdict[str(nn)] = Ip

            outdict2[str(nn) + ' Ex'] = Vout[:, 0, 0]
            outdict2[str(nn) + ' Ey'] = Vout[:, 1, 0]
            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        df = pd.DataFrame(outdict)
        df.to_csv(strfile1, index=False)
        df2 = pd.DataFrame(outdict2)
        df2.to_csv(strfile1 + "_S", index=False)
        fig2, ax2, lines = spunfiber.plot_error(strfile1)

        labelTups = [('Lamming method with small step (dz = SP/25)', 1),
                     ('Lamming method for whole fiber (dz = L)', 2),
                     ('Iter specification', 4)]

        # ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right', bbox_to_anchor=(0.7, .8))
        ax2.legend(lines, [lt[0] for lt in labelTups], loc='upper right')
        ax2.set(xlim=(0, 4e6), ylim=(0, 0.2))
        ax2.xaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-1, "%1.1f"))
    if mode == 4:
        LB = 0.009
        SP = 0.005
        # dz = SP / 1000
        dz = 0.5
        len_lf = 0  # lead fiber
        len_ls = 1  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        strfile1 = 'b1.csv'
        strfile2 = 'b2.csv'

        fig1, ax1 = spunfiber.init_plot_SOP()

        #vV_I = [1e6, 5e6, 15e6]
        vV_I = arange(0, 6e6, 1e6)

        nM_vib = 0
        ang_FM = 45
        Vin = np.array([[1], [0]])

        E = Jones_vector('Output')
        S_dL = create_Stokes('Laming_dL')
        S_L = create_Stokes('Laming_L')

        for V_I in vV_I:

            spunfiber.dz = 1
            print(spunfiber.dz)
            Vout = spunfiber.single_rotation4(V_I, Vin)  # cal rotation angle using lamming method (variable dL)
            V_L = S_dL.from_Jones(E.from_matrix(Vout)).parameters.matrix()
            draw_stokes_points(fig1[0], S_dL, kind='line', color_line='r')

            spunfiber.dz= 0.0001
            print(spunfiber.dz)
            Vout = spunfiber.single_rotation4(V_I, Vin)  # cal rotation angle using lamming method (variable dL)
            V_dL = S_dL.from_Jones(E.from_matrix(Vout)).parameters.matrix()
            draw_stokes_points(fig1[0], S_dL, kind='line', color_line='b')

            print(V_I)
            print(V_L.T)
            print(V_dL.T)

        figure, ax = plt.subplots(3, figsize=(5, 8))
        figure.subplots_adjust(left=0.179, bottom=0.15, right=0.94, hspace=0.226, top=0.938)

        # # ax[0].plot(var_dL, V_dL[..., 1], 'r', label='Laming')
        # # ax[0].plot(var_dL, V_L[1, ...], 'k--', label='Laming(w/o slicing)')
        # # ax[0].plot(var_dL, V_StL[...,1], 'm--', label='Laming(w/o slicing)')
        #
        # # ax[0].set_xscale('log')
        # # ax[0].set_ylabel('S1')
        # # ax[0].legend(loc='upper left')
        # # # ax[0].set_title('S1')
        # # ax[0].set_xticklabels('')
        # #
        # # ax[1].plot(var_dL, -V_dL[..., 2], 'r', label='Laming')
        # # ax[1].plot(var_dL, V_St[..., 2], 'b', label='Stacking')
        # # ax[1].plot(var_dL, V_L[2, ...], 'k--', label='Laming(w/o slicing)')
        # # # ax[1].plot(var_dL, V_StL[..., 2], 'm--', label='Laming(w/o slicing)')
        # # ax[1].set_ylabel('S2')
        # # ax[1].set_xscale('log')
        # # ax[1].legend(loc='upper left')
        # # # ax[1].set_title('S2')
        # # ax[1].set_xticklabels('')
        #
        # ax[2].plot(var_dL, V_dL[..., 3], 'r', label='Laming')
        # ax[2].plot(var_dL, V_L[3, ...], 'k--', label='Laming(w/o slicing)')
        # ax[2].set_xscale('log')
        # ax[2].set_xlabel('dL [m]')
        # ax[2].set_ylabel('S3')
        # ax[2].legend(loc='lower left')
        # # ax[2].set_title('S3')
        # ax[2].set_xticks(var_dL)
        # str_xtick = ['SP/50', 'SP/100', 'SP/500', 'SP/1000', 'SP/5000']
        # ax[2].set_xticklabels(str_xtick, minor=False, rotation=-45)

    if mode == 5:
        L = 1
        dz = 1
        LB = 1
        SP = 0.005

        v_dz = [1, 0.0001]
        V = 0.54 * 4 * pi * 1e-7
        E = Jones_vector('Output')
        S = create_Stokes('Output')
        tmp = 0
        for dz in v_dz:
            V_z = arange(0, L + dz, dz)
            delta = 2 * pi / LB

            mm = 0
            n = 0
            m = 0
            n2 = 0
            m2 = 0

            vV_I = [1e6, 0e6, 15e6]
            H = vV_I[1] / L
            rho = V * H

            # --------Laming: orientation of the local slow axis ------------
            # --------Laming matrix on spun fiber --------------------------

            # define forward
            # The sign of farday rotation is opposite to that of the Laming paper, in order
            # to be consistant with anti-clockwise (as in Jones paper) orientation for both
            # spin and farday rotation.
            s_t_r = 2 * pi / SP  # spin twist ratio
            V_theta_1s = V_z * s_t_r
            qu = 2 * (s_t_r - rho) / delta
            gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5

            R_zf = 2 * arcsin(sin(gma * dz) / ((1 + qu ** 2) ** 0.5))

            Le = 2 * pi / gma
            #V_nf = -((V_z / (Le / 4)).astype(int) / 2).astype(int)
            nf = -int(gma*dz/pi)-1 if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) > 0 else -int(gma*dz/pi)

            Omega_zf = s_t_r * dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) + nf * pi
            Phi_zf = ((s_t_r * dz) - Omega_zf) / 2 + m * (pi / 2)

            # forward
            MF = np.array([[1, 0], [0, 1]])
            for kk in range(len(V_theta_1s) - 1):
                Omega_z2 = Omega_zf
                Phi_z2 = Phi_zf + V_theta_1s[kk]
                R_z2 = R_zf
                n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                M_R_f = np.array([[n11, n12], [n21, n22]])
                M_Omega_f = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
                MF = M_Omega_f @ M_R_f @ MF
            # print("Omega_z2= ", Omega_z2)

            print("### Forward")
            print("gma*dz= ", gma * dz)
            print("xx +nf*pi = ", arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) + nf * pi)
            # define backward
            # spin and farday rotation.
            s_t_r = -2 * pi / SP  # spin twist ratio
            # V_theta_1s = V_z * s_t_r
            qu = 2 * (s_t_r - rho) / delta
            gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5

            R_zb = 2 * arcsin(sin(gma * dz) / ((1 + qu ** 2) ** 0.5))
            nb = int(gma * dz / pi) if arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) > 0 else int(
                gma * dz / pi)+1

            Omega_zb = s_t_r * dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) + nb * pi
            Phi_zb = ((s_t_r * dz) - Omega_zf) / 2 + m * (pi / 2)

            # backward
            MB = np.array([[1, 0], [0, 1]])
            for kk in range(len(V_theta_1s) - 1):
                Omega_z2 = Omega_zb
                Phi_z2 = Phi_zb + V_theta_1s[-1-kk]
                R_z2 = R_zb
                n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                M_R_b = np.array([[n11, n12], [n21, n22]])
                M_Omega_b = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
                MB = M_Omega_b @ M_R_b @ MB
                # print("Omega_z2= ", Omega_z2)

            print("### Backward")
            print("gma*dz= ", gma * dz)
            print("xx +nb*pi = ", arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * dz)) + nb * pi)

            ksi =45*pi/180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            V_in = np.array([[1],[0]])

            V_out = MB@M_FR@MF@V_in

            E.from_matrix(V_out)
            print(S.from_Jones(E).parameters.matrix()[1])
            print(S.from_Jones(E).parameters.matrix()[2])
            print(S.from_Jones(E).parameters.matrix()[3])

            print(S.parameters.azimuth()*180/pi)

    if mode == 6:
    # to Check Magnetic field in Bridge
        LB = 0.009
        SP = 0.005
        s_t_r = 2 * pi / SP
        dz = 0.5
        len_lf = 6  # lead fiber
        len_ls = 28  # sensing fiber
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)

        fig1, ax1 = spunfiber.init_plot_SOP()

        vV_I = [1e6]
        # vV_I = arange(0, 6e6, 1e6)

        mm = 0
        V_rho=np.array([])
        for iter_I in vV_I:
            # Lead fiber vector with V_theta_lf
            spunfiber.dz = spunfiber.LF / 100
            V_L_lf = arange(0, spunfiber.LF + spunfiber.dz, spunfiber.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            spunfiber.dz = spunfiber.SP / 100
            V_L = arange(0,spunfiber.L + spunfiber.dz, spunfiber.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Another lead fiber vector with V_theta_lf2
            spunfiber.dz = spunfiber.LF / 100
            V_L_lf2 = arange(0, spunfiber.LF + spunfiber.dz, spunfiber.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf2 * s_t_r


            spunfiber.dz = spunfiber.LF / 100
            V_rho1 = spunfiber.f_in_bridge(-iter_I, 1, 1, V_theta_lf, V_L_lf)
            V_rho2 = spunfiber.f_in_bridge(iter_I, 1, 2, V_theta_lf2, V_L_lf2)
            V_rho3 = spunfiber.f_in_bridge(iter_I, -1, 2, V_theta_lf2, V_L_lf2)
            V_rho4 = spunfiber.f_in_bridge(-iter_I, -1, 1, V_theta_lf, V_L_lf)
            V_rho = np.hstack((V_rho1, V_rho2, V_rho3, V_rho4))
        print(V_rho)
        f = open('V_rho', 'w')
        writer = csv.writer(f)
        writer.writerow(V_rho)
        f.close()
        figure, ax = plt.subplots(figsize=(5, 8))
        ax.plot(V_rho)
plt.show()
