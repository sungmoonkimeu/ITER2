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

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt
import os

#from .basis_correction_lib import calib_basis3
#from .draw_poincare_plotly import *

# import parmap
import tqdm


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


class SPUNFIBER:
    def __init__(self, beat_length, spin_pitch, delta_l, len_lf, len_sf):
        self.LB = beat_length
        self.SP = spin_pitch
        self.dz = delta_l
        self.V = 0.54 * 4* pi * 1e-7
        self.LF = len_lf
        self.L = len_sf

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

    def stacking_matrix_rotation(self, V_Ip, Vin=None):
        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        Ip = zeros(len(V_Ip))
        dq = 2 * pi / self.SP  # spin twist ratio
        delta = 2 * pi / self.LB
        LC = 1*2*pi*100000000000000
        I = 1
        Len_SF = 1
        V = 0.54
        rho_C = 2*pi/LC
        rho_F = V * 4 * pi * 1e-7 / (Len_SF * I)  # Non reciprocal circular birefringence for unit ampare and unit length[rad/m·A]

        delta_L = 0.0001
        V_L = arange(delta_L, Len_SF + delta_L, delta_L)
        n_V_L = len(V_L)

        # ------------------------------ Variable forward--------------
        rho_1 = rho_C + rho_F * V_Ip
        delta_Beta_1 = 2 * (rho_1 ** 2 + (delta ** 2) / 4) ** 0.5

        alpha_1 = cos(delta_Beta_1 / 2 * delta_L)
        beta_1 = delta / delta_Beta_1 * sin(delta_Beta_1 / 2 * delta_L)
        gamma_1 = 2 * rho_1 / delta_Beta_1 * sin(delta_Beta_1 / 2 * delta_L)

        # ------------------------------ Variable backward--------------
        rho_2 = rho_C + rho_F * V_Ip
        delta_Beta_2 = 2 * (rho_2 ** 2 + (delta ** 2) / 4) ** 0.5

        alpha_2 = cos(delta_Beta_2 / 2 * delta_L)
        beta_2 = delta / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)
        gamma_2 = 2 * rho_2 / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)

        # ------------------------------ Variable FRM--------------
        # print(J)

        E = Jones_vector('Output')

        Ip = zeros(len(V_Ip))
        V_ang = zeros(len(V_Ip))
        JF = np.mat([[0, 1], [-1, 0]])

        m = 0
        for nn in range(len(V_Ip)):
            #print("mm = ", mm, " nn = ", nn)
            q = 0
            J = np.mat([[1, 0], [0, 1]])
            JT = np.mat([[1, 0], [0, 1]])
            for kk in range(len(V_L)):
                q = q + dq * delta_L

                J11 = alpha_1[nn] + 1j * beta_1[nn] * cos(2 * q)
                J12 = -gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
                J21 = gamma_1[nn] + 1j * beta_1[nn] * sin(2 * q)
                J22 = alpha_1[nn] - 1j * beta_1[nn] * cos(2 * q)

                J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J

                J11 = alpha_2[nn] + 1j * beta_2[nn] * cos(2 * q)
                J12 = -gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
                J21 = gamma_2[nn] + 1j * beta_2[nn] * sin(2 * q)
                J22 = alpha_2[nn] - 1j * beta_2[nn] * cos(2 * q)

                JT = JT @ np.vstack((J11, J12, J21, J22)).T.reshape(2, 2)

            # print(q)
            V_out[nn] = JT @ JF @ J @ Vin
            E.from_matrix(M=V_out[nn])

            if nn > 2 and E.parameters.azimuth() + m * pi - V_ang[nn - 1] < -pi * 0.9:
                m = m + 1
            V_ang[nn] = E.parameters.azimuth() + m * pi
            Ip[nn] = (V_ang[nn] - pi / 2) / (2 * V * 4 * pi * 1e-7)

        return Ip, V_out

    def total_rotation(self, V_Ip, fig=None, Vin=None):

        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        Ip = zeros(len(V_Ip))
        delta = 2 * pi / self.LB

        mm = 0
        n, m, n2, m2 = 0, 0, 0, 0

        H = V_Ip/self.L
        rho = self.V * H

        # define forward
        s_t_r = 2 * pi / self.SP  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zf = 2 * arcsin(sin(gma * self.L) / ((1 + qu ** 2) ** 0.5))
        Omega_zf, Phi_zf = zeros(len(V_Ip)), zeros(len(V_Ip))
        for nn in range(len(V_Ip)):
            tmp_Omega = s_t_r * self.L + \
                        arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * self.L)) - n * pi
            if nn > 1 and tmp_Omega - Omega_zf[nn-1] > pi/2:
                n += 1
            elif nn > 1 and tmp_Omega - Omega_zf[nn-1] < -pi/2:
                n -= 1
            Omega_zf[nn] = s_t_r * self.L + \
                           arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * self.L)) - n * pi

            tmp_Phi = ((s_t_r * self.L) - Omega_zf[nn]) / 2 + m * (pi / 2)
            if nn > 1 and tmp_Phi - Phi_zf[nn - 1] > pi / 2:
                m += 1
            elif nn > 1 and tmp_Phi - Phi_zf[nn - 1] < -pi / 2:
                m -= 1
            Phi_zf[nn] = ((s_t_r * self.L) - Omega_zf[nn]) / 2 + m * (pi / 2)

        V_theta_1s = self.L * s_t_r


        # define backward
        s_t_r = -s_t_r  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zb = 2 * arcsin(sin(gma * self.L) / ((1 + qu ** 2) ** 0.5))
        Omega_zb, Phi_zb = zeros(len(V_Ip)), zeros(len(V_Ip))
        for nn in range(len(V_Ip)):
            tmp_Omega = s_t_r * self.L + \
                        arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * self.L)) - n2 * pi

            if nn > 1 and tmp_Omega - Omega_zb[nn-1] > pi/2:
                n2 += 1
            elif nn > 1 and tmp_Omega - Omega_zb[nn-1] < -pi/2:
                n2 -= 1
            Omega_zb[nn] = s_t_r * self.L + \
                           arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * self.L)) - n2 * pi

            tmp_Phi = ((s_t_r * self.L) - Omega_zb[nn]) / 2 + m2 * (pi / 2)
            if nn > 1 and tmp_Phi - Phi_zb[nn - 1] > pi / 2:
                m2 += 1
            elif nn > 1 and tmp_Phi - Phi_zb[nn - 1] < -pi / 2:
                m2 -= 1
            Phi_zb[nn] = ((s_t_r * self.L) - Omega_zb[nn]) / 2 + m2 * (pi / 2)

        # Faraday mirror
        ang_FM = 45
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        for nn in range(len(V_Ip)):
            # forward
            Phi_z2 = Phi_zf[nn]
            R_z2 = R_zf[nn]
            Omega_z2 = Omega_zf[nn]
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_f = np.array([[n11, n12], [n21, n22]])
            M_Omega_f = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MF = M_Omega_f @ M_R_f

            # Backward
            Phi_z2 = Phi_zb[nn] + V_theta_1s
            R_z2 = R_zb[nn]
            Omega_z2 = Omega_zb[nn]
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_b = np.array([[n11, n12], [n21, n22]])
            M_Omega_b = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MB = M_Omega_b @ M_R_b

            V_out[mm] = MB @ M_FR @ MF @ Vin
            mm += 1

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
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = (E[kk].parameters.azimuth() + m * pi)
            Ip[kk] = (V_ang[kk] - V_ang[0]) / (2 * self.V)
            #print(V_ang[kk], Ip[kk])
        return Ip, V_out

    def stacking_laming(self, V_Ip, fig=None, Vin=None):

        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        Ip = zeros(len(V_Ip))

        delta = 2 * pi / self.LB

        mm = 0
        n = 0
        m = 0
        n2 = 0
        m2 = 0

        V_z = arange(0, self.L+self.dz, self.dz)

        H = V_Ip/self.L
        rho = self.V * H

        # --------Laming: orientation of the local slow axis ------------
        # --------Laming matrix on spun fiber --------------------------

        # define forward
        s_t_r = 2 * pi / self.SP  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zf = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))
        Omega_zf = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) - n * pi
        Phi_zf = ((s_t_r * self.dz) - Omega_zf) / 2 + m * (pi / 2)
        V_theta_1s = V_z * s_t_r

        # define backward
        s_t_r = -s_t_r  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zb = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))
        Omega_zb = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) - n2 * pi
        Phi_zb = ((s_t_r * self.dz) - Omega_zb) / 2 + m2 * (pi / 2)

        # Faraday mirror
        ang_FM = 45
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        for nn in range(len(V_Ip)):
            # forward
            MF = np.array([[1, 0], [0, 1]])
            for kk in range(len(V_theta_1s)-1):
                Phi_z2 = Phi_zf[nn] + V_theta_1s[kk]
                #print(Phi_z2)
                R_z2 = R_zf[nn]
                Omega_z2 = Omega_zf[nn]
                n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                M_R_f = np.array([[n11, n12], [n21, n22]])
                M_Omega_f = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
                MF = M_Omega_f @ M_R_f @ MF

            # Backward
            MB = np.array([[1, 0], [0, 1]])
            for kk in range(len(V_theta_1s)-1):
                Phi_z2 = Phi_zb[nn] + V_theta_1s[-1-kk]
                R_z2 = R_zb[nn]
                Omega_z2 = Omega_zb[nn]
                n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
                n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
                M_R_b = np.array([[n11, n12], [n21, n22]])
                M_Omega_b = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
                MB = M_R_b @ M_Omega_b @ MB

            V_out[mm] = MB @ M_FR @ MF @ Vin
            #V_out[mm] = M_FR @ MF @ Vin
            mm += 1

        E = Jones_vector('Output')
        E.from_matrix(M=V_out)
        V_ang = zeros(len(V_Ip))

        # SOP evolution in Lead fiber (Forward)
        S = create_Stokes('Output_S')
        S.from_Jones(E)

        if fig is not None:
            draw_stokes_points(fig[0], S, kind='line', color_line='m')
        else:
            fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line',
                                      color_line='m')
        m = 0
        for kk in range(len(V_I)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = (E[kk].parameters.azimuth() + m * pi)
            Ip[kk] = (V_ang[kk] - V_ang[0]) / (2 * self.V)
            #print(V_ang[kk], Ip[kk])
        return Ip, V_out

    def lamming(self, Ip, DIR, V_theta, M_vib=None):
        """
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :return: M matrix calculated from N matrix
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
        M_vib = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)

        return M_vib

    def cal_2ndBridge(self, ang_FM, num, M_vib=None, Vin=None):

        s_t_r = 2 * pi / self.SP
        #Vin = np.array([[1], [0]])

        # Faraday mirror
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        nM_vib = 5
        V_out = zeros((num,2,1))*1j

        for mm in range(num):

            M_vib = self.create_Mvib(nM_vib, 1, 1)
            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, self.LF+self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            M_lf_f = self.lamming(0, 1, V_theta_lf, M_vib)
            M_lf_b = self.lamming(0, -1, V_theta_lf, M_vib)

            V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ Vin

        return V_out

    def cal_rotation(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None, Vin=None):
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
        #print("done")

        Vout_dic[num] = V_out

    def cal_rotation2(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None, Vin=None):
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

            # Faraday mirror
            ksi = ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming(iter_I, 1, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, V_theta)
            M_b = self.lamming(iter_I, -1, V_theta)
            M_lf_b = self.lamming(0, -1, V_theta_lf, M_vib)

            # M_lf_f = self.lamming(iter_I, 1, LF, V_theta_lf)
            # M_lf_b = self.lamming(iter_I, -1, LF, V_theta_lf)
            # M_f = self.lamming(iter_I, 1, L, V_theta)
            # M_b = self.lamming(iter_I, -1, L, V_theta)

            # if num == 0 and iter_I == V_plasmaCurrent[0]:
                #print("M_lf_f = ", M_lf_f[0, 1], M_lf_f[1, 0])
                #print("M_lf_b = ", M_lf_b[0, 1], M_lf_b[1, 0])
                #print("abs() = ", abs(M_lf_f[0, 1])-abs(M_lf_b[1, 0]))
                #print("Norm (MLead_f - MLead_b.T) = ", norm(M_lf_f - M_lf_b.T))
                # print("M_f = ", M_f[0, 1], M_f[1, 0])
                # print("M_b = ", M_b[0, 1], M_b[1, 0])
                #print("Norm (Msens_f - Msens_b) = ", norm(M_f - M_b))

            #V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ Vin
            V_out[mm] = M_lf_f @ Vin
            # V_out[mm] =  M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")

        Vout_dic[num] = V_out

    def calc_mp(self, num_processor, V_I, ang_FM, M_vib=None, fig=None, Vin=None):
        spl_I = np.array_split(V_I, num_processor)

        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        Ip = zeros(len(V_I))
        #print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            # proc = Process(target=self.cal_rotation,
            proc = Process(target=self.cal_rotation,
                           args=(spl_I[num], ang_FM, num, Vout_dic, M_vib, Vin))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        Vout = Vout_dic[0]
        for kk in range(num_processor - 1):
            Vout = np.vstack((Vout, Vout_dic[kk + 1]))

        E = Jones_vector('Output')
        E.from_matrix(M=Vout)
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
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi
            Ip[kk] = (V_ang[kk] - V_ang[0]) / (2 * self.V)

        return Ip, Vout

    def single_rotation1(self, V_Ip, Vin=None):
        # cal rotation angle using lamming method (variable dL)
        s_t_r = 2 * pi / self.SP

        # Sensing fiber vector with V_theta
        V_L = arange(0, self.L + self.dz, self.dz)
        V_theta = V_L * s_t_r

        # Faraday mirror
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        M_f = self.lamming(V_Ip, 1, V_theta)
        M_b = self.lamming(V_Ip, -1, V_theta)
        Vout = M_b @ M_FR @ M_f  @ Vin
        #Vout = M_f @ Vin

        return Vout

    def single_rotation2(self, V_Ip, Vin=None):
        # cal rotation angle using lamming method (dL=Fiber length)

        s_t_r = 2 * pi / self.SP  # spin twist ratio
        delta = 2 * pi / self.LB

        mm = 0
        n = 0
        m = 0
        n2 = 0
        m2 = 1

        tmp_f_Omega_z = 0
        tmp_f_Phi_z = 0
        tmp_b_Omega_z = 0
        tmp_b_Phi_z = 0

        I = V_Ip

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        H = I / self.L
        rho = self.V * H

        # --------Laming: orientation of the local slow axis ------------
        # --------Laming matrix on spun fiber --------------------------
        qu = 2 * (s_t_r - rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5

        R_z = 2 * arcsin(sin(gma * self.L) / ((1 + qu ** 2) ** 0.5))
        Omega_z = s_t_r * self.L + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.L)) + n2 * pi
        Phi_z = ((s_t_r * self.L) - Omega_z) / 2 + m2 * (pi / 2)

        # Forward
        n11 = cos(R_z / 2) + 1j * sin(R_z / 2) * cos(2 * Phi_z)
        n12 = 1j * sin(R_z / 2) * sin(2 * Phi_z)
        n21 = 1j * sin(R_z / 2) * sin(2 * Phi_z)
        n22 = cos(R_z / 2) - 1j * sin(R_z / 2) * cos(2 * Phi_z)
        M_R_f = np.array([[n11, n12], [n21, n22]])
        M_Omega_f = np.array([[cos(Omega_z), -sin(Omega_z)], [sin(Omega_z), cos(Omega_z)]])
        #print("Omega_z=", Omega_z)
        # Backward
        #rho = -self.V * H
        s_t_r = -s_t_r
        qu = 2 * (s_t_r - rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r - rho) ** 2)) ** 0.5

        R_z = 2 * arcsin(sin(gma * self.L) / ((1 + qu ** 2) ** 0.5))
        Omega_z = s_t_r * self.L + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.L)) + n2 * pi
        Phi_z = ((s_t_r * self.L) - Omega_z) / 2 + m2 * (pi / 2) + -s_t_r * self.L

        n11 = cos(R_z / 2) + 1j * sin(R_z / 2) * cos(2 * Phi_z)
        n12 = 1j * sin(R_z / 2) * sin(2 * Phi_z)
        n21 = 1j * sin(R_z / 2) * sin(2 * Phi_z)
        n22 = cos(R_z / 2) - 1j * sin(R_z / 2) * cos(2 * Phi_z)
        M_R_b = np.array([[n11, n12], [n21, n22]])
        M_Omega_b = np.array([[cos(Omega_z), -sin(Omega_z)], [sin(Omega_z), cos(Omega_z)]])
        #print("M_R_b=", M_R_b)
        #print("M_R_f=", M_R_f)
        # Faraday mirror
        ang_FM = 45
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot
        V_out = M_Omega_b @ M_R_b @ M_FR @ M_Omega_f @ M_R_f @ Vin
        #V_out = M_Omega_f @ Vin
        #print("M_OMega_f =", M_Omega_f, "V_I = ", V_Ip, "A")
        return V_out

    def single_rotation3(self, V_Ip, Vin=None):
        # cal rotation angle using stacking method (dL=variable)

        dq = 2 * pi / self.SP  # spin twist ratio
        delta = 2 * pi / self.LB

        LC = 1 * 2 * pi * 100000000000000
        rho_C = 0       # 2 * pi / LC
        rho_F = self.V

        delta_L = self.dz
        V_L = arange(delta_L, self.L + delta_L, delta_L)
        n_V_L = len(V_L)

        # ------------------------------ Variable forward--------------
        rho_1 = rho_C + rho_F * V_Ip
        delta_Beta_1 = 2 * (rho_1 ** 2 + (delta ** 2) / 4) ** 0.5

        alpha_1 = cos(delta_Beta_1 / 2 * delta_L)
        beta_1 = delta / delta_Beta_1 * sin(delta_Beta_1 / 2 * delta_L)
        gamma_1 = 2 * rho_1 / delta_Beta_1 * sin(delta_Beta_1 / 2 * delta_L)
        # ------------------------------ Variable backward--------------
        rho_2 = -rho_C + rho_F * V_Ip
        delta_Beta_2 = 2 * (rho_2 ** 2 + (delta ** 2) / 4) ** 0.5

        alpha_2 = cos(delta_Beta_2 / 2 * delta_L)
        beta_2 = delta / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)
        gamma_2 = 2 * rho_2 / delta_Beta_2 * sin(delta_Beta_2 / 2 * delta_L)

        # ------------------------------ Variable FRM--------------
        JF = np.mat([[0, 1], [-1, 0]])
        m = 0
        # print("mm = ", mm, " nn = ", nn)
        q = 0
        J = np.mat([[1, 0], [0, 1]])
        JT = np.mat([[1, 0], [0, 1]])
        for kk in range(len(V_L)):
            q = q + dq * delta_L

            J11 = alpha_1 + 1j * beta_1 * cos(2 * q)
            J12 = -gamma_1 + 1j * beta_1 * sin(2 * q)
            J21 = gamma_1 + 1j * beta_1 * sin(2 * q)
            J22 = alpha_1 - 1j * beta_1 * cos(2 * q)

            #J = np.vstack((J11, J12, J21, J22)).T.reshape(2, 2) @ J
            J = np.array([[J11, J12], [J21, J22]]) @ J

            J11 = alpha_2 + 1j * beta_2 * cos(2 * q)
            J12 = -gamma_2 + 1j * beta_2 * sin(2 * q)
            J21 = gamma_2 + 1j * beta_2 * sin(2 * q)
            J22 = alpha_2 - 1j * beta_2 * cos(2 * q)

            JT = JT @ np.array([[J11, J12], [J21, J22]])

        # print(q)
        V_out = JT @ JF @ J @ Vin
        #print("J = ", J, "in V_I of ", V_Ip, "A")
        return V_out

    def single_rotation4(self, V_Ip, Vin=None):

        delta = 2 * pi / self.LB

        mm = 0
        n = 0
        m = 0
        n2 = 0
        m2 = 0

        V_z = arange(0, self.L + self.dz, self.dz)

        H = V_Ip / self.L
        rho = self.V * H

        # --------Laming: orientation of the local slow axis ------------
        # --------Laming matrix on spun fiber --------------------------

        # define forward
        s_t_r = 2 * pi / self.SP  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zf = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))
        Omega_zf = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) - n * pi
        Phi_zf = ((s_t_r * self.dz) - Omega_zf) / 2 + m * (pi / 2)
        V_theta_1s = V_z * s_t_r

        # define backward
        s_t_r = -s_t_r  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zb = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))
        Omega_zb = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) - n2 * pi
        Phi_zb = ((s_t_r * self.dz) - Omega_zb) / 2 + m2 * (pi / 2)

        # Faraday mirror
        ang_FM = 45
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        # forward
        MF = np.array([[1, 0], [0, 1]])
        for kk in range(len(V_theta_1s) - 1):
            Phi_z2 = Phi_zf + V_theta_1s[kk]
            # print(Phi_z2)
            R_z2 = R_zf
            Omega_z2 = Omega_zf
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_f = np.array([[n11, n12], [n21, n22]])
            M_Omega_f = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MF = M_Omega_f @ M_R_f @ MF

        # Backward
        MB = np.array([[1, 0], [0, 1]])
        for kk in range(len(V_theta_1s) - 1):
            Phi_z2 = Phi_zb + V_theta_1s[-1 - kk]
            R_z2 = R_zb
            Omega_z2 = Omega_zb
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_b = np.array([[n11, n12], [n21, n22]])
            M_Omega_b = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MB = M_R_b @ M_Omega_b @ MB

        V_out = MB @ M_FR @ MF @ Vin
        # V_out[mm] = M_FR @ MF @ Vin

        return Ip, V_out


    def plot_error(self, filename):

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

        fig, ax = plt.subplots(figsize=(6, 3))
        lines = []
        for col_name in data:
            if col_name != 'Ip':
                if V_I[0] == 0:
                    lines += ax.plot(V_I[1:], abs((data[col_name][1:] - V_I[1:]) / V_I[1:]))
                #lines += ax.plot(V_I, abs((data[col_name]-V_I)/V_I), label=col_name)
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
        #plt.show()

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
    #print("init angle? = ",V_ang[0])
    return (Ip[1:] - V_I[1:]) / V_I[1:]

# Progress bar is not easy/
# Todo comparison between transmission and reflection
# Todo FM effect
# Todo Ip calculation method change --> azimuth angle --> arc length


if __name__ == '__main__':
    mode = 0
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
        LB = 0.009
        SP = 0.0048
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
            V_StL = np.array([])

            Vout = spunfiber.single_rotation2(V_I, Vin)  # cal rotation angle using lamming method (dL=Fiber length)
            V_L = S_L.from_Jones(E.from_matrix(Vout)).parameters.matrix()
            draw_stokes_points(fig1[0], S_L, kind='scatter', color_scatter='r')

            print(V_L.T)
            var_dL = SP*10**(-np.arange(1.5, 4, 0.5, dtype=float))

            for nn, var in enumerate(var_dL):
                spunfiber.dz = var

                #Vout = spunfiber.single_rotation1(V_I, Vin)         # cal rotation angle using lamming method (variable dL)
                Vout = spunfiber.single_rotation1(V_I, Vin)         # cal rotation angle using lamming method (variable dL)
                V_dL = np.append(V_dL, S_dL.from_Jones(E.from_matrix(Vout)).parameters.matrix())
                draw_stokes_points(fig1[0], S_dL, kind='scatter', color_scatter='b')

                Vout = spunfiber.single_rotation3(V_I, Vin)         # cal rotation angle using stacking method (dL=variable)
                V_St = np.append(V_St, S_S.from_Jones(E.from_matrix(Vout)).parameters.matrix())
                draw_stokes_points(fig1[0], S_S, kind='scatter', color_scatter='k')

                Vout = spunfiber.single_rotation3(V_I, Vin)  # cal rotation angle using stacking method (dL=variable)
                V_StL = np.append(V_St, S_STL.from_Jones(E.from_matrix(Vout)).parameters.matrix())
                draw_stokes_points(fig1[0], S_STL, kind='scatter', color_scatter='k')

        V_dL = V_dL.reshape(len(var_dL), 4)
        V_St = V_St.reshape(len(var_dL), 4)
        V_StL = V_StL.reshape(len(var_dL), 4)

        print(V_dL)
        print(V_St)
        V_L = np.ones(len(var_dL))*V_L
        figure, ax = plt.subplots(3, figsize=(5, 8))
        figure.subplots_adjust(left=0.179, bottom=0.15, right=0.94, hspace=0.226, top=0.938)

        ax[0].plot(var_dL, V_dL[...,1], 'r', label='Laming')
        ax[0].plot(var_dL, V_St[...,1], 'b', label='Stacking')
        ax[0].plot(var_dL, V_L[1,...], 'k--', label='Laming(w/o slicing)')
        ax[0].plot(var_dL, V_StL[...,1], 'm--', label='Laming(w/o slicing)')

        ax[0].set_xscale('log')
        ax[0].set_ylabel('S1')
        ax[0].legend(loc='upper left')
        #ax[0].set_title('S1')
        ax[0].set_xticklabels('')

        ax[1].plot(var_dL, V_dL[..., 2], 'r',  label='Laming')
        ax[1].plot(var_dL, V_St[..., 2], 'b', label='Stacking')
        ax[1].plot(var_dL, V_L[2,...], 'k--', label='Laming(w/o slicing)')
        ax[1].plot(var_dL, V_StL[..., 2], 'm--', label='Laming(w/o slicing)')
        ax[1].set_ylabel('S2')
        ax[1].set_xscale('log')
        ax[1].legend(loc='upper left')
        #ax[1].set_title('S2')
        ax[1].set_xticklabels('')

        ax[2].plot(var_dL, V_dL[..., 3], 'r', label='Laming')
        ax[2].plot(var_dL, V_St[..., 3], 'b', label='Stacking')
        ax[2].plot(var_dL, V_L[3,...], 'k--', label='Laming(w/o slicing)')
        ax[2].plot(var_dL, V_StL[..., 3], 'm--', label='Laming(w/o slicing)')
        ax[2].set_xscale('log')
        ax[2].set_xlabel('dL [m]')
        ax[2].set_ylabel('S3')
        ax[2].legend(loc='lower left')
        #ax[2].set_title('S3')
        ax[2].set_xticks(var_dL)
        str_xtick = ['SP/50', 'SP/100', 'SP/500', 'SP/1000', 'SP/5000']
        ax[2].set_xticklabels(str_xtick, minor=False, rotation=-45)

    if mode==2:
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
                #Vout = M_vib[..., nn].T @ M_Ip @ M_vib2[..., nn].T @ M_vib2[..., nn] @ M_Ip @ M_vib[..., nn] @ Vin
                #Vout  = M_vib[..., nn].T@M_Ip@M_vib2[..., nn].T @ M_FR @ M_vib2[..., nn] @M_Ip@M_vib[..., nn]@ Vin
                #Vout = M_vib[..., nn].T @ M_Ip @M_FR @ M_Ip @ M_vib[..., nn] @ Vin

                M_v = M_vib[...,4]@M_vib[...,3]@M_vib[...,2]@M_vib[...,1]@M_vib[...,0]

                Vout = M_v.T@ M_FR @M_v @ Vin
                Eo.from_matrix(Vout)
                tmp = np.hstack((tmp, Eo.parameters.ellipticity_angle()*180/pi))
                #tmp2 = np.hstack((tmp2, Eo.parameters.azimuth()*180/pi + ang_FM*2))
                So.from_Jones(Eo)
                draw_stokes_points(fig1[0], So, kind='scatter', color_line='r')

            SOPchange_mean = np.hstack((SOPchange_mean, tmp.mean()))
            SOPchange_std = np.hstack((SOPchange_std, tmp.std()))
            SOPchange_max = np.hstack((SOPchange_max, tmp.max()))

        fig, ax = plt.subplots()
        ax.plot(V_ang, SOPchange_mean)
        ax.plot(V_ang, SOPchange_std)
        ax.plot(V_ang, SOPchange_max)
    if mode==3:
        LB = 0.009
        SP = 0.005
        # dz = SP / 1000
        dz = 0.0001
        len_lf = 1  # lead fiber
        len_ls = 1  # sensing fiber
        L= len_ls
        spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
        V = spunfiber.V
        V_Ip = arange(0e6, 4e6 + 0.1e6, 0.2e6)
        Vin = np.array([[1], [0]])
        fig = None
        fig1, ax1 = spunfiber.init_plot_SOP()

        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        Ip = zeros(len(V_Ip))

        delta = 2 * pi / LB

        mm = 0
        n, m, n2, m2 = 0, 0, 0, 0

        H = V_Ip / L
        rho = V * H

        # define forward
        s_t_r = 2 * pi / SP  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zf = 2 * arcsin(sin(gma * L) / ((1 + qu ** 2) ** 0.5))
        Omega_zf, Phi_zf = zeros(len(V_Ip)), zeros(len(V_Ip))
        for nn in range(len(V_Ip)):
            tmp_Omega = s_t_r * L + \
                        arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * L)) - n * pi
            if nn > 1 and tmp_Omega - Omega_zf[nn - 1] > pi / 2:
                n += 1
            elif nn > 1 and tmp_Omega - Omega_zf[nn - 1] < -pi / 2:
                n -= 1
            Omega_zf[nn] = s_t_r * L + \
                           arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * L)) - n * pi

            tmp_Phi = ((s_t_r * L) - Omega_zf[nn]) / 2 + m * (pi / 2)
            if nn > 1 and tmp_Phi - Phi_zf[nn - 1] > pi / 2:
                m += 1
            elif nn > 1 and tmp_Phi - Phi_zf[nn - 1] < -pi / 2:
                m -= 1
            Phi_zf[nn] = ((s_t_r * L) - Omega_zf[nn]) / 2 + m * (pi / 2)

        V_theta_1s = L * s_t_r

        # define backward
        s_t_r = -s_t_r  # spin twist ratio
        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5

        R_zb = 2 * arcsin(sin(gma * L) / ((1 + qu ** 2) ** 0.5))
        Omega_zb, Phi_zb = zeros(len(V_Ip)), zeros(len(V_Ip))
        for nn in range(len(V_Ip)):
            tmp_Omega = s_t_r * L + \
                        arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * L)) - n2 * pi

            if nn > 1 and tmp_Omega - Omega_zb[nn - 1] > pi / 2:
                n2 += 1
            elif nn > 1 and tmp_Omega - Omega_zb[nn - 1] < -pi / 2:
                n2 -= 1
            Omega_zb[nn] = s_t_r * L + \
                           arctan((-qu[nn] / ((1 + qu[nn] ** 2) ** 0.5)) * tan(gma[nn] * L)) - n2 * pi

            tmp_Phi = ((s_t_r * L) - Omega_zb[nn]) / 2 + m2 * (pi / 2)
            if nn > 1 and tmp_Phi - Phi_zb[nn - 1] > pi / 2:
                m2 += 1
            elif nn > 1 and tmp_Phi - Phi_zb[nn - 1] < -pi / 2:
                m2 -= 1
            Phi_zb[nn] = ((s_t_r * L) - Omega_zb[nn]) / 2 + m2 * (pi / 2)

        # Faraday mirror
        ang_FM = 45
        ksi = ang_FM * pi / 180
        Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
        Jm = np.array([[1, 0], [0, 1]])
        M_FR = Rot @ Jm @ Rot

        for nn in range(len(V_Ip)):
            # forward
            Phi_z2 = Phi_zf[nn]
            R_z2 = R_zf[nn]
            Omega_z2 = Omega_zf[nn]
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_f = np.array([[n11, n12], [n21, n22]])
            M_Omega_f = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MF = M_Omega_f @ M_R_f
            #
            # Backward
            Phi_z2 = Phi_zb[nn] + V_theta_1s
            R_z2 = R_zb[nn]
            Omega_z2 = Omega_zb[nn]
            n11 = cos(R_z2 / 2) + 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            n12 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n21 = 1j * sin(R_z2 / 2) * sin(2 * Phi_z2)
            n22 = cos(R_z2 / 2) - 1j * sin(R_z2 / 2) * cos(2 * Phi_z2)
            M_R_b = np.array([[n11, n12], [n21, n22]])
            M_Omega_b = np.array([[cos(Omega_z2), -sin(Omega_z2)], [sin(Omega_z2), cos(Omega_z2)]])
            MB = M_Omega_b @ M_R_b

            V_out[mm] = MB  @ Vin
            #V_out[mm] = M_FR @ MF @ Vin
            mm += 1

        E = Jones_vector('Output')
        E.from_matrix(M=V_out)
        V_ang = zeros(len(V_Ip))

        # SOP evolution in Lead fiber (Forward)
        S = create_Stokes('Output_S')
        S.from_Jones(E)

        if fig is not None:
            draw_stokes_points(fig[0], S, kind='line', color_line='b')
        else:
            fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line',
                                      color_line='b')
        m = 0
        for kk in range(len(V_Ip)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = (E[kk].parameters.azimuth() + m * pi)
            Ip[kk] = (V_ang[kk] - V_ang[0]) / (V)
            # print(V_ang[kk], Ip[kk])

plt.show()
