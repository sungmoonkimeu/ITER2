import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse

import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt

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

        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5
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
        N = np.array([[n11, n12], [n21, n22]])
        N_integral = self._eigen_expm(N.T)
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

    def cal_rotation(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None, M_vib2=None, Vin=None):
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

            # Lead fiber2 vector with V_theta_lf2
            V_L_lf = arange(0, self.LF + self.dz, self.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf * s_t_r

            # Faraday mirror
            ksi = ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming(0, 1, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, V_theta)
            M_lf2_f = self.lamming(0, 1, V_theta_lf2, M_vib2)
            M_lf2_b = self.lamming(0, -1, V_theta_lf2, M_vib2)
            M_b = self.lamming(iter_I, -1, V_theta)
            M_lf_b = self.lamming(0, -1, V_theta_lf, M_vib)

            # M_lf_f = self.lamming(iter_I, 1, LF, V_theta_lf)
            # M_lf_b = self.lamming(iter_I, -1, LF, V_theta_lf)
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

            V_out[mm] = M_lf_b @ M_b @ M_lf2_b @ M_FR @ M_lf2_f @ M_f @ M_lf_f @ Vin
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ Vin
            # V_out[mm] =  M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")

        Vout_dic[num] = V_out

    def cal_rotation_trans(self, V_Ip, num, Vout_dic, M_vib=None, Vin=None):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        s_t_r = 2 * pi / self.SP
        V_in = Vin
        mm = 0
        for iter_I in V_plasmaCurrent:

            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, self.LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, self.L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # lead fiber vector with V_theta_lf2
            V_L_lf2 = arange(0, self.LF + self.dz, self.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf2 * s_t_r

            # Faraday mirror

            M_lf_f = self.lamming(0, 1, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, V_theta)
            M_lf_b = self.lamming(0, 1, V_theta_lf2, M_vib)


            #M_f = self.lamming1(iter_I, 1, L, V_theta)

            V_out[mm] = M_lf_b @ M_f @ M_lf_f @ V_in
            #V_out[mm] = M_lf_b @ M_f @ M_lf_f @ V_in
            #V_out[mm] = M_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")
        Vout_dic[num] = V_out

    def calc_mp(self, num_processor, V_I, ang_FM, M_vib=None, M_vib2=None, fig=None, Vin=None):
        spl_I = np.array_split(V_I, num_processor)

        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        Ip = zeros(len(V_I))
        #print("Vin_calc_mp", Vin)
        for num in range(num_processor):
            proc = Process(target=self.cal_rotation,
                           args=(spl_I[num], ang_FM, num, Vout_dic, M_vib, M_vib2, Vin))
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

    def calc_mp_trans(self, num_processor, V_I, M_vib=None, fig=None, Vin=None):
        spl_I = np.array_split(V_I, num_processor)
        '''
        f = open('mp1.txt', 'w')
        savetxt(f, V_I, newline="\t")
        f.write("\n")
        f.close()
        '''
        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        # f = open('mp1.txt', 'a')

        Ip = zeros(len(V_I))

        for num in range(num_processor):
            proc = Process(target=self.cal_rotation_trans,
                           args=(spl_I[num], num, Vout_dic, M_vib, Vin))
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
            fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')

        m = 0

        for kk in range(len(V_I)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi
            Ip[kk] = -(V_ang[kk]-V_ang[0]) / self.V

        return Ip, Vout

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

    def plot_errorbar(self, filename, fig=None, ax=None, label=None):
        is_reuse = bool(fig)
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

        if fig is None:
            fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
            fig.set_dpi(91.79)  # DPI of My office monitor

        if is_reuse is False:
            if V_I[0] == 0:
                ax.plot(V_I[1:], relErrorlimit[1:], 'r--', label='ITER specification')
                ax.plot(V_I[1:], -relErrorlimit[1:], 'r--')
            else:
                ax.plot(V_I, relErrorlimit, 'r--', label='ITER specification')
                ax.plot(V_I, -relErrorlimit, 'r--')

        color = 'k'
        if V_I[0] == 0:
            if is_reuse is True:
                color = 'b'
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1).drop(0, axis=0)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1).drop(0, axis=0)
            ax.plot(V_I[1:], df_mean[:], color, label=label)
            ax.errorbar(V_I[1::3], df_mean[::3], yerr=df_std[::3], ls='None', c='black', ecolor=color, capsize=4)
        else:
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1)
            ax.plot(V_I, df_mean, color, label=label)
            ax.errorbar(V_I[::2], df_mean[::2], yerr=df_std[::2], ls='None', c='black', ecolor=color, capsize=4)

        lines = []

        ax.legend(loc="lower right")

        #plt.rc('text', usetex=True)
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
        fig.subplots_adjust(left = 0.195, hspace=0.4, right=0.95, top=0.93, bottom=0.2)
        # fig.set_size_inches(6,4)
        # plt.rc('text', usetex=False)

        return fig, ax, lines

    def plot_errorbar_inset(self, filename, ax=None):
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

        if V_I[0] == 0:
            ax.plot(V_I[1:], relErrorlimit[1:], 'r', label='ITER specification')
            ax.plot(V_I[1:], -relErrorlimit[1:], 'r')
        else:
            ax.plot(V_I, relErrorlimit, 'r', label='ITER specification')
            ax.plot(V_I, -relErrorlimit, 'r')

        if V_I[0] == 0:
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1).drop(0,
                                                                                                                  axis=0)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1).drop(0,
                                                                                                                axis=0)
            ax.plot(V_I[1:], df_mean[:], 'k')
            ax.errorbar(V_I[1::3], df_mean[::3], yerr=df_std[::3], ls='None', c='black', ecolor='k', capsize=4)
        else:
            df_mean = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).mean(axis=1)
            df_std = data.drop(['Ip'], axis=1).sub(data['Ip'], axis=0).div(data['Ip'], axis=0).std(axis=1)
            ax.plot(V_I, df_mean, 'r')
            ax.errorbar(V_I[::2], df_mean[::2], yerr=df_std[::2], ls='None', c='black', ecolor='r', capsize=4)

        return ax
    #plt.show()


    def plot_errorbar_byStokes(self, filename, fig=None, ax=None, label=None, V_custom=None, cal_init=None):
        is_reuse = bool(fig)
        # print(is_reuse)
        data = pd.read_csv(filename)
        V_I = np.array(data['Ip'])
        E = Jones_vector('Output')
        S = create_Stokes('Output_S')
        V_ang = zeros(len(V_I))

        Ip = zeros([int((data.shape[1] - 1) / 2), len(V_I)])

        if fig is None:
            fig, ax = plt.subplots(figsize=(12/2.54, 10/2.54))
            fig.set_dpi(91.79)  # DPI of My office monitor
            fig.subplots_adjust(hspace=0.4, left=0.195, right=0.95, top=0.93, bottom=0.2)

        # Requirement specification for ITER
        absErrorlimit = zeros(len(V_I))
        relErrorlimit = zeros(len(V_I))

        # Calculation ITER specification
        for nn in range(len(V_I)):
            if V_I[nn] < 1e6:
                absErrorlimit[nn] = 10e3
            else:
                absErrorlimit[nn] = V_I[nn] * 0.01
            if V_I[nn] == 0:
                pass
            else:
                relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

        if is_reuse is False:
            if V_I[0] == 0:
                ax.plot(V_I[1:], relErrorlimit[1:], 'r--', label='ITER specification')
                ax.plot(V_I[1:], -relErrorlimit[1:], 'r--')
            else:
                ax.plot(V_I, relErrorlimit, 'r--', label='ITER specification')
                ax.plot(V_I, -relErrorlimit, 'r--')

        for nn in range(int((data.shape[1] - 1) / 2)):
            str_Ex = str(nn) + ' Ex'
            str_Ey = str(nn) + ' Ey'
            Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                            [complex(y) for y in data[str_Ey].to_numpy()]])

            E.from_matrix(M=Vout)
            S.from_Jones(E)
            # Azimuth angle calculation

            m = 0
            for kk in range(len(V_I)):
                if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                    m = m + 1
                elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                    m = m - 1
                V_ang[kk] = E[kk].parameters.azimuth() + m * pi

                if cal_init is None:
                    c = pi/2
                else:
                    c = V_ang[0]
                if V_custom is not None:
                    Ip[nn][kk] = (V_ang[kk] - c) / (2 * spunfiber.V) * V_custom
                    #print(cal_init)
                else:
                    Ip[nn][kk] = (V_ang[kk] - c) / (2 * spunfiber.V)

        color = 'k'
        if V_I[0] == 0:
            if is_reuse is True:
                color = 'b'
            df_mean = ((Ip[...,1:]-V_I[1:])/V_I[1:]).mean(axis=0)
            df_std = ((Ip[...,1:]-V_I[1:])/V_I[1:]).std(axis=0)

            ax.plot(V_I[1:], df_mean, color, label=label)
            ax.errorbar(V_I[2::3], df_mean[1::3], yerr=df_std[1::3], ls='None', c='black', ecolor=color, capsize=3)
        else:
            df_mean = Ip.mean(axis=1)
            df_std = Ip.std(axis=1)
            ax.plot(V_I, df_mean, color, label=label)
            ax.errorbar(V_I[::2], df_mean[::2], yerr=df_std[::2], ls='None', c='black', ecolor=color, capsize=4)

        ax.legend(loc="lower right")
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
        plt.rc('text', usetex=False)
        plt.grid(True)
        return fig, ax

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

    def draw_PS(self):
        '''
           plot Poincare Sphere, ver. 20/03/2020
           return:
           ax, fig
           '''
        fig = plt.figure(figsize=(6, 6))
        #    plt.figure(constrained_layout=True)
        ax = Axes3D(fig)
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
                        # edgecolor='k',
                        edgecolor=(3 / 256, 3 / 256, 3 / 256),
                        linestyle=(0, (5, 5)),
                        rstride=3, cstride=3,
                        linewidth=.5, alpha=0.8, shade=
                        0)

        # main circles
        # ax.plot(np.sin(u), np.cos(u), np.zeros_like(u), 'g-.', linewidth=0.75)  # equator
        #    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), 'b-', linewidth=0.5)
        #    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), 'b-', linewidth=0.5)

        # axes and captions
        amp = 1.5 * sprad
        ax.plot([-amp, amp], [0, 0], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=1)
        ax.plot([0, 0], [-amp, amp], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=1)
        ax.plot([0, 0], [0, 0], [-amp*1.2/1.5, amp*1.2/1.5], 'k-.', lw=1, alpha=0.5, zorder=1)

        distance = 1.5 * sprad
        ax.text(-distance*1.3, 0, 0, '$S_1$', fontsize=18)
        ax.text(0, distance, 0, '$S_2$', fontsize=18)
        ax.text(0, 0, distance/1.5*1.2, '$S_3$', fontsize=18)

        # points
        px = [1, -1, 0, 0, 0, 0]
        py = [0, 0, 1, -1, 0, 0]
        pz = [0, 0, 0, 0, 1, -1]

        ax.plot(px, py, pz,
                color='black', marker='o', markersize=4, alpha=1, linewidth=0)
        #

        max_size = 1.05 * sprad
        ax.set_xlim(-max_size, max_size)
        ax.set_ylim(-max_size, max_size)
        ax.set_zlim(-max_size, max_size)

        # ax.view_init(elev=-21, azim=-54)
        ax.view_init(elev=18, azim=-148)
        #    ax.view_init(elev=0/np.pi, azim=0/np.pi)

        #    ax.set_title(label = shot, loc='left', pad=10)
        #    ax.set_title(label="  " + shot, loc='left', pad=-10, fontsize=8)

        #    ax.legend()

        ax.set_box_aspect([1, 1, 1])

        return ax, fig

# Progress bar is not easy/
# Todo comparison between transmission and reflection
# Todo FM effect
# Todo Ip calculation method change --> azimuth angle --> arc length


if __name__ == '__main__':
    LB = 0.09
    SP = 0.048
    # dz = SP / 1000
    dz = 0.00005
    len_lf = 1  # lead fiber
    len_ls = 1   # sensing fiber
    spunfiber = SPUNFIBER(LB, SP, dz, len_lf, len_ls)
    mode =5

    # 44FM_Errdeg1x5_0 : length of leadfiber 10 m
    # 44FM_Errdeg1x5_1 : length of leadfiber 10->20 m
    if mode == 0:
        num_iter = 50
        strfile1 = 'Hibi_IdealFM_errdeg1x5.csv'
        strfile2 = 'A.csv'
        num_processor = 16
        V_I = arange(0e6, 18e6 + 0.1e6, 0.1e6)
        outdict = {'Ip': V_I}
        outdict2 = {'Ip': V_I}
        nM_vib = 5
        start = pd.Timestamp.now()
        ang_FM = 45
        Vin = np.array([[1], [0]])

        fig1, ax1 = spunfiber.init_plot_SOP()
        for nn in range(num_iter):
            M_vib = spunfiber.create_Mvib(nM_vib, 1, 1)
            M_vib2 = spunfiber.create_Mvib(nM_vib, 1, 1)
            Ip, Vout = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, M_vib2, fig1, Vin)
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
        fig, ax, lines = spunfiber.plot_errorbar(strfile1)

        '''
        fig3, ax3 = spunfiber.init_plot_SOP()
        for nn in range(num_iter):
            M_vib = spunfiber.create_Mvib(nM_vib, 1, 1)
            Ip, Vout = spunfiber.calc_mp_trans(num_processor, V_I, M_vib, fig3, Vin)

            outdict[str(nn)] = Ip
            outdict2[str(nn) + ' Ex'] = Vout[:, 0, 0]
            outdict2[str(nn) + ' Ey'] = Vout[:, 1, 0]
            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        df = pd.DataFrame(outdict)
        df.to_csv(strfile2, index=False)
        fig4, ax4, lines = spunfiber.plot_error(strfile2)
        df2 = pd.DataFrame(outdict2)
        df2.to_csv(strfile2 + "_S", index=False)
        '''
    elif mode == 1:

        strfile0 = 'IdealFM_Errdeg1x5_2.csv'
        #strfile1 = '44FM_errdeg1x5_0_2.csv'
        fig, ax, lines = spunfiber.plot_error(strfile0)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile0, label='Lo-bi spun fiber')

        strfile1 = 'IdealFM_Hibi_Errdeg1x5_0.csv'
        fig, ax, lines = spunfiber.plot_error(strfile1)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile1, fig2, ax2, 'hi-bi spun fiber')
        ax2.set(xlim=(0, 18e6), ylim=(-0.08, 0.08))
        #ax2.legend().set_visible(False)

        ax2ins = inset_axes(ax2, width="45%", height=0.8, loc=1)

        x1,x2,y1,y2 = 0, 8e6, -0.005, 0.005
        ax2ins.set_xlim(x1,x2)
        ax2ins.set_ylim(y1,y2)
        ax2ins = spunfiber.plot_errorbar_inset(strfile0, ax2ins)
        '''
        strfile2 = '42FM_errdeg1x5_0_2.csv'
        fig, ax, lines = spunfiber.plot_error(strfile2)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile2, fig2, ax2)

        strfile3 = '40FM_errdeg1x5_0_2.csv'
        fig, ax, lines = spunfiber.plot_error(strfile3)
        fig2, ax2, lines2 = spunfiber.plot_errorbar(strfile3, fig2, ax2)


        strfile10 = 'Hibi_44FM_errdeg1x5.csv'
        fig, ax, lines = spunfiber.plot_error(strfile10)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile10)

        strfile11 = 'Hibi_42FM_errdeg1x5.csv'
        fig, ax, lines = spunfiber.plot_error(strfile11)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile11, fig2, ax2)
        '''
    elif mode == 2:
        strfile0 = 'IdealFM_errdeg1x5_2.csv'
        # strfile1 = '44FM_errdeg1x5_0_2.csv'
        fig, ax, lines = spunfiber.plot_error(strfile0)
        fig2, ax2, lines = spunfiber.plot_errorbar(strfile0, label='Lo-bi spun fiber')

        strfile1 = 'Hibi_IdealFM_errdeg1x5.csv'
        #fig, ax, lines = spunfiber.plot_error(strfile1)
        strfile1 = 'Hibi_IdealFM_errdeg1x5.csv_S'
        #fig2, ax2, lines = spunfiber.plot_errorbar(strfile1, fig2, ax2, 'Hi-bi spun fiber')
        fig2, ax2 = spunfiber.plot_errorbar_byStokes(strfile1, fig2, ax2, label='Hi-bi spun fiber', V_custom=1.0365)

        ax2.set(xlim=(0, 18e6), ylim=(-0.08, 0.08))
        # ax2.legend().set_visible(False)

        ax2ins = inset_axes(ax2, width="45%", height=0.8, loc=1)

        x1, x2, y1, y2 = 0, 8e6, -0.005, 0.005
        ax2ins.set_xlim(x1, x2)
        ax2ins.set_ylim(y1, y2)
        ax2ins = spunfiber.plot_errorbar_inset(strfile0, ax2ins)
    elif mode == 3:
        strfile0 = 'Hibi_44FM_errdeg1x5.csv'
        strfile1 = 'Hibi_44FM_errdeg1x5.csv_S'

        #fig, ax, lines = spunfiber.plot_errorbar(strfile0, label='with considering FM error')
        fig, ax = spunfiber.plot_errorbar_byStokes(strfile1, label='w/o considering FM error', V_custom=1.038)

        strfile1 = 'Hibi_44FM_errdeg1x5.csv_S'
        fig, ax = spunfiber.plot_errorbar_byStokes(strfile1, fig, ax, label='wtih considering FM error', V_custom=1.038, cal_init=1)
        #fig, ax = spunfiber.plot_errorbar_byStokes(strfile1, fig, ax, label='w/o considering FM error2', V_custom=1.037, cal_init=1)
        ax.set(xlim=(0, 18e6), ylim=(-0.08, 0.08))
        ax.legend(loc='upper right')


        strfile0 = '44FM_errdeg1x5_0.csv'
        fig, ax, lines = spunfiber.plot_errorbar(strfile0, label='with considering FM error')
        ax2ins = inset_axes(ax, width="45%", height=0.8, loc=1)
        x1, x2, y1, y2 = 0, 8e6, -0.005, 0.005
        ax2ins.set_xlim(x1, x2)
        ax2ins.set_ylim(y1, y2)
        ax2ins = spunfiber.plot_errorbar_inset(strfile0, ax2ins)

        strfile1 = '44FM_errdeg1x5_0.csv_S'
        fig, ax = spunfiber.plot_errorbar_byStokes(strfile1, fig, ax, label='w/o considering FM error')
        ax.set(xlim=(0, 18e6), ylim=(-0.08, 0.08))
    else:
        strfile1 = 'IdealFM_Errdeg1x5_2.csv_S'
        #strfile1 = 'Hibi_IdealFM_errdeg1x5.csv_S'
        data = pd.read_csv(strfile1)
        V_I = data['Ip']
        E = Jones_vector('Output')
        V_ang = zeros(len(V_I))
        ax, fig = spunfiber.draw_PS()

        pnt = [0, 3, 7, 11, 15, 19]
        for nn in range(50):

            str_Ex = str(nn) + ' Ex'
            str_Ey = str(nn) + ' Ey'
            Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                             [complex(y) for y in data[str_Ey].to_numpy()]])

            E.from_matrix(M=Vout)

            # SOP evolution in Lead fiber (Forward)
            S = create_Stokes('Output_S')
            S.from_Jones(E)
            # Azimuth angle calcuation
            V_ang1 = zeros(len(V_I))

            S1 = S[pnt].parameters.components()[1]
            S2 = S[pnt].parameters.components()[2]
            S3 = S[pnt].parameters.components()[3]
            line1 = ax.plot(S1, S2, S3, marker='o', markersize=5, alpha=1.0, linewidth=0, zorder=3, color='b', label='Ideal FM')

        strfile1 = '44FM_Errdeg1x5_0.csv_S'
        #strfile1 = 'Hibi_44FM_errdeg1x5.csv_S'
        data = pd.read_csv(strfile1)
        V_I = data['Ip']
        E = Jones_vector('Output')
        V_ang = zeros(len(V_I))

        # for nn in range(int((data.shape[1] - 1) / 2)):

        pnt = [0, 3, 7, 11, 15, 19]
        #plt.rc('text', usetex=True)

        for nn in range(50):
            str_Ex = str(nn) + ' Ex'
            str_Ey = str(nn) + ' Ey'
            Vout = np.array([[complex(x) for x in data[str_Ex].to_numpy()],
                             [complex(y) for y in data[str_Ey].to_numpy()]])

            E.from_matrix(M=Vout)

            # SOP evolution in Lead fiber (Forward)
            S = create_Stokes('Output_S')
            S.from_Jones(E)
            # Azimuth angle calcuation
            V_ang1 = zeros(len(V_I))

            S1 = S[pnt].parameters.components()[1]
            S2 = S[pnt].parameters.components()[2]
            S3 = S[pnt].parameters.components()[3]
            line2 = ax.plot(S1, S2, S3, marker='x', markersize=5, alpha=1.0, linewidth=0, zorder=3, color='r',
                            label=r'Nonideal FM ($\theta_{err}=1^{\circ}$)')

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        #plt.rc('text', usetex=False)
plt.show()
