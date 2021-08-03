import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import matplotlib.pyplot as plt
# import tdqm



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
    def __init__(self, beat_length, spin_pitch, delta_l):
        self.LB = beat_length
        self.SP = spin_pitch
        self.dz = delta_l

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

    def lamming1(self, Ip, DIR, L, V_theta):
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
        H = Ip / L
        V = 0.54 * 4 * pi * 1e-7
        rho = V * H

        # ----------------------Laming parameters--------------------------------- #
        n = 0
        m = 0
        # --------Laming: orientation of the local slow axis ------------

        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5
        omega = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) + n * pi

        R_z = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))

        M = np.array([[1, 0], [0, 1]])

        kk = 0  # for counting M_err
        for nn in range(len(V_theta) - 1):
            if DIR == 1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
            elif DIR == -1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[-1 - nn]

            n11 = R_z / 2 * 1j * cos(2 * phi)
            n12 = R_z / 2 * 1j * sin(2 * phi) - omega
            n21 = R_z / 2 * 1j * sin(2 * phi) + omega
            n22 = R_z / 2 * -1j * cos(2 * phi)
            N = np.array([[n11, n12], [n21, n22]])
            N_integral = self._eigen_expm(N)
            M = N_integral @ M

        return M

    def lamming_vib(self, Ip, DIR, L, V_theta, M_err):
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
        H = Ip / L
        V = 0.54 * 4 * pi * 1e-7
        rho = V * H

        # ----------------------Laming parameters--------------------------------- #
        n = 0
        m = 0
        # --------Laming: orientation of the local slow axis ------------

        qu = 2 * (s_t_r + rho) / delta
        gma = 0.5 * (delta ** 2 + 4 * ((s_t_r + rho) ** 2)) ** 0.5
        omega = s_t_r * self.dz + arctan((-qu / ((1 + qu ** 2) ** 0.5)) * tan(gma * self.dz)) + n * pi

        R_z = 2 * arcsin(sin(gma * self.dz) / ((1 + qu ** 2) ** 0.5))

        M = np.array([[1, 0], [0, 1]])

        kk = 0  # for counting M_err
        for nn in range(len(V_theta) - 1):
            if DIR == 1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
            elif DIR == -1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[-1 - nn]

            n11 = R_z / 2 * 1j * cos(2 * phi)
            n12 = R_z / 2 * 1j * sin(2 * phi) - omega
            n21 = R_z / 2 * 1j * sin(2 * phi) + omega
            n22 = R_z / 2 * -1j * cos(2 * phi)
            N = np.array([[n11, n12], [n21, n22]])
            N_integral = self._eigen_expm(N)
            M = N_integral @ M

            # If vibration matrix (Merr) is presence, it will be inserted automatically.
            # For example, if Merr.shape[2] == 2, two Merr will be inserted
            # in the 1/3, 2/3 position of L

            nVerr = M_err.shape[2]
            nSet = int(len(V_theta) / (nVerr + 1))
            if nVerr > 0:
                if DIR == 1 and (nn + 1) % nSet == 0:
                    if kk != nVerr:
                        M = M_err[..., kk] @ M
                        kk = kk + 1
                elif DIR == -1 and int((len(V_theta) - nn) % nSet) == 0:
                    if kk != nVerr:
                        M = M_err[..., -1 - kk].T @ M
                        kk = kk + 1
        return M

    def first_calc(self):

        V_plasmaCurrent = arange(1e5, 1e6, 1e5)
        # V_plasmaCurrent = np.append(V_plasmaCurrent, arange(1e6, 18e6, 5e5))
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        V = 0.54 * 4 * pi * 1e-7
        s_t_r = 2 * pi / self.SP
        LF = 1
        L = 1

        V_in = np.array([[1], [0]])
        mm = 0
        for iter_I in V_plasmaCurrent:
            #  Preparing M_err
            n_M_err = 1
            theta = (np.random.rand(n_M_err) - 0.5) * 2 * pi / 2  # random axis of LB
            phi = (np.random.rand(n_M_err) - 0.5) * 2 * 0.8 * pi / 180  # ellipticity angle change from experiment
            theta_e = (np.random.rand(n_M_err) - 0.5) * 2 * 0.8 * pi / 180  # azimuth angle change from experiment

            M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,n_M_err)
            M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,n_M_err)
            M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,n_M_err)

            # Create (2,2,n_M_err) Birefringence matrix
            IB = np.zeros((2, 2, n_M_err))
            np.einsum('iij->ij', IB)[:] = 1
            Bexp = np.exp(1j * np.vstack((phi, -phi)))
            M_phi = einsum('ijk, ...ik -> ijk', IB, Bexp)

            # Random birefringence(circular + linear), random optic axis matrix calculation
            M_err = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)

            # Empty matrix => Error matrix (Merr) with no effect
            M_empty = np.array([]).reshape(2, 2, 0)

            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Faraday mirror
            ksi = 44 * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming_vib(0, 1, LF, V_theta_lf, M_err)
            M_f = self.lamming_vib(iter_I, 1, L, V_theta, M_empty)
            M_b = self.lamming_vib(iter_I, -1, L, V_theta, M_empty)
            M_lf_b = self.lamming_vib(0, -1, LF, V_theta_lf, M_err)

            V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        print("done")

        # -------------- Using py_pol module -----------------------------------
        E = Jones_vector('Output_J')
        S = create_Stokes('Output_S')

        E.linear_light(azimuth=0 * abs(V_out))
        S.linear_light(azimuth=0 * abs(V_out))

        # SOP evolution in Lead fiber (Forward)
        E.from_matrix(V_out)
        S.from_Jones(E)
        fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='scatter',
                                  color_line='b')

        ell_V_out = E.parameters.ellipticity_angle() * 180 / pi
        print("ell=", ell_V_out.max() - ell_V_out.min())
        azi_V_out = E.parameters.azimuth() * 180 / pi

        for nn, v in enumerate(azi_V_out):
            if v > 90:
                azi_V_out[nn] = azi_V_out[nn] - 180

        print("azi=", azi_V_out.max() - azi_V_out.min())

        abs_error = zeros([len(V_out)])
        rel_error = zeros([len(V_out)])
        Ip = zeros(len(V_out))
        V_ang = zeros(len(V_out))

        m = 0
        for nn in range(len(V_out)):
            if nn > 2 and E[nn].parameters.azimuth() + m * pi - V_ang[nn - 1] < -pi * 0.5:
                m = m + 1
            elif nn > 2 and E[nn].parameters.azimuth() + m * pi - V_ang[nn - 1] > pi * 0.5:
                m = m - 1
            V_ang[nn] = E[nn].parameters.azimuth() + m * pi
            # Ip[nn] = -(V_ang[nn] - pi / 2) / (2 * V)
            Ip[nn] = -(V_ang[nn] - 2 * ksi) / (2 * V)  # Calibration the FM angle error
            abs_error[nn] = abs(Ip[nn] - V_plasmaCurrent[nn])
            if V_plasmaCurrent[nn] == 0:
                rel_error[nn] = 100
            else:
                rel_error[nn] = abs_error[nn] / V_plasmaCurrent[nn]

        # Requirement specificaion for ITER
        absErrorlimit = zeros(len(V_out))
        relErrorlimit = zeros(len(V_out))

        # Calcuation ITER specification
        for nn in range(len(V_plasmaCurrent)):
            if V_plasmaCurrent[nn] < 1e6:
                absErrorlimit[nn] = 10e3
            else:
                absErrorlimit[nn] = V_plasmaCurrent[nn] * 0.01

            if V_plasmaCurrent[nn] == 0:
                relErrorlimit[nn] = 100
            else:
                relErrorlimit[nn] = absErrorlimit[nn] / V_plasmaCurrent[nn]

        # Ploting graph
        fig, ax = plt.subplots(figsize=(6, 3))

        ax.plot(V_plasmaCurrent, rel_error, lw='1')
        ax.plot(V_plasmaCurrent, relErrorlimit, 'r', label='ITER specification', lw='1')
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

        plt.show()

    def create_Merr(self, num_Merr, max_phi, max_theta_e):
        theta = (np.random.rand(num_Merr) - 0.5) * 2 * pi / 2  # random axis of LB
        phi = (np.random.rand(num_Merr) - 0.5) * 2 * max_phi * pi / 180  # ellipticity angle change from experiment
        theta_e = (np.random.rand(num_Merr) - 0.5) * 2 * max_theta_e * pi / 180  # azimuth angle change from experiment

        M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,n_M_err)
        M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,n_M_err)
        M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,n_M_err)
        # print(theta)
        # Create (2,2,n_M_err) Birefringence matrix
        IB = np.zeros((2, 2, num_Merr))
        np.einsum('iij->ij', IB)[:] = 1
        Bexp = np.exp(1j * np.vstack((phi, -phi)))
        M_phi = einsum('ijk, ...ik -> ijk', IB, Bexp)

        # Random birefringence(circular + linear), random optic axis matrix calculation
        M_err = einsum('ij..., jk..., kl...,lm...-> im...', M_rot, M_theta, M_phi, M_theta_T)

        return M_err

    def cal_rotation(self, V_Ip, num, Vout_dic):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        V = 0.54 * 4 * pi * 1e-7
        s_t_r = 2 * pi / self.SP
        LF = 1
        L = 1
        V_in = np.array([[1], [0]])

        mm = 0
        for iter_I in V_plasmaCurrent:
            # Empty matrix => Error matrix (Merr) with no effect
            M_empty = np.array([]).reshape(2, 2, 0)

            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Faraday mirror
            ksi = 45 * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming1(0, 1, LF, V_theta_lf)
            M_f = self.lamming1(iter_I, 1, L, V_theta)
            M_b = self.lamming1(iter_I, -1, L, V_theta)
            M_lf_b = self.lamming1(0, -1, LF, V_theta_lf)

            V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in

            '''
            # QWP
            M_qwp = Rot @ np.array([[1, 0], [0, 1j]]) @ Rot.T
            M_qwp_b = M_qwp.T

            M_f = self.lamming1(iter_I, 1, LF, V_theta_lf)
            M_b = self.lamming1(iter_I, -1, LF, V_theta_lf)

            V_out[mm] = M_qwp_b @ M_b @ M_FR @ M_f @ M_qwp @ V_in
            '''
            mm = mm + 1
        #print("done")
        Vout_dic[num] = V_out

    def calc_mp1(self, num_processor, V_I):
        V = 0.54
        spl_I = np.array_split(V_I, num_processor)

        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        # f = open('mp1.txt', 'a')

        Ip = zeros(len(V_I))

        for num in range(num_processor):
            proc = Process(target=self.cal_rotation,
                           args=(spl_I[num], num, Vout_dic))
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

        m = 0
        for kk in range(len(V_I)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi
            Ip[kk] = -(V_ang[kk] - 2*44*pi/180) / (2 * V * 4 * pi * 1e-7)

        return Ip

    def cal_rotation_Merr(self, V_Ip, num, Vout_dic, M_err):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        V = 0.54 * 4 * pi * 1e-7
        s_t_r = 2 * pi / self.SP
        LF = 1
        L = 1
        V_in = np.array([[1], [0]])

        mm = 0
        for iter_I in V_plasmaCurrent:
            # Empty matrix => Error matrix (Merr) with no effect
            M_empty = np.array([]).reshape(2, 2, 0)

            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Faraday mirror
            ksi = 44 * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming_vib(0, 1, LF, V_theta_lf, M_err)
            M_f = self.lamming_vib(iter_I, 1, L, V_theta, M_empty)
            M_b = self.lamming_vib(iter_I, -1, L, V_theta, M_empty)
            M_lf_b = self.lamming_vib(0, -1, LF, V_theta_lf, M_err)

            V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")
        Vout_dic[num] = V_out

    def calc_mp_Merr(self, num_processor, V_I, M_err):
        V = 0.54
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
            proc = Process(target=self.cal_rotation_Merr,
                           args=(spl_I[num], num, Vout_dic, M_err))
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

        m = 0
        for kk in range(len(V_I)):
            if kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] < -pi * 0.8:
                m = m + 1
            elif kk > 2 and E[kk].parameters.azimuth() + m * pi - V_ang[kk - 1] > pi * 0.8:
                m = m - 1
            V_ang[kk] = E[kk].parameters.azimuth() + m * pi
            Ip[kk] = -(V_ang[kk] - 2*44*pi/180) / (2 * V * 4 * pi * 1e-7)

        return Ip

    def plot_error(self, filename):

        data = pd.read_csv(filename)

        V_I = data['Ip']
        DataIN = data['1']

        ## Requirement specificaion for ITER
        absErrorlimit = zeros(len(V_I))
        relErrorlimit = zeros(len(V_I))

        # Calcuation ITER specification
        for nn in range(len(V_I)):
            if V_I[nn] < 1e6:
                absErrorlimit[nn] = 10e3
            else:
                absErrorlimit[nn] = V_I[nn] * 0.01
            relErrorlimit[nn] = absErrorlimit[nn] / V_I[nn]

        fig, ax = plt.subplots(figsize=(6, 3))
        for col_name in data:
            if col_name != 'Ip':
                ax.plot(V_I, abs((data[col_name]-V_I)/V_I), label=col_name)

        ax.plot(V_I, relErrorlimit, 'r', label='ITER specification')
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
        plt.show()


if __name__ == '__main__':
    LB = 1.000
    SP = 0.200
    # dz = SP / 1000
    dz = 0.0001
    spunfiber = SPUNFIBER(LB, SP, dz)
    #spunfiber.first_calc()

    num_processor = 8
    V_I = arange(0.2e6, 18e6 + 0.2e6, 0.2e6)
    outdict = {'Ip': V_I}
    num_Merr = 1
    start = pd.Timestamp.now()
    for nn in range(100):
        M_err = spunfiber.create_Merr(num_Merr, 0.2, 0.2)
        Ip = spunfiber.calc_mp_Merr(num_processor, V_I, M_err)
        outdict[str(nn)] = Ip
        checktime = pd.Timestamp.now() - start
        print(nn, "/100, ", checktime)
        start = pd.Timestamp.now()

    df = pd.DataFrame(outdict)
    df.to_csv('mp2.csv', index=False)

    '''
    
    num_processor = 8
    V_I = arange(0.2e6, 18e6 + 0.2e6, 0.2e6)
    outdict = {'Ip': V_I}
    Ip = spunfiber.calc_mp1(num_processor, V_I)
    outdict['1'] = Ip
    df = pd.DataFrame(outdict)
    df.to_csv('mp3.csv', index=False)
    '''
    spunfiber.plot_error('mp2.csv')
