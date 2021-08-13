import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan, savetxt
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

    def lamming(self, Ip, DIR, L, V_theta, M_vib=None):
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

        kk = 0  # for counting M_vib
        tmp = np.array([])  # for test
        if M_vib is not None:
            nM_vib = M_vib.shape[2]
            nSet = int((len(V_theta)-1) / (nM_vib + 1))
            rem = (len(V_theta)-1) % nSet

        for nn in range(len(V_theta)-1):

            if DIR == 1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
                '''
                strM = "M" + str(nn)        # For indexing matrix to indicate the position of Merr  
                tmp = np.append(tmp, strM)  
                '''
            elif DIR == -1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[-1 - nn]
                '''
                strM = "M" + str(len(V_theta) - 1 - nn)
                tmp = np.append(tmp, strM)  # for test
                '''
            # phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
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
            # nVerr = M_vib.shape[2]
            # nSet = int((len(V_theta) - 1) / (nVerr + 1))
            # rem = (len(V_theta) - 1) % nSet
            if M_vib is not None:
                if DIR == 1 and (nn + 1) % nSet == 0:
                    if kk != nM_vib and (nn + 1 - rem) != 0:
                        M = M_vib[..., kk] @ M
                        '''
                        print(nn+1, "번째에 에러 매트릭스 추가")
                        strM = "Merr" + str(kk)
                        tmp = np.append(tmp, strM)  # for test
                        '''
                        kk = kk + 1

                elif DIR == -1 and (nn + 1 - rem) % nSet == 0:
                    if kk != nM_vib and (nn + 1 - rem) != 0:
                        M = M_vib[..., -1 - kk].T @ M
                        '''
                        print(len(V_theta) - 1 - nn, "번째에 에러 매트릭스 추가 (-backward)")
                        strM = "Merr" + str(nVerr - kk - 1)
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

    def cal_rotation(self, V_Ip, ang_FM, num, Vout_dic, M_vib=None):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        V = 0.54 * 4 * pi * 1e-7
        s_t_r = 2 * pi / self.SP
        LF = 10
        L = 10
        V_in = np.array([[1], [0]])

        mm = 0
        for iter_I in V_plasmaCurrent:
            # Empty matrix => Error matrix (Merr) with no effect
            M_empty = np.array([]).reshape(2, 2, 0)

            # Lead fiber vector with V_theta_lf
            V_L_lf = arange(0, LF+self.dz , self.dz)
            V_theta_lf = V_L_lf * s_t_r

            # Sensing fiber vector with V_theta
            V_L = arange(0, L+self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            # Faraday mirror
            ksi = ang_FM * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming(0, 1, LF, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, L, V_theta)
            M_b = self.lamming(iter_I, -1, L, V_theta)
            M_lf_b = self.lamming(0, -1, LF, V_theta_lf, M_vib)

            if num == 0 and iter_I == V_plasmaCurrent[-1]:
                #print("M_lf_f = ", M_lf_f[0, 1], M_lf_f[1, 0])
                #print("M_lf_b = ", M_lf_b[0, 1], M_lf_b[1, 0])
                #print("abs() = ", abs(M_lf_f[0, 1])-abs(M_lf_b[1, 0]))
                print("Norm (MLead_f - MLead_b.T) = ", norm(M_lf_f - M_lf_b.T))

                #print("M_f = ", M_f[0, 1], M_f[1, 0])
                #print("M_b = ", M_b[0, 1], M_b[1, 0])
                print("Norm (Msens_f - Msens_b) = ", norm(M_f - M_b))


            V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            #V_out[mm] =  M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")


        Vout_dic[num] = V_out

    def cal_rotation_trans(self, V_Ip, num, Vout_dic, M_vib=None):
        V_plasmaCurrent = V_Ip
        V_out = np.einsum('...i,jk->ijk', ones(len(V_plasmaCurrent)) * 1j, np.mat([[0], [0]]))

        V = 0.54 * 4 * pi * 1e-7
        s_t_r = 2 * pi / self.SP
        LF = 10
        L = 10
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

            # lead fiber vector with V_theta_lf2
            V_L_lf2 = arange(0, LF + self.dz, self.dz)
            V_theta_lf2 = V_theta[-1] + V_L_lf2 * s_t_r

            # Faraday mirror

            M_lf_f = self.lamming(0, 1, LF, V_theta_lf, M_vib)
            M_f = self.lamming(iter_I, 1, L, V_theta)
            M_lf_b = self.lamming(0, 1, LF, V_theta_lf2, M_vib)


            #M_f = self.lamming1(iter_I, 1, L, V_theta)

            V_out[mm] = M_lf_b @ M_f @ M_lf_f @ V_in
            #V_out[mm] = M_lf_b @ M_f @ M_lf_f @ V_in
            #V_out[mm] = M_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        #print("done")
        Vout_dic[num] = V_out

    def calc_mp(self, num_processor, V_I, ang_FM, M_vib=None, fig=None):
        V = 0.54
        spl_I = np.array_split(V_I, num_processor)

        procs = []
        manager = Manager()
        Vout_dic = manager.dict()

        Ip = zeros(len(V_I))

        for num in range(num_processor):
            proc = Process(target=self.cal_rotation,
                           args=(spl_I[num], ang_FM, num, Vout_dic, M_vib))
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
            Ip[kk] = -(V_ang[kk] - V_ang[0]) / (2 * V * 4 * pi * 1e-7)

        return Ip

    def calc_mp_trans(self, num_processor, V_I, M_vib=None, fig=None):
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
            proc = Process(target=self.cal_rotation_trans,
                           args=(spl_I[num], num, Vout_dic, M_vib))
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
            Ip[kk] = -(V_ang[kk]-V_ang[0]) / (V * 4 * pi * 1e-7)

        return Ip

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

# Progress bar is not easy/
# Todo comparison between transmission and reflection
# Todo FM effect
# Todo Ip calculation method change --> azimuth angle --> arc length


if __name__ == '__main__':
    LB = 1
    SP = 0.2
    # dz = SP / 1000
    dz = 0.001
    spunfiber = SPUNFIBER(LB, SP, dz)
    mode = 0
    num_iter = 5

    if mode == 0:
        strfile1 = 'IdealFM_Vib5ref3.csv'
        strfile2 = 'IdealFM_Vib5trans3.csv'
        num_processor = 16
        V_I = arange(0e6, 18e6 + 0.1e6, 0.1e6)
        outdict = {'Ip': V_I}
        nM_vib = 1
        start = pd.Timestamp.now()
        ang_FM = 45

        fig1, ax1 = spunfiber.init_plot_SOP()

        for nn in range(num_iter):
            M_vib = spunfiber.create_Mvib(nM_vib, 5, 5)
            Ip = spunfiber.calc_mp(num_processor, V_I, ang_FM, M_vib, fig1)
            outdict[str(nn)] = Ip
            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        df = pd.DataFrame(outdict)
        df.to_csv(strfile1, index=False)
        fig2, ax2, lines = spunfiber.plot_error(strfile1)

        fig3, ax3 = spunfiber.init_plot_SOP()

        for nn in range(num_iter):
            M_vib = spunfiber.create_Mvib(nM_vib, 5, 5)
            Ip = spunfiber.calc_mp_trans(num_processor, V_I, M_vib, fig3)

            outdict[str(nn)] = Ip
            checktime = pd.Timestamp.now() - start
            print(nn, "/", num_iter, checktime)
            start = pd.Timestamp.now()

        df = pd.DataFrame(outdict)
        df.to_csv(strfile2, index=False)
        fig4, ax4, lines = spunfiber.plot_error(strfile2)

    elif mode == 1:
        pass
    else:
        strfile1 = 'IdealFM_Vib5ref2.csv'
        strfile2 = 'IdealFM_Vib5trans2.csv'
        fig, ax, lines = spunfiber.plot_error(strfile1)
        fig, ax, lines = spunfiber.plot_error(strfile2)
        #ax.legend(lines[:], ['line A', 'line B'], loc='upper right')

        #spunfiber.add_plot('mp3.csv', ax, '45')
    '''
    num_processor = 8
    V_I = arange(0.2e6, 18e6 + 0.2e6, 0.2e6)
    outdict = {'Ip': V_I}
    Ip = spunfiber.calc_mp1(num_processor, V_I)
    outdict['1'] = Ip
    df = pd.DataFrame(outdict)
    df.to_csv('mp3.csv', index=False)
    '''
plt.show()