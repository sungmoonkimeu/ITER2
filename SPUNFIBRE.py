import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, arcsin, arctan, tan
from numpy.linalg import norm, eig


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

    def lamming1(self, Ip, DIR):
        """
        :param LB: beatlength
        :param SP: spin period
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :param n_div: division number
        :param vib_azi: [0, pi]: Azimuth. Default: 0.
        :param vib_ell: [-pi/4, pi/4]: Ellipticity. Default: 0.
        :return: M matrix calculated from N matrix
        """

        s_t_r = 2 * pi / self.SP * DIR  # spin twist ratio
        delta = 2 * pi / self.LB

        # magnetic field in unit length
        # H = Ip / (2 * pi * r)
        H = Ip / self.LF
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

        V_L = arange(0, self.LF + self.dz, self.dz)
        V_theta = V_L * s_t_r

        M = np.array([[1, 0], [0, 1]])

        kk = 0  # for counting M_err
        for nn in range(len(V_theta) - 1):
            if self.DIR == 1:
                phi = ((s_t_r * self.dz) - omega) / 2 + m * (pi / 2) + V_theta[nn]
            elif self.DIR == -1:
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
        :param LB: beatlength
        :param SP: spin period
        :param DIR: direction (+1: forward, -1: backward)
        :param Ip: plasma current
        :param L: fiber length
        :param V_theta: vector of theta (angle of optic axes)
        :param n_div: division number
        :param vib_azi: [0, pi]: Azimuth. Default: 0.
        :param vib_ell: [-pi/4, pi/4]: Ellipticity. Default: 0.
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

            # Empty matrix => there is no error matrix(Merr)
            M_empty = np.array([]).reshape(2, 2, 0)

            V_L_lf = arange(0, LF + self.dz, self.dz)
            V_theta_lf = V_L_lf * s_t_r

            V_L = arange(0, L + self.dz, self.dz)
            V_theta = V_theta_lf[-1] + V_L * s_t_r

            ksi = 44 * pi / 180
            Rot = np.array([[cos(ksi), -sin(ksi)], [sin(ksi), cos(ksi)]])
            Jm = np.array([[1, 0], [0, 1]])
            M_FR = Rot @ Jm @ Rot

            M_lf_f = self.lamming_vib(0, 1, LF, V_theta_lf, M_err)
            M_f = self.lamming_vib(iter_I, 1, L, V_theta, M_empty)
            M_b = self.lamming_vib(iter_I, -1, L, V_theta, M_empty)
            M_lf_b = self.lamming_vib(iter_I, -1, L, V_theta_lf, M_err)
            '''
            M_lf_f = lamming(LB, SP, 1, 0, L, V_theta_lf, M_empty)
            M_f = lamming(LB, SP, 1, iter_I, L, V_theta, M_empty)
            M_b = lamming(LB, SP, -1, iter_I, L, V_theta, M_empty)
            M_lf_b = lamming(LB, SP, -1, 0, L, V_theta_lf, M_empty)
            '''
            V_out[mm] = M_lf_b @ M_b @ M_FR @ M_f @ M_lf_f @ V_in
            # V_out[mm] = M_lf_b @ M_FR @ M_lf_f @ V_in
            # V_out[mm] = M_lf_f @ V_in
            mm = mm + 1
        print("done")


if __name__ == '__main__':
    LB = 1.000
    SP = 0.005
    # dz = SP / 1000
    dz = 0.0001
    spunfiber = SPUNFIBER(LB, SP, dz)
    spunfiber.first_calc()

