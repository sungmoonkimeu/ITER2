
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:54:00 2022

@author: sungmoon

Simulation for evalutate the optimization of FOCS.
Calculate the maximum iterations for input SOP optimization

you can change "mode" variable in main function to run:

mode == 0:
    Run Monte-Carlo simulation for randomly generated MF, MB matrices.
    Data will be saved with "strfile" variable.
    If you wan to run the ideal case(without uncertainty),
        put theta0, phi0, theta_e0 = np.array([]), np.array([]), np.array([]) in line 186


mode == 1:
    Plot the datafile as a histogram.


"""

#SOP measurement uncertainty
# --> Polarimeter's uncertainty ~ 0.2 deg

#SOP control uncertainty
# --> SOP controller's uncertainty
# --> Rotatable polarizer's uncertainty ~ 0.2 deg?

#Calibration current uncertainty
# --> difference between measured value and real value
# --> noise signal on the calibratino current ?
# --> maximum value of calibration current has an error of x%

# 1D optimization
# --> random variable
# --> how many iteration is required for each run
# bisection?


from scipy import optimize
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes

import pandas as pd
import os

def cal_arclength(S):
    L = 0
    for nn in range(len(S)-1):
        c = pi/2 - S.parameters.ellipticity_angle()[nn]*2
        b = pi/2 - S.parameters.ellipticity_angle()[nn+1]*2

        A0 = S.parameters.azimuth()[nn]*2
        A1 = S.parameters.azimuth()[nn+1]*2
        A = A1 - A0
        if A == np.nan:
            A = 0

        L = L + arccos(cos(b) * cos(c) + sin(b) * sin(c) * cos(A))
        #print("c",c,"b",b,"A0",A0,"A1",A1, "L",L)

    return L

def create_M_arb(theta, phi, theta_e):

    M_rot = np.array([[cos(theta_e), -sin(theta_e)], [sin(theta_e), cos(theta_e)]])  # shape (2,2,nM_vib)
    M_theta = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
    M_theta_T = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])  # shape (2,2,nM_vib)
    M_phi = np.array([[exp(1j*phi), 0],[0, exp(-1j*phi)]])

    return M_rot @ M_theta @ M_phi @ M_theta_T

def f(x, Mci, Mco, strfile = None):
    # 1st Optimization function without uncertainty
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.54 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp / 100
    V_Ip = arange(0e6, MaxIp + dIp, dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        # Faraday rotation matirx
        th_FR = iter_I * V * 2
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    S = create_Stokes('output')
    S.from_Jones(E1)

    # print(S.parameters.ellipticity_angle()[0])
    # print(S.parameters.azimuth()[-1])
    L = cal_arclength(S)  # Arc length is orientation angle psi -->
    #Veff = L / 2 / (MaxIp * 2)  # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    #errV = abs((Veff - V) / V)
    Veff = L / 2 / (MaxIp * 2) * 180/pi * 1e6  # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = L / 2 / (MaxIp * 2)  * 180/pi * 1e6 * -1

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    # df = pd.DataFrame(outdict)
    # df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    return errV

# Noise included fuction
def f2(x, Mci, Mco):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    x = x + (np.random.rand(1)-0.5)*pi/180 # 0.5 deg SOP control uncertainty
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.54 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/50
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        [theta, phi, theta_e] = (np.random.rand(3) *
                                [90, 0.01, 0.01]-[45, .005, 0.005])*np.pi/180
        Mn = create_M_arb(theta, phi, theta_e)

        # Faraday rotation matirx
        th_FR = iter_I * V*2 * (1+np.random.rand(1)*0.01-0.005)[0] # 1% error including
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mn @ Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)

    # estimation value for optimization,
    # algorithm is basically set to find minimum, so multiply it by -1 to find maximum
    est_V = L / 2 / (MaxIp * 2) * 180/pi * 1e6 * -1

    #errV = abs((Veff-V)/V)
    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return est_V

if __name__ == '__main__':

    start = pd.Timestamp.now()
    mode = 0

    ## 2nd step
    #strfile = 'Multiple_Cal_ideal.csv'
    #strfile = 'Multiple_Cal_with_SOPnoise.csv'
    #strfile = 'Multiple_Cal_with_Cur40kA_noise_0.1.csv'
    strfile = 'Multiple_Cal_with_Cur40kA_noise_0.05.csv'


    if mode == 0:

        n_iter = 10
        n_iter2 = 10
        fig, ax = plt.subplots(figsize=(6, 6))
        for mm in range(n_iter2):

            v_out = np.zeros(n_iter)
            for nn in range(n_iter):
                theta0, phi0, theta_e0 = np.random.rand(3)*360
                # theta0, phi0, theta_e0 = np.array([]), np.array([]), np.array([]) # for non uncertainty cases
                Mci = create_M_arb(theta0*pi/180, phi0*pi/180, theta_e0*pi/180)

                theta1, phi1, theta_e1 = np.random.rand(3)*360
                Mco = create_M_arb(theta1*pi/180, phi1*pi/180, theta_e1*pi/180)

                # initial point
                init_polstate = np.array([[0], [pi / 4]])

                fmin_result = optimize.fmin(f2, pi/6, (Mci, Mco), maxiter=30, xtol=1, ftol=0.05,
                                    initial_simplex=init_polstate, retall=True, full_output=1)

                v_out[nn] = fmin_result[3]
                print("mm=", mm, " nn=", nn)

            ax.plot(v_out)

            outdict = {'out': v_out}
            df = pd.DataFrame(outdict)
            df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    elif mode ==1:
        # Plotting

        bins = np.arange(1, 40, 2)
        #bins = np.append(bins, 40)
        #bins = [3, 7, 11, 15, 19]
        fig, ax = plt.subplots(figsize=(6, 6))

        #strfile = 'Multiple_Cal_ideal.csv'
        data = pd.read_csv(strfile)
        #ax.hist(data['out'], bins, label='ideal', alpha=0.7, facecolor = 'g')
        ax.hist(data['out'], bins, label='ftol = 0.1', alpha=0.7, facecolor='g')

        # strfile = 'Multiple_Cal_with_SOPnoise_0.001.csv'
        # data = pd.read_csv(strfile)
        # ax.hist(data['out'], bins, label='>99.9%',alpha=0.7, facecolor = 'r')
        #
        # strfile = 'Multiple_Cal_with_SOPnoise_0.0005.csv'
        # data = pd.read_csv(strfile)
        # ax.hist(data['out'], bins, label='>99.95%', alpha=0.7, facecolor='b')

        # strfile ='Multiple_Cal_with_Cur5kA_noise_0.01.csv'
        # data = pd.read_csv(strfile)
        # ax.hist(data['out'], bins, label='>99%',alpha=0.7, facecolor = 'b')

        ax.set_xlabel('iteration')
        ax.set_ylabel('n')
        ax.legend(loc='upper right')


    end = pd.Timestamp.now()
    print("Total time = ", (end-start).total_seconds())

    plt.show()

