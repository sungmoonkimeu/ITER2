
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:54:00 2022

@author: sungmoon

Simulation of FOCS calibration.
To show how calibration works for each iterations.

you can change "mode" variale in main function to run:

mode == 0:
    Calbiratoin in 1D space with or without uncertainties
    Run optimization function in 1D space

    Choose desired function to evalute
    f : ideal cases (without any uncertainity, with poincare sphere drawing)
    f2 : with input SOP control uncertainty (x) + SOP measurment uncertainty (Mn)

    'calibration_log.csv' will be generated

mode == 1:
    Scanning FOCS response for each input azimuth from 0 to 180 deg
    'scanning1d_xy.csv' will be generated

mode == 2:
    overlap the complete response (scanning1d_xy.csv) and optimization (calibration_log.csv)
    The optimization process will be shown on the graph for each iterations and
    recorded as a gif file

mode == 3:
    Scanning FOCS response with noise
    for each input azimuth from 0 to 180 deg
    'scanning1d_noise_xy.csv' will be generated

mode == 5:
    overlap the noise included simulation result
    FOCS comlete response (scanning1d_noise_xy.csv)
    and optimization generated using f2 in mode 0 (calibration_log.csv)
    The optimization process will be shown on the graph for each iterations and
    recorded as a gif file



"""
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
import matplotlib.pylab as pl
from matplotlib.colors import rgb2hex
import os
from scipy import optimize

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import matplotlib as mpl
import matplotlib.ticker
from matplotlib.ticker import (MaxNLocator,
                               FormatStrFormatter, ScalarFormatter)
from multiprocessing import Process, Queue, Manager,Lock
import pandas as pd
import glob
import imageio


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def create_gif(gif_name = None, remove = True):

    if gif_name is None:
        gif_name = 'mygif.gif'

    file_list = glob.glob('*.png')  # Get all the pngs in the current directory
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for img in file_list:
        images.append(imageio.imread(img))
        if remove == True:
            os.remove(img)
    imageio.mimsave(gif_name, images, fps=2)

def show_result_poincare(strfile, Mci, Mco, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')

    E0.general_azimuth_ellipticity(azimuth=data['x'], ellipticity=0)
    S = create_Stokes('output')

    Et = Jones_vector('trace')
    St = create_Stokes('trace')

    V_out = np.einsum('...i,jk->ijk', ones(len(E0)) * 1j, np.mat([[0], [0]]))

    V = 0.54 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/20
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out_trace = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x'])))
    Stmp = create_Stokes('tmp')

    for mm, val in enumerate(E0):
        V_out[mm] = Mco @ Mci @ E0[mm].parameters.matrix()

        for nn, iter_I in enumerate(V_Ip):
            th_FR = iter_I * V * 2
            M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
            V_out_trace[nn] = Mco @ M_FR @ Mci @ E0[mm].parameters.matrix()

        Et.from_matrix(M=V_out_trace)
        St.from_Jones(Et)
        draw_stokes_points(ax, St[0], kind='scatter', color_scatter=rgb2hex(colors[mm]))
        draw_stokes_points(ax, St, kind='line', color_scatter=rgb2hex(colors[mm]))

        if mm > 0:
            x0 = St[0].parameters.matrix()
            arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=rgb2hex(colors[mm]), shrinkA=0, shrinkB=0)
            a = Arrow3D([o[1][0], x0[1][0]], [o[2][0], x0[2][0]], [o[3][0], x0[3][0]], **arrow_prop_dict)
            ax.add_artist(a)
            if mm > 1:
                plt.savefig(str(mm)+'.png')
        o = St[0].parameters.matrix()

    create_gif('mygif3.gif')

def show_result_cal_azimuth(strfile_background, strfile_calibration, fig, ax):
    V = 0.54 * 4 * pi * 1e-7
    maxVI = 40e3

    # draw background
    data_bg = pd.read_csv(strfile_background)
    azi_bg = np.array(data_bg['azi'])
    l_bg = np.array(data_bg['l'])
    #Ip_bg = l_bg / 4 / V
    #sensitivity_bg = Ip_bg / maxVI
    FOCSresponse_bg = l_bg/4/maxVI*180/pi *1e6
    ax.plot(azi_bg*180/pi, FOCSresponse_bg)
    ax.set_xlabel('azimuth angle [deg]')
    ax.set_ylabel('FOCS response (deg/MA)')
    ax.set(xlim=(0, 180),ylim=(0,55))
    # draw calibration footstep
    prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", shrinkA=0, shrinkB=0)
    data = pd.read_csv(strfile_calibration)
    x0 = data['x'][0]
    #y0 = data['L'][0]/(4*V)/maxVI
    y0 = data['L'][0] / 4 / maxVI * 180/pi * 1e6

    for nn in range(len(data['x'])-1):
        x = data['x'][nn]
        y = data['L'][nn]/4/maxVI* 180/pi *1e6

        if nn < 11:
            ax.annotate("", xy=(x*180/pi, y),
                    xytext=(x0*180/pi, y0), arrowprops=prop)
            if nn == 10:
                plt.cla()
                ax.plot(azi_bg * 180 / pi, FOCSresponse_bg)
                ax.set_xlabel('azimuth angle [deg]')
                ax.set_ylabel('FOCS response (deg/MA)')
                ax.set(xlim=(55/2, 69/2), ylim=(38.5, 39))
        else:
            ax.annotate("", xy=(x*180/pi, y),
                    xytext=(x0*180/pi, y0), arrowprops=prop)
        x0 = x
        y0 = y
        plt.savefig(str(nn)+'.png')

    create_gif('mygif2.gif', remove=False)

def f(x, Mci, Mco, strfile):
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
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    return errV

def f2(x, Mci, Mco, strfile):
    # 2nd Optimization function SOP control and measurement uncertainty
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    x = x + (np.random.rand(1)-0.5)*pi/180 # 0.5 deg SOP control uncertainty
    # x = x + (np.random.rand(1)*2 - 1) * pi / 180  # 1 deg SOP control uncertainty
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.54 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/100
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        [theta, phi, theta_e] = (np.random.rand(3) *
        #                         [90, 1, 1] - [45, .5, 0.5]) * np.pi / 180
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
    errV = L / 2 / (MaxIp * 2) * 180/pi * 1e6 * -1

    #errV = abs((Veff-V)/V)

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV

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

def eval_result_gif(strfile):
    data = pd.read_csv(strfile)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.subplots_adjust(hspace=0.4, right=0.95, top=0.93, bottom=0.17)

    ax.plot(data['errV'] * 100, label='Err = (Vmeas - V)/V ')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Error (%)')
    ax.legend()
    ax.set_yscale('log')

    for nn in range(len(data['errV'])):
        ax.scatter(nn, data['errV'][nn] * 100)
        if nn > 2:
            plt.savefig(str(nn) + '.png')
    create_gif('mygif4.gif')

def eval_result(strfile):

    data = pd.read_csv(strfile)
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(data['errV']*100, label='Err = (Vmeas - V)/V ')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Error (%)')
    ax.legend()

if __name__ == '__main__':

    # _______________________________Parameters#1___________________________________#
    # Circulator input matrix
    '''
    theta = 40* pi / 180   # birefringence axis of LB
    phi = -10 * pi / 180  # ellipticity angle change from experiment
    theta_e = 30* pi / 180  # azimuth angle change from experiment
    '''
    inputa = [30, 35, 30]
    [theta, phi, theta_e] = np.array(inputa)*pi/180
    Mci = create_M_arb(theta, phi, theta_e)

    # Circulator output matrix

    inputb = [15, 25, 10]
    [theta, phi, theta_e] = np.array(inputb)*pi/180
    Mco = create_M_arb(theta, phi, theta_e)

    mode =1
    if mode == 0:
        #Calbiratoin_1D space with or without uncertainties
        strfile = 'calibration_log.csv'

        if os.path.exists(strfile):
            print("previous data(", strfile, ") has been deleted")
            os.remove(strfile)

        # initial point
        init_polstate = np.array([[0], [pi / 4]])

        # f ==> without uncertainty
        # fmin_result = optimize.fmin(f, pi / 6, (Mci, Mco, strfile), maxiter=30, xtol=1, ftol=0.0001,
        #                             initial_simplex=init_polstate, retall=True, full_output=1)
        # f2 ==> with uncertainty of SOP measurement and SOP control and Calibration current uncertainty
        fmin_result = optimize.fmin(f2, pi/6, (Mci, Mco, strfile), maxiter=30, xtol=1, ftol=0.05,
                                   initial_simplex=init_polstate, retall=True, full_output=1)

        print(fmin_result[0])
    elif mode == 1:
        #Scanning 1D space w/o noise
        strfile = 'scanning1D.csv'
        n_azi = 1  # 20

        V_I = arange(0e6, 40e3 + 1e3, 5e3)

        V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
        V = 0.54 * 4 * pi * 1e-7

        E0 = Jones_vector('input')
        E = Jones_vector('Output')

        azi = np.linspace(0, 180, n_azi) * pi / 180
        colors = pl.cm.hsv(np.linspace(0, 1, len(azi)))

        E0.general_azimuth_ellipticity(azimuth=azi, ellipticity=0)

        OV = np.array([])
        midpnt = int(len(V_I) / 2)
        length_S = []
        S = create_Stokes('Output_S')

        for nn in range(len(E0)):
            for mm, iter_I in enumerate(V_I):
                # Faraday rotation matirx
                th_FR = iter_I * V * 2
                M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])
                # V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]
                V_out[mm] = Mco @ M_FR @ Mci @ E0[nn].parameters.matrix()

            E.from_matrix(M=V_out)
            S.from_Jones(E)

            length_S.append(cal_arclength(S)[0])

            if nn != 0:
                draw_stokes_points(fig[0], S, kind='line', color_line=rgb2hex(colors[nn]))
                draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
            else:
                fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[23 * pi / 180, 32 * pi / 180], kind='line',
                                          color_line=rgb2hex(colors[nn]))
                draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
            print(S[-1])

        print(length_S)

        l_measured = np.array(length_S)

        outdict0 = {'azi': azi, 'l' : l_measured}
        df0 = pd.DataFrame(outdict0)
        df0.to_csv(strfile.split('.')[0] + '_xy.' + strfile.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile))

        fig, ax = plt.subplots(figsize=(5,3))

        Ip_measured = l_measured / 4 / V
        sensitivity = Ip_measured / max(V_I)
        errV = abs(sensitivity - 1)

        # ax.plot(azi*180/pi*2,sensitivity)
        ax.plot(azi*180/pi, l_measured/4/max(V_I)*180/pi*1e6)
        ax.set_xlabel('azimuth [deg]')
        #ax.set_ylabel('normalized sensitivity')
        ax.set_ylabel('FOCS response (deg/MA)')
        ax.set(xlim=(0,180), ylim=(0, 55))

    elif mode == 2:
        # overlap complete response and optimization w/o noise
        strfile1 = 'scanning1D_xy.csv'
        fig, ax = plt.subplots(figsize=(6,5))
        #fig.subplots_adjust(left=0.2, bottom=0.25)

        strfile2 = 'calibration_log.csv'
        show_result_cal_azimuth(strfile1, strfile2, fig, ax)

        #Stmp = create_Stokes('tmp')
        #fig, ax = Stmp.draw_poincare(figsize=(5, 5), angle_view=[4 * pi / 180, 124 * pi / 180], kind='line')
        #show_result_poincare(strfile2, Mci, Mco, fig[0], ax)
    elif mode == 3:
        #scanning 1D space with noise
        strfile = 'scanning1D_noise.csv'
        n_azi = 100  # 20

        MaxIp = 40e3
        dIp = MaxIp / 100
        V_Ip = arange(0e6, MaxIp + dIp, dIp)
        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        V = 0.54 * 4 * pi * 1e-7

        E0 = Jones_vector('input')
        E = Jones_vector('Output')

        azi = np.linspace(0, 180, n_azi) * pi / 180
        azi_noise = azi + (np.random.rand(len(azi)) - 0.5) * pi / 180
        colors = pl.cm.hsv(np.linspace(0, 1, len(azi)))

        E0.general_azimuth_ellipticity(azimuth=azi_noise, ellipticity=0)

        OV = np.array([])
        midpnt = int(len(V_Ip) / 2)
        length_S = []
        S = create_Stokes('Output_S')

        for nn in range(len(E0)):
            for mm, iter_I in enumerate(V_Ip):
                [theta, phi, theta_e] = (np.random.rand(3) *
                                        [90, 0.01, 0.01]-[45, .005, 0.005])*np.pi/180
                Mn = create_M_arb(theta, phi, theta_e)
                # Faraday rotation matirx
                th_FR = iter_I * V * 2 * (1+np.random.rand(1)*0.01-0.005)[0] # 1% error including
                M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])
                # V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]
                V_out[mm] = Mn @ Mco @ M_FR @ Mci @ E0[nn].parameters.matrix()

            E.from_matrix(M=V_out)
            S.from_Jones(E)

            #L= cal_arclength(S)[0]
            #L = L * (1 + np.random.rand(1) * 0.01 - 0.005)  # 1% error including
            #length_S.append(L)
            length_S.append(cal_arclength(S)[0])

            #
            # if nn != 0:
            #     draw_stokes_points(fig[0], S, kind='line', color_line=rgb2hex(colors[nn]))
            #     draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
            # else:
            #     fig, ax = S.draw_poincare(figsize=(7, 7), angle_view=[23 * pi / 180, 32 * pi / 180], kind='line',
            #                               color_line=rgb2hex(colors[nn]))
            #     draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter=rgb2hex(colors[nn]))
            # print(S[-1])

        print(length_S)
        l_measured = np.array(length_S)
        print(l_measured)

        fig, ax = plt.subplots(figsize=(5, 3))

        Ip_measured = l_measured / 4 / V

        # ax.plot(azi*180/pi*2,sensitivity)
        ax.plot(azi * 180 / pi, l_measured / 4 / MaxIp * 180 / pi * 1e6, label='with SOP uncertainties', zorder = 10)
        ax.set_xlabel('azimuth [deg]')
        # ax.set_ylabel('normalized sensitivity')
        ax.set_ylabel('FOCS response (deg/MA)')
        ax.set(xlim=(0, 180), ylim=(0, 55))

        outdict0 = {'azi': azi, 'l': l_measured}
        df0 = pd.DataFrame(outdict0)
        df0.to_csv(strfile.split('.')[0] + '_xy.' + strfile.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile))



        # Overlapping ideal case (without uncertainty)
        strfile_ideal = 'scanning1D_xy.csv'

        data_ideal = pd.read_csv(strfile_ideal)
        azi_ideal = np.array(data_ideal['azi'])
        l_ideal = np.array(data_ideal['l'])

        FOCSresponse_ideal = l_ideal / 4 / MaxIp * 180 / pi * 1e6
        ax.plot(azi_ideal * 180 / pi, FOCSresponse_ideal, label='w/o SOP uncertainties', zorder = 0)
        ax.legend(loc='lower right')

    elif mode == 5:
        strfile1 = 'scanning1D_noise_xy.csv'
        fig, ax = plt.subplots(figsize=(5,4))
        strfile2 = 'calibration_log.csv'
        show_result_cal_azimuth(strfile1, strfile2, fig, ax)

        #Stmp = create_Stokes('tmp')
        #fig, ax = Stmp.draw_poincare(figsize=(5, 5), angle_view=[4 * pi / 180, 124 * pi / 180], kind='line')
        #show_result_poincare(strfile2, Mci, Mco, fig[0], ax)

    plt.show()

