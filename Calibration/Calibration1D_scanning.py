
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:45:38 2020

@author: sungmoon

Simulation of FOCS calibration.
To show how calibration works for each iterations.

you can change "mode" variale in main function to run:

mode == 0:
    Scanning 2D space
    FOCS output response for each (azi, ell) of input SOP
    'scanning.csv' will be generated

mode == 1:
    Calbiratoin in 1D space
    Run optimization function in 1D space

    Choose desired function to evalute
    f : ideal cases (without any uncertainity, with poincare sphere drawing)
    f2 : ideal cases (without any uncertainity)
    f3 : with input SOP control uncertainty (x) + SOP measurment uncertainty (Mn)
    f4 : f3 + calibration current uncertainty
    'calibration_log.csv' will be generated


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

def create_gif(gif_name = None):

    if gif_name is None:
        gif_name = 'mygif.gif'

    file_list = glob.glob('*.png')  # Get all the pngs in the current directory
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for img in file_list:
        images.append(imageio.imread(img))
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

    V = 0.7 * 4 * pi * 1e-7
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

def show_result_aziellspace(strfile, aziell, sensitivity, ax, fig):
    data = pd.read_csv(strfile)

    azi0 = data['x'][0]
    ell0 = 0
    prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0)

    for nn in range(len(data['x'])-1):
        if nn> 1:
            plt.cla()
            plot_contour(aziell, sensitivity, fig,ax, True)
        azi = data['x'][nn+1]
        ell = 0
        print("azi0, azi ", azi0*180/pi, azi*180/pi)

        ax.annotate("", xy=(azi*180/pi, ell*180/pi),
                    xytext=(azi0*180/pi, ell0*180/pi), arrowprops=prop)

        azi0 = azi
        ell0 = ell

        plt.savefig(str(nn)+'.png')

    create_gif('mygif2.gif')

def show_result_cal_azimuth(strfile_background, strfile_calibration, fig, ax):
    V = 0.7 * 4 * pi * 1e-7
    maxVI = 40e3

    # draw background
    data_bg = pd.read_csv(strfile_background)
    azi_bg = np.array(data_bg['azi'])
    l_bg = np.array(data_bg['l'])
    Ip_bg = l_bg / 4 / V
    sensitivity_bg = Ip_bg / maxVI
    ax.plot(azi_bg*180/pi*2, sensitivity_bg)
    ax.set_xlabel('azimuth angle [deg]')
    ax.set_ylabel('Normalized sensitivity')
    ax.set(xlim=(0,360),ylim=(0,1.1))
    # draw calibration footstep
    prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", shrinkA=0, shrinkB=0)
    data = pd.read_csv(strfile_calibration)
    x0 = data['x'][0]
    y0 = data['L'][0]/(4*V)/maxVI

    for nn in range(len(data['x'])-1):
        x = data['x'][nn]
        y = data['L'][nn]/(4*V)/maxVI

        ax.annotate("", xy=(x*180/pi*2, y),
                    xytext=(x0*180/pi*2, y0), arrowprops=prop)
        x0 = x
        y0 = y
        plt.savefig(str(nn)+'.png')

    create_gif('mygif2.gif')


def f(x, Mci, Mco, fig, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=x[0], ellipticity=x[1])
    V = 0.7 * 4 * pi * 1e-7
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

    draw_stokes_points(fig[0], S, kind='line', color_scatter='k')
    draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter='b')
    draw_stokes_points(fig[0], S[-1], kind='scatter', color_scatter='r')
    # print(S.parameters.azimuth()[-1])
    L = cal_arclength(S)  # Arc length is orientation angle psi -->
    Veff = L / 2 / (MaxIp * 2)  # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff - V) / V)
    # Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V,
          "errV=", errV)

    outdict = {'x0': np.array([x[0]]), 'x1': np.array([x[1]]), 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    return errV

def f2(x, Mci, Mco, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.7 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/50
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff-V)/V)

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV

def f3(x, Mci, Mco, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    #x = x + (np.random.rand(1)-0.5)*pi/180 # 0.5 deg SOP control uncertainty
    x = x + (np.random.rand(1)*2 - 1) * pi / 180  # 1 deg SOP control uncertainty
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.7 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/50
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        [theta, phi, theta_e] = (np.random.rand(3) *
                                 [90, 1, 1] - [45, .5, 0.5]) * np.pi / 180
        #                        [90, 0.01, 0.01]-[45, .005, 0.005])*np.pi/180

        Mn = create_M_arb(theta, phi, theta_e)

        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mn @ Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff-V)/V)

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV

def f4(x, Mci, Mco, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    #x = x + (np.random.rand(1)-0.5)*pi/180 # 0.5 deg SOP control uncertainty
    x = x + (np.random.rand(1)*2 - 1) * pi / 180  # 1 deg SOP control uncertainty
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
    V = 0.7 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/50
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        [theta, phi, theta_e] = (np.random.rand(3) *
                                 #[90, 1, 1] - [45, .5, 0.5]) * np.pi / 180
                                [90, 0.01, 0.01]-[45, .005, 0.005])*np.pi/180

        Mn = create_M_arb(theta, phi, theta_e)

        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mn @ Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    L0 = L
    L = L*(1+np.random.rand(1)*0.01-0.005) # 1% error including
    #print((L0 - L) /L0)
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    #print("veff=", Veff)

    errV = abs((Veff-V)/V)
    #print("err=", errV)

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV

#def plot_sensitivity_1D(aziell, sensitivity, fig,ax):


def plot_contour(aziell, sensitivity, fig, ax, redraw):

    contour = ax.contourf(aziell[0] * 180 / pi * 2, aziell[1] * 180 / pi * 2, sensitivity, levels=np.linspace(0, 1, 21),
                          cmap='Reds')
    contour2 = ax.contour(aziell[0] * 180 / pi * 2, aziell[1] * 180 / pi * 2, sensitivity, levels=[0.995], colors=('y'),
                          linestyles=('-',), linewidths=(2,))
    ax.clabel(contour2, fmt='%4.3f', colors='y', fontsize=14)  # contour line labels
    ax.set_xlabel('azimuth angle [deg]')
    ax.set_ylabel('elevation angle [deg]')
    if redraw is False:
        # ax.clabel(contour3, fmt='%4.3f', colors='y', fontsize=14) #contour line labels
        fig.colorbar(contour, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])  # Add a colorbar to a plot

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
    inputa = [30, -35, 30]
    [theta, phi, theta_e] = np.array(inputa)*pi/180
    Mci = create_M_arb(theta, phi, theta_e)

    # Circulator output matrix

    inputb = [15, -25, 10]
    [theta, phi, theta_e] = np.array(inputb)*pi/180
    Mco = create_M_arb(theta, phi, theta_e)

    mode = 4
    if mode == 0:
        # scanning 2D space
        # FOCS output response for each (azi, ell) of input SOP

        strfile = 'scanning.csv'
        n_azi = 20 #20
        n_ell = 25 #25
        # input matrix
        '''
        strfile0 = 'Filteredsignal.csv'
        data = pd.read_csv(strfile0)
        V_I = np.array(data)
        V_I = V_I.reshape(V_I.size,)
        '''
        V_I = arange(0e6, 40e3 + 1e3, 5e3)

        V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
        V = 0.7 * 4 * pi * 1e-7

        E0 = Jones_vector('input')
        E = Jones_vector('Output')

        azi = np.linspace(0,180,n_azi)*pi/180
        ell = np.linspace(-45,45,n_ell)*pi/180
        aziell = np.meshgrid(azi,ell)

        colors = pl.cm.hsv(np.linspace(0, 1, len(aziell[0].reshape(aziell[0].shape[0] * aziell[0].shape[1]))))

        E0.general_azimuth_ellipticity(azimuth=aziell[0].reshape(aziell[0].shape[0] * aziell[0].shape[1],),
                                       ellipticity=aziell[1].reshape(aziell[1].shape[0] * aziell[1].shape[1],))

        OV = np.array([])
        midpnt = int(len(V_I)/2)
        length_S = []
        S = create_Stokes('Output_S')

        for nn in range(len(E0)):
            for mm, iter_I in enumerate(V_I):
                # Faraday rotation matirx
                th_FR = iter_I * V*2
                M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])
                #V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]
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

        l_measured = np.array(length_S).reshape(np.array(aziell).shape[1], np.array(aziell).shape[2])

        outdict0 = {'azi': azi}
        df0 = pd.DataFrame(outdict0)
        df0.to_csv(strfile.split('.')[0]+'_x.'+strfile.split('.')[1], index=False, mode='w', header=not os.path.exists(strfile))

        outdict1 = {'ell': ell}
        df0 = pd.DataFrame(outdict1)
        df0.to_csv(strfile.split('.')[0] + '_y.' + strfile.split('.')[1], index=False, mode='w',header=not os.path.exists(strfile))

        df1 = pd.DataFrame(l_measured)
        df1.to_csv(strfile.split('.')[0]+'_z.'+strfile.split('.')[1], index=False, mode='w', header=not os.path.exists(strfile))

        '''
        outdict0 = {'azi': azi_o}
        df0 = pd.DataFrame(outdict0)
        df0.to_csv(strfile2.split('.')[0] + '_x.' + strfile2.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile2))

        outdict1 = {'ell': ell_o}
        df0 = pd.DataFrame(outdict1)
        df0.to_csv(strfile2.split('.')[0] + '_y.' + strfile2.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile2))

        df1 = pd.DataFrame(l_measured)
        df1.to_csv(strfile2.split('.')[0] + '_z.' + strfile2.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile2))

        '''
        #labelTups = [('LP0', 0), ('LP45', 1), ('RCP', 2)]
        #colors = color_code
        #custom_lines = [plt.Line2D([0], [0], ls="", marker='.',mec='k', mfc=c, mew=.1, ms=20) for c in colors]
        #ax.legend(custom_lines, [lt[0] for lt in labelTups],loc='center left', bbox_to_anchor=(0.7, .8))

        #print(V_out)

        #fig = plt.figure(figsize=(14,9))
        #ax = plt.axes(projection='3d')

        fig, ax = plt.subplots(1,1)
        l_measured = np.array(length_S).reshape(np.array(aziell).shape[1],np.array(aziell).shape[2])
        Ip_measured = l_measured/4/V
        sensitivity = Ip_measured/max(V_I)
        #surf = ax.plot_surface(aziell[0], aziell[1],B)
        '''
        contour = ax.contourf(aziell[0]*180/pi,aziell[1]*180/pi*2,B/4/V, levels = np.linspace(0,40000,41),cmap='Reds')
        contour2 = ax.contour(aziell[0]*180/pi,aziell[1]*180/pi*2,B/4/V, levels=[39600], colors=('y'), linestyles=('-',), linewidths=(2,))
        '''
        contour = ax.contourf(aziell[0]*180/pi*2,aziell[1]*180/pi*2,sensitivity, levels=np.linspace(0,1,21),cmap='Reds')
        contour2 = ax.contour(aziell[0]*180/pi*2,aziell[1]*180/pi*2,sensitivity, levels=[0.995], colors=('y'), linestyles=('-',), linewidths=(2,))
        #contour3 = ax.contour(aziell[0]*180/pi,aziell[1]*180/pi*2,sensitivity, levels=[0.999], colors=('y'), linestyles=('-',), linewidths=(2,))

        ax.clabel(contour2, fmt='%4.3f', colors='y', fontsize=14) #contour line labels
        #ax.clabel(contour3, fmt='%4.3f', colors='y', fontsize=14) #contour line labels

        fig.colorbar(contour, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]) # Add a colorbar to a plot

        ax.set_xlabel('azimuth angle [deg]')
        ax.set_ylabel('ellipticity angle [deg]')
    elif mode == 1:
        #Calbiratoin_1D space
        strfile = 'calibration_log.csv'

        if os.path.exists(strfile):
            print("previous data(", strfile, ") has been deleted")
            os.remove(strfile)

        # initial point
        init_polstate = np.array([[pi/6], [pi / 3]])

        #fmin_result = optimize.fmin(f2, pi/6, (Mci, Mco, strfile), maxiter=30, xtol=1, ftol=0.0001,
        #                            initial_simplex=init_polstate, retall=True, full_output=1)
        fmin_result = optimize.fmin(f4, pi / 6, (Mci, Mco, strfile), maxiter=30, xtol=1, ftol=0.01,
                                    initial_simplex=init_polstate, retall=True, full_output=1)
        print(fmin_result[0])
    elif mode == 2:
        #Scanning 1D space w/o noise
        strfile = 'scanning1D.csv'
        n_azi = 100  # 20

        V_I = arange(0e6, 40e3 + 1e3, 5e3)

        V_out = np.einsum('...i,jk->ijk', ones(len(V_I)) * 1j, np.mat([[0], [0]]))
        V = 0.7 * 4 * pi * 1e-7

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

        ax.plot(azi*180/pi*2,sensitivity)
        #ax.plot(azi*180/pi*2, errV)
        ax.set_xlabel('azimuth angle [deg]')
        ax.set_ylabel('normalized sensitivity')
        ax.set(xlim=(0,360), ylim=(0, 1.1))
    elif mode == 3:
        #scanning 1D space with noise
        strfile = 'scanning1D_noise.csv'
        n_azi = 100  # 20

        MaxIp = 40e3
        dIp = MaxIp / 50
        V_Ip = arange(0e6, MaxIp + dIp, dIp)
        V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))
        V = 0.7 * 4 * pi * 1e-7

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
                th_FR = iter_I * V * 2
                M_FR = np.array([[cos(th_FR), sin(th_FR)], [-sin(th_FR), cos(th_FR)]])
                # V_out[mm] = M_co @ M_FR @ M_ci @ V_in[nn]
                V_out[mm] = Mn @ Mco @ M_FR @ Mci @ E0[nn].parameters.matrix()

            E.from_matrix(M=V_out)
            S.from_Jones(E)

            #L= cal_arclength(S)[0]
            #L = L * (1 + np.random.rand(1) * 0.01 - 0.005)  # 1% error including
            #length_S.append(L)
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
        print(length_S)
        print(l_measured)


        fig, ax = plt.subplots(figsize=(5, 3))

        Ip_measured = l_measured / 4 / V
        sensitivity = Ip_measured / MaxIp
        errV = abs(sensitivity - 1)
        ax.plot(azi * 180 / pi * 2, sensitivity, label='with SOP uncertainty')

        # Calibration current noise
        l_measured = l_measured * (1 + np.random.rand(len(l_measured)) * 0.1 - 0.05)
        Ip_measured = l_measured / 4 / V
        sensitivity = Ip_measured / MaxIp
        errV = abs(sensitivity - 1)

        ax.plot(azi * 180 / pi * 2, sensitivity, label='with Ical. uncertainty')

        outdict0 = {'azi': azi, 'l': l_measured}
        df0 = pd.DataFrame(outdict0)
        df0.to_csv(strfile.split('.')[0] + '_xy.' + strfile.split('.')[1], index=False, mode='w',
                   header=not os.path.exists(strfile))

        # ax.plot(azi*180/pi*2, errV)
        ax.set_xlabel('azimuth angle [deg]')
        ax.set_ylabel('normalized sensitivity')
        ax.set(xlim=(0, 360), ylim=(0, 1.1))
        ax.legend(loc='lower right')
    elif mode == 4:
        strfile1 = 'scanning1D_noise_xy.csv'
        fig, ax = plt.subplots(figsize=(5,4))
        strfile2 = 'calibration_log.csv'
        show_result_cal_azimuth(strfile1, strfile2, fig, ax)

        #Stmp = create_Stokes('tmp')
        #fig, ax = Stmp.draw_poincare(figsize=(5, 5), angle_view=[4 * pi / 180, 124 * pi / 180], kind='line')
        #show_result_poincare(strfile2, Mci, Mco, fig[0], ax)
    elif mode == 5:

        #comparing noise included normalized sensitivity
        V = 0.7 * 4 * pi * 1e-7
        maxVI = 40e3
        fig, ax = plt.subplots(figsize=(5, 4))
        # draw strfile1
        strfile1 = 'scanning1D_noise_xy_2.5kA.csv'
        data = pd.read_csv(strfile1)
        azi = np.array(data['azi'])
        maxVI = 2.5e3
        l = np.array(data['l'])
        Ip = l / 4 / V
        sensitivity = Ip / maxVI
        ax.plot(azi * 180 / pi * 2, sensitivity, label='Ical = 2.5kA')

        strfile1 = 'scanning1D_noise_xy_5kA.csv'
        data = pd.read_csv(strfile1)
        azi = np.array(data['azi'])
        maxVI = 5e3
        l = np.array(data['l'])
        Ip = l / 4 / V
        sensitivity = Ip / maxVI
        ax.plot(azi * 180 / pi * 2, sensitivity, label='Ical = 5kA')

        strfile1 = 'scanning1D_noise_xy_10kA.csv'
        data = pd.read_csv(strfile1)
        azi = np.array(data['azi'])
        maxVI = 10e3
        l = np.array(data['l'])
        Ip = l / 4 / V
        sensitivity = Ip / maxVI
        ax.plot(azi * 180 / pi * 2, sensitivity, label='Ical = 10kA')

        strfile1 = 'scanning1D_noise_xy_40kA.csv'
        data = pd.read_csv(strfile1)
        azi = np.array(data['azi'])
        maxVI = 40e3
        l = np.array(data['l'])
        Ip = l / 4 / V
        sensitivity = Ip / maxVI
        ax.plot(azi * 180 / pi * 2, sensitivity, label='Ical = 40kA')

        ax.set_xlabel('azimuth angle [deg]')
        ax.set_ylabel('Normalized sensitivity')
        ax.set(xlim=(0, 360), ylim=(0, 2))
        ax.legend(loc="upper right")

    plt.show()

