from scipy import optimize
import numpy as np
from numpy import pi, cos, sin, ones, zeros, einsum, arange, exp,arcsin, arctan, tan, arccos, savetxt
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse
import pandas as pd
import os

def simplex_trace(strfile, ax):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])

    numpnt = [0,1,2]

    for nn in range(len(data['x0']-5)):

        azi0 = data['x0'][numpnt[0]]*2
        azi1 = data['x0'][numpnt[1]]*2
        ell0 = data['x1'][numpnt[0]]*2
        ell1 = data['x1'][numpnt[1]]*2
        p0 = np.array([cos(azi0)*sin(pi/2-ell0), sin(azi0)*sin(pi/2-ell0), cos(pi/2-ell0)])
        p1 = np.array([cos(azi1)*sin(pi/2-ell1), sin(azi1)*sin(pi/2-ell1), cos(pi/2-ell1)])
        ax.plot( [p0[0],p1[0]],[p0[1],p1[1]], [p0[2],p1[2]])
        '''
            V_azi = np.linspace(azi0, azi1, 100)
            V_ell = np.linspace(ell0,ell1,100)
            p = np.array([np.cos(V_azi)*np.sin(pi/2-V_ell), np.sin(V_azi)*np.sin(pi/2-V_ell), np.cos(pi/2-V_ell)])
            ax.plot(p[0,:], p[1,:], p[2,:], 'k--')
        '''
        azi0 = data['x0'][numpnt[1]]*2
        azi1 = data['x0'][numpnt[2]]*2
        ell0 = data['x1'][numpnt[1]]*2
        ell1 = data['x1'][numpnt[2]]*2
        p0 = np.array([cos(azi0) * sin(pi / 2 - ell0), sin(azi0) * sin(pi / 2 - ell0), cos(pi / 2 - ell0)])
        p1 = np.array([cos(azi1) * sin(pi / 2 - ell1), sin(azi1) * sin(pi / 2 - ell1), cos(pi / 2 - ell1)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])

        azi0 = data['x0'][numpnt[2]]*2
        azi1 = data['x0'][numpnt[0]]*2
        ell0 = data['x1'][numpnt[2]]*2
        ell1 = data['x1'][numpnt[0]]*2
        p0 = np.array([cos(azi0)*sin(pi/2-ell0), sin(azi0)*sin(pi/2-ell0), cos(pi/2-ell0)])
        p1 = np.array([cos(azi1)*sin(pi/2-ell1), sin(azi1)*sin(pi/2-ell1), cos(pi/2-ell1)])
        ax.plot( [p0[0],p1[0]],[p0[1],p1[1]], [p0[2],p1[2]])
        print(data['L'][numpnt])
        minindex = np.argmin(data['L'][numpnt])
        print(minindex, data['L'][minindex])
        numpnt[minindex] = nn+3

    #ax.view_init(elev=-150, azim=60)
    ax.set_axis_off()
    #plt.sho`w()

def cal_arclength(S):
    L = 0
    for nn in range(len(S)-1):
        c = pi / 2 - S.parameters.ellipticity_angle()[nn]
        b = pi / 2 - S.parameters.ellipticity_angle()[nn+1]
        #gg
        A0 = S.parameters.azimuth()[nn]
        A1 = S.parameters.azimuth()[nn+1]
        A = A1-A0
        if A < 0:
            A = A+pi
        L = L + arccos(cos(b) * cos(c) + sin(b) * sin(c) * cos(A))
    return L

def create_M(r, omega, phi):

    M_rot = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    M_theta = np.array([[cos(omega), -sin(omega)], [sin(omega), cos(omega)]])
    M_phi = np.array([[exp(1j * r/2), 0], [0, exp(-1j * r/2)]])

    return M_rot @ M_theta.T @ M_phi @ M_theta

def f(x, Mci, Mco, fig, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=x[0], ellipticity=x[1])
    V = 0.7 * 4 * pi * 1e-7
    MaxIp = 40e3
    dIp = MaxIp/100
    V_Ip = arange(0e6,MaxIp+dIp,dIp)
    V_out = np.einsum('...i,jk->ijk', ones(len(V_Ip)) * 1j, np.mat([[0], [0]]))

    for mm, iter_I in enumerate(V_Ip):
        # Faraday rotation matirx
        th_FR = iter_I * V*2
        M_FR = np.array([[cos(th_FR), -sin(th_FR)], [sin(th_FR), cos(th_FR)]])
        V_out[mm] = Mco @ M_FR @ Mci @ E0.parameters.matrix()

    E1.from_matrix(M=V_out)
    #print(E1)
    S = create_Stokes('output')
    S.from_Jones(E1)

    #print(S.parameters.ellipticity_angle()[0])

    draw_stokes_points(fig[0], S, kind='line', color_scatter='k')
    draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter='b')
    draw_stokes_points(fig[0], S[-1], kind='scatter', color_scatter='r')
    #print(S.parameters.azimuth()[-1])
    L = cal_arclength(S)
    Veff = L/(2*MaxIp)
    errV = abs((Veff-V)/V)
    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    print("arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    outdict = {'x0': np.array([x[0]]), 'x1': np.array([x[1]]), 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    return errV

if __name__ == '__main__':
    strfile = 'calibration_log.csv'
    mode = 1
    if mode == 0:

        #Mci = create_M(pi/6,pi/8,pi/10)
        Mci = create_M(pi/6,pi/8,pi/10)
        Mco = create_M(pi/6,pi/8,pi/10)
        #Mco = create_M(0,0,0)
        init_polstate = np.array([[0,0], [pi/4,0], [pi/4, pi/4]])
        Stmp = create_Stokes('tmp')


        fig, ax = Stmp.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')

        minimum = optimize.fmin(f, [0,0], (Mci, Mco, fig, strfile), maxiter=20, xtol=1, ftol=0.001, initial_simplex=init_polstate, retall=True)

        print(minimum[0])

    elif mode ==1:
        Stmp = create_Stokes('tmp')
        fig, ax = Stmp.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        simplex_trace(strfile, fig[0])
        # show figure

    #track = minimum[1]
    #fig2, ax = Stmp.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')
    #for nn, val in enumerate(track):
    #    f(val, Mci, Mco, fig2)
    plt.show()

# todo 1. to correct the length calculation (when Mco != 0 --> error!)
# todo 2. to save the iteration in a file
# todo 3. to show the iteration in the poincare sphere


