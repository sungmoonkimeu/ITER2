#todo Uncertainty of system
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
from py_pol.drawings import draw_stokes_points, draw_poincare, draw_ellipse, draw_empty_sphere
import matplotlib.pylab as pl
from matplotlib.colors import rgb2hex
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation, PillowWriter

import pandas as pd
import os
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


def create_gif(gif_name = None):

    if gif_name is None:
        gif_name = 'mygif.gif'

    file_list = glob.glob('*.png')  # Get all the pngs in the current directory
    file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for img in file_list:
        images.append(imageio.imread(img))
        os.remove(img)
    imageio.mimsave(gif_name, images, fps=5)

def eval_result3_1D(strfile, Mci, Mco, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=data['x'], ellipticity=0)
    S = create_Stokes('output')

    V_out = np.einsum('...i,jk->ijk', ones(len(E0)) * 1j, np.mat([[0], [0]]))
    for mm, val in enumerate(E0):
        V_out[mm] = Mco @ Mci @ E0[mm].parameters.matrix()

    E1.from_matrix(V_out)
    S.from_Jones(E1)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x'])))
    for nn in range(len(data['x'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        if nn > 0:
            x0 = S[nn].parameters.matrix()
            print(x0[1])
            arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=rgb2hex(colors[nn]), shrinkA=0, shrinkB=0)
            a = Arrow3D([o[1][0], x0[1][0]], [o[2][0], x0[2][0]], [o[3][0], x0[3][0]], **arrow_prop_dict)
            ax.add_artist(a)
            plt.savefig(str(nn)+'.png')
        o = S[nn].parameters.matrix()
    create_gif('mygif2.gif')

def eval_result3(strfile, Mci, Mco, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])
    S = create_Stokes('output')

    V_out = np.einsum('...i,jk->ijk', ones(len(E0)) * 1j, np.mat([[0], [0]]))
    for mm, val in enumerate(E0):
        V_out[mm] = Mco @ Mci @ E0[mm].parameters.matrix()

    E1.from_matrix(V_out)
    S.from_Jones(E1)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x0'])))
    for nn in range(len(data['x0'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        if nn > 0:
            x0 = S[nn].parameters.matrix()
            print(x0[1])
            arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=rgb2hex(colors[nn]), shrinkA=0, shrinkB=0)
            a = Arrow3D([o[1][0], x0[1][0]], [o[2][0], x0[2][0]], [o[3][0], x0[3][0]], **arrow_prop_dict)
            ax.add_artist(a)
            plt.savefig(str(nn)+'.png')
        o = S[nn].parameters.matrix()
    create_gif('mygif2.gif')

def eval_result2(strfile, Mci, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])
    S = create_Stokes('output')

    V_out = np.einsum('...i,jk->ijk', ones(len(E0)) * 1j, np.mat([[0], [0]]))
    for mm, val in enumerate(E0):
        V_out[mm] = Mci @ E0[mm].parameters.matrix()

    E1.from_matrix(V_out)
    S.from_Jones(E1)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x0'])))
    for nn in range(len(data['x0'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        if nn > 0:
            x0 = S[nn].parameters.matrix()
            print(x0[1])
            arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=rgb2hex(colors[nn]), shrinkA=0, shrinkB=0)
            a = Arrow3D([o[1][0], x0[1][0]], [o[2][0], x0[2][0]], [o[3][0], x0[3][0]], **arrow_prop_dict)
            ax.add_artist(a)
            plt.savefig(str(nn)+'.png')
        o = S[nn].parameters.matrix()
    create_gif()

def eval_result2_1D(strfile, Mci, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=data['x'], ellipticity=0)
    S = create_Stokes('output')

    V_out = np.einsum('...i,jk->ijk', ones(len(E0)) * 1j, np.mat([[0], [0]]))
    for mm, val in enumerate(E0):
        V_out[mm] = Mci @ E0[mm].parameters.matrix()

    E1.from_matrix(V_out)
    S.from_Jones(E1)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x'])))
    for nn in range(len(data['x'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        if nn > 0:
            x0 = S[nn].parameters.matrix()
            print(x0[1])
            arrow_prop_dict = dict(mutation_scale=15, arrowstyle='-|>', color=rgb2hex(colors[nn]), shrinkA=0, shrinkB=0)
            a = Arrow3D([o[1][0], x0[1][0]], [o[2][0], x0[2][0]], [o[3][0], x0[3][0]], **arrow_prop_dict)
            ax.add_artist(a)
            plt.savefig(str(nn)+'.png')
        o = S[nn].parameters.matrix()
    #create_gif()

def eval_result(strfile):

    data = pd.read_csv(strfile)
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(data['errV']*100, label='Err = (Vmeas - V)/V ')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Error (%)')
    ax.legend()

def simplex_trace2(strfiel, ax,fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])
    S = create_Stokes('output')
    Stmp = create_Stokes('tmp')

    S.from_Jones(E0)

    numpnt = [0, 1, 2]

    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x0'])))
    # pl.cm.
    # todo to replace the way showing simplex triangle on the sphere

    u = np.linspace(0, 2 * np.pi, 61)  # azimuth

    for nn in range(len(data['x0']) - 3):

        if nn > 1:
            plt.cla()
            draw_empty_sphere(ax, angle_view=[24 * pi / 180, 31 * pi / 180])
            ax.plot([xp0[0], xp1[0]], [xp0[1], xp1[1]], [xp0[2], xp1[2]], color=colors[nn-1])
            draw_stokes_points(ax, S[xnumpnt[0]], kind='scatter', color_scatter=rgb2hex(colors[nn-1]))
            ax.plot([xp1[0], xp2[0]], [xp1[1], xp2[1]], [xp1[2], xp2[2]], color=colors[nn - 1])
            draw_stokes_points(ax, S[xnumpnt[1]], kind='scatter', color_scatter=rgb2hex(colors[nn - 1]))
            ax.plot([xp2[0], xp0[0]], [xp2[1], xp0[1]], [xp2[2], xp0[2]], color=colors[nn - 1])
            draw_stokes_points(ax, S[xnumpnt[2]], kind='scatter', color_scatter=rgb2hex(colors[nn - 1]))

        ax.plot(np.sin(u)*1.1, np.cos(u)*1.1, np.zeros_like(u), 'g-.', linewidth=2)  # equator

        azi0 = data['x0'][numpnt[0]] * 2
        azi1 = data['x0'][numpnt[1]] * 2
        ell0 = data['x1'][numpnt[0]] * 2
        ell1 = data['x1'][numpnt[1]] * 2
        p0 = np.array([cos(azi0) * sin(pi / 2 - ell0), sin(azi0) * sin(pi / 2 - ell0), cos(pi / 2 - ell0)])
        p1 = np.array([cos(azi1) * sin(pi / 2 - ell1), sin(azi1) * sin(pi / 2 - ell1), cos(pi / 2 - ell1)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[nn])
        draw_stokes_points(ax, S[numpnt[0]], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        xp0 = p0
        '''
            V_azi = np.linspace(azi0, azi1, 100)
            V_ell = np.linspace(ell0,ell1,100)
            p = np.array([np.cos(V_azi)*np.sin(pi/2-V_ell), np.sin(V_azi)*np.sin(pi/2-V_ell), np.cos(pi/2-V_ell)])
            ax.plot(p[0,:], p[1,:], p[2,:], 'k--')
        '''
        azi0 = data['x0'][numpnt[1]] * 2
        azi1 = data['x0'][numpnt[2]] * 2
        ell0 = data['x1'][numpnt[1]] * 2
        ell1 = data['x1'][numpnt[2]] * 2
        p0 = np.array([cos(azi0) * sin(pi / 2 - ell0), sin(azi0) * sin(pi / 2 - ell0), cos(pi / 2 - ell0)])
        p1 = np.array([cos(azi1) * sin(pi / 2 - ell1), sin(azi1) * sin(pi / 2 - ell1), cos(pi / 2 - ell1)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[nn])
        draw_stokes_points(ax, S[numpnt[1]], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        xp1 = p0

        azi0 = data['x0'][numpnt[2]] * 2
        azi1 = data['x0'][numpnt[0]] * 2
        ell0 = data['x1'][numpnt[2]] * 2
        ell1 = data['x1'][numpnt[0]] * 2
        p0 = np.array([cos(azi0) * sin(pi / 2 - ell0), sin(azi0) * sin(pi / 2 - ell0), cos(pi / 2 - ell0)])
        p1 = np.array([cos(azi1) * sin(pi / 2 - ell1), sin(azi1) * sin(pi / 2 - ell1), cos(pi / 2 - ell1)])
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[nn])
        draw_stokes_points(ax, S[numpnt[2]], kind='scatter', color_scatter=rgb2hex(colors[nn]))

        xp2 = p0

        # print(data['L'][numpnt[0]],data['L'][numpnt[1]],data['L'][numpnt[2]])
        xnumpnt = numpnt
        minindex = np.argmin(data['L'][numpnt])
        numpnt[minindex] = nn + 3
        plt.savefig(str(nn) + '.png')
    create_gif('mygif3.gif')

def simple_trace(strfile, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])
    S = create_Stokes('output')
    S.from_Jones(E0)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x0'])))
    for nn in range(len(data['x0'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))
    ax.set_axis_off()

def simple_trace1D(strfile, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E0.general_azimuth_ellipticity(azimuth=data['x'], ellipticity=0)
    S = create_Stokes('output')
    S.from_Jones(E0)
    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x'])))
    for nn in range(len(data['x'])):
        draw_stokes_points(ax, S[nn], kind='scatter', color_scatter=rgb2hex(colors[nn]))
    ax.set_axis_off()

def simplex_trace(strfile, ax, fig):
    data = pd.read_csv(strfile)
    E0 = Jones_vector('input')
    E0.general_azimuth_ellipticity(azimuth=data['x0'], ellipticity=data['x1'])

    numpnt = [0,1,2]

    colors = pl.cm.BuPu(np.linspace(0.2, 1, len(data['x0'])))
    #pl.cm.
    # todo to replace the way showing simplex triangle on the sphere
    for nn in range(len(data['x0'])-3):

        azi0 = data['x0'][numpnt[0]]*2
        azi1 = data['x0'][numpnt[1]]*2
        ell0 = data['x1'][numpnt[0]]*2
        ell1 = data['x1'][numpnt[1]]*2
        p0 = np.array([cos(azi0)*sin(pi/2-ell0), sin(azi0)*sin(pi/2-ell0), cos(pi/2-ell0)])
        p1 = np.array([cos(azi1)*sin(pi/2-ell1), sin(azi1)*sin(pi/2-ell1), cos(pi/2-ell1)])
        ax.plot( [p0[0],p1[0]],[p0[1],p1[1]], [p0[2],p1[2]], color=colors[nn])
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
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[nn])

        azi0 = data['x0'][numpnt[2]]*2
        azi1 = data['x0'][numpnt[0]]*2
        ell0 = data['x1'][numpnt[2]]*2
        ell1 = data['x1'][numpnt[0]]*2
        p0 = np.array([cos(azi0)*sin(pi/2-ell0), sin(azi0)*sin(pi/2-ell0), cos(pi/2-ell0)])
        p1 = np.array([cos(azi1)*sin(pi/2-ell1), sin(azi1)*sin(pi/2-ell1), cos(pi/2-ell1)])
        ax.plot( [p0[0],p1[0]],[p0[1],p1[1]], [p0[2],p1[2]], color=colors[nn])

        #print(data['L'][numpnt[0]],data['L'][numpnt[1]],data['L'][numpnt[2]])
        minindex = np.argmin(data['L'][numpnt])

        numpnt[minindex] = nn+3
        plt.savefig(str(nn) + '.png')

    ax.set_axis_off()

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


def create_M(r, omega, phi):

    M_rot = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    M_theta = np.array([[cos(omega), -sin(omega)], [sin(omega), cos(omega)]])
    M_phi = np.array([[exp(1j * r/2), 0], [0, exp(-1j * r/2)]])

    return M_rot @ M_theta.T @ M_phi @ M_theta


def f(x, Mci, Mco, strfile):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    #E0.general_azimuth_ellipticity(azimuth=x[0], ellipticity=x[1])
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
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
    S = create_Stokes('output')
    S.from_Jones(E1)

    #print(S.parameters.ellipticity_angle()[0])

    #draw_stokes_points(fig[0], S  , kind='line', color_scatter='k')
    #draw_stokes_points(fig[0], S[0], kind='scatter', color_scatter='b')
    #draw_stokes_points(fig[0], S[-1], kind='scatter', color_scatter='r')
    #print(S.parameters.azimuth()[-1])
    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff-V)/V)
    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    outdict = {'x': x, 'L': np.array(L), 'errV': np.array(errV)}
    df = pd.DataFrame(outdict)
    df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    return errV


def f2(x, Mci, Mco):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    E0.general_azimuth_ellipticity(azimuth=x, ellipticity=0)
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
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff-V)/V)
    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV


def f3(x, Mci, Mco):
    E0 = Jones_vector('input')
    E1 = Jones_vector('output')
    noise = np.random.rand(1)*0 * pi/180
    E0.general_azimuth_ellipticity(azimuth=x+noise, ellipticity=0)

    noise2 = 1+np.random.rand(1) * 0.01
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
    S = create_Stokes('output')
    S.from_Jones(E1)

    L = cal_arclength(S)*noise2    # Arc length is orientation angle psi -->
    Veff = L/2/(MaxIp*2)    # Ip = V * psi *2 (Pol. rotation angle is 2*psi)
    errV = abs((Veff-V)/V)
    #Lazi = S.parameters.azimuth()[-1]-S.parameters.azimuth()[0]
    #print("E=", E0.parameters.matrix()[0], E0.parameters.matrix()[1], "arc length= ", L, "Veff = ", Veff, "V=", V, "errV=", errV)

    return errV


if __name__ == '__main__':

    ## first step
    '''
    strfile = 'calibration1D_log.csv'
    if os.path.exists(strfile):
        os.remove(strfile)

    # Define Minput Moutput
    theta0, phi0, theta_e0 = np.random.rand(3)*360
    Mci = create_M(theta0*pi/180, phi0*pi/180, theta_e0*pi/180)

    theta1, phi1, theta_e1 = np.random.rand(3)*360
    Mco = create_M(theta1*pi/180, phi1*pi/180, theta_e1*pi/180)

    # initial point
    init_polstate = np.array([[0], [pi / 4]])

    out = optimize.fmin(f, 0, (Mci, Mco, strfile), maxiter=20, xtol=1, ftol=0.0001,
                            initial_simplex=init_polstate, retall=True, full_output=1)

    #print("minimum[0]", minimum[0])
    print(out[3])

    Stmp = create_Stokes('tmp')
    fig, ax = Stmp.draw_poincare(figsize=(7, 7), angle_view=[24 * pi / 180, 31 * pi / 180], kind='line')

    eval_result(strfile)
    eval_result3_1D(strfile, Mci, Mco, fig[0], ax)
    '''
    ''' 
    ## 2nd step

    start = pd.Timestamp.now()
    strfile = 'Multiple_Calibration.csv'

    # Define Minput Moutput

    n_iter = 100
    n_iter2 = 100
    fig, ax = plt.subplots(figsize=(6, 6))
    for mm in range(n_iter2):

        v_out = np.zeros(n_iter)
        for nn in range(n_iter):
            theta0, phi0, theta_e0 = np.random.rand(3)*360
            Mci = create_M(theta0*pi/180, phi0*pi/180, theta_e0*pi/180)

            theta1, phi1, theta_e1 = np.random.rand(3)*360
            Mco = create_M(theta1*pi/180, phi1*pi/180, theta_e1*pi/180)

            # initial point
            init_polstate = np.array([[0], [pi / 4]])

            fmin_result = optimize.fmin(f2, 0, (Mci, Mco), maxiter=50, xtol=1, ftol=0.0001,
                                initial_simplex=init_polstate, retall=True, full_output=1)

            v_out[nn] = fmin_result[3]


        ax.plot(v_out)

        outdict = {'out': v_out}
        df = pd.DataFrame(outdict)
        df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))

    end = pd.Timestamp.now()
    print("Total time = ", (end-start).total_seconds())

    '''

    ## 3rd step

    start = pd.Timestamp.now()
    strfile = 'Multiple_Calibration_withnoise.csv'

    # Define Minput Moutput

    n_iter = 10
    n_iter2 = 1000
    fig, ax = plt.subplots(figsize=(6, 6))
    for mm in range(n_iter2):

        v_out = np.zeros(n_iter)
        for nn in range(n_iter):
            theta0, phi0, theta_e0 = np.random.rand(3) * 360
            Mci = create_M(theta0 * pi / 180, phi0 * pi / 180, theta_e0 * pi / 180)

            theta1, phi1, theta_e1 = np.random.rand(3) * 360
            Mco = create_M(theta1 * pi / 180, phi1 * pi / 180, theta_e1 * pi / 180)

            # initial point
            init_polstate = np.array([[0], [pi / 4]])

            fmin_result = optimize.fmin(f3, 0, (Mci, Mco), maxiter=100, xtol=1, ftol=0.0001,
                                        initial_simplex=init_polstate, retall=True, full_output=1)

            v_out[nn] = fmin_result[3]

        ax.plot(v_out)

        outdict = {'out': v_out}
        df = pd.DataFrame(outdict)
        df.to_csv(strfile, index=False, mode='a', header=not os.path.exists(strfile))
        ax.plot(v_out)

    end = pd.Timestamp.now()
    print("Total time = ", (end - start).total_seconds())


plt.show()

