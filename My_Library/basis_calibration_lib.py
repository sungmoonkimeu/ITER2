# Compensate the randomly rotated basis by adjusting the SOP rotation is aligned to the parallel to the equator.
# 1. Calculate the normal vector of SOP rotation
# 2. Basis rotation based on the axis 'R (S3)'
# 3. Another basis rotation base don the axis '+(S2)'

import numpy as np
from numpy import pi, cos, sin, ones, exp
from numpy.linalg import norm
from py_pol.jones_matrix import Jones_matrix
from py_pol.jones_vector import Jones_vector
from py_pol.mueller import Mueller
from py_pol.stokes import create_Stokes
from py_pol.drawings import draw_stokes_points
import matplotlib.pyplot as plt


def calib_basis1(S):
    a = S.parameters.matrix()[1:]  # convert 4x1 Stokes vectors to 3x1 cartesian vectors

    mean_a = np.array([a[0, :].sum(), a[1, :].sum(), a[2, :].sum()])
    mean_a = mean_a / (np.linalg.norm(mean_a))
    # 평균 벡터와 모든 점 사이의 거리
    dist_a_mean_a = np.linalg.norm(a.T - mean_a, axis=1)
    # 평균벡터와 가장 가까운 벡터 --> 대표 벡터 ?
    std_a = a[:, np.argmin(dist_a_mean_a)]
    # 대표 벡터 와 나머지 벡터 연결
    diff_a = a.T - std_a
    # 대표 벡터와 나머지 벡터가 이루는 벡터 끼리 외적
    cross_a = np.cross(diff_a[0], diff_a)

    # filtering out small vectors
    cross_a2 = cross_a[np.linalg.norm(cross_a, axis=1) > np.linalg.norm(cross_a, axis=1).mean() / 10]
    # 반대 방향 vector 같은 방향으로 변환
    cross_an = cross_a2.T / np.linalg.norm(cross_a2, axis=1)
    # Normalize
    cross_an_abs = cross_an * abs(cross_an.sum(axis=0)) / cross_an.sum(axis=0)
    # average after summation whole vectors
    c = cross_an_abs.sum(axis=1) / np.linalg.norm(cross_an_abs.sum(axis=1))

    # print("new c", c)
    # fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1, )

    z = [0, 0, 1]
    y = [0, 1, 0]
    x = [1, 0, 0]

    th_x = np.arccos(np.dot(x, c))
    th_y = np.arccos(np.dot(y, c))
    th_z = np.arccos(np.dot(z, c))
    # print("x=", th_x * 180 / pi, "y=", th_y * 180 / pi, "z=", th_z * 180 / pi)

    Rx = np.array([[cos(th_x), -sin(th_x), 0], [sin(th_x), cos(th_x), 0], [0, 0, 1]])
    Ry = np.array([[1, 0, 0], [0, cos(th_y), -sin(th_y)], [0, sin(th_y), cos(th_y)]])
    Rz = np.array([[cos(th_z), 0, sin(th_z)], [0, 1, 0], [-sin(th_z), 0, cos(th_z)]])

    th = th_x
    if th_y > pi / 2:
        th = -th_x
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

    th = th_z
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + 기준 rotation

    th = 0
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation

    TT = R45.T @ Rh.T @ Rr.T @ a
    zT = ones(np.shape(TT)[1])

    Sp = np.vstack((zT, TT))
    S.from_matrix(Sp)
    return S


def calib_basis2(S):  # first Point to -45 (S2)

    a = S.parameters.matrix()[1:]  # convert 4x1 Stokes vectors to 3x1 cartesian vectors
    c = a[..., 0]

    z = [0, 0, 1]
    y = [0, 1, 0]
    x = [1, 0, 0]

    th_x = np.arccos(np.dot(x, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_y = np.arccos(np.dot(y, [c[0], c[1], 0] / np.linalg.norm([c[0], c[1], 0])))
    th_z = np.arccos(np.dot(z, c))

    th = -th_y
    if th_x > pi / 2:
        th = th_y
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

    th = 0
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + 기준 rotation

    th = pi / 2 - th_z + pi
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation

    TT = R45.T @ Rh.T @ Rr.T @ a
    zT = ones(np.shape(TT)[1])

    Sp = np.vstack((zT, TT))
    S.from_matrix(Sp)
    return S


def calib_basis3(S):
    a = S.parameters.matrix()[1:]  # convert 4x1 Stokes vectors to 3x1 cartesian vectors

    init_a = np.array([a[0, 0], a[1, 0], a[2, 0]])
    # mean_a = np.array([a[0, :].sum(), a[1, :].sum(), a[2, :].sum()])
    # mean_a = mean_a / (np.linalg.norm(mean_a))
    # # 평균 벡터와 모든 점 사이의 거리
    # dist_a_mean_a = np.linalg.norm(a.T - mean_a, axis=1)
    # # 평균벡터와 가장 가까운 벡터 --> 대표 벡터 ?
    # std_a = a[:, np.argmin(dist_a_mean_a)]
    # # 첫번째 벡터 와 나머지 벡터 연결
    #diff_a = a.T - std_a
    diff_a = a.T - init_a
    # 대표 벡터와 나머지 벡터가 이루는 벡터 끼리 외적
    cross_a = np.cross(diff_a[1], diff_a[2:])

    # filtering out small vectors
    cross_a2 = cross_a[np.linalg.norm(cross_a, axis=1) > np.linalg.norm(cross_a, axis=1).mean() / 10]
    # 반대 방향 vector 같은 방향으로 변환
    cross_an = cross_a2.T / np.linalg.norm(cross_a2, axis=1)
    # Normalize
    cross_an_abs = cross_an * abs(cross_an.sum(axis=0)) / cross_an.sum(axis=0)
    # average after summation whole vectors
    c = cross_an_abs.sum(axis=1) / np.linalg.norm(cross_an_abs.sum(axis=1))

    # print("new c", c)
    # fig[0].plot([0, c[0]], [0, c[1]], [0, c[2]], 'r-', lw=1, )

    z = [0, 0, 1]
    y = [0, 1, 0]
    x = [1, 0, 0]

    th_x = np.arccos(np.dot(x, c))
    th_y = np.arccos(np.dot(y, c))
    th_z = np.arccos(np.dot(z, c))
    print("x=", th_x * 180 / pi, "y=", th_y * 180 / pi, "z=", th_z * 180 / pi)

    Rx = np.array([[cos(th_x), -sin(th_x), 0], [sin(th_x), cos(th_x), 0], [0, 0, 1]])
    Ry = np.array([[1, 0, 0], [0, cos(th_y), -sin(th_y)], [0, sin(th_y), cos(th_y)]])
    Rz = np.array([[cos(th_z), 0, sin(th_z)], [0, 1, 0], [-sin(th_z), 0, cos(th_z)]])
    #print(th_x, th_y, th_z)
    th = th_x
    if th_y > pi / 2:
        th = -th_x
    Rr = np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])  # S3, R 기준 rotation

    th = th_z
    R45 = np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])  # S2, + 기준 rotation

    th = 0
    Rh = np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])  # S1, H 기준 rotation

    TT = R45.T @ Rh.T @ Rr.T @ a
    zT = ones(np.shape(TT)[1])

    Sp = np.vstack((zT, TT))
    S.from_matrix(Sp)
    return S

if __name__ == '__main__':

    E = Jones_vector('Input')
    S = create_Stokes('Output')
    S2 = create_Stokes('cal')
    J1 = Jones_matrix('Random element')
    J2 = Jones_matrix('Random element')
    M = Mueller('cal')

    azi = np.arange(0, pi/0.5, 0.1)
    # ell = np.arange(0, pi/6, 0.1)
    E.general_azimuth_ellipticity(azimuth=azi, ellipticity=pi/12)
    S.from_Jones(E)
    fig, ax = S.draw_poincare(kind='line', color_line='k')

    phi0 = -pi/12
    Mp = np.array([[exp(1j*phi0/2), 0], [0, exp(-1j*phi0/2)]])
    phi1 = pi/12
    Mr = np.array([[cos(phi1), -sin(phi1)], [sin(phi1), cos(phi1)]])

    J1.from_matrix(Mr)
    J2.from_matrix(Mp)
    Out = J2*J1*E
    S.from_Jones(Out)
    # S.from_Jones(Out).draw_poincare()

    draw_stokes_points(fig[0], S, kind='line', color_line='r')
    S2 = calib_basis3(S)
    draw_stokes_points(fig[0], S2, kind='line', color_line='b')

    plt.show()
