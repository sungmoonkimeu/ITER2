import numpy as np
from numpy import pi, array


def cart2sph(x, y, z):
    # r, theta, phi cooridinate
    # theta is inclination angle between the zenith direction and the line segment (0,0,0) - (x,y,z)
    # phi is the azimuthal angle between the azimuth reference direction and
    # the orthogonal projection of the line segment OP on the reference plane.

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def sph2cart(r, theta, phi):

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


if __name__ == '__main__':
    th = pi/2
    phi = pi/4
    v0_s = array([1, th, phi])
    v1_s = array([1, pi/2, 0])
    v2_s = array([1, th, -phi])

    v0_c = sph2cart(v0_s[0], v0_s[1], v0_s[2])
    v1_c = sph2cart(v1_s[0], v1_s[1], v1_s[2])
    v2_c = sph2cart(v2_s[0], v2_s[1], v2_s[2])

    v01_c = v1_c - v0_c
    v02_c = v2_c - v1_c

    print('V0_sphe = ', v0_s, 'V0_cart = ', v0_c)
    print('V1_sphe = ', v1_s, 'V1_cart = ', v1_c)
    print('V2_sphe = ', v2_s, 'V2_cart = ', v2_c)

    print('V01_cart = ', v01_c)
    print('V02_cart = ', v02_c)

    print(np.cross(v01_c, v02_c))