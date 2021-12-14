import numpy as np
from numpy import pi, array


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(r, az, el):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z])


th = 0
phi = pi/6
v0_s = array([1, th, phi])
v1_s = array([1, 0, 0])
v2_s = array([1, -th, -phi])

v0_c = sph2cart(v0_s[0], v0_s[1], v0_s[2])
v1_c = sph2cart(v1_s[0], v1_s[1], v1_s[2])
v2_c = sph2cart(v2_s[0], v2_s[1], v2_s[2])

v01_c = v1_c - v0_c
v02_c = v2_c - v1_c

print(v0_s, v0_c)
print(v1_s, v1_c)
print(v2_s, v2_c)
print(v01_c)
print(v02_c)
print(np.cross(v01_c, v02_c))



