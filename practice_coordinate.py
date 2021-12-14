import numpy as np
from numpy import pi, array
from oct2py import octave

th = pi/6
phi = pi/6
v0_s = array([1, th, phi])
v1_s = array([1, 0, 0])

v0_c = octave.sph2cart(v0_s)
v1_c = octave.sph2cart(v0_s)

print(v0_s, v0_c)
print(v0_s, v0_c)


