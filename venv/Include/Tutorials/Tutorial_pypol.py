import numpy as np
from numpy import sin, cos, linspace, pi
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


from py_pol.jones_vector import Jones_vector, degrees
from py_pol.stokes import Stokes, create_Stokes

S = Stokes('Linear light')
E = Jones_vector('test')
E.linear_light(azimuth = linspace(0,1,5))
#S.from_distribution(Ex=a[0], Ey=b[0])
S.from_Jones(E)
#S.linear_light(azimuth=np.linspace(0,180,13)*degrees)
#print(S)
print(S[0][0])
fig, ax = S[0:5].draw_poincare(figsize=(10,10), angle_view=[0.5,-1])
plt.show()
