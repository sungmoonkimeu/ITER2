import numpy as np
from scipy.interpolate import UnivariateSpline, Rbf, interp1d, CubicSpline
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pandas as pd
import os
print(os.getcwd())

try:
    sys.path.append(os.getcwd() + '\My_library')
    os.chdir(os.getcwd() + '\My_library')
    print("os path is changed")
except:
    print("os path is changed")

# Function that will convert any given function 'f' defined in a given range '[li,lf]' to a periodic function of period 'lf-li'
def periodicf(li, lf, f, x):
    if x >= li and x <= lf:
        return f(x)
    elif x > lf:
        x_new = x - (lf - li)
        return periodicf(li, lf, f, x_new)
    elif x < (li):
        x_new = x + (lf - li)
        return periodicf(li, lf, f, x_new)

if __name__ == "__main__":
    data = pd.read_csv('VVtemp.csv', delimiter=';')
    l_vv = data['L'].to_numpy()/100
    temp_vv = data['TEMP'].to_numpy()

    #interp_func = interp1d(l_vv, temp_vv, 'cubic')
    interp_func = CubicSpline(l_vv, temp_vv)

    x = np.arange(0, 0.2, 0.001)
    y = interp_func(x)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(l_vv, temp_vv, label='raw data')
    ax.plot(x, y, label='interpolated data ' )
    ax.set_xlabel('Fiber position (m)')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='best')

    li = 0
    lf = 0.2
    step_size = 0.001
    #
    x_l = 0
    x_u = 28

    x = np.arange(x_l, x_u, step_size)
    #y1 = [temp_dist_VV(li, lf, interp_func, xi) for xi in x]
    y1 = np.array([periodicf(li, lf, interp_func, xi) for xi in x])
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x,y1)
    ax.set_xlabel('Fiber position (m)')
    ax.set_ylabel('Temperature (K)')
    #
    # x_plot = []
    # y_plot1 = []
    #
    # x_l_plot = x_l - 15
    # x_u_plot = x_l_plot + 20
    # plt.xlim(x_l_plot, x_u_plot)
    # plt.ylim(-6, 7)
    #
    # for i in range(x.size):
    #     x_plot.append(x[i])
    #     y_plot1.append(y1[i])
    #
    #     # Sawtooth
    #     plt.plot(x_plot, y_plot1, c='darkkhaki')
    #     x_l_plot = x_l_plot + step_size
    #     x_u_plot = x_u_plot + step_size
    #     plt.xlim(x_l_plot, x_u_plot)
    #     plt.pause(0.01)
    plt.show()