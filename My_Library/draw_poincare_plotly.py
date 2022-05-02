# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:50:03 2022

@author: agoussar
Updated on Apr 26 2022
SMK has added another PS drawing method with plotly library
Please install plotly before running code

    plotly may be installed using pip:
    $ pip install plotly==5.7.0
    or conda:
    $ conda install -c plotly plotly=5.7.0

more information about plotly is here (https://plotly.com/python/getting-started/)

"""

import pickle
import scipy.io
import scipy.optimize
from scipy import stats
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time
from itertools import chain
import warnings
import plotly.graph_objects as go
import plotly.express as px

#plt.style.use('seaborn-whitegrid')
#plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", message="This figure includes Axes")
warnings.filterwarnings("ignore", message="This figure was using")
warnings.filterwarnings("ignore", message="Calling figure.constrained_layout")


def PS3(shot):
    '''
    plot Poincare Sphere, ver. 20/03/2020
    return:
    ax, fig
    '''
    fig = plt.figure(figsize=(6, 6))
    #    plt.figure(constrained_layout=True)
    ax = Axes3D(fig)
    # white panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # no ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # no panes
    ax.set_axis_off()

    # plot greed
    u = np.linspace(0, 2 * np.pi, 61)  # azimuth
    v = np.linspace(0, np.pi, 31)  # elevation
    sprad = 1
    x = sprad * np.outer(np.cos(u), np.sin(v))
    y = sprad * np.outer(np.sin(u), np.sin(v))
    z = sprad * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,
                    color='w',  # (0.5, 0.5, 0.5, 0.0),
                    edgecolor='k',
                    linestyle=(0, (5, 5)),
                    rstride=3, cstride=3,
                    linewidth=.5, alpha=0.0, shade=0, zorder=1)

    # main circles
    ax.plot(np.sin(u), np.cos(u), np.zeros_like(u), 'g-.', linewidth=0.75, zorder=0)  # equator
    #    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), 'b-', linewidth=0.5)
    #    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), 'b-', linewidth=0.5)

    # axes and captions
    amp = 1.3 * sprad
    ax.plot([-amp, amp], [0, 0], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=0)
    ax.plot([0, 0], [-amp, amp], [0, 0], 'k-.', lw=1, alpha=0.5, zorder=0)
    ax.plot([0, 0], [0, 0], [-amp, amp], 'k-.', lw=1, alpha=0.5, zorder=0)

    distance = 1.3 * sprad
    ax.text(distance, 0, 0, '$S_1$', fontsize=18)
    ax.text(0, distance, 0, '$S_2$', fontsize=18)
    ax.text(0, 0, distance, '$S_3$', fontsize=18)

    # points
    px = [1, -1, 0, 0, 0, 0]
    py = [0, 0, 1, -1, 0, 0]
    pz = [0, 0, 0, 0, 1, -1]

    ax.plot(px, py, pz,
            color='black', marker='o', markersize=4, alpha=1.0, linewidth=0, zorder=22)
    #

    max_size = 1.05 * sprad
    ax.set_xlim(-max_size, max_size)
    ax.set_ylim(-max_size, max_size)
    ax.set_zlim(-max_size, max_size)

    #    plt.tight_layout()            #not compatible
    ax.set_box_aspect([1, 1, 1])  # for the same aspect ratio

    ax.view_init(elev=90 / np.pi, azim=90 / np.pi)
    #    ax.view_init(elev=0/np.pi, azim=0/np.pi)

    #    ax.set_title(label = shot, loc='left', pad=10)
    ax.set_title(label="  " + shot, loc='left', pad=-10, fontsize=8)

    #    ax.legend()

    return ax, fig


def PS5(alpha=0.5):
    '''
    plot Poincare Sphere, ver. 26/04/2022
    return:
    fig
    '''

    u = np.linspace(0, 2 * np.pi, 61)  # azimuth
    v = np.linspace(0, np.pi, 31)  # elevation
    sprad = 1
    x = sprad * np.outer(np.cos(u), np.sin(v))
    y = sprad * np.outer(np.sin(u), np.sin(v))
    z = sprad * np.outer(np.ones(np.size(u)), np.cos(v))
    print(x)
    color1 = 'whitesmoke'
    color2 = 'red'
    fig = go.Figure()
    colorscale = [[0, color1],[0.5,color1],[1, color1]]
    fig.add_surface(x=x, y=y, z=z, opacity=alpha, showscale=False, colorscale=colorscale,
                    showlegend=False, lighting=dict(ambient=1))
    #fig.add_surface(x=x, y=y, z=z, opacity=0.2, showscale=False)
    #fig.update(layout_coloraxis_showscale=False)

    sprad = 1
    x = (sprad * np.outer(np.cos(u), np.sin(v)))[::3]
    y = (sprad * np.outer(np.sin(u), np.sin(v)))[::3]
    z = (sprad * np.outer(np.ones(np.size(u)), np.cos(v)))[::3]

    line_marker = dict(color='#000000', width=4, dash='dot')
    for xx, yy, zz in zip(x,y,z):
        fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='',showlegend=False)

    u = np.linspace(0, np.pi, 41)  # azimuth
    v = np.linspace(0, 2 * np.pi, 81)  # elevation
    sprad = 1
    x = (sprad * np.outer(np.sin(u), np.cos(v)))[::4]
    y = (sprad * np.outer(np.sin(u), np.sin(v)))[::4]
    z = (sprad * np.outer(np.cos(u), np.ones(np.size(v))))[::4]

    for xx, yy, zz in zip(x, y, z):
        fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='', showlegend=False)

    # axes and captions
    amp = 1.2 * sprad

    line_marker2 = dict(color='#000000', width=5, dash = 'dashdot')
    fig.add_scatter3d(x=[-amp, amp], y=[0, 0], z=[0, 0], mode='lines', line=line_marker2,
                      name='', showlegend=False)
    fig.add_scatter3d(x=[0, 0], y=[-amp, amp], z=[0, 0], mode='lines', line=line_marker2,
                      name='', showlegend=False)
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[-amp, amp], mode='lines', line=line_marker2,
                      name='', showlegend=False)

    distance = 1.2 * sprad
    fig.add_scatter3d(x=[distance],y=[0],z=[0], text='S<sub>1</sub>',
                      mode="text",
                      textposition="top left",
                      textfont=dict(size=24),showlegend=False)
    fig.add_scatter3d(x=[0], y=[distance], z=[0], text='S<sub>2</sub>',
                      mode="text",
                      textposition="top left",
                      textfont=dict(size=24),showlegend=False)
    fig.add_scatter3d(x=[0], y=[0], z=[distance], text='S<sub>3</sub>',
                      mode="text",
                      textposition="middle right",
                      textfont=dict(size=24),showlegend=False)

    fig.add_scatter3d(x=[-sprad, sprad], y=[0, 0], z=[0, 0], mode="markers",
                      marker=dict(color='#000000', size=2),showlegend=False)
    fig.add_scatter3d(x=[0, 0], y=[sprad, -sprad], z=[0, 0], mode="markers",
                      marker=dict(color='#000000', size=2),showlegend=False)
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[sprad, -sprad], mode="markers",
                      marker=dict(color='#000000', size=2),showlegend=False)

    fig.update_layout(
        scene={
            "xaxis": {"showbackground": False, "showticklabels":False, "visible":False},
            "yaxis": {"showbackground": False, "showticklabels":False, "visible":False},
            "zaxis": {"showbackground": False, "showticklabels":False, "visible":False},
            'camera_eye': {"x": 0, "y": -1, "z": 1},
            "aspectratio": {"x": 1, "y": 1, "z": 1}
        },
        showlegend=False,
    )
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    return fig

def main():
    #PS3('3')
    fig = PS5()
    #fig.show()

if (__name__ == "__main__"):
    fig = PS5()
    fig.show()
    #main()
    #plt.show()
