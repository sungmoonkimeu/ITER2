import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from numpy import pi
from scipy.signal import butter, filtfilt
import pandas as pd

time = np.linspace(0,200,201)
azi = np.linspace(0, pi, 21)
ele = np.linspace(-pi/2, pi/2, 21)
y0 = []
for nn in range(201):
    y0 = [y0, ]
y0 = np
fig, ax = plt.subplot(2,1,1)


