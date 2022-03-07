import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from numpy import pi
from scipy.signal import butter, filtfilt
import pandas as pd

'''
# Switching OS folder
path2 = 'C:/Users/Iter/PycharmProjects/loaddata'
path1 = 'C:/Users/SMK/PycharmProjects/loaddata'
import os
def switch_osfolder():
    try:
        if os.path.exists(path1):
            os.chdir(path1)
        else:
            os.chdir(path2)
    except OSError:
        print('Error: Changing OS directory')

switch_osfolder()
'''

def rc_low_pass(x_new, y_old, sample_rate_hz, lowpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*lowpass_cutoff_hz)
    alpha = dt/(rc + dt)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new


def rc_high_pass(x_new, x_old, y_old, sample_rate_hz, highpass_cutoff_hz):
    dt = 1/sample_rate_hz
    rc = 1/(2*np.pi*highpass_cutoff_hz)
    alpha = rc/(rc + dt)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new


def rc_filters(xs, sample_rate_hz,
               highpass_cutoff_hz,
               lowpass_cutoff_hz):
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in xs:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz,
                                   highpass_cutoff_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz,
                                 lowpass_cutoff_hz)
        x_prev = x
        yield y_prev_high, y_prev_low

if __name__ == "__main__":
    """
    # RC filters for continuous signals
    """
    sample_rate = 1000
    duration_points = 1000*18
    sec_duration = duration_points/sample_rate

    frequency_low = 1/12
    #frequency_high = 100

    # Design the cutoff
    #number_octaves = 3
    highpass_cutoff = 100
    tc = 0.5 # time constant 0.5 s
    fc = 1/tc/(2*pi)
    lowpass_cutoff = fc

    print('Sample rate, Hz:', sample_rate)
    print('Record duration, s:', sec_duration)
    #print('Low, high tone frequency:', frequency_low, frequency_high)

    time_s = np.arange(duration_points)/sample_rate

    #sig = np.sin(2*np.pi*frequency_low*time_s) + \
    #      np.sin(2*np.pi*frequency_high*time_s)
    sig = -0.5 * signal.square(2 * pi * 1/12 * time_s) + 0.5

    filt_signals = np.array([[high, low]
                             for high, low in
                             rc_filters(sig, sample_rate,
                                        highpass_cutoff, lowpass_cutoff)])

    plt.plot(time_s, 10*sig, label="Supply current")
    #plt.plot(time_s,filt_signals[:, 0], label="High-pass")
    plt.plot(time_s, 40*filt_signals[:, 1], label="Calibration current outside of VV (tc = 0.5s)")
    plt.title("VS3 coil current & Calibration current")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("Ampare [kA]")
    #plt.xlim([5.98, 6.06])
    #plt.ylim([-0.05, 1.05])
    #plt.plot(6.01,1-np.exp(-1), 'ro')
    #plt.text(6.011,1-np.exp(-1), '(6.01, 0.632) = (tc, 1-exp(-1)')
    plt.show()

    sig_dir = 'Filteredsignal.csv'
    outx = time_s
    outy = filt_signals[:, 1]*40000
    df = pd.DataFrame(outy)
    df.to_csv(sig_dir, index=False)