import time

import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import convolve as sig_convolve, fftconvolve, lfilter, firwin
from scipy.ndimage import convolve1d
from pylab import grid, show, legend, loglog, xlabel, ylabel, figure
# import unittest
# from  ECGPreprocessor import Signalpreprocessor
#
# class Test_test_filter(unittest.TestCase):
#     filename = r'D:\ALL_Male_Seated\Aortic\aortic2.wav'
#     def test_wavread(self):
#         process = Signalpreprocessor()
#         assert process.sample_rate == 1e3
#         process.wavread(self.filename)
#         assert  process.sample_rate == 44100
#     def test_downsample(self):
#         process = Signalpreprocessor()
#         process.wavread(self.filename)
#         process.down_sample()
#         assert process.sample_rate  == 1e3
#         assert process.raw_data != process.downsampled_data
# if __name__ == '__main__':
#     unittest.main()

# Create the m by n data to be filtered.
m = 4
n = 2 ** 17
x = np.random.random(size=(m, n))

conv_time = []
npconv_time = []
fftconv_time = []
conv1d_time = []
lfilt_time = []

diff_list = []
diff2_list = []
diff3_list = []

ntaps_list = 2 ** np.arange(2, 13)

for ntaps in ntaps_list:
    # Create a FIR filter.
    b = firwin(ntaps, [0.05, 0.95], width=0.05, pass_zero=False)

    if ntaps <= 2 ** 9:
        # --- signal.convolve ---
        # We know this is slower than the others when ntaps is
        # large, so we only compute it for small values.
        tstart = time.time()
        conv_result = sig_convolve(x, b[np.newaxis, :], mode='valid')
        conv_time.append(time.time() - tstart)

    # --- numpy.convolve ---
    tstart = time.time()
    npconv_result = np.array([np_convolve(xi, b, mode='valid') for xi in x])
    npconv_time.append(time.time() - tstart)

    # --- signal.fftconvolve ---
    tstart = time.time()
    fftconv_result = fftconvolve(x, b[np.newaxis, :], mode='valid')
    fftconv_time.append(time.time() - tstart)

    # --- convolve1d ---
    tstart = time.time()
    # convolve1d doesn't have a 'valid' mode, so we explicitly slice out
    # the valid part of the result.
    conv1d_result = convolve1d(x, b)[:, (len(b)-1)//2 : -(len(b)//2)]
    conv1d_time.append(time.time() - tstart)

    # --- lfilter ---
    tstart = time.time()
    lfilt_result = lfilter(b, [1.0], x)[:, len(b) - 1:]
    lfilt_time.append(time.time() - tstart)

    diff = np.abs(fftconv_result - lfilt_result).max()
    diff_list.append(diff)

    diff2 = np.abs(conv1d_result - lfilt_result).max()
    diff2_list.append(diff2)

    diff3 = np.abs(npconv_result - lfilt_result).max()
    diff3_list.append(diff3)

# Verify that np.convolve and lfilter gave the same results.
print "Did np.convolve and lfilter produce the same results?",
check = all(diff < 1e-13 for diff in diff3_list)
if check:
    print "Yes."
else:
    print "No!  Something went wrong."

# Verify that fftconvolve and lfilter gave the same results.
print "Did fftconvolve and lfilter produce the same results?",
check = all(diff < 1e-13 for diff in diff_list)
if check:
    print "Yes."
else:
    print "No!  Something went wrong."

# Verify that convolve1d and lfilter gave the same results.
print "Did convolve1d and lfilter produce the same results?",
check = all(diff2 < 1e-13 for diff2 in diff2_list)
if check:
    print "Yes."
else:
    print "No!  Something went wrong."

figure(1, figsize=(8, 5.5))
loglog(ntaps_list, npconv_time, 'c-s', label='numpy.convolve')
loglog(ntaps_list, conv1d_time, 'k-p', label='ndimage.convolve1d')
loglog(ntaps_list, fftconv_time, 'g-*', markersize=8, label='signal.fftconvolve')
loglog(ntaps_list[:len(conv_time)], conv_time, 'm-d', label='signal.convolve')
loglog(ntaps_list, lfilt_time, 'b-o', label='signal.lfilter')
legend(loc='best', numpoints=1)
grid(True)
xlabel('Number of taps')
ylabel('Time to filter (seconds)')
show()

Filter_design.py
from __future__ import division
import numpy as np
import numba as nb
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show

filter_para = {'fpass': 5, 'fstop': 30, 'Rp': 1.0, 'As': 60}
filter_para = {'fpass': 5, 'fstop': 30, 'Rp': 1.0, 'As': 60}
ftype = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']


def wavread(filename):
    sample_rate, data = wavfile.read(filename, mmap=True)  # mmap:read data as memory mapped
    t = np.arange(len(data)) / sample_rate
    return sample_rate, t, data[:, 0]


def test_signal():
    """
     Create a signal for demonstration.
    """

    sample_rate = 100.0
    nsamples = 400
    t = np.arange(nsamples) / sample_rate
    x = np.cos(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 2.5 * t + 0.1) + \
        0.2 * np.sin(2 * np.pi * 15.3 * t) + 0.1 * np.sin(2 * np.pi * 16.7 * t + 0.1) + \
        0.1 * np.sin(2 * np.pi * 23.45 * t + .8)

    return sample_rate, t, x


def SignalPreprocessing(object):
    self.ny = sample_rate / 2.0

    def __int__(self):
        pass

    def downsample(self, x, sample_rate, new_rate):
        """ Downsample the signal by using a filter.
        By default, an order 8 Chebyshev type I filter is used.
        A 30 point FIR filter with hamming window is used if ftype is

        input:
        x: the input signal
        sample_rate: the input signal sampling rate
        new_rate: the new sample rate
        return:
            the downsampled signal
        """
        # Decimation Rate
        factor = np.round(sample_rate / new_rate)
        return signal.decimate(data, factor)

    def smooth(self, x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        return:
            the smoothed signal


        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return (y[(window_len / 2 - 1):-(window_len / 2)])


class ECGfilter(object):

    def designIIR(self, filter_para, ftype='butter', btype='Bandpass'):
        """
        IIR filter design ulitity class, uses scipy.signal.irrdesign
        input:
            filter_para: disc
            fpass: starting frequency(Hz)
            fstop: stopping frequency (Hz)will be changed to (Ws = start/nyquistWs)

            Rp: passband maximum loss (gpass)
            As: stoppand min attenuation (gstop)
            btaps: type of filter defalut is bandpass

        usage:
            frequencies will be normalized from 0 to 1, where 1 is the Nyquist
            frequency, pi radians/sample.
            (wp and ws are thus in half-cycles / sample.)
                Lowpass: wp = 0.2, ws = 0.3
                Highpass: wp = 0.3, ws = 0.2
                Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]
                Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]
        return

        """

        # The butter and cheby1 need less constraint spec
        ecgfilter = signal.iirdesign(Wp, Ws, Rp, As, btype, btype, ftype=ftype)

        # btype = ['bandpass', 'lowpass', 'highpass', 'bandstop']

        return ecgfilter

    def designFIR(self, filter_para, pass_zero=True):
        """
        FIR filter design ulitity class, uses scipy.signal.irrdesign
        input:
            filter_para: disc
            fpass: starting frequency(Hz)
            fstop: stopping frequency (Hz)
            Rp: passband maximum loss (gpass)
            As: stoppand min attenuation (gstop)
            btaps: type of filter defalut is bandpass
             pass_zero : bool
            If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
             Otherwise the DC gain is 0. deafult true

        usage:
            frequencies will be normalized from 0 to 1, where 1 is the Nyquist
            frequency, pi radians/sample.
            (wp and ws are thus in half-cycles / sample.)
                Lowpass: pass_zero = True
                Highpass: pass_zero = False
                Bandpass: pass_zero = False
                Bandstop: pass_zero = True
            return

        """

        # Compute the order and Kaiser parameter for the FIR filter.
        M, beta = signal.kaiserord(ripple_db, (fstop + fpass) / nyq_rate)

        # Use signal.firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta),
                             pass_zero)

        # btype = ['bandpass', 'lowpass', 'highpass', 'bandstop']

        return ecgfilter


def filter_lp(sample_rate, width, cutoff_hz, ripple_db):
    """
     Create a Low pass FIR filter
    # The cutoff frequency of the filter.
     # The desired attenuation in the stop band, in dB.
     # The width of the transition from pass to stop,relative to the Nyquist rate.
    """
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = signal.kaiserord(ripple_db, width / nyq_rate)

    # Use signal.firwin with a Kaiser window to create a lowpass FIR filter.
    taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    return taps, N


def filter_hp(sample_rate, width, cutoff_hz, ripple_db):
    """
     Create a Low pass FIR filter
    # The cutoff frequency of the filter.
     # The desired attenuation in the stop band, in dB.
     # The width of the transition from pass to stop,relative to the Nyquist rate.
    """
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = signal.kaiserord(ripple_db, width / nyq_rate)

    # Use signal.firwin with a Kaiser window to create a lowpass FIR filter.
    taps = signal.firwin(N, cutoff_hz / nyq_rate, pass_zero=False,
                         window=('kaiser', beta))

    return taps, N


def filter_bp(sample_rate, width, lowercutoff_hz, uppercutoff_hz, ripple_db):
    """
     Create a Band pass FIR filter
    # The cutoff frequency of the filter.
     # The desired attenuation in the stop band, in dB.
     # The width of the transition from pass to stop,relative to the Nyquist rate.
    """
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = signal.kaiserord(ripple_db, width / nyq_rate)

    # Use signal.firwin with a Kaiser window to create a lowpass BP filter.

    taps = signal.firwin(N, [lowercutoff_hz / nyq_rate, uppercutoff_hz / nyq_rate],
                         window=('kaiser', beta), pass_zero=False)

    return taps, N


def plot_taps(taps, N):
    # ------------------------------------------------
    # Plot the FIR filter coefficients.
    # ------------------------------------------------
    nyq_rate = sample_rate / 2.0
    plt.figure(1)
    plot(taps, 'bo-', linewidth=2)
    title('Filter Coefficients (%d taps)' % N)
    grid(True)


def plot_response(taps, N):
    # ------------------------------------------------
    # Plot the magnitude response of the filter.
    # ------------------------------------------------
    nyq_rate = sample_rate / 2.0
    plt.figure(2)
    clf()
    # w, h = signal.freqz(taps, worN=8000)
    w, h = signal.freqz(taps, worN=8000)
    plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
    xlabel('Frequency (Hz)')
    ylabel('Gain')
    title('Frequency Response')
    ylim(-0.05, 1.05)
    grid(True)

    # Upper inset plot.
    ax1 = axes([0.42, 0.6, .45, .25])
    plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
    xlim(0, 15)
    ylim(0.9985, 1.001)
    grid(True)

    # Lower inset plot
    ax2 = axes([0.42, 0.25, .45, .25])
    plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
    xlim(12.0, 30.0)
    ylim(0.0, 0.0025)
    grid(True)
    plt.show()


def spectral(data, sample_rate):
    plt.figure(4)
    f, Pxx_den = signal.welch(data, sample_rate, nperseg=1024)
    plt.semilogy(f, Pxx_den)
    plt.grid(True)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    print
    np.mean(Pxx_den[30:])

    show()

    f, Pxx_spec = signal.welch(data, sample_rate, 'flattop', 1024, scaling='spectrum')
    plt.figure(5)
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()


def iirfilter(data):
    """Bandpass filter the ECG from 5 to 15 Hz"""
    # TODO: Explore - different filter designs
    nyq_rate = sample_rate / 2.0
    # wn = [30/ nyq_rate, 40/ nyq_rate]
    wn = [30 / nyq_rate]
    b, a = signal.butter(2, wn, btype='lowpass')
    return signal.filtfilt(b, a, data)


if __name__ == "__main__":
    filename = r'D:\ALL_Male_Seated\Aortic\aortic2.wav'
    sample_rate, t, data = wavread(filename)

    Newsample_rate = 1000
    sample_rate, t, data = downSample(1000.0, sample_rate, data)

    # sample_rate, t, data = test_signal()

    # The cutoff frequency of the filter.
    lowercutoff_hz = 5
    uppercutoff_hz = 20
    width = 5
    ripple_db = 60

    # taps,N = filter_lp(sample_rate, width, lowercutoff_hz, ripple_db)
    taps, N = filter_hp(sample_rate, width, uppercutoff_hz, ripple_db)
    # taps, N = filter_bp(sample_rate, width, lowercutoff_hz, uppercutoff_hz,ripple_db)

    # Filters=iirdesign(sample_rate)
    # plot_response(Filters['butter'] [0],Filters['butter'] [1])

    ##filtered_x = signal.lfilter(taps, 1.0, data)
    filtered_x = iirfilter(data)

    smoooth_x = smooth(data, window_len=21, window='hanning')
    ##------------------------------------------------
    ## Plot the original and filtered signals.
    ##------------------------------------------------

    ## The phase delay of the filtered signal.
    # delay = 0.5 * (N - 1) / sample_rate

    # plt.figure(3)
    # Plot the original signal.
    plot(t, data)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    # plot(t - delay, filtered_x, 'r-')
    plot(t, filtered_x, 'r-')
    plot(smoooth_x, 'g-')
    ## Plot just the "good" part of the filtered signal.  The first N-1
    ## samples are "corrupted" by the initial conditions.
    ##plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=4)
    ##plot(t[N - 1:] - delay, data[N-1:]-filtered_x[N - 1:], 'r')

    # xlabel('t')
    # grid(True)
    show()

    spectral(data, sample_rate)
    y = filtered_x
    spectral(y, sample_rate)
