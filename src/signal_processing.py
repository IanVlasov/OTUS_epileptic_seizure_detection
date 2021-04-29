from collections import Counter

import scipy
import pywt
import numpy as np

from tqdm.autonotebook import tqdm

from scipy.signal import welch
from scipy.fftpack import fft
from src.detect_peaks import detect_peaks

global sample_rate


def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = np.array([sample_rate * kk for kk in range(0, len(y_values))])
    return x_values, y_values


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


def get_first_n_peaks(x, y, no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks


def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y


def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in tqdm(range(0, len(dataset)), leave=False):
        features = []
        list_of_labels.append(labels[signal_no])
        signal = dataset.iloc[signal_no].values

        signal_min = np.nanpercentile(signal, percentile)
        signal_max = np.nanpercentile(signal, 100-percentile)
        #  ijk = (100 - 2*percentile)/10
        mph = signal_min + (signal_max - signal_min)/denominator

        features += get_features(*get_psd_values(signal, T, N, f_s), mph)
        features += get_features(*get_fft_values(signal, T, N, f_s), mph)
        features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)

        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

# Wavelets
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(
        np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_wavelet_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in tqdm(range(0, len(dataset)), leave=False):
        features = []
        signal = dataset.iloc[signal_no].values
        list_coeff = pywt.wavedec(signal, waveletname)
        for coeff in list_coeff:
            features += get_wavelet_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y