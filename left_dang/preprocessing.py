# Importing numpy 
import numpy as np
# Importing Scipy 
import scipy as sp
from skimage.restoration import denoise_wavelet
from scipy.signal import savgol_filter
from scipy.signal import medfilt

#band pass filter between 0.5 and 40 hz
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def median(signal, kernel_size=3):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=sp.signal.medfilt(array, kernel_size=kernel_size) # applying the median filter order3(kernel_size=3)
    return  med_filtered # return the med-filtered signal: numpy array 1D
#notch filter apllied at 50hz
def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    fs   = 256#1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def filter_teeth(x):
    fs = 256
    # lowcut = 20
    # highcut = 49
    lowcut = 0.20 * 128
    highcut = 0.30 *128
    x=butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    x=median(x, 9)
    x=savgol_filter(x, 10, polyorder=5 ,mode='nearest')

    return x

def filter_eyebrows(x):
    fs = 256
    lowcut = 20
    highcut = 49
    x=butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    x=median(x, 5)
    x=savgol_filter(x, 10, polyorder=5 ,mode='nearest')
    # fs = 256
    # lowcut = 128 * 0.15
    # highcut = 128 * 0.30
    # x=butter_bandpass_filter(x, lowcut, highcut, fs, order=13)
    # x=median(x, 5)

    return x

def filter_right(x):
    fs = 256
    x=median(x)
    x=butter_bandpass_filter(x, lowcut=0.5, highcut=30, fs=fs, order=2)
    x=denoise_wavelet(x, method='BayesShrink',mode='hard',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
    x=savgol_filter(x, 120, polyorder=3,mode='constant')

    return x

def filter_left(x):
    fs = 256
    lowcut = 0.5
    highcut = 5
    x=median(x)
    x=butter_bandpass_filter(x, lowcut, highcut, fs, order=2)
    # x=denoise_wavelet(x,method='BayesShrink',mode='soft',wavelet='sym9',wavelet_levels=5,rescale_sigma=True)
    x=savgol_filter(x, 20, polyorder=5 ,mode='nearest')

    return x

def filter_both(x):
    fs = 256
    med_size,lowcut,highcut = 11, 3, 12
    x=butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    x=median(x, med_size)
    # clean_signnal = log_compression(clean_signal)
    # clean_signal = power_law_transform(clean_signal, 1.5)
    x=savgol_filter(x, 10, polyorder=5 ,mode='nearest')

    return x