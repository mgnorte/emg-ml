import pywt
import numpy as np
from scipy import signal

def apply_bandpass_filter(data, sample_rate, lowpass_cutoff=350, highpass_cutoff=30, order=5):
    """
    Apply a band-pass filter to the input data, removing motion artifacts, flicker noise, and irrelevant frequencies

    Parameters:
    data (array-like): Input signal data.
    sample_rate (float): Sampling rate of the signal in Hz.
    lowpass_cutoff (float): Cutoff frequency for the low-pass filter in Hz (default is 350)
    highpass_cutoff (float): Cutoff frequency for the high-pass filter in Hz (default is 30)
    order (int): Order of the filters (default is 5).

    Returns:
    numpy.ndarray: Band-pass filtered signal
    """

    #prevent misinterpretation of high-frequency signals as lower-frequency signals
    nyquist = 0.5 * sample_rate
    if lowpass_cutoff != None:
        low = lowpass_cutoff / nyquist
    if highpass_cutoff != None:
        high = highpass_cutoff / nyquist

    # Design filters
    if lowpass_cutoff != None:
        b_low, a_low = signal.butter(order, low, btype='low', analog=False)
    if highpass_cutoff != None:
        b_high, a_high = signal.butter(order, high, btype='high', analog=False)

    # Apply filters sequentially
    if lowpass_cutoff == None:
        bandpass_filtered = signal.filtfilt(b_high, a_high, data)
    elif highpass_cutoff == None:
        bandpass_filtered = signal.filtfilt(b_low, a_low, data)
    else:
        bandpass_filtered = signal.filtfilt(b_high, a_high, signal.filtfilt(b_low, a_low, data))

    return bandpass_filtered

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def hard_threshold(x, threshold):
    if abs(x) < threshold: return 0
    else: return x

def known_noise_donoho_filter(signal, noiseStartIndex, noiseEndIndex, wavelet='db4', level=1, hard=False):
    """
    Apply the thresholding filter outlined in DE-NOISING BY SOFT-THRESHOLDING by David L. Donoho to remove gaussian noise
    Takes the noise standard deviation from the noise, rather than approximating it from the WT detail coefficients

    Parameters:
    signal (array-like): Signal data.
    startNoiseIndex (int): The index of the signal where noise domination begins
    noiseEndIndex (int): The index of the signal where noise domination ends
    wavelet (string): The type of wavelet used in the wavelet transform (default db4)
    level (int): The level in which the wavelet transform is performed (default 1)
    hard (bool): If True, hard thresholding is used instead of soft thresholding (default false)

    Returns:
    numpy.ndarray: Denoised signal
    """

    # Decompose the signal
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=level)

    #extract noise
    noise = signal[noiseStartIndex:noiseEndIndex]

    # fast approximation of the standard deviation of the noise
    # assumes noise is gaussian (usually is)
    sigma_hat = np.median(np.abs(noise)) / 0.6745
    threshold = sigma_hat * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to the coefficients (default soft)
    if not hard:
        new_coeff = [soft_threshold(c, threshold) for c in coeff]
    else:
        new_coeff = [hard_threshold(c, threshold) for c in coeff]

    # Reconstruct the denoised signal
    denoised = pywt.waverec(new_coeff, wavelet, mode="per")

    return denoised

def donoho_denoise_signal(y, wavelet='db4', level=1, threshold=None, hard=False):
    """
    Apply the thresholding filter outlined in DE-NOISING BY SOFT-THRESHOLDING by David L. Donoho to remove gaussian noise

    Parameters:
    signal (array-like): Signal data.
    wavelet (string): The type of wavelet used in the wavelet transform (default db4)
    level (int): The level in which the wavelet transform is performed (default 1)
    hard (bool): If True, hard thresholding is used instead of soft thresholding (default false)

    Returns:
    numpy.ndarray: Denoised signal
    """

    # Decompose the signal
    coeff = pywt.wavedec(y, wavelet, mode="per", level=level)

    # Calculate the threshold
    if threshold is None:
        #fast approximation of the standard deviation of the noise (detail coefficients of wavelet transform)
        #assumes noise is gaussian (usually is)
        sigma = np.median(np.abs(coeff[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(y)))

    # Apply thresholding to the coefficients (default soft)
    if not hard:
        new_coeff = [soft_threshold(c, threshold) for c in coeff]
    else:
        new_coeff = [hard_threshold(c, threshold) for c in coeff]

    # Reconstruct the denoised signal
    denoised = pywt.waverec(new_coeff, wavelet, mode="per")

    return denoised

def denoise_known_noise(signal, sampleRate, noiseStartIndex, noiseEndIndex, highpassFrequency=30, lowpassFrequency=350, bandpassFilterOrder=5, wavelet='db4', waveletTransformOrder=1, hard=False):
    """
    Denoises the signal with a bandpass filter and WT based soft threshold

    Parameters:
    signal (array-like): Input signal data.
    sampleRate (int): Sampling rate of the signal in Hz.
    noiseStartIndex (int): The index of the signal where noise domination begins
    noiseEndIndex (int): The index of the signal where noise domination ends
    highpassCutoff (int): Cutoff frequency for the low-pass filter in Hz (default is 350)
    lowpassCutoff (int): Cutoff frequency for the high-pass filter in Hz (default is 30)
    bandpassFilterOrder (int): Order of the bandpass (lowpass + highpass) filters (default is 5).
    wavelet (string): The type of wavelet used in the wavelet transform (default db4)
    waveletTransformOrder (int): The level in which the wavelet transform is performed (default 1)
    hard (bool): If True, hard thresholding is used instead of soft thresholding (default false)

    Returns:
    numpy.ndarray: Denoised signal
    """

    bandPassFiltered = apply_bandpass_filter(signal, sampleRate, lowpassFrequency, highpassFrequency, bandpassFilterOrder)
    denoised = known_noise_donoho_filter(bandPassFiltered, noiseStartIndex, noiseEndIndex, wavelet, waveletTransformOrder, hard)

    return denoised

def denoise_unknown_noise(signal, sampleRate, highpassCutoff=30, lowpassCutoff=350, bandpassFilterOrder=5, wavelet='db4', waveletTransformOrder=1, hard=False):
    """
    Denoises the signal with a bandpass filter and WT based soft threshold

    Parameters:
    signal (array-like): Input signal data.
    sampleRate (float): Sampling rate of the signal in Hz.
    highpassCutoff (float): Cutoff frequency for the low-pass filter in Hz (default is 350)
    lowpassCutoff (float): Cutoff frequency for the high-pass filter in Hz (default is 30)
    bandpassFilterOrder (int): Order of the bandpass (lowpass + highpass) filters (default is 5).
    wavelet (string): The type of wavelet used in the wavelet transform (default db4)
    waveletTransformOrder (int): The level in which the wavelet transform is performed (default 1)
    hard (bool): If True, hard thresholding is used instead of soft thresholding (default false)

    Returns:
    numpy.ndarray: Denoised signal
    """

    bandPassFiltered = apply_bandpass_filter(signal, sampleRate, lowpassCutoff, highpassCutoff, bandpassFilterOrder)
    denoised = donoho_denoise_signal(bandPassFiltered, wavelet, waveletTransformOrder, hard)

    return denoised