# DT2119, Lab 1 Feature Extraction
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.spatial import distance_matrix

from lab1_tools import lifter, trfbank

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    frames = []
    for i in range(0, len(samples), winshift):
        if i + winlen > len(samples):
            break
        frames.append(samples[i:i+winlen])
    return np.array(frames)
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return scipy.signal.lfilter([1, -p], 1, input)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    # If the scipy.signal.hamming function does not exist (which should happen if 
    # you are using a recent version of scipy), we have to use scipy.signal.windows.hamming
    if not hasattr(scipy.signal, 'hamming'):
        window = scipy.signal.windows.hamming(input.shape[1], sym=False)
    else:
        window = scipy.signal.hamming(input.shape[1], sym=False)
    return input * window

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.abs(fft(input, nfft))**2

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    return np.log(np.dot(input, trfbank(samplingrate, input.shape[1]).T))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input)[:, :nceps]
    
def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    LD = distance_matrix(x, y, dist)
    AD = np.zeros(LD.shape)
    # Initialize the first row and column of the accumulated distance matrix
    # with the local distance matrix
    AD[0, 0] = LD[0, 0]
    for i in range(1, LD.shape[0]):
        AD[i, 0] = AD[i-1, 0] + LD[i, 0]
    for j in range(1, LD.shape[1]):
        AD[0, j] = AD[0, j-1] + LD[0, j]
        
    # Fill the accumulated distance matrix and track the predecessor for each cell
    pred = np.zeros(AD.shape)
    
    for i in range(1, LD.shape[0]):
        for j in range(1, LD.shape[1]):
            AD[i, j] = LD[i, j] + min(AD[i-1, j], AD[i, j-1], AD[i-1, j-1])
            if AD[i-1, j] < AD[i, j-1] and AD[i-1, j] < AD[i-1, j-1]:
                pred[i, j] = 1
            elif AD[i, j-1] < AD[i-1, j] and AD[i, j-1] < AD[i-1, j-1]:
                pred[i, j] = 2
            else:
                pred[i, j] = 3
                
    # Backtrack the best path
    path = []
    i = LD.shape[0] - 1
    j = LD.shape[1] - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if pred[i, j] == 1:
            i -= 1
        elif pred[i, j] == 2:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.append((0, 0))
    path.reverse()
    
    return AD[-1, -1] / (len(x) + len(y)), LD, AD, path
