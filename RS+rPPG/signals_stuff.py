import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft,rfft
import torch.nn as nn
from scipy import signal

def psd_(sig):
    fs = 30
    sig = np.pad(sig,16)
    sig = (sig * signal.windows.hann(sig.shape[0]))[16:-16]
    Pxx = np.abs(rfft(sig,int(len(sig)*5*fs)))
    f = np.linspace(0,15,len(Pxx))
    Pxx[(f<0.7)*(f>3)] = 0
    return Pxx,f


def specsim(sig1,sig2):
    Pxx1, f1 = psd_(sig1)
    Pxx2, f2 = psd_(sig2)
    sim = np.sum(Pxx1*Pxx2)/(np.linalg.norm(Pxx1)*np.linalg.norm(Pxx2))
    return sim

def butter_bandpass(sig, lowcut, highcut, fs, order=7):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

def norm(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def hr_fft(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = butter_bandpass(sig, 0.7, 3, fs)
    #sig = detrend(sig)
    #sig = np.pad(sig,128)
    #sig = (sig * signal.windows.hann(sig.shape[0]))[128:-128]
    #Pxx = np.abs(rfft(sig,int(len(sig)*5*fs)))
    f, Pxx = welch(sig, 30, nperseg=160,nfft=2048)
    #f = np.linspace(0,15,len(Pxx))
    #f, Pxx = welch(sig, 1, nperseg=64,nfft=1024)
    #f = f * fs

    peak_idx, _ = signal.find_peaks(Pxx)
    tr_c = Pxx.shape[0]/(fs/2)
    high_idx = int(round(3*tr_c))
    low_idx = int(round(0.5*tr_c))
    peak_idx = peak_idx[(peak_idx>low_idx) * (peak_idx<high_idx)]

    if len(peak_idx) == 0:
        return 0, Pxx, f, sig
    if len(peak_idx) == 1:
        hr = 60*peak_idx[0]/tr_c
    if len(peak_idx) > 1:
        sort_idx = np.argsort(Pxx[peak_idx])
        sort_idx = sort_idx[::-1]
        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        hr1 = 60*peak_idx1/tr_c
        hr2 = 60*peak_idx2/tr_c


        p1 = Pxx[peak_idx1]
        p2 = Pxx[peak_idx2]

        th_p = 0.8
        th_hr = 10
        hr = hr1
        if harmonics_removal:
            if p2/p1 > th_p and np.abs(hr1-2*hr2)<th_hr:
                hr = hr2
        if (p2/p1 > 0.85) and abs(hr1-hr2) < 30:
            hr = np.mean([hr1,hr2])
    return hr, Pxx, f, sig


def old_hr_fft(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = butter_bandpass(sig, 0.7, 3, fs)
    sig = detrend(sig)
    sig = np.pad(sig,16)
    sig = (sig * signal.windows.hann(sig.shape[0]))[16:-16]
    Pxx = np.abs(rfft(sig,int(len(sig)*5*fs)))
    f = np.linspace(0,15,len(Pxx))
    #f, Pxx = welch(sig, 1, nperseg=64,nfft=1024)
    #f = f * fs

    peak_idx, _ = signal.find_peaks(Pxx)
    tr_c = Pxx.shape[0]/(fs/2)
    high_idx = int(round(2.7*tr_c))
    low_idx = int(round(0.75*tr_c))
    peak_idx = peak_idx[(peak_idx>low_idx) * (peak_idx<high_idx)]

    if len(peak_idx) == 0:
        return 0, Pxx, f, sig
    if len(peak_idx) == 1:
        hr = 60*peak_idx[0]/tr_c
    if len(peak_idx) > 1:
        sort_idx = np.argsort(Pxx[peak_idx])
        sort_idx = sort_idx[::-1]
        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        hr1 = 60*peak_idx1/tr_c
        hr2 = 60*peak_idx2/tr_c


        p1 = Pxx[peak_idx1]
        p2 = Pxx[peak_idx2]


        th_p = 0.8
        th_hr = 10
        hr = hr1
        if harmonics_removal:
            if p2/p1 > th_p and np.abs(hr1-2*hr2)<th_hr:
                hr = hr2
        if p2/p1 > 0.95 and abs(hr1-hr2) < 15:
            hr = min(hr1,hr2)
    return hr, Pxx, f, sig


class NegPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(NegPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            pearson = torch.stack((preds[i],labels[i]),axis=0)
            loss += 1 - torch.corrcoef(pearson)[0,1]
        loss = loss / preds.shape[0]
        return loss
