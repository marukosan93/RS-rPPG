import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from tqdm import tqdm
from scipy import signal
from scipy.fft import fft, rfft
from scipy.signal import butter, filtfilt, welch,resample
import heartpy as hp
import os
from sklearn.decomposition import PCA
from scipy import sparse
import math
from scipy import linalg
from scipy.stats import pearsonr


def get_stats(hr_pred, hr_gt):
    hr_pred = np.array(hr_pred)
    hr_gt = np.array(hr_gt)
    error = np.abs(hr_pred-hr_gt)
    mae = round(np.mean(error),3)
    rmse = round(np.sqrt(np.mean(np.square(error))),3)
    std = round(np.std(error),3)
    #print(hr_pred)
    #print(hr_gt)
    r_score = round(pearsonr(hr_pred,hr_gt)[0],3)
    #print(r)
    return mae,rmse,std,r_score

def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal

"""
GREEN
Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
"""
def GREEN_rppg(input_sig):
    green = input_sig[:,1]
    return green
"""
ICA
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.
"""
def ICA_rppg(input_sig, FS=30):
    # Cut off frequency.
    LPF = 0.5
    HPF = 3

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(input_sig.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = detrend(input_sig[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))
    BVP = BVP_F[0]
    return BVP

def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S


def norm(arrr):
    return (arrr-np.min(arrr))/(np.max(arrr)-np.min(arrr))


"""
POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

def POS_rppg(signal):
    mean_rgb = signal
    l = int(30 * 1.6)
    H = np.zeros(mean_rgb.shape[0])
    for t in range(0, (mean_rgb.shape[0]-l)):
        C = mean_rgb[t:t+l-1,:].T
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.pinv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv,C)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        P = np.matmul(std,S)
        H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P)
    signal = H
    return H

"""
CHROM
De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
"""

def CHROM_rppg(signal):
    X = signal.astype(np.float64())
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = np.std(Xcomp,axis=-1)
    sY = np.std(Ycomp,axis=-1)
    alpha = (sX/sY)
    bvp = Xcomp - alpha*Ycomp
    return norm(bvp)

"""
LGI
Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
"""
def LGI_rppg(input_sig):
    precessed_data = input_sig.transpose(1, 0).reshape(1, 3, -1)
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp

"""
PBV
De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.
"""

def PBV_rppg(input_sig):
    precessed_data = input_sig.transpose(1, 0).reshape(1, 3, -1)
    sig_mean = np.mean(precessed_data, axis=2)

    signal_norm_r = precessed_data[:, 0, :] / np.expand_dims(sig_mean[:, 0], axis=1)
    signal_norm_g = precessed_data[:, 1, :] / np.expand_dims(sig_mean[:, 1], axis=1)
    signal_norm_b = precessed_data[:, 2, :] / np.expand_dims(sig_mean[:, 2], axis=1)

    pbv_n = np.array([np.std(signal_norm_r, axis=1), np.std(signal_norm_g, axis=1), np.std(signal_norm_b, axis=1)])
    pbv_d = np.sqrt(np.var(signal_norm_r, axis=1) + np.var(signal_norm_g, axis=1) + np.var(signal_norm_b, axis=1))
    pbv = pbv_n / pbv_d

    C = np.swapaxes(np.array([signal_norm_r, signal_norm_g, signal_norm_b]), 0, 1)
    Ct = np.swapaxes(np.swapaxes(np.transpose(C), 0, 2), 1, 2)
    Q = np.matmul(C, Ct)
    W = np.linalg.solve(Q, np.swapaxes(pbv, 0, 1))

    A = np.matmul(Ct, np.expand_dims(W, axis=2))
    B = np.matmul(np.swapaxes(np.expand_dims(pbv.T, axis=2), 1, 2), np.expand_dims(W, axis=2))
    bvp = A / B
    return bvp.squeeze(axis=2).reshape(-1)

"""
PCA
Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.
"""

"""def PCA_rppg(window):
    sig_out = []
    for i in range(window.shape[0]):
        X = window[i].reshape(-1, 1)
        pca = PCA(n_components=3)
        pca.fit(X)

        # selector
        if kargs['component']=='all_comp':
            sig_out.append(pca.components_[0] * pca.explained_variance_[0])
            sig_out.append(pca.components_[1] * pca.explained_variance_[1])
        elif kargs['component']=='second_comp':
            sig_out.append(pca.components_[1] * pca.explained_variance_[1])
    sig_out = np.array(sig_out)
    return sig_out"""

def PCA_rppg(window):
    #normalized = window
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    normalized = (window - mean) / std
    normalized[:,0] = butter_bandpass(normalized[:,0], 0.5, 3, 30)
    normalized[:,1] = butter_bandpass(normalized[:,1], 0.5, 3, 30)
    normalized[:,2] = butter_bandpass(normalized[:,2], 0.5, 3, 30)
    pca = PCA(n_components=3)
    srcSig = pca.fit_transform(normalized)
    return srcSig[:,1]

def butter_bandpass(sig, lowcut, highcut, fs, order=5):
    # butterworth bandpass filter
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, sig)
    return y

def calc_hr(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = butter_bandpass(sig, 0.7, 3, fs)
    #sig = signal.detrend(sig)
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
        """if (p2/p1 > 0.85) and abs(hr1-hr2) < 30:
            hr = np.mean([hr1,hr2])"""
    return hr, Pxx, f, sig

def calc_hr_fast(signal_tensor,fps=30):  #might be more efficient with just torch, but runs just on eval so it's ok
    listino = []
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()
        ### WELCH
        #signal = butter_bandpass(signal, 0.5, 3, 30)
        #f, Pxx = welch(signal, fps, nperseg=160,nfft=2048)
        #hr = f[np.argmax(Pxx)]*60
        hr,_,_,_ = calc_hr(signal,fps,harmonics_removal=True)
        listino.append(hr)

    return np.array(listino)
