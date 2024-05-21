import random
import torch
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import torchvision.transforms as T
import more_itertools as mit
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

class NegativeMaxCrossCov(nn.Module):
    def __init__(self, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels,Fs):
        # Normalize
        preds_norm = preds - torch.mean(preds,dim=3, keepdim=True)
        labels_norm = labels - torch.mean(labels,dim=3, keepdim=True)
        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])

        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2

        preds_pad = F.pad(preds_norm, (0,padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = torch.view_as_real(X)

        Fn = Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])

        # Determine ratio of energy between relevant and non-relevant regions

        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)

        use_energy = torch.sum(torch.linalg.norm(X_real[:,:,:,use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:,:,:,zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)

        #for (i,j,k) in [(i,j,k) for i in range(0,denom.size()[0]) for j in range(0,denom.size()[1]) for k in range(0,denom.size()[2])]:
        #    if denom[i,j,k] < 0:
        #        energy_ratio[i,j,k] = use_energy[i,j,k] / denom[i,j,k]
        energy_ratio = use_energy/denom

        # Zero out irrelevant freqs
        X[:,:,:,zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0]/energy_ratio

        return -max_cc


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, high_pass, low_pass):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(high_pass, low_pass)

    def forward(self, preds, labels,Fs):
        cov = self.cross_cov(preds, labels,Fs)

        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)

        output = torch.zeros_like(cov)
        output = cov/denom
        output = torch.mean(output)
        return output

class MapPSDMSE(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(MapPSDMSE, self).__init__()
        return

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        crit_fft = nn.MSELoss()
        #fig ,ax = plt.subplots(6,1)
        preds_fft = torch.fft.rfft(preds,dim=3)
        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)
        labels_fft = torch.fft.rfft(labels,dim=3)
        labels_psd = torch.real(labels_fft)*torch.real(labels_fft)+torch.imag(labels_fft)*torch.imag(labels_fft)

        f = torch.fft.rfftfreq(labels.size(3),1/fps[0])
        indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

        preds_psd = preds_psd[:,:,:,indices]
        labels_psd = labels_psd[:,:,:,indices]

        preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
        labels_psd = torch.div(labels_psd,torch.sum(labels_psd,3,keepdim=True)) #normalise

        loss = crit_fft(preds_psd,labels_psd)
        #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        #loss = loss/power_labels
        return loss

class MapPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(MapPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        mode = 0
        if mode == 1:
            loss = 0
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    for k in range(preds.shape[3]):
                        pearson = torch.stack((preds[i,j,k,:],labels[i,j,k,:]),axis=0)
                        loss += 1 - torch.corrcoef(pearson)[0,1]
            loss = loss / (preds.shape[0]*preds.shape[1]*preds.shape[3])
            return loss
        else:   #Slightly different value but much faster
            preds_flat = torch.flatten(preds)
            labels_flat = torch.flatten(labels)
            pearson = torch.stack((preds_flat,labels_flat),axis=0)
            loss = 1 - torch.corrcoef(pearson)[0,1]
            return loss

class MapFFTMSE(nn.Module):
    def __init__(self):
        super(MapFFTMSE, self).__init__()
        return

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        crit_fft = nn.MSELoss()
        preds_fft = torch.abs(torch.fft.rfft(preds,dim=3))
        labels_fft = torch.abs(torch.fft.rfft(labels,dim=3))
        f = torch.fft.rfftfreq(labels.size(3),1/fps[0])
        indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]
        loss = crit_fft(preds_fft[:,:,:,indices],labels_fft[:,:,:,indices])
        power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        loss = loss/power_labels
        return loss

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Acc(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.error = 0
        self.mae = 0
        self.std = 0
        self.rmse = 0
        self.sum = 0
        self.count = 0
        self.sqr_sum = 0

    def update(self, error, n=1):
        self.error = error
        self.sum += torch.sum(torch.abs(error)).item()
        self.sqr_sum += torch.sum(torch.square(error)).item()
        self.count += n
        self.mae = self.sum / self.count
        self.rmse = math.sqrt(self.sqr_sum / self.count)
        self.std = math.sqrt(self.sqr_sum / self.count-(self.sum / self.count)**2)

def list_dirs(dir,extension):
    r = []
    if extension == "bvm_map.npy" or extension == "bvp.npy":
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                for file in os.listdir(dirpath):
                    if file[-len(extension):] == extension:
                        r.append(dirpath)
                        break
    return r

def create_datasets(dataset,fold,train_stride=576,seq_len=576,train_temp_aug=False):
    if train_temp_aug:
        if fold != "whole":
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+"_aug.pkl",'rb')
        else:
            file = open("./folds"+str(seq_len)+"/"+dataset+"_"+fold+"_aug.pkl",'rb')
    else:
        if fold != "whole":
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'rb')
        else:
            file = open("./folds"+str(seq_len)+"/"+dataset+"_"+fold+".pkl",'rb')

    train_dirs = pickle.load(file)
    valid_dirs = pickle.load(file)
    return train_dirs, valid_dirs
