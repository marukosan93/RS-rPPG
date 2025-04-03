import os
import time
import random
import numpy as np
import torch
#from torchsummary import summary
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from torchsummary import summary
from MST_tmap2_mv_bg_sunet_nosq import mst
import more_itertools as mit
import torchvision.transforms as T
from torch.utils.data import DataLoader
import PIL
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
import math
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch,filtfilt
import timm
from utils_sunet import setup_seed, AverageMeter, Acc,create_datasets, MapPearson, MapFFTMSE,  NegativeMaxCrossCorr
import argparse
from scipy.stats import pearsonr
from itertools import chain
from signals_stuff import butter_bandpass, norm,NegPearson,hr_fft
import pickle
from itertools import permutations

from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft,rfft

from scipy import signal




def get_perms(n,unique):
    indices = list(np.arange(0,n))
    perms = list(permutations(indices))
    fp = []
    for p in perms:
        flag = True
        for i in range(0,len(p)):
            if indices[i] == p[i]:
                flag = False
        if flag:
            if unique:
                sum = list(np.array(indices)+np.array(p))
                if len(sum) == len(set(sum)):
                    fp.append(p)
            else:
                fp.append(p)
    return fp


class FPeak(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(FPeak, self).__init__()
        return

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        crit_fft = nn.L1Loss()

        #fig ,ax = plt.subplots(6,1)
        preds_fft = torch.fft.rfft(preds,dim=3)
        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)
        labels_fft = torch.fft.rfft(labels,dim=3)
        labels_psd = torch.real(labels_fft)*torch.real(labels_fft)+torch.imag(labels_fft)*torch.imag(labels_fft)

        freqs = torch.fft.rfftfreq(labels.size(3),1/30)
        use_freqs = torch.logical_and(freqs >= f_min, freqs <= f_max)
        not_use_freqs = torch.logical_or(freqs < f_min, freqs > f_max)

        df = freqs[use_freqs][1]-freqs[use_freqs][0]
        df.requires_grad_()

        preds_psd = preds_psd[:,:,:,use_freqs]
        labels_psd = labels_psd[:,:,:,use_freqs]

        preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
        labels_psd = torch.div(labels_psd,torch.sum(labels_psd,3,keepdim=True)) #normalise

        preds_peaks = preds_psd.max(3, keepdim=True).indices*df
        labels_peaks = labels_psd.max(3, keepdim=True).indices*df
        loss = crit_fft(preds_peaks,labels_peaks)
        return loss

list_feats = []
stats_train = []
loss_global = []
loss_global_pos= []
loss_global_neg1= []
loss_global_neg2= []
loss_global_neg3= []
loss2_global = []
loss_global_reg = []
loss_global_neg= []
loss_global_rppg= []
loss_global_pos_hr= []
loss_global_neg_hr= []
loss_ = []

alpha = 1
gamma = 1# 1
delta = 20#20#20
f_min = 0.5
f_max = 3

class MapPSDMSE(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(MapPSDMSE, self).__init__()
        return

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        crit_fft = nn.L1Loss()
        #fig ,ax = plt.subplots(6,1)
        preds_fft = torch.fft.rfft(preds,dim=3)
        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)
        labels_fft = torch.fft.rfft(labels,dim=3)
        labels_psd = torch.real(labels_fft)*torch.real(labels_fft)+torch.imag(labels_fft)*torch.imag(labels_fft)

        f = torch.fft.rfftfreq(labels.size(3),1/30)
        indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

        preds_psd = preds_psd[:,:,:,indices]
        labels_psd = labels_psd[:,:,:,indices]

        preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
        labels_psd = torch.div(labels_psd,torch.sum(labels_psd,3,keepdim=True)) #normalise

        loss = crit_fft(preds_psd,labels_psd)
        #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        #loss = loss/power_labels
        return loss


def torch_fft(preds,n=2048):  # tensor [Batch, Temporal]
    f_min = 0.5
    f_max = 3
    fps = 30
    #fig ,ax = plt.subplots(6,1)
    preds_fft = torch.fft.rfft(preds,dim=3,n=n)
    preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)

    f = torch.fft.rfftfreq(n,1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

    preds_psd = preds_psd[:,:,:,indices]

    preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
    #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
    #loss = loss/power_labels
    return preds_psd

def remove_harmonics(bvp_fft_tensor,fs=30):
    list_hr = []
    for b in range(bvp_fft_tensor.size(0)):
        bvp_fft = bvp_fft_tensor[b].detach().cpu().numpy()

        peak_idx, _ = signal.find_peaks(bvp_fft)
        #tr_c = bvp_fft.shape[0]/(fs/2)
        #high_idx = int(round(2.7*tr_c))
        #low_idx = int(round(0.6*tr_c))
        #peak_idx = peak_idx[(peak_idx>low_idx) * (peak_idx<high_idx)]

        if len(peak_idx) == 0:
            hr = 0
        if len(peak_idx) == 1:
            hr = peak_idx[0]
        if len(peak_idx) > 1:
            sort_idx = np.argsort(bvp_fft[peak_idx])
            sort_idx = sort_idx[::-1]
            peak_idx1 = peak_idx[sort_idx[0]]
            peak_idx2 = peak_idx[sort_idx[1]]

            hr1 = peak_idx1
            hr2 = peak_idx2

            p1 = bvp_fft[peak_idx1]
            p2 = bvp_fft[peak_idx2]

            th_p = 0.8
            th_hr = 10
            hr = hr1
            if p2/p1 > th_p and np.abs(hr1-2*hr2)<th_hr:
                hr = hr2
            #if p2/p1 > 0.95 and abs(hr1-hr2) < 15:
            #    hr = min(hr1,hr2)
        list_hr.append(hr)
    hr = np.array(list_hr)
    return hr



def torch_hr(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048

    preds_mean = torch.mean(preds,dim=[1,2])
    bvp_mean = torch.mean(bvpmap,dim=[1,2])

    gt = calc_hr(bvp_mean,harmonics_removal=True)
    hr = calc_hr(preds_mean,harmonics_removal=False)
    return hr,gt



def hr_fft(sig, fs=30, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = butter_bandpass(sig, 0.7, 3, fs)
    #sig = detrend(sig)
    #sig = np.pad(sig,128)
    #sig = (sig * signal.windows.hann(sig.shape[0]))[128:-128]
    #Pxx = np.abs(rfft(sig,int(len(sig)*5*fs)))
    f, Pxx = welch(sig, 30, nperseg=90,nfft=2048)
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

        th_p = 0.6
        th_hr = 10
        hr = hr1
        if harmonics_removal:
            if p2/p1 > th_p and np.abs(hr1-2*hr2)<th_hr:
                hr = hr2
        if (p2/p1 > 0.85) and abs(hr1-hr2) < 30:
            hr = np.mean([hr1,hr2])
    return hr, Pxx, f, sig

def torch_hr_plot(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048

    preds_mean = torch.mean(preds,dim=[1,2])
    bvp_mean = torch.mean(bvpmap,dim=[1,2])

    gt = calc_hr(bvp_mean)
    hr = calc_hr(preds_mean)

    fig,ax = plt.subplots(4,1)
    signal_tensor = preds_mean
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()
        hr, Pxx, f, sig = hr_fft(signal)

        ax[b].plot(f,norm(Pxx),color="green")
    signal_tensor = bvp_mean
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()

        hr, Pxx, f, sig = hr_fft(signal)

        ax[b].plot(f,norm(Pxx),color="black")

    plt.show()
    print(abs(gt-hr))
    return hr,gt


class FreqReg(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(FreqReg, self).__init__()
        return

    def forward(self, preds,fps,f_min,f_max):  # tensor [Batch, Temporal]

        preds_long = preds
        #plt.imshow(preds_long[0].permute(1,2,0).detach().cpu().numpy())
        #plt.show()
        #fig ,ax = plt.subplots(6,1)
        preds_fft = torch.fft.rfft(preds_long,dim=3)

        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)

        preds_psd -= preds_psd.min(3, keepdim=True)[0]
        preds_psd /= preds_psd.max(3, keepdim=True)[0]

        freqs = torch.fft.rfftfreq(preds_long.size(3),1/fps[0])
        use_freqs = torch.logical_and(freqs >= f_min, freqs <= f_max)
        not_use_freqs = torch.logical_or(freqs < f_min, freqs > f_max)

        preds_not_psd = preds_psd[:,:,:,not_use_freqs]
        preds_use_psd = preds_psd[:,:,:,use_freqs]
        preds_total_psd = preds_psd[:,:,:,:]

        preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
        n = torch.sum(preds_not_psd,dim=3)
        d = torch.sum(preds_total_psd,dim=3)
        d2 = torch.sum(preds_use_psd,dim=3)


        preds_use_psd -= preds_use_psd.min(3, keepdim=True)[0]
        preds_use_psd /= preds_use_psd.max(3, keepdim=True)[0]

        loss2 = torch.sqrt(torch.mean(torch.square(preds_use_psd)))

        term1 = torch.mean(n)
        term2 = torch.mean(d)
        loss = torch.mean(n/d)

        loss = torch.log(1+torch.exp(loss))
        loss2 = torch.log(1+torch.exp(loss2))
        return loss,term1,term2,loss2

class TPL_softplus(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(TPL_softplus, self).__init__()
        return

    def forward(self, A,P,N):  # tensor [Batch, Temporal]
        dap = torch.mean(torch.abs(A-P)) #L1 first
        dan = torch.mean(torch.abs(A-N)) #L1 first
        loss = torch.log(1+torch.exp(dap-dan))
        return loss


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
        self.std = 0#math.sqrt(self.sqr_sum / self.count-(self.sum / self.count)**2)

def calc_hr(signal_tensor,fps=30,harmonics_removal=True):
    listino = []
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()
        hr, Pxx, f, sig = hr_fft(signal,harmonics_removal=harmonics_removal)

        listino.append(hr)
    return np.array(listino)

def accuracy_bvp(output,bvpmap):
    B,C,H,W = output.size()
    pred = []
    gts = []
    pred,gt = torch_hr(output,bvpmap)
    return np.abs(pred-gt),pred,gt

def norm(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def random_ind_diff_pos(length):
    indices = np.arange(0,length)
    new = indices.copy()
    random.shuffle(new)
    while np.count_nonzero(new == indices) > 0:
        random.shuffle(new)
    return new

def tensor_random_stack(tenz,numeretto,indices):
    chunks = torch.split(tenz,64,dim=3)
    tenz = torch.cat(chunks,dim=2)
    chunks = list(torch.split(tenz,16,dim=2))
    chunks = [chunks[x] for x in indices]
    tenz = torch.cat(chunks,dim=2)
    return tenz

def third_random_stack(tenz,numeretto,indices):
    chunks = list(torch.split(tenz,64,dim=2))
    chunks = [chunks[x] for x in indices]
    tenz = torch.cat(chunks,dim=2)
    return tenz


def tensor_stack(tenz,numeretto):
    chunks = torch.split(tenz,64,dim=3)
    tenz = torch.cat(chunks,dim=2)
    return tenz

class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        freqs = torch.fft.rfftfreq(x.size(3),1/self.Fs)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = torch.fft.rfft(x,dim=3)
        x = torch.sqrt(torch.real(x)*torch.real(x)+torch.imag(x)*torch.imag(x))

        use_freqs = torch.logical_and(freqs >= f_min, freqs <= f_max)
        x = x[:,:,:,use_freqs]
        return freqs[use_freqs],x

def validate(valid_loader, model,epoch):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc = Acc()

    # switch to train mode
    model.eval()

    errors = np.zeros(190)
    error_count = np.ones(190)

    list_preds = []
    list_gt = []
    list_error = []

    end = time.time()
    for i, (patched_map,tmap,mvmap,bgmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        patched_map = patched_map.to(device=device, dtype=torch.float)
        tmap = tmap.to(device=device, dtype=torch.float)
        masked_map = masked_map.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)

        # compute output
        with torch.no_grad():
            output,output_hr,feat = model(tmap)

        error,preds,gts = accuracy_bvp(output,bvpmap)

        list_preds = list_preds + preds.tolist()
        list_gt = list_gt + gts.tolist()
        list_error = list_error + error.tolist()


        if i % 1 == 0:
            for b in range(0,patched_map.size(0)):
                if error[b] > 1000:
                    print(error[b])
                    print(i,b)

                    print(valid_dirs[i*4+b])
                    torch_hr_plot(output,bvpmap)



        acc.update(torch.Tensor(error),output.size()[0])
        if i % 10 == 0 or i == len(valid_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
            'MAE {acc.mae:.4f}\t'
            'RMSE {acc.rmse:.4f}\t'
            'STD {acc.std:.4f}\n'.format(
              epoch, i, len(valid_loader),acc=acc))
            with open(name_of_run+".txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                'MAE {acc.mae:.4f}\t'
                'RMSE {acc.rmse:.4f}\t'
                'STD {acc.std:.4f}\n'.format(
                  epoch, i, len(valid_loader),acc=acc))
    r_score = pearsonr(np.array(list_gt),np.array(list_preds))[0]

    errs = np.array(list_error)
    gtss = np.array(list_gt)
    idx   = np.argsort(gtss)
    errs = errs[idx]
    gtss = gtss[idx]
    return acc,r_score


parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str,required=True)
parser.add_argument('-f','--fold', type=str,required=True)
parser.add_argument('-b','--batch', type=int,required=True)
parser.add_argument('-pt','--whatpt', type=str,required=True)
args = parser.parse_args()

dataset = args.data
name_of_run = "test_"+dataset+args.fold+"_b"+str(args.batch)+"_"+args.whatpt
fold = int(args.fold)-1

BATCH_SIZE = args.batch
NUM_WORKERS = BATCH_SIZE

train_stride = 576
seq_len = 576

train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)

transforms = [ T.ToTensor(),T.Resize((64,576))]
transforms = T.Compose(transforms)

train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms,seq_len=seq_len)

#train_loader_no_temp = DataLoader(train_dataset_no_temp,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

model = SwinTransformerSys(img_size=(64,576),
                                patch_size=4,
                                in_chans=6,
                                num_classes=6,
                                embed_dim=96,
                                depths=[2, 2, 2, 2],
                                depths_decoder=[1, 2, 2, 2],
                                num_heads=[3,6,12,24],
                                window_size=4,
                                mlp_ratio=2,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model.to(device)

model.load_state_dict(torch.load(args.whatpt,map_location=torch.device('cpu')))

criterion_map =MapPearson()
criterion_map = criterion_map.to(device)

criterion_freqreg = FreqReg()
criterion_freqreg = criterion_freqreg.to(device)

criterion_hr = torch.nn.L1Loss()#focal_mse()
criterion_hr = criterion_hr.to(device)

crit_tpl = TPL_softplus()
crit_tpl = crit_tpl.to(device)

criterion_map_fft = MapPSDMSE()
criterion_map_fft = criterion_map_fft.to(device)

criterion_map_fpeak = MapPSDMSE()
criterion_map_fpeak = criterion_map_fpeak.to(device)

for epoch in range(0,1):
    acc,r_score = validate(valid_loader, model,epoch)

print('MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  STD: '+str(acc.std)+'\n')

with open(dataset+"_stats.txt", "a") as file_object:
    # Append 'hello' at the end of file
    file_object.write('Model :'+name_of_run+'  -  MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(acc.std)+'\n')
