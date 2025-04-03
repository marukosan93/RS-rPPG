import os
import time
import random
import numpy as np
import torch
#from torchsummary import summary
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from torchsummary import summary
from MST_tmap2_mv_bg_aug2_sunet_snr_nosq import mst#_dirs import mst
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
from utils_sunet import setup_seed, AverageMeter, Acc,create_datasets,  MapFFTMSE
import argparse
from scipy.stats import pearsonr
from itertools import chain
from signals_stuff import butter_bandpass, norm,NegPearson
import pickle
from itertools import permutations
import pandas as pd
from utils_dl import MapPearson, NegativeMaxCrossCorr
from scipy import signal

np.random.seed(10)
random.seed(10)

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

list_feats = []
stats_train = []
loss_global = []
loss_p_global = []
loss_n1_global = []
loss_n2_global = []
loss_n3_global = []
loss_n4_global = []
loss_sp_global = []
loss_bw_global = []
valid_loss_global = []
valid_loss_p_global = []
valid_loss_n1_global = []
valid_loss_n2_global = []
valid_loss_n3_global = []
valid_loss_n4_global = []
valid_loss_sp_global = []
valid_loss_bw_global = []

f_min = 0.5
f_max = 3

def torch_fft(preds,n=2048):  # tensor [Batch, Temporal]
    f_min = 0.5
    f_max = 3
    fps = 30
    preds_fft = torch.fft.rfft(preds,dim=3,n=n)
    preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)

    f = torch.fft.rfftfreq(n,1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

    preds_psd = preds_psd[:,:,:,indices]

    preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
    return preds_psd

def hr_fft(sig, fs=30, harmonics_removal=True):
    sig = butter_bandpass(sig, 0.7, 3, fs)
    f, Pxx = welch(sig, 30, nperseg=90,nfft=2048)

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


def calc_hr(signal_tensor,fps=30,harmonics_removal=True):
    listino = []
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()
        hr, Pxx, f, sig = hr_fft(signal,harmonics_removal=harmonics_removal)

        listino.append(hr)
    return np.array(listino)

def torch_hr(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048

    preds_mean = torch.mean(preds,dim=[1,2])
    bvp_mean = torch.mean(bvpmap,dim=[1,2])

    gt = calc_hr(bvp_mean)
    hr = calc_hr(preds_mean)
    return hr,gt

def torch_hr_snr(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048
    preds_fft = torch_fft(preds,n)

    bvpmap = bvpmap[:,0,0,:].unsqueeze(1).unsqueeze(1)
    bvpmap_fft = torch_fft(bvpmap,n)

    f = torch.fft.rfftfreq(n,1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]
    f = f[indices].detach().cpu().numpy()
    max_inds = torch.argmax(preds_fft,dim=-1)
    max_ind = np.rint(torch.mean(max_inds.float(),dim=[1,2]).detach().cpu().numpy()).astype(np.int16())

    #if it's too slow make it non-iterative
    masksnr = torch.zeros_like(preds_fft).to(device)
    for b in range(0,preds_fft.size()[0]):
        for c in range(0,preds_fft.size()[1]):
            for r in range(0,preds_fft.size()[2]):
                indice = max_inds[b,c,r]
                masksnr[b,c,r,indice-2:indice+2] = 1
                masksnr[b,c,r,2*indice-2:2*indice+2] = 1


    num = masksnr * preds_fft
    denum = (1-masksnr) * preds_fft

    power_ratio = torch.sum(num,dim=-1)/torch.sum(denum,dim=-1)
    pr = torch.mean(power_ratio,dim=[1,2])

    mean_preds_fft = torch.mean(preds_fft,dim=[1,2])
    mean_gt_fft = bvpmap_fft.squeeze()
    if len(mean_gt_fft.size()) == 1:
        mean_gt_fft = mean_gt_fft.unsqueeze(0)

    hr = f[max_ind]*60

    max_ind_gt = torch.argmax(mean_gt_fft,dim=1).detach().cpu().numpy()
    gt = f[max_ind_gt]*60
    pr = pr.detach().cpu().numpy()
    return hr,gt, pr


def get_snr(preds):  # tensor [Batch, Temporal]
    n = 2048
    preds_fft = torch_fft(preds,n)

    f = torch.fft.rfftfreq(n,1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]
    f = f[indices].detach().cpu().numpy()
    max_inds = torch.argmax(preds_fft,dim=-1)
    max_ind = np.rint(torch.mean(max_inds.float(),dim=[1,2]).detach().cpu().numpy()).astype(np.int16())

    #if it's too slow make it non-iterative
    masksnr = torch.zeros_like(preds_fft).to(device)
    for b in range(0,preds_fft.size()[0]):
        for c in range(0,preds_fft.size()[1]):
            for r in range(0,preds_fft.size()[2]):
                indice = max_inds[b,c,r]
                masksnr[b,c,r,indice-2:indice+2] = 1
                masksnr[b,c,r,2*indice-2:2*indice+2] = 1
    num = masksnr * preds_fft
    denum = (1-masksnr) * preds_fft

    power_ratio = torch.sum(num,dim=-1)/torch.sum(denum,dim=-1)
    pr = torch.mean(power_ratio,dim=[1,2])
    pr = pr.detach()
    return pr

class FreqReg(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(FreqReg, self).__init__()
        return

    def forward(self, preds,fps,f_min,f_max):  # tensor [Batch, Temporal]
        preds_fft = torch.fft.rfft(preds,dim=3)

        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)

        preds_psd -= preds_psd.min(3, keepdim=True)[0]
        preds_psd /= preds_psd.max(3, keepdim=True)[0]

        freqs = torch.fft.rfftfreq(preds.size(3),1/fps[0])
        use_freqs = torch.logical_and(freqs >= f_min, freqs <= f_max)
        not_use_freqs = torch.logical_or(freqs < f_min, freqs > f_max)

        preds_not_psd = preds_psd[:,:,:,not_use_freqs]
        preds_use_psd = preds_psd[:,:,:,use_freqs]
        preds_total_psd = preds_psd[:,:,:,:]

        preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
        n = torch.sum(preds_not_psd,dim=3)
        d = torch.sum(preds_total_psd,dim=3)

        preds_use_psd -= preds_use_psd.min(3, keepdim=True)[0]
        preds_use_psd /= preds_use_psd.max(3, keepdim=True)[0]

        loss_sp = torch.sqrt(torch.mean(torch.square(preds_use_psd)))

        loss_bw = torch.mean(n/d)

        return loss_sp,loss_bw

def accuracy_bvp_snr(output,bvpmap):
    B,C,H,W = output.size()
    pred = []
    gts = []
    pred,gt,pr = torch_hr_snr(output,bvpmap)
    return np.abs(pred-gt),pred,gt,pr


def accuracy_bvp(output,bvpmap):
    B,C,H,W = output.size()
    pred = []
    gts = []
    pred,gt = torch_hr(output,bvpmap)
    return np.abs(pred-gt),pred,gt

def random_ind_diff_pos(length):
    indices = np.arange(0,length)
    new = indices.copy()
    random.shuffle(new)
    while np.count_nonzero(new == indices) > 0:
        random.shuffle(new)
    return new

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

def exhange_places(inp_arr):
    if inp_arr.shape[0] == 8:
        out_arr = inp_arr.copy()
        out_arr[np.where(inp_arr==0)] = 7
        out_arr[np.where(inp_arr==1)] = 6
        out_arr[np.where(inp_arr==2)] = 5
        out_arr[np.where(inp_arr==3)] = 4
        out_arr[np.where(inp_arr==4)] = 2
        out_arr[np.where(inp_arr==5)] = 3
        out_arr[np.where(inp_arr==6)] = 0
        out_arr[np.where(inp_arr==7)] = 1
    else:
        if inp_arr.shape[0] == 4:
            out_arr = inp_arr.copy()
            out_arr[np.where(inp_arr==0)] = 3
            out_arr[np.where(inp_arr==1)] = 2
            out_arr[np.where(inp_arr==2)] = 0
            out_arr[np.where(inp_arr==3)] = 1
        else:
            print("NEED to define for batches that aint 4 or 8")
    return out_arr

def train(train_loader, model,criterion_freqreg, optimizer, epoch,name_of_run,crit_sim,tau,enable_p,enable_n1,enable_fr,bg_enable,mv_enable,enable_roic,enable_tc,coeff_snr,enable_bssnr):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_p = AverageMeter()
    losses_n1 = AverageMeter()
    losses_n2 = AverageMeter()
    losses_n3 = AverageMeter()
    losses_n4 = AverageMeter()
    losses_sp = AverageMeter()
    losses_bw = AverageMeter()
    acc = Acc()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (mstmap_in,tmap,mvmap,bgmap,fr_mstmap_in,masked_map,bvpmap,gt_hr,fps,wave,pr,idx) in enumerate(train_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap_in = mstmap_in.to(device=device, dtype=torch.float)
        fr_mstmap_in = fr_mstmap_in.to(device=device, dtype=torch.float)
        tmap = tmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bgmap = bgmap.to(device=device, dtype=torch.float)
        mvmap = mvmap.to(device=device, dtype=torch.float)
        snr = pr.to(device=device, dtype=torch.float)

        output,output_hr,feat = model(mstmap_in)

        mstmap_in2 = tmap
        output_tmap,output_tmap_hr,feat2 = model(mstmap_in2)

        mstmap_in3 = fr_mstmap_in
        output_fr,output_fr_hr,feat3 = model(mstmap_in3)

        mstmap_in4 = bgmap
        output_bg,output_bg_hr,feat4 = model(mstmap_in4)

        mstmap_in5 = mvmap
        output_mv,output_mv_hr,feat5 = model(mstmap_in5)


        len_temp = mstmap_in.size()[3]
        half = int(len_temp/2)
        offset = int(random.uniform(0, 1)*half)
        if enable_tc == 1:
            offset2 = int(random.uniform(0, 1)*half)
        else:
            offset2 = offset

        neg2 = output_fr[:,:,:,offset:offset+half]
        neg3 = output_bg[:,:,:,offset:offset+half]
        neg4 = output_mv[:,:,:,offset:offset+half]




        anchor = output
        if enable_p == 0:
            pos = output
        else:
            pos = output_tmap

        if enable_bssnr == 1:
            indices_sort = np.argsort(pr.detach().cpu().numpy())
            neg1 = (output[indices_sort,:,:,:])[exhange_places(np.argsort(indices_sort)),:,:,:]
        else:
            list_n = get_perms(mstmap_in.size()[0],True)
            random.shuffle(list_n)
            neg1 = output[list_n[0],:,:,:]

        neg1 = neg1[:,:,:,offset:offset+half]

        if enable_roic == 1:
            shifted_ind = np.array(list(0*64+random_ind_diff_pos(64)))
            shift_ch_ind = np.array(list(random_ind_diff_pos(6)))
            shifted_pos = pos[:,:,shifted_ind,:]
            shifted_pos = shifted_pos[:,shift_ch_ind,:,:]
        else:
            shifted_pos = pos

        anchor = anchor[:,:,:,offset:offset+half]
        shifted_pos = shifted_pos[:,:,:,offset2:offset2+half]

        fft_anchor = torch_fft(anchor)
        fft_shifted_pos = torch_fft(shifted_pos)
        fft_neg1 = torch_fft(neg1)
        fft_neg2 = torch_fft(neg2)
        fft_neg3 = torch_fft(neg3)
        fft_neg4 = torch_fft(neg4)

        loss_p = crit_sim(fft_anchor,fft_shifted_pos)
        loss_n1 = crit_sim(fft_anchor,fft_neg1)
        loss_n2 = crit_sim(fft_anchor,fft_neg2)


        loss_sp,loss_bw = criterion_freqreg(anchor,fps,f_min,f_max)

        if coeff_snr == 0:
            loss_n3 = crit_sim(fft_anchor,fft_neg3)
            loss_n4 = crit_sim(fft_anchor,fft_neg4)
        else:
            inv_snr = 1/snr
            bgmv_w = inv_snr/coeff_snr#F.softmax(snr)

            loss_n3 = torch.mean(bgmv_w*torch.mean(torch.abs(fft_anchor-fft_neg3),dim=(1,2,3)))
            loss_n4 = torch.mean(bgmv_w*torch.mean(torch.abs(fft_anchor-fft_neg4),dim=(1,2,3)))
        #tau = 10
        num = torch.exp(loss_p/tau)
        denum = enable_n1*torch.exp(loss_n1/tau) + enable_fr*torch.exp(loss_n2/tau) +bg_enable*torch.exp(loss_n3/tau)+mv_enable*torch.exp(loss_n4/tau)
        if denum == 0:
            denum = 1
        loss = torch.log(num/denum) #+ enable_sp*loss_sp + enable_bw*loss_bw


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        losses.update(loss.item(), mstmap_in.size(0))
        losses_p.update(loss_p.item(), mstmap_in.size(0))
        losses_n1.update(loss_n1.item(), mstmap_in.size(0))
        losses_n2.update(loss_n2.item(), mstmap_in.size(0))
        losses_n3.update(loss_n3.item(), mstmap_in.size(0))
        losses_n4.update(loss_n4.item(), mstmap_in.size(0))
        losses_sp.update(loss_sp.item(), mstmap_in.size(0))
        losses_bw.update(loss_bw.item(), mstmap_in.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        error,preds,gts = accuracy_bvp(output,bvpmap)
        acc.update(torch.Tensor(error),output.size()[0])

        if i % 100 == 0 or i == len(train_loader)-1:
            #print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_sp {loss_sp.val:.4f} ({loss_sp.avg:.4f})\t'
                  'Loss_bw {loss_bw.val:.4f} ({loss_bw.avg:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_sp=losses_sp,loss_bw=losses_bw))

            with open(name_of_run+".txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_sp {loss_sp.val:.4f} ({loss_sp.avg:.4f})\t'
                  'Loss_bw {loss_bw.val:.4f} ({loss_bw.avg:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_sp=losses_sp,loss_bw=losses_bw))


    loss_global.append(losses.avg)
    loss_p_global.append(losses_p.avg)
    loss_n1_global.append(losses_n1.avg)
    loss_n2_global.append(losses_n2.avg)
    loss_n3_global.append(losses_n3.avg)
    loss_n4_global.append(losses_n4.avg)
    loss_sp_global.append(losses_sp.avg)
    loss_bw_global.append(losses_bw.avg)
    return acc

def validate(valid_loader, model, epoch,name_of_run,tau,crit_sim,enable_p,enable_n1,enable_fr,bg_enable,mv_enable,enable_roic,enable_tc,coeff_snr):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_p = AverageMeter()
    losses_n1 = AverageMeter()
    losses_n2 = AverageMeter()
    losses_n3 = AverageMeter()
    losses_n4 = AverageMeter()
    losses_sp = AverageMeter()
    losses_bw = AverageMeter()

    acc = Acc()

    # switch to train mode
    model.eval()

    list_preds = []
    list_gt = []
    list_error = []

    end = time.time()
    for i, (mstmap_in,tmap,mvmap,bgmap,fr_mstmap_in,masked_map,bvpmap,gt_hr,fps,wave,pr,idx) in enumerate(valid_loader):
        data_time.update(time.time() - end)


        mstmap_in = mstmap_in.to(device=device, dtype=torch.float)
        fr_mstmap_in = fr_mstmap_in.to(device=device, dtype=torch.float)
        tmap = tmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bgmap = bgmap.to(device=device, dtype=torch.float)
        mvmap = mvmap.to(device=device, dtype=torch.float)

        snr = pr.to(device=device, dtype=torch.float)
        ### Generate random permutation for batch that has less repetitions
        if mstmap_in.size(0) <= 2:
            temp_ind = random_ind_diff_pos(mstmap_in.size(0))
        else:
            list_n = get_perms(mstmap_in.size(0),True)
            random.shuffle(list_n)
            temp_ind = list_n[0]

        mstmap_in2 = tmap

        mstmap_in3 = fr_mstmap_in
        mstmap_in4 = bgmap
        mstmap_in5 = mvmap

        # compute output
        with torch.no_grad():
            output,output_hr,feat = model(mstmap_in)

        with torch.no_grad():
            output_tmap,output_tmap_hr,feat2 = model(mstmap_in2)
            output_fr,output_fr_hr,feat3 = model(mstmap_in3)
            output_bg,output_bg_hr,feat4 = model(mstmap_in4)
            output_mv,output_mv_hr,feat5 = model(mstmap_in5)

        len_temp = mstmap_in.size()[3]
        half = int(len_temp/2)
        offset = int(random.uniform(0, 1)*half)
        if enable_tc == 1:
            offset2 = int(random.uniform(0, 1)*half)
        else:
            offset2 = offset

        neg2 = output_fr[:,:,:,offset:offset+half]
        neg3 = output_bg[:,:,:,offset:offset+half]
        neg4 = output_mv[:,:,:,offset:offset+half]

        anchor = output
        if enable_p == 0:
            pos = output
        else:
            pos = output_tmap
        neg1 = output[temp_ind,:,:,:]


        if enable_roic == 1:
            shifted_ind = np.array(list(0*64+random_ind_diff_pos(64)))
            shift_ch_ind = np.array(list(random_ind_diff_pos(6)))
            shifted_pos = pos[:,:,shifted_ind,:]
            shifted_pos = shifted_pos[:,shift_ch_ind,:,:]
        else:
            shifted_pos = pos

        anchor = anchor[:,:,:,offset:offset+half]
        shifted_pos = shifted_pos[:,:,:,offset2:offset2+half]
        neg1 = neg1[:,:,:,offset:offset+half]

        fft_anchor = torch_fft(anchor)
        fft_shifted_pos = torch_fft(shifted_pos)

        loss_sp,loss_bw = criterion_freqreg(anchor,fps,f_min,f_max)

        fft_neg1 = torch_fft(neg1)
        fft_neg2 = torch_fft(neg2)
        fft_neg3 = torch_fft(neg3)
        fft_neg4 = torch_fft(neg4)

        loss_p = crit_sim(fft_anchor,fft_shifted_pos)
        loss_n1 = crit_sim(fft_anchor,fft_neg1)
        loss_n2 = crit_sim(fft_anchor,fft_neg2)

        if coeff_snr == 0:
            loss_n3 = crit_sim(fft_anchor,fft_neg3)
            loss_n4 = crit_sim(fft_anchor,fft_neg4)
        else:
            inv_snr = 1/snr
            bgmv_w = inv_snr/coeff_snr#F.softmax(snr)

            loss_n3 = torch.mean(bgmv_w*torch.mean(torch.abs(fft_anchor-fft_neg3),dim=(1,2,3)))
            loss_n4 = torch.mean(bgmv_w*torch.mean(torch.abs(fft_anchor-fft_neg4),dim=(1,2,3)))

        num = torch.exp(loss_p/tau)# + enable_sp*torch.exp(loss_sp/tau) + enable_bw*torch.exp(loss_bw/tau)
        denum = enable_n1*torch.exp(loss_n1/tau) + enable_fr*torch.exp(loss_n2/tau) +bg_enable*torch.exp(loss_n3/tau)+mv_enable*torch.exp(loss_n4/tau)
        if denum == 0:
            denum = 1
        loss = torch.log(num/denum)#+enable_bw*loss_bw+enable_sp*loss_sp

        loss = loss.float()

        losses.update(loss.item(), mstmap_in.size(0))
        losses_p.update(loss_p.item(), mstmap_in.size(0))
        losses_n1.update(loss_n1.item(), mstmap_in.size(0))
        losses_n2.update(loss_n2.item(), mstmap_in.size(0))
        losses_n3.update(loss_n3.item(), mstmap_in.size(0))
        losses_n4.update(loss_n4.item(), mstmap_in.size(0))
        losses_sp.update(loss_sp.item(), mstmap_in.size(0))
        losses_bw.update(loss_bw.item(), mstmap_in.size(0))

        error,preds,gts = accuracy_bvp(output,bvpmap)

        list_preds = list_preds + preds.tolist()
        list_gt = list_gt + gts.tolist()
        list_error = list_error + error.tolist()

        r_score = pearsonr(np.array(list_gt),np.array(list_preds))[0]

        acc.update(torch.Tensor(error),output.size()[0])
        if i % 10 == 0 or i == len(valid_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
            'MAE {acc.mae:.4f}\t'
            'RMSE {acc.rmse:.4f}\t'
            'R {r_score:.4f}\t'
            'STD {acc.std:.4f}\n'.format(
              epoch, i, len(valid_loader),acc=acc,r_score=r_score))
            with open(name_of_run+".txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                'MAE {acc.mae:.4f}\t'
                'RMSE {acc.rmse:.4f}\t'
                'R {r_score:.4f}\t'
                'STD {acc.std:.4f}\n'.format(
                  epoch, i, len(valid_loader),acc=acc,r_score=r_score))

    r_score = pearsonr(np.array(list_gt),np.array(list_preds))[0]

    valid_loss_global.append(losses.avg)
    valid_loss_p_global.append(losses_p.avg)
    valid_loss_n1_global.append(losses_n1.avg)
    valid_loss_n2_global.append(losses_n2.avg)
    valid_loss_n3_global.append(losses_n3.avg)
    valid_loss_n4_global.append(losses_n4.avg)
    valid_loss_sp_global.append(losses_sp.avg)
    valid_loss_bw_global.append(losses_bw.avg)
    return acc, r_score


def tinit(train_loader):
    #Run one train epoch
    batch_time = AverageMeter()
    acc = Acc()

    list_preds = []
    list_pr = []
    list_gt = []
    list_error = []

    end = time.time()
    for i, (mstmap_in,tmap,mvmap,bgmap,fr_mstmap_in,masked_map,bvpmap,gt_hr,fps,wave,pr,idx) in enumerate(train_loader):
        # measur data loading time
        for b in range(0,2):
            fig,ax = plt.subplots(5,1)
            ax[0].imshow(mstmap_in[b,:,:,:].permute(1,2,0)[:,:,:3].detach().cpu().numpy())
            ax[1].imshow(tmap[b,:,:,:].permute(1,2,0)[:,:,:3].detach().cpu().numpy())
            ax[2].imshow(fr_mstmap_in[b,:,:,:].permute(1,2,0)[:,:,:3].detach().cpu().numpy())
            ax[3].imshow(bgmap[b,:,:,:].permute(1,2,0)[:,:,:3].detach().cpu().numpy())
            ax[4].imshow(mvmap[b,:,:,:].permute(1,2,0)[:,:,:3].detach().cpu().numpy())
            plt.show()

        tmap = tmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)

        #print(1/get_snr(tmap))
        batch_time.update(time.time() - end)
        end = time.time()
        output = tmap

        error,preds,gts,pr = accuracy_bvp_snr(output,bvpmap)

        list_preds = list_preds + preds.tolist()
        list_pr= list_pr + pr.tolist()
        list_gt = list_gt + gts.tolist()
        list_error = list_error + error.tolist()

        #for b in range(0,out_feat.size()[0]):
        #    list_feats.append(out_feat[b].detach().cpu().numpy())
    return list_preds,list_pr

def new_order(new_train_dirs,new_list_preds,new_list_pr,phase,BATCH_SIZE):

    total_len = len(new_list_pr)
    crop_n = total_len#int(round((total_len/2)*phase))
    new_train_dirs = new_train_dirs[0:crop_n]
    new_list_preds = new_list_preds[0:crop_n]
    new_list_pr = new_list_pr[0:crop_n]

    #after cropping to right phase just run old code
    new_train_dirs = mit.sort_together([new_list_preds, new_train_dirs])[1]

    split_train_dirs = ([list(x) for x in mit.divide(2*BATCH_SIZE,new_train_dirs)])
    indices = np.arange(0,int(2*BATCH_SIZE))
    indices = np.array(list(indices[::2])+list(indices[1::2]))
    split_train_dirs = [split_train_dirs[i] for i in indices]
    for indc in range(0,len(split_train_dirs)):
        random.shuffle(split_train_dirs[indc])
    new_train_dirs=[val for tup in zip(*split_train_dirs) for val in tup]

    #CHECK IF ANY OF THE SAME DIRS AND EXCHANGE
    num_batches = int(len(new_train_dirs)/4)
    for b in range(0,num_batches):
        cazzino = new_train_dirs[b*4:4*(b+1)]
        cavolo = []
        for c in range(0,4):
            cavolo.append(cazzino[c][0])
        if len(set(cavolo))<4:
            index = np.where(pd.Series(cavolo).duplicated())[0][0]
            #temp = new_train_dirs[b*4:index]
            rnd_b = random.choice(np.arange(0,num_batches))
            temp = new_train_dirs[rnd_b*4+index]
            new_train_dirs[rnd_b*4+index] = new_train_dirs[b*4+index]
            new_train_dirs[b*4+index] = temp

    new_train_dataset = mst(data=new_train_dirs,stride=train_stride,shuffle=False, Training = True, transform=transforms,seq_len=seq_len)
    train_loader = DataLoader(new_train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
    return train_loader

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', type=str,required=True)
parser.add_argument('-d','--data', type=str,required=True)
parser.add_argument('-f','--fold', type=str,required=True)
parser.add_argument('-m','--margin', type=float,required=True)
parser.add_argument('-lr','--lr', type=str,required=True)
parser.add_argument('-b','--batch', type=int,required=True)
parser.add_argument('-pre','--pre', type=int,required=True)
parser.add_argument('-e','--epochs', type=int,required=True)
parser.add_argument('-bg','--bg', type=int,required=True)
parser.add_argument('-mov','--mov', type=int,required=True)
parser.add_argument('-fr','--fr', type=int,required=True)
parser.add_argument('-n1','--n1', type=int,required=True)
parser.add_argument('-p1','--p1', type=int,required=True)
parser.add_argument('-roic','--roic', type=int,required=True)
parser.add_argument('-tc','--tc', type=int,required=True)
parser.add_argument('-ord','--ord', type=int,required=True)
parser.add_argument('-wsnr','--wsnr', type=float,required=True)
parser.add_argument('-bssnr','--bssnr', type=int,required=True)
parser.add_argument('-shuff','--shuff', type=int,required=True)
args = parser.parse_args()

dataset = args.data
name_of_run = args.name+"_rs+rppg_v2_"+dataset+args.fold+"_"+str(args.margin).replace(".","_")+"_b"+str(args.batch)+"_lr"+str(args.lr)+"_pre"+str(args.pre)+"_e"+str(args.epochs)+"_p1"+str(args.p1)+"_n1"+str(args.n1)+"_fr"+str(args.fr)+"_bg"+str(args.bg)+"_mv"+str(args.mov)+"_roic"+str(args.roic)+"_tc"+str(args.tc)+"_ord"+str(args.ord)+"_wsnr"+str(args.wsnr)+"_bssnr"+str(args.bssnr)+"_shuff"+str(args.shuff)
if args.fold != "whole":
    fold = int(args.fold)-1
else:
    fold = args.fold

BATCH_SIZE = args.batch
NUM_WORKERS = 2*BATCH_SIZE
if NUM_WORKERS > 10:
    NUM_WORKERS = 10
train_stride = 576
seq_len = 576

if dataset != "scamps":
    train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)
else:
    train_dirs = []
    valid_dirs = []
    for gigio in range(1,2800+1):
        train_dirs.append((os.path.join("./MSTmaps/scamps_mstmaps/","P"+str(gigio).zfill(6)),12,1))
        if gigio % 28 == 0:
            valid_dirs.append((os.path.join("./MSTmaps/scamps_mstmaps/","P"+str(gigio).zfill(6)),12,1))

transforms = [ T.ToTensor(),T.Resize((64,576))]
transforms = T.Compose(transforms)

#train_dirs = train_dirs[:10]
#valid_dirs = valid_dirs[:2]

train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms,seq_len=seq_len)

#train_loader_no_temp = DataLoader(train_dataset_no_temp,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)

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
                                drop_rate=0.2,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model.to(device)

#model = torch.nn.DataParallel(model)

criterion_freqreg = FreqReg()
criterion_freqreg = criterion_freqreg.to(device)

crit_sim = torch.nn.L1Loss()

optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=float(args.lr), weight_decay=0.05) #from SWIN paper

if args.ord == 1:
    list_preds,list_pr = tinit(train_loader)

    new_train_dirs = mit.sort_together([list_pr, train_dirs])[1][::-1]
    new_list_preds = mit.sort_together([list_pr, list_preds])[1][::-1]
    new_list_pr = mit.sort_together([list_pr, list_preds])[0][::-1]

    train_loader = new_order(new_train_dirs,new_list_preds,new_list_pr,1,BATCH_SIZE)

if args.pre == 1:
    model.load_state_dict(torch.load("trained/mstmap2tmap2_"+dataset+args.fold+".pt"))
rmse_list = []
mae_list = []
rmse_list_train = []
mae_list_train = []
best_loss_bw = 99
best_acc = 0
best_r_score = 0

for epoch in range(0, args.epochs):
    shuff = int(args.shuff)
    if shuff != 0 and args.ord == 1:
        if epoch % shuff == 0:
            train_loader = new_order(new_train_dirs,new_list_preds,new_list_pr,1,BATCH_SIZE)

    acc_tr = train(train_loader, model,criterion_freqreg, optimizer, epoch,name_of_run,crit_sim,args.margin,args.p1,args.n1,args.fr,args.bg,args.mov,args.roic,args.tc,args.wsnr,args.bssnr)
    rmse_list_train.append(acc_tr.rmse)
    mae_list_train.append(acc_tr.mae)

    if args.fold != "whole":

        acc,r_score = validate(valid_loader, model, epoch,name_of_run,args.margin,crit_sim,args.p1,args.n1,args.fr,args.bg,args.mov,args.roic,args.tc,args.wsnr)
        rmse_list.append(acc.rmse)
        mae_list.append(acc.mae)

        best_acc = acc
        best_r_score = r_score

        fig,ax = plt.subplots(1,1)
        plt.plot(rmse_list,'b-')
        plt.plot(mae_list,'r-')
        plt.plot(rmse_list_train,'b--')
        plt.plot(mae_list_train,'r--')
        plt.ylim(top=12)
        plt.grid()
        plt.savefig(name_of_run+"_rmse_mae.png")
        plt.close()

        fig,ax = plt.subplots(1,1)

        plt.plot(loss_global,'r-')
        plt.plot(valid_loss_global,'r--')

        plt.plot(loss_sp_global,'y-')
        plt.plot(valid_loss_sp_global,'y--')

        plt.plot(loss_bw_global,'g-')
        plt.plot(valid_loss_bw_global,'g--')
        plt.grid()
        plt.savefig(name_of_run+"_loss_main.png")
        plt.close()

        fig,ax = plt.subplots(1,1)
        #plt.plot(loss_global,'b-')
        plt.plot(loss_p_global,'r-')
        plt.plot(valid_loss_p_global,'r--')

        plt.plot(loss_n1_global,'y-')
        plt.plot(valid_loss_n1_global,'y--')

        plt.plot(loss_n2_global,'g-')
        plt.plot(valid_loss_n2_global,'g--')

        plt.plot(loss_n3_global,'m-')
        plt.plot(valid_loss_n3_global,'m--')

        plt.plot(loss_n4_global,'o-')
        plt.plot(valid_loss_n4_global,'o--')
        plt.grid()
        plt.savefig(name_of_run+"_loss_pn.png")
        plt.close()
        #if len(rmse_list)>1:
        #    if rmse_list[-1] < min(rmse_list[:-1]):
    print("Save")
    torch.save(model.state_dict(), name_of_run+".pt")

if args.fold != "whole":
    with open("ssl_stats.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write('Model :'+name_of_run+'  -  MAE: '+str(best_acc.mae)+'  -  RMSE: '+str(best_acc.rmse)+'  -  R: '+str(best_r_score)+'  -  STD: '+str(best_acc.std)+'\n')
