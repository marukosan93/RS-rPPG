import os
import time
import random
import numpy as np
import torch
#from torchsummary import summary
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from torchsummary import summary
from MST_tmap_sunet_nosq import mst#_dirs import mst
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
import heartpy as hp
from scipy.stats import pearsonr
from itertools import chain
from utils_signals import butter_bandpass, norm, hr_fft,NegPearson
import pickle
from itertools import permutations
import pandas as pd

#########
# Contrastive learning using RS-rPPG method
#########

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
loss_sp_global = []
loss_bw_global = []
loss_tf_global = []

f_min = 0.5
f_max = 3

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

def calc_hr(signal_tensor,fps=30):
    listino = []
    for b in range(signal_tensor.size(0)):
        signal = signal_tensor[b].detach().cpu().numpy()
        ### WELCH
        signal = butter_bandpass(signal, 0.5, 3, 30)
        f, Pxx = welch(signal, fps, nperseg=160,nfft=2048)
        hr = f[np.argmax(Pxx)]*60
        #hr, Pxx, f, sig = hr_fft(signal)
        listino.append(hr)

    return np.array(listino)

def torch_hr(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048

    preds_mean = torch.mean(preds,dim=[1,2])
    bvp_mean = torch.mean(bvpmap,dim=[1,2])

    gt = calc_hr(bvp_mean)
    hr = calc_hr(preds_mean)
    return hr,gt

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

def accuracy_bvp(output,bvpmap):
    B,C,H,W = output.size()
    pred = []
    gts = []
    """for b in range(0,B):
        bvp = bvpmap[b].permute(1,2,0).detach().cpu().numpy()
        gt,_,_,_ = hr_fft(bvp[10,:,0],harmonics_removal=False)
        gts.append(gt)
    gt = np.array(gts)"""
    pred,gt = torch_hr(output,bvpmap)
    return np.abs(pred-gt),pred,gt

def random_ind_diff_pos(length):
    indices = np.arange(0,length)
    new = indices.copy()
    random.shuffle(new)
    while np.count_nonzero(new == indices) > 0:
        random.shuffle(new)
    return new

def train(train_loader, model,crit_tpl,criterion_freqreg, optimizer, epoch,name_of_run):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sp = AverageMeter()
    losses_bw = AverageMeter()
    losses_tf = AverageMeter()
    acc = Acc()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (patched_map,tmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(train_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        patched_map = patched_map.to(device=device, dtype=torch.float)
        tmap = tmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)

        output,output_hr,feat = model(patched_map)

        patched_map2 = tmap
        output2,output2_hr,feat2 = model(patched_map2)

        anchor = output
        pos = output2

        ### Generate random permutation for batch that has less repetitions
        if patched_map.size(0) <= 2:
            neg = output[random_ind_diff_pos(patched_map.size(0)),:,:,:]
        else:
            list_n = get_perms(patched_map.size(0),True)
            random.shuffle(list_n)
            neg = output[list_n[0],:,:,:]

        loss_sp,loss_bw = criterion_freqreg(anchor,fps,f_min,f_max)
        loss_sp2,loss_bw2 = criterion_freqreg(pos,fps,f_min,f_max)

        loss_sp = loss_sp+loss_sp2
        loss_bw = loss_bw+loss_bw2

        shifted_ind = np.array(list(0*64+random_ind_diff_pos(64)))
        shifted_pos = pos[:,:,shifted_ind,:]
        shifted_pos = shifted_pos[:,np.array([2,0,1]),:,:]

        loss_tf = crit_tpl(torch_fft(anchor),torch_fft(shifted_pos),torch_fft(neg))
        loss = loss_sp+loss_bw+loss_tf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        losses.update(loss.item(), patched_map.size(0))
        losses_sp.update(loss_sp.item(), patched_map.size(0))
        losses_bw.update(loss_bw.item(), patched_map.size(0))
        losses_tf.update(loss_tf.item(), patched_map.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(train_loader)-1:
            #print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_tf {loss_tf.val:.4f} ({loss_tf.avg:.4f})\t'
                  'Loss_sp {loss_sp.val:.4f} ({loss_sp.avg:.4f})\t'
                  'Loss_bw {loss_bw.val:.4f} ({loss_bw.avg:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_sp=losses_sp,loss_bw=losses_bw,loss_tf=losses_tf))

            with open(name_of_run+".txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_tf {loss_tf.val:.4f} ({loss_tf.avg:.4f})\t'
                  'Loss_sp {loss_sp.val:.4f} ({loss_sp.avg:.4f})\t'
                  'Loss_bw {loss_bw.val:.4f} ({loss_bw.avg:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_sp=losses_sp,loss_bw=losses_bw,loss_tf=losses_tf))


    loss_global.append(losses.avg)
    loss_sp_global.append(losses_sp.avg)
    loss_bw_global.append(losses_bw.avg)
    loss_tf_global.append(losses_tf.avg)

def validate(valid_loader, model, epoch,name_of_run):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = Acc()

    # switch to train mode
    model.eval()

    list_preds = []
    list_gt = []
    list_error = []

    end = time.time()
    for i, (patched_map,tmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        patched_map = patched_map.to(device=device, dtype=torch.float)
        masked_map = masked_map.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)

        # compute output
        with torch.no_grad():
            output,output_hr,feat = model(patched_map)

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

    errs = np.array(list_error)
    gtss = np.array(list_gt)
    idx   = np.argsort(gtss)
    errs = errs[idx]
    gtss = gtss[idx]
    plt.plot(errs)
    plt.plot(gtss)
    plt.savefig(name_of_run+'error.png')
    plt.close()
    return acc, r_score

def tinit(train_loader):
    #Run one train epoch
    batch_time = AverageMeter()
    acc = Acc()

    list_preds = []
    list_gt = []
    list_error = []

    end = time.time()
    for i, (patched_map,tmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(train_loader):
        # measur data loading time
        tmap = tmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)

        batch_time.update(time.time() - end)
        end = time.time()
        output = tmap

        error,preds,gts = accuracy_bvp(output,bvpmap)

        list_preds = list_preds + preds.tolist()
        list_gt = list_gt + gts.tolist()
        list_error = list_error + error.tolist()

        #for b in range(0,out_feat.size()[0]):
        #    list_feats.append(out_feat[b].detach().cpu().numpy())
    return list_preds

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', type=str,required=True)
parser.add_argument('-d','--data', type=str,required=True)
parser.add_argument('-f','--fold', type=str,required=True)
parser.add_argument('-m','--margin', type=float,required=True)
parser.add_argument('-b','--batch', type=int,required=True)
parser.add_argument('-pre','--pre', type=int,required=True)
parser.add_argument('-e','--epochs', type=int,required=True)
args = parser.parse_args()

dataset = args.data
name_of_run = args.name+"_singlet_hrfft_"+dataset+args.fold+"_"+str(args.margin).replace(".","_")+"_b"+str(args.batch)+"_pre"+str(args.pre)+"_e"+str(args.epochs)
if args.fold != "whole":
    fold = int(args.fold)-1
else:
    fold = args.fold

BATCH_SIZE = args.batch
NUM_WORKERS = 2*BATCH_SIZE

train_stride = 576
seq_len = 576

train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)
transforms = [ T.ToTensor(),T.Resize((64,576))]
transforms = T.Compose(transforms)

train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=False, Training = True, transform=transforms,seq_len=seq_len)
valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms,seq_len=seq_len)

#train_loader_no_temp = DataLoader(train_dataset_no_temp,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

model = SwinTransformerSys(img_size=(64,576),
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
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

#model = torch.nn.DataParallel(model)

criterion_freqreg = FreqReg()
criterion_freqreg = criterion_freqreg.to(device)

crit_tpl = torch.nn.TripletMarginLoss(margin=args.margin,p=1)
crit_tpl = crit_tpl.to(device)

optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=1e-5, weight_decay=0.05) #from SWIN paper

list_preds = tinit(train_loader)
new_train_dirs = mit.sort_together([list_preds, train_dirs])[1]

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

if args.pre == 1:
    model.load_state_dict(torch.load("trained/mstmap2tmap_"+dataset+args.fold+".pt"))
rmse_list = []
mae_list = []
for epoch in range(0, args.epochs):
    train(train_loader, model,crit_tpl,criterion_freqreg, optimizer, epoch,name_of_run)
    if args.fold != "whole":

        acc,r_score = validate(valid_loader, model, epoch,name_of_run)
        rmse_list.append(acc.rmse)
        mae_list.append(acc.mae)
        fig,ax = plt.subplots(1,1)
        plt.plot(rmse_list,'b-')
        plt.plot(mae_list,'r-')
        plt.grid()
        plt.savefig(name_of_run+"_rmse_mae")
        plt.close()

        #if len(rmse_list)>1:
        #    if rmse_list[-1] < min(rmse_list[:-1]):
    print("Save")
    torch.save(model.state_dict(), name_of_run+".pt")

if args.fold != "whole":
    with open("ssl_stats.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write('Model :'+name_of_run+'  -  MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(acc.std)+'\n')
