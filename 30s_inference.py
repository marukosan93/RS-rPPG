import os
import time
import random
import numpy as np
import torch
#from torchsummary import summary
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from torchsummary import summary
from MST_tmap_sunet_nosq import mst
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
from utils_signals import butter_bandpass, norm,NegPearson, hr_fft
import pickle
from itertools import permutations

from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft,rfft

from scipy import signal

list_pred_sig = []
list_bvp_sig = []

##############################
# Once a model has been trained, this script can be used to run inference on dataset and calculate error over 30s segments
##############################


def torch_hr(preds,bvpmap):  # tensor [Batch, Temporal]
    n = 2048

    preds_mean = torch.mean(preds,dim=[1,2])
    bvp_mean = torch.mean(bvpmap,dim=[1,2])

    gt = calc_hr(bvp_mean,harmonics_removal=True)
    hr = calc_hr(preds_mean,harmonics_removal=False)
    return hr,gt

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
    for i, (patched_map,tmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(valid_loader):
        #if i < 18:
        #    continue
        #gt_hr = (gt_hr-40)/140
        # measur data loading time
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

        r_score = pearsonr(np.array(list_gt),np.array(list_preds))[0]

        if i % 1 == 0:
            for b in range(0,patched_map.size(0)):
                if error[b] > 1000:
                #if (i == 13 and b == 3) or (i==14 and b ==0):
                    print(error[b])
                    print(i,b)

                    print(valid_dirs[i*4+b])



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


def stats_30s(vids,pred_sig,bvp_sig,dataset):
    list_pred_hr = []
    list_gt_hr = []
    sl = 30*30
    for vid_num in range(0,np.max(vids)+1):
        pred_arr = pred_sig[vids == vid_num]
        bvp_arr = bvp_sig[vids == vid_num]
        pred = np.concatenate(pred_arr,axis=0)
        bvp = np.concatenate(bvp_arr,axis=0)
        pred = butter_bandpass(pred, 0.5, 3, 30)
        bvp = butter_bandpass(bvp, 0.5, 3, 30)

        if dataset == "obf":
            n_30s = int(len(pred)/(sl))
            diff = len(pred)-n_30s*sl
            offset = int(diff/2)
            for index in range(0,n_30s):
                pred_part = pred[offset+index*sl:offset+(index+1)*sl]
                bvp_part = bvp[offset+index*sl:offset+(index+1)*sl]
                hr, Pxx_hr, f, sig = hr_fft(pred_part)

                gt, Pxx_gt, f, sig = hr_fft(bvp_part)
                if abs(hr-gt) >1000:
                    print(hr)
                    print(gt)
                    print("-")
                    fig,ax = plt.subplots(2,1)
                    ax[0].plot(norm(bvp_part),color="black")
                    ax[0].plot(norm(pred_part))
                    ax[1].plot(f,norm(Pxx_gt),color="black")
                    ax[1].plot(f,norm(Pxx_hr))
                    plt.show()

                list_pred_hr.append(hr)
                list_gt_hr.append(gt)
        if dataset == "pure":
            if len(pred) > 1800:
                half = 900
            else:
                half = int(len(pred)/2)
            for index in range(0,2):
                pred_part = pred[index*half:(index+1)*half]
                bvp_part = bvp[index*half:(index+1)*half]
                hr, Pxx_hr, f, sig = hr_fft(pred_part)

                gt, Pxx_gt, f, sig = hr_fft(bvp_part)
                if abs(hr-gt) > 10000:
                    print(hr)
                    print(gt)
                    print("-")
                    fig,ax = plt.subplots(2,1)
                    ax[0].plot(bvp_part,color="black")
                    ax[0].plot(pred_part)
                    ax[1].plot(Pxx_gt,color="black")
                    ax[1].plot(Pxx_hr)
                    plt.show()
                list_pred_hr.append(hr)
                list_gt_hr.append(gt)

    r_score = pearsonr(np.array(list_pred_hr),np.array(list_gt_hr))[0]
    error = np.abs(np.array(list_pred_hr)-np.array(list_gt_hr))
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(np.square(error)))
    std = np.std(error)
    return mae,rmse,r_score,std

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str,required=True)
parser.add_argument('-f','--fold', type=str,required=True)
parser.add_argument('-b','--batch', type=int,required=True)
parser.add_argument('-pt','--whatpt', type=str,required=True)
args = parser.parse_args()

dataset = args.data
name_of_run = "test_"+dataset+args.fold+"_b"+str(args.batch)+"_"+args.whatpt
if args.fold != "whole":
    fold = int(args.fold)-1
else:
    fold = args.fold
BATCH_SIZE = args.batch
NUM_WORKERS = BATCH_SIZE

train_stride = 576
seq_len = 576

train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)

list_vids = []
numeretto = 0
prev = ""
for indice,direc in enumerate(valid_dirs):
    direc = direc[0][-5:]
    if indice > 0 and prev != direc:
        numeretto+=1
    list_vids.append(numeretto)
    prev = direc

transforms = [ T.ToTensor(),T.Resize((64,576))]
transforms = T.Compose(transforms)

train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
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
model.load_state_dict(torch.load(args.whatpt))

for epoch in range(0,1):
    acc,r_score = validate(valid_loader, model,epoch)

print('MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  STD: '+str(acc.std)+'\n')

mae, rmse, r_score, std = stats_30s(np.array(list_vids),np.array(list_pred_sig),np.array(list_bvp_sig),dataset)

print('MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(std)+'\n')

with open(dataset+"_stats.txt", "a") as file_object:
    file_object.write('Model :'+name_of_run+'  -  MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(std)+'\n')
