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
from utils_signals import butter_bandpass, norm,NegPearson,hr_fft
import pickle
from itertools import permutations

from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft,rfft

from scipy import signal

##############################
# Once a model has been trained, this script can be used to run inference on dataset and calculate error over 576 frame input (19.2s)
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

        if i % 1 == 0:
            for b in range(0,patched_map.size(0)):
                if error[b] > 1000:
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

with open(dataset+"_stats.txt", "a") as file_object:
    # Append 'hello' at the end of file
    file_object.write('Model :'+name_of_run+'  -  MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(acc.std)+'\n')
