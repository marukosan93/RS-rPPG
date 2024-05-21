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
from utils_signals import butter_bandpass, norm, hr_fft,NegPearson
import pickle

#########
# Use this to pre-train the Network to predict Tmaps from MSTmaps, then the pre-trained network if futher refined with train_rsppg by training contrastively
#########


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


def torch_fft(preds):  # tensor [Batch, Temporal]
    f_min = 0.5
    f_max = 3
    fps = 30
    #fig ,ax = plt.subplots(6,1)
    preds_fft = torch.fft.rfft(preds,dim=3)
    preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)

    f = torch.fft.rfftfreq(preds.size(3),1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

    preds_psd = preds_psd[:,:,:,indices]

    preds_psd = torch.div(preds_psd,torch.sum(preds_psd,3,keepdim=True)) #normalise
    #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
    #loss = loss/power_labels
    return preds_psd



class FreqReg(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self):
        super(FreqReg, self).__init__()
        return

    def forward(self, preds,fps,f_min,f_max):  # tensor [Batch, Temporal]


        """preds_long = torch.ones((preds.size()[0],preds.size()[1],int(preds.size()[2]/4),int(preds.size()[3]*4)))
        preds_long[:,:,:,0:224] = preds[:,:,0:56,:]
        preds_long[:,:,:,224:448] = preds[:,:,56:112,:]
        preds_long[:,:,:,448:672] = preds[:,:,112:168,:]
        preds_long[:,:,:,672:896] = preds[:,:,168:224,:]"""
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
        #plt.plot(preds_use_psd[0,0,0,:].detach().cpu().numpy())


        preds_use_psd -= preds_use_psd.min(3, keepdim=True)[0]
        preds_use_psd /= preds_use_psd.max(3, keepdim=True)[0]
        #plt.plot(preds_use_psd[0,0,0,:].detach().cpu().numpy())

        #preds_use_psd /= torch.sum(preds_use_psd,dim=3,keepdim=True)

        loss2 = torch.sqrt(torch.mean(torch.square(preds_use_psd)))
        #loss2 = torch.mean(preds_use_psd)


        #exit()
        #loss2 = torch.mean(preds_use_psd.max(3, keepdim=True)[0])

        """peaks = preds_use_psd.max(3, keepdim=True).indices
        peaks_left = torch.clip(peaks-torch.ones_like(peaks),min=0)
        peaks_right = torch.clip(peaks+torch.ones_like(peaks),max=preds_use_psd.size(3)-1)
        preds_peak = torch.gather(preds_use_psd,3,peaks)
        preds_peak_left = torch.gather(preds_use_psd,3,peaks_left)
        preds_peak_right = torch.gather(preds_use_psd,3,peaks_right)"""

        """peak_power = preds_peak+preds_peak_left+preds_peak_right
        peak_power = peak_power.squeeze()"""

        #loss2 = torch.mean(peak_power/d2)
        term1 = torch.mean(n)
        term2 = torch.mean(d)
        loss = torch.mean(n/d)

        #print(term1)
        #print(term2)
        #print(loss)
        """#preds_psd = nn.functional.softmax(preds_psd,dim=3)
        term1 = torch.sqrt(torch.mean(torch.square(preds_not_psd)))
        term2 = -torch.sqrt(torch.mean((torch.square(preds_psd))))

        #loss = crit_fft(preds_psd,labels_psd)
        #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        #loss = loss/power_labels
        rc1 = 1
        rc2 = 1
        loss = rc1*term1+rc2*term2"""
        return loss,term1,term2,loss2

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
        self.std = math.sqrt(self.sqr_sum / self.count-(self.sum / self.count)**2)

def calc_hr(signal,fps):
        eccezione = False
        try:
            working_data, measures = hp.process(signal, fps)
            hr = measures['bpm']
        except:
            eccezione = True
            hr = 0
        if math.isnan(hr) or eccezione:
            f, Pxx = welch(signal, fps, nperseg=160,nfft=2048)
            hr = f[np.argmax(Pxx)]*60
        return hr

def accuracy_bvp(output,bvpmap):
    B,C,H,W = output.size()
    pred = []
    gts = []

    for b in range(0,B):
        list_hr = []
        bvp = bvpmap[b].permute(1,2,0).detach().cpu().numpy()
        gt,_,_,_ = hr_fft(bvp[10,:,0])
        gts.append(gt)

        out = output[b].permute(1,2,0).detach().cpu().numpy()
        for c in range(0,C):
            for h in range(0,64):
                hr,_,_,_ = hr_fft(out[h,:,c])
                #hr = np.mean([hr1,hr2,hr3])
                list_hr.append(hr)
        list_hr.sort()
        list_hr = list_hr[48:-48]
        #list_hr2 = list_hr2[16:48]
        #list_hr3 = list_hr3[16:48]
        hr = np.mean(list_hr)
        pred.append(hr)
    pred = np.array(pred)
    gt = np.array(gts)
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
    """fig,ax = plt.subplots(3,1)
    ax[0].imshow(tenz[0].permute(1,2,0).detach().cpu().numpy())
    print(tenz.size())"""
    chunks = torch.split(tenz,64,dim=3)
    """print(len(chunks))"""
    tenz = torch.cat(chunks,dim=2)
    """ax[1].imshow(tenz[0].permute(1,2,0).detach().cpu().numpy())

    print(tenz.size())"""
    chunks = list(torch.split(tenz,16,dim=2))
    chunks = [chunks[x] for x in indices]
    tenz = torch.cat(chunks,dim=2)
    """ax[2].imshow(tenz[0].permute(1,2,0).detach().cpu().numpy())

    plt.show()
    print(chunks[0].size())
    exit()"""
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

def train(train_loader, model,criterion_map,criterion_hr,criterion_map_fft,crit_tpl,criterion_freqreg, optimizer, epoch,name_of_run,random_36_list,random_n_list):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_rppg = AverageMeter()
    losses_reg = AverageMeter()
    losses2 = AverageMeter()
    losses_pos = AverageMeter()
    losses_neg = AverageMeter()
    losses_neg1 = AverageMeter()
    losses_neg2 = AverageMeter()
    losses_neg3 = AverageMeter()
    losses_pos_hr = AverageMeter()
    losses_neg_hr = AverageMeter()
    acc = Acc()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (patched_map,tmap,masked_map,bvpmap,gt_hr,fps,wave,idx) in enumerate(train_loader):
        gt_hr = (gt_hr-40)/140
        # measur data loading time
        data_time.update(time.time() - end)
        patched_map = patched_map.to(device=device, dtype=torch.float)
        tmap = tmap.to(device=device, dtype=torch.float)
        masked_map = masked_map.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        gt_hr = gt_hr.to(device=device, dtype=torch.float)


        output,output_hr,feat = model(patched_map)


        loss = delta*criterion_map_fft(output,tmap,fps,f_min,f_max)

        #loss_neg = loss_neg+loss_neg2+loss_neg3
        #criterion_map_fft(pos,out_stacked,fps,f_min,f_max)+criterion_map_fft(pos2,output,fps,f_min,f_max)

        #loss =0.001*loss_reg+torch.log((torch.exp(loss_pos))/(torch.exp(loss_pos)+torch.exp(loss_neg)+torch.exp(loss_neg2)+torch.exp(loss_neg3)))
        #print(loss)
        #loss = delta*(loss_pos-loss_neg)+0.2*loss_reg+1*loss2#+loss2#-loss_neg)#+delta*loss_reg#delta*loss_mapfft+gamma*loss_maprppg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        loss_maprppg = criterion_map(output,bvpmap)#1+criterion_map(output,bvpmap,fps[0].detach().cpu().numpy())

        # measure accuracy and record loss
        losses.update(loss.item(), patched_map.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        """normpsd = CalculateNormPSD(30, 0.7, 6)
        f,patched_map_fft =normpsd(patched_map)
        f,output_fft =normpsd(output)
        f,bvpmap_fft =normpsd(bvpmap)"""

        if i % 100 == 0 or i == len(train_loader)-1:
            #print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_g {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

            with open(name_of_run+".txt", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_g {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))
    loss_global_pos.append(losses.avg)

def validate(valid_loader, model,criterion_map,criterion_hr,criterion_map_fft, epoch,name_of_run,random_36_list,random_n_list):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_rppg = AverageMeter()
    losses_pos = AverageMeter()
    losses_neg = AverageMeter()
    losses_pos_hr = AverageMeter()
    losses_neg_hr = AverageMeter()
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
        gt_hr = (gt_hr-40)/140
        # measur data loading time
        data_time.update(time.time() - end)
        patched_map = patched_map.to(device=device, dtype=torch.float)
        masked_map = masked_map.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        gt_hr = gt_hr.to(device=device, dtype=torch.float)

        # compute output
        output,output_hr,feat = model(patched_map)


        output_lin = output
        bvpmap_lin = bvpmap

        error,preds,gts = accuracy_bvp(output_lin,bvpmap_lin)

        list_preds = list_preds + preds.tolist()
        list_gt = list_gt + gts.tolist()
        list_error = list_error + error.tolist()

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

    errs = np.array(list_error)
    gtss = np.array(list_gt)
    idx   = np.argsort(gtss)
    errs = errs[idx]
    gtss = gtss[idx]
    plt.plot(errs)
    plt.plot(gtss)
    plt.savefig(name_of_run+'error.png')
    plt.close()
    return acc



parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str,required=True)
parser.add_argument('-f','--fold', type=str,required=True)
args = parser.parse_args()

dataset = args.data
if args.fold != "whole":
    fold = int(args.fold)-1
else:
    fold = args.fold
name_of_run = "mstmap2tmap_"+dataset+args.fold

BATCH_SIZE = 4
NUM_WORKERS = 0

train_stride = 576
seq_len = 576

train_dirs, valid_dirs = create_datasets(dataset,fold,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)
transforms = [ T.ToTensor(),T.Resize((64,576))]
transforms = T.Compose(transforms)

"""eq_train_dirs = []
eq_valid_dirs = []
for cavolo in train_dirs:
    eq_train_dirs.append((cavolo[0].replace("VIPL-HR","equalised_VIPL-HR"),cavolo[1],cavolo[2]))
for cavolo in valid_dirs:
    eq_valid_dirs.append((cavolo[0].replace("VIPL-HR","equalised_VIPL-HR"),cavolo[1],cavolo[2]))
train_dirs = eq_train_dirs
valid_dirs = eq_valid_dirs"""

train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms,seq_len=seq_len)

N_BATCHES = int(len(train_dataset)/BATCH_SIZE)+1
random_36_list = []
random_n_list = []
for nb in range(0,N_BATCHES):
    random_36_list.append(random_ind_diff_pos(36))
    random_n_list.append(random_ind_diff_pos(BATCH_SIZE))

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

criterion_map =MapPearson() #NegativeMaxCrossCorr(180,42) #
criterion_map = criterion_map.to(device)
criterion_freqreg = FreqReg()

criterion_hr = torch.nn.L1Loss()#focal_mse()
criterion_hr = criterion_hr.to(device)

crit_tpl = torch.nn.TripletMarginLoss(margin=1,p=1)
crit_tpl = crit_tpl.to(device)

criterion_map_fft = MapPSDMSE()
criterion_map_fft = criterion_map_fft.to(device)

optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=1e-5, weight_decay=0.05) #from SWIN paper
#model.load_state_dict(torch.load("obf1.pt"))
rmse_list = []
mae_list = []
for epoch in range(0, 10):
    train(train_loader, model,criterion_map,criterion_hr,criterion_map_fft,crit_tpl,criterion_freqreg, optimizer, epoch,name_of_run,random_36_list,random_n_list)
    if epoch == 9:
        acc = validate(valid_loader, model,criterion_map,criterion_hr,criterion_map_fft, epoch,name_of_run,random_36_list,random_n_list)
    print("Save")
    torch.save(model.state_dict(), name_of_run+".pt")

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_global,'b-')
    plt.grid()
    plt.savefig(name_of_run+"_loss")
    plt.close()

    """fig,ax = plt.subplots(1,1)
    plt.plot(rmse_list,'b-')
    plt.plot(mae_list,'r-')
    plt.grid()
    plt.savefig(name_of_run+"_rmse_mae")
    plt.close()"""
