import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import math
from PIL import Image
import torch
from scipy.fft import fft,fftfreq
import torchvision.transforms.functional as transF
import heartpy as hp
from skimage.transform import resize
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

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


def get_snr(preds):  # tensor [Batch, Temporal]
    preds = preds.unsqueeze(0)
    n = 2048
    preds_fft = torch_fft(preds,n)
    f_min = 0.5
    f_max = 3

    f = torch.fft.rfftfreq(n,1/30)
    indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]
    f = f[indices].detach().cpu().numpy()
    max_inds = torch.argmax(preds_fft,dim=-1)
    max_ind = np.rint(torch.mean(max_inds.float(),dim=[1,2]).detach().cpu().numpy()).astype(np.int16())

    #if it's too slow make it non-iterative
    masksnr = torch.zeros_like(preds_fft)
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
    pr = pr.detach().squeeze()
    return pr


def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

class mst(Dataset):
    def __init__(self, data,stride,shuffle=True, Training=True, transform=None,seq_len=576):
        self.train = Training
        self.data = data
        self.transform = transform
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        dir = self.data[idx]

        long = False
        shift = 0
        sr = 1
        if type(dir) is tuple:
            if len(dir) > 2:
                long = True
                shift = dir[1]
            sr = dir[2]
            dir = dir[0]

        dataset=""
        if "VIPL" in dir:
            dataset = "vipl"
            tmap = np.load(os.path.join(dir.replace("VIPL-HR","tmaps_VIPL-HR"),"tmap2.npy"))

        if "OBF" in dir:
            dataset = "obf"
            tmap = np.load(os.path.join(dir,"tmap2.npy"))
        if "PURE" in dir:
            dataset = "pure"
            tmap = np.load(os.path.join(dir.replace("PURE_map","tmaps2_PURE_map"),"tmap2.npy"))
        if "MMSE" in dir:
            dataset = "mmse"
            tmap = np.load(os.path.join(dir.replace("MMSE_map","tmaps2_MMSE_map"),"tmap2.npy"))
        if "scamps" in dir:
            dataset = "scamps"
        if "UBFC" in dir:
            dataset = "ubfc"
        if "EquiPleth" in dir:
            dataset = "equipleth"

        shift = int(shift)
        mstmap = np.load(os.path.join(dir,"mstmap.npy"))[:,:,0:3]
        if dataset == "vipl":
            bgmap = np.load(os.path.join(dir,"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir,"mvmap.npy"))[:,:,:]
        if dataset == "pure":
            bgmap = np.load(os.path.join(dir.replace("PURE_map","PURE_mvbg"),"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir.replace("PURE_map","PURE_mvbg"),"mvmap.npy"))[:,:,0:3]
        if dataset == "mmse":
            bgmap = np.load(os.path.join(dir.replace("MMSE_map","MMSE_mvbg"),"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir.replace("MMSE_map","MMSE_mvbg"),"mvmap.npy"))[:,:,0:3]
        if dataset == "obf":
            bgmap = np.load(os.path.join(dir,"bgmap.npy"))[:,:,0:3] #### WTF NOBG MAP
            mvmap = np.load(os.path.join(dir,"mvmap.npy"))[:,:,0:3]
        if dataset == "scamps":
            bgmap = np.load(os.path.join(dir.replace("scamps_mstmaps","scamps_mvbg_maps"),"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir.replace("scamps_mstmaps","scamps_mvbg_maps"),"mvmap.npy"))[:,:,0:3]
            tmap = np.load(os.path.join(dir,"tmap2.npy"))[:,:,:]
        if dataset == "ubfc":
            bgmap = np.load(os.path.join(dir.replace("UBFC_map","UBFC_mvbg_map"),"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir.replace("UBFC_map","UBFC_mvbg_map"),"mvmap.npy"))[:,:,0:3]
            tmap = np.load(os.path.join(dir,"tmap2.npy"))[:,:,:]
        if dataset == "equipleth":
            bgmap = np.load(os.path.join(dir.replace("EquiPleth_map","EquiPleth_mvbg_map"),"bgmap.npy"))[:,:,0:3]
            mvmap = np.load(os.path.join(dir.replace("EquiPleth_map","EquiPleth_mvbg_map"),"mvmap.npy"))[:,:,0:3]
            tmap = np.load(os.path.join(dir,"tmap2.npy"))[:,:,:]

        if dataset == "vipl":
            fps = np.load(os.path.join(dir,"fps.npy"))[0]
            bvm_map = np.load(os.path.join(dir,"bvm_map.npy"))[:,:,0:3]
            wave = bvm_map[0,:,0]
        if dataset == "obf" or dataset == "pure" or dataset == "mmse" or dataset == "scamps"  or dataset == "ubfc" or dataset == "equipleth":
            wave = np.load(os.path.join(dir,"bvp.npy"))
            #ecg = np.load(os.path.join(dir,"ecg.npy"))[:len(wave)+120]
            #ecg = ecg[60:-60]
            fps = 30

        #if mstmap.shape[1]>=self.seq_len:   REMEMBER FOR VIPLHR THIS IS NOT TRUE
        mstmap = mstmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        tmap = tmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        bgmap = bgmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        mvmap = mvmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        wave = wave[int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift]
        #ecg = ecg[int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift]
        yuv_mstmap = RGB2YUV(mstmap)
        yuv_bgmap = RGB2YUV(bgmap)

        wave = (wave-np.min(wave))/(np.max(wave)-np.min(wave))
        #wave = butter_bandpass(wave, 0.5, 3, fps) #low pass filter to remove DC component (introduced by normalisation)
        #ecg = butter_bandpass(ecg, 0.05, 3, fps)

        bvpmap = np.stack([wave]*64,axis=0)
        bvpmap = np.stack([bvpmap]*6,axis=2)

        mstmap = mstmap -  np.min(mstmap,axis=1,keepdims=True)
        mstmap = 255*(mstmap / np.max(mstmap,axis=1,keepdims=True))
        mstmap = mstmap.astype(np.uint8())

        #resize_rate = 2/3
        resize_rate = np.random.choice(np.concatenate((np.linspace(2/3,0.9,50),np.linspace(1.1,4/3,50))))

        resize_size = int(mstmap.shape[1]*resize_rate)
        if resize_rate < 1:
            offset = int((mstmap.shape[1]-resize_size)/2)
            aug2_mstmap = mstmap[:,offset:offset+resize_size,:]
            aug2_mstmap = cv2.resize(aug2_mstmap, dsize=(mstmap.shape[1], mstmap.shape[0]), interpolation=cv2.INTER_CUBIC)
        if resize_rate > 1:
            sidepad = int((resize_rate-1)*(mstmap.shape[1]/2))
            aug2_mstmap = cv2.copyMakeBorder(mstmap, 0, 0, sidepad, sidepad, cv2.BORDER_REFLECT)
            aug2_mstmap = cv2.resize(aug2_mstmap, dsize=(mstmap.shape[1], mstmap.shape[0]), interpolation=cv2.INTER_CUBIC)

        aug2_mstmap = aug2_mstmap -  np.min(aug2_mstmap,axis=1,keepdims=True)
        aug2_mstmap = 255*(aug2_mstmap / np.max(aug2_mstmap,axis=1,keepdims=True))
        aug2_mstmap = aug2_mstmap.astype(np.uint8())
        yuv_aug2_mstmap = RGB2YUV(aug2_mstmap)

        yuv_aug2_mstmap = yuv_aug2_mstmap -  np.min(yuv_aug2_mstmap,axis=1,keepdims=True)
        yuv_aug2_mstmap = 255*(yuv_aug2_mstmap / np.max(yuv_aug2_mstmap,axis=1,keepdims=True))
        yuv_aug2_mstmap = yuv_aug2_mstmap.astype(np.uint8())

        yuv_mstmap = yuv_mstmap -  np.min(yuv_mstmap,axis=1,keepdims=True)
        yuv_mstmap = 255*(yuv_mstmap / np.max(yuv_mstmap,axis=1,keepdims=True))
        yuv_mstmap = yuv_mstmap.astype(np.uint8())

        bgmap = bgmap -  np.min(bgmap,axis=1,keepdims=True)
        bgmap = 255*(bgmap / np.max(bgmap,axis=1,keepdims=True))
        bgmap = bgmap.astype(np.uint8())

        yuv_bgmap = yuv_bgmap -  np.min(yuv_bgmap,axis=1,keepdims=True)
        yuv_bgmap = 255*(yuv_bgmap / np.max(yuv_bgmap,axis=1,keepdims=True))
        yuv_bgmap = yuv_bgmap.astype(np.uint8())

        mvmap = mvmap -  np.min(mvmap,axis=1,keepdims=True)
        mvmap = 255*(mvmap / np.max(mvmap,axis=1,keepdims=True))
        mvmap = mvmap.astype(np.uint8())

        tmap = tmap -  np.min(tmap,axis=1,keepdims=True)
        tmap = 255*(tmap / np.max(tmap,axis=1,keepdims=True))
        tmap = tmap.astype(np.uint8())

        stacked_bvpmap = bvpmap
        stacked_bvpmap = ((stacked_bvpmap-np.min(stacked_bvpmap))/(np.max(stacked_bvpmap)-np.min(stacked_bvpmap)))*255

        mstmap  = np.concatenate([mstmap,yuv_mstmap],axis=-1)
        bgmap  = np.concatenate([bgmap,yuv_bgmap],axis=-1)

        aug2_mstmap  = np.concatenate([aug2_mstmap,yuv_aug2_mstmap],axis=-1)

        mstmap1 = mstmap[:,:,0:3].astype(np.uint8())
        mstmap2 = mstmap[:,:,3:6].astype(np.uint8())
        aug2_mstmap1 = aug2_mstmap[:,:,0:3].astype(np.uint8())
        aug2_mstmap2 = aug2_mstmap[:,:,3:6].astype(np.uint8())
        tmap1 = tmap[:,:,0:3].astype(np.uint8())
        tmap2 = tmap[:,:,3:6].astype(np.uint8())
        bgmap1 = bgmap[:,:,0:3].astype(np.uint8())
        bgmap2 = bgmap[:,:,3:6].astype(np.uint8())
        mvmap = mvmap[:,:,0:3].astype(np.uint8())
        #mstmap2 = mstmap[:,:,3:6].astype(np.uint8())
        bvpmap1 = stacked_bvpmap[:,:,0:3].astype(np.uint8())
        #bvpmap2 = stacked_bvpmap[:,:,3:6].astype(np.uint8())

        masked_map = (mstmap1)#*mask(mask_size=48,mask_patch_size=4,channels=3,mask_ratio=0.75)).astype(np.uint8())

        mstmap1 = Image.fromarray(mstmap1)
        mstmap2 = Image.fromarray(mstmap2)
        aug2_mstmap1 = Image.fromarray(aug2_mstmap1)
        aug2_mstmap2 = Image.fromarray(aug2_mstmap2)
        tmap1 = Image.fromarray(tmap1)
        tmap2 = Image.fromarray(tmap2)
        masked_map = Image.fromarray(masked_map)
        mvmap = Image.fromarray(mvmap)
        bgmap1 = Image.fromarray(bgmap1)
        bgmap2 = Image.fromarray(bgmap2)
        bvpmap1 = Image.fromarray(bvpmap1)
        #bvpmap2 = Image.fromarray(bvpmap2)

        #mstmap2 = Image.fromarray(mstmap2)

        #bvpmap2 = Image.fromarray(bvpmap2)
        mstmap1 = self.transform(mstmap1)
        mstmap2 = self.transform(mstmap2)
        aug2_mstmap1 = self.transform(aug2_mstmap1)
        aug2_mstmap2 = self.transform(aug2_mstmap2)
        tmap1 = self.transform(tmap1)
        tmap2 = self.transform(tmap2)
        mvmap = self.transform(mvmap)
        bgmap1 = self.transform(bgmap1)
        bgmap2 = self.transform(bgmap2)
        masked_map = self.transform(masked_map)
        bvpmap1 = self.transform(bvpmap1)
        #bvpmap2 = self.transform(bvpmap2)

        mstmap1 = torch.cat([mstmap1,mstmap2],dim=0)
        aug2_mstmap1 = torch.cat([aug2_mstmap1,aug2_mstmap2],dim=0)
        tmap1 = torch.cat([tmap1,tmap2],dim=0)
        bgmap1 = torch.cat([bgmap1,bgmap2],dim=0)
        bvpmap1 = torch.cat([bvpmap1,bvpmap1],dim=0)
        mvmap = torch.cat([mvmap,mvmap],dim=0)

        pr = get_snr(tmap1)


        hr = 0
        sample = (mstmap1,tmap1,mvmap,bgmap1,aug2_mstmap1,masked_map,bvpmap1,hr,fps,wave.copy(),pr,idx)
        return sample
