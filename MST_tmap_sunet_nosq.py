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
        if "OBF" in dir:
            dataset = "obf"
        if "PURE" in dir:
            dataset = "pure"
        if "MMSE" in dir:
            dataset = "mmse"


        shift = int(shift)
        mstmap = np.load(os.path.join(dir,"mstmap.npy"))[:,:,0:6]
        tmap = np.load(os.path.join(dir,"tmap.npy"))[:,:,0:3]

        if dataset == "vipl":
            fps = np.load(os.path.join(dir,"fps.npy"))[0]
            bvm_map = np.load(os.path.join(dir,"bvm_map.npy"))[:,:,0:6]
            wave = bvm_map[0,:,0]
        if dataset == "obf" or dataset == "pure" or dataset == "mmse":
            wave = np.load(os.path.join(dir,"bvp.npy"))
            fps = 30

        mstmap = mstmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        tmap = tmap[:,int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift,:]
        wave = wave[int(round(self.seq_len*(1-sr)))+shift:int(self.seq_len)+shift]

        wave = (wave-np.min(wave))/(np.max(wave)-np.min(wave))

        bvpmap = np.stack([wave]*64,axis=0)
        bvpmap = np.stack([bvpmap]*6,axis=2)

        mstmap = mstmap -  np.min(mstmap,axis=1,keepdims=True)
        mstmap = 255*(mstmap / np.max(mstmap,axis=1,keepdims=True))
        mstmap = mstmap.astype(np.uint8())

        tmap = tmap -  np.min(tmap,axis=1,keepdims=True)
        tmap = 255*(tmap / np.max(tmap,axis=1,keepdims=True))
        tmap = tmap.astype(np.uint8())

        stacked_bvpmap = bvpmap
        stacked_bvpmap = ((stacked_bvpmap-np.min(stacked_bvpmap))/(np.max(stacked_bvpmap)-np.min(stacked_bvpmap)))*255

        mstmap1 = mstmap[:,:,0:3].astype(np.uint8())
        tmap = tmap[:,:,0:3].astype(np.uint8())
        bvpmap1 = stacked_bvpmap[:,:,0:3].astype(np.uint8())

        masked_map = (mstmap1)

        mstmap1 = Image.fromarray(mstmap1)
        tmap = Image.fromarray(tmap)
        masked_map = Image.fromarray(masked_map)
        #mstmap2 = Image.fromarray(mstmap2)

        bvpmap1 = Image.fromarray(bvpmap1)
        #bvpmap2 = Image.fromarray(bvpmap2)
        mstmap1 = self.transform(mstmap1)
        tmap = self.transform(tmap)
        masked_map = self.transform(masked_map)
        bvpmap1 = self.transform(bvpmap1)
        hr = 0
        sample = (mstmap1,tmap,masked_map,bvpmap1,hr,fps,wave.copy(),idx)
        return sample
