import random
import torch
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch,filtfilt
import torchvision.transforms as T
import more_itertools as mit
import math
import torch.nn as nn
from MST_sunet import mst
import matplotlib.pyplot as plt
from einops import rearrange
from torch.autograd import Variable
import torch.nn.functional as F
import heartpy as hp
import pickle
from scipy import signal
from scipy.fft import fft
from utils_signals import hr_fft, butter_bandpass


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

def create_datasets(dataset,howmany,train_stride=576,seq_len=576,train_temp_aug=False):
    if dataset == "vipl":
        input_dir = "./MSTmaps/VIPL-HR"
        extension = "bvm_map.npy"
        list_subj = []
        for s in list(np.arange(1,108,1).astype(str)):
            s = "/p"+s+"/"
            list_subj.append(s)
    if dataset == "obf":
        input_dir = "./MSTmaps/OBF_video"
        extension = "bvp.npy"
        list_subj = []
        for s in list(np.arange(1,101,1).astype(str)):
            if len(s) == 1:
                s = "00"+s
            if len(s) == 2:
                s = "0"+s
            list_subj.append(s)
    if dataset == "pure":
        input_dir = "./MSTmaps/PURE_map"
        extension = "bvp.npy"
        list_subj = []

        for s in range(1,11):
            list_subj.append(str(s).zfill(2))
    if dataset == "mmse":
        input_dir = "./MSTmaps/MMSE_map"
        extension = "bvp.npy"
        list_subj = []

        for s in range(5,28):
            list_subj.append("F"+str(s).zfill(3))
        for s in range(1,18):
            list_subj.append("M"+str(s).zfill(3))

    all_dirnames = list_dirs(input_dir,extension)   #gets all filepaths that contain extension
    all_dirnames.sort()

    if dataset == "obf":
        all_dirnames.remove("./MSTmaps/OBF_video/073_2")  #GROUNTURHT DATA IS Wrong
        all_dirnames.remove("./MSTmaps/OBF_video/057_2")  #also weird for some reason

    #VIPL-HR videos with bad ground truth
    #blacklist_vipl = ['./MSTmaps/VIPL-HR/p10/v3/source1/video', './MSTmaps/VIPL-HR/p10/v3/source2/video', './MSTmaps/VIPL-HR/p10/v3/source3/video', './MSTmaps/VIPL-HR/p11/v4/source1/video', './MSTmaps/VIPL-HR/p11/v4/source3/video', './MSTmaps/VIPL-HR/p11/v5/source1/video', './MSTmaps/VIPL-HR/p11/v5/source3/video', './MSTmaps/VIPL-HR/p16/v8/source2/video', './MSTmaps/VIPL-HR/p22/v1/source1/video', './MSTmaps/VIPL-HR/p22/v1/source2/video', './MSTmaps/VIPL-HR/p22/v1/source3/video', './MSTmaps/VIPL-HR/p24/v6/source1/video', './MSTmaps/VIPL-HR/p24/v6/source3/video', './MSTmaps/VIPL-HR/p26/v8/source2/video', './MSTmaps/VIPL-HR/p29/v6/source2/video', './MSTmaps/VIPL-HR/p37/v8/source2/video', './MSTmaps/VIPL-HR/p37/v9/source2/video', './MSTmaps/VIPL-HR/p38/v3/source1/video', './MSTmaps/VIPL-HR/p38/v3/source2/video', './MSTmaps/VIPL-HR/p38/v3/source3/video', './MSTmaps/VIPL-HR/p40/v3/source2/video', './MSTmaps/VIPL-HR/p41/v2/source1/video', './MSTmaps/VIPL-HR/p41/v2/source3/video', './MSTmaps/VIPL-HR/p43/v3/source1/video', './MSTmaps/VIPL-HR/p43/v3/source2/video', './MSTmaps/VIPL-HR/p43/v3/source3/video', './MSTmaps/VIPL-HR/p44/v3/source1/video', './MSTmaps/VIPL-HR/p44/v3/source2/video', './MSTmaps/VIPL-HR/p44/v3/source3/video', './MSTmaps/VIPL-HR/p45/v3/source1/video', './MSTmaps/VIPL-HR/p45/v3/source2/video', './MSTmaps/VIPL-HR/p45/v3/source3/video', './MSTmaps/VIPL-HR/p46/v2/source1/video', './MSTmaps/VIPL-HR/p46/v2/source2/video', './MSTmaps/VIPL-HR/p46/v2/source3/video', './MSTmaps/VIPL-HR/p48/v9/source2/video', './MSTmaps/VIPL-HR/p49/v2/source1/video', './MSTmaps/VIPL-HR/p49/v2/source2/video', './MSTmaps/VIPL-HR/p49/v2/source3/video', './MSTmaps/VIPL-HR/p49/v3/source1/video', './MSTmaps/VIPL-HR/p49/v3/source2/video', './MSTmaps/VIPL-HR/p49/v3/source3/video', './MSTmaps/VIPL-HR/p50/v6/source1/video', './MSTmaps/VIPL-HR/p50/v6/source3/video', './MSTmaps/VIPL-HR/p51/v7/source2/video', './MSTmaps/VIPL-HR/p59/v3/source1/video', './MSTmaps/VIPL-HR/p59/v3/source2/video', './MSTmaps/VIPL-HR/p59/v3/source3/video', './MSTmaps/VIPL-HR/p68/v2/source1/video', './MSTmaps/VIPL-HR/p68/v2/source2/video', './MSTmaps/VIPL-HR/p68/v2/source3/video', './MSTmaps/VIPL-HR/p71/v7/source2/video', './MSTmaps/VIPL-HR/p83/v6/source1/video', './MSTmaps/VIPL-HR/p83/v6/source2/video', './MSTmaps/VIPL-HR/p83/v6/source3/video', './MSTmaps/VIPL-HR/p88/v2/source1/video', './MSTmaps/VIPL-HR/p88/v2/source2/video', './MSTmaps/VIPL-HR/p88/v2/source3/video', './MSTmaps/VIPL-HR/p88/v3/source1/video', './MSTmaps/VIPL-HR/p88/v3/source2/video', './MSTmaps/VIPL-HR/p88/v3/source3/video', './MSTmaps/VIPL-HR/p88/v9/source2/video', './MSTmaps/VIPL-HR/p90/v5/source2/video', './MSTmaps/VIPL-HR/p97/v2/source2/video', './MSTmaps/VIPL-HR/p97/v2/source3/video', './MSTmaps/VIPL-HR/p97/v7/source2/video', './MSTmaps/VIPL-HR/p97/v7/source3/video']
    blacklist_vipl = []
    if dataset == "vipl":
        for b in blacklist_vipl:
            all_dirnames.remove(b)

    #randomly shuffle,but with seed so that it's reproducible
    if dataset != "pure":
        np.random.seed(4)
        np.random.shuffle(list_subj)

    if howmany == 'whole':
        train_subj = list_subj
        valid_subj = []

        train_dirs = []
        valid_dirs = []


        for dir in all_dirnames:
            for s in train_subj:
                if s in dir:
                    train = True
            for s in valid_subj:
                if s in dir:
                    train = False
            if train:
                train_dirs.append(dir)
            else:
                valid_dirs.append(dir)


        train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
        train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

        valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
        valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

        #file = open("./folds/"+dataset+"_whole_aug2.pkl",'wb')
        file = open("./folds"+str(seq_len)+"/"+dataset+"_whole.pkl",'wb')
        pickle.dump(train_dirs, file)
        pickle.dump(valid_dirs, file)

        print(len(train_dirs))
        print(len(valid_dirs))

    if howmany == 'purezd':
        train_subj = ['01', '02', '03', '04', '05', '07']
        valid_subj = ['06', '08', '09', '10']

        train_dirs = []
        valid_dirs = []

        for dir in all_dirnames:
            for s in train_subj:
                if s in dir[:-3]:
                    train = True
            for s in valid_subj:
                if s in dir[:-3]:
                    train = False
            if train:
                train_dirs.append(dir)
            else:
                valid_dirs.append(dir)

        train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
        train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

        valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
        valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)


        #file = open("./folds/"+dataset+"_whole_aug2.pkl",'wb')
        file = open("./folds"+str(seq_len)+"/"+dataset+"_fold1.pkl",'wb')
        pickle.dump(train_dirs, file)
        pickle.dump(valid_dirs, file)
        print(len(train_dirs))
        print(len(valid_dirs))

    if howmany == '5fold':
        divided = ([list(x) for x in mit.divide(5,list_subj)])
        for fold in range(0,5):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,5))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]],*divided[train_div[3]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []


            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    if howmany == '3fold':
        divided = ([list(x) for x in mit.divide(3,list_subj)])
        for fold in range(0,3):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,3))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []

            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    if howmany == 'skin':
        fold11_indices =['001', '008', '011', '012', '013', '014', '016', '018', '019', '025', '030', '032', '033', '034', '036', '037', '042', '043', '052', '055', '060', '070', '073', '074', '077', '078', '079', '092', '093', '095', '100']
        fold12_indices =['003', '006', '007', '009', '010', '015', '017', '020', '022', '023', '024', '026', '027', '038', '039', '040', '044', '045', '047', '048', '059', '061', '064', '067', '069', '071', '072', '076', '081', '083', '084', '085', '086', '087', '088', '089', '090', '091', '094', '098', '099']
        fold13_indices =['002', '004', '005', '021', '028', '029', '031', '035', '041', '046', '049', '050', '051', '053', '054', '056', '057', '058', '062', '063', '065', '066', '068', '075', '080', '082', '096', '097']

        divided = [fold11_indices,fold12_indices,fold13_indices]
        #divided = ([list(x) for x in mit.divide(3,list_subj)])
        for fold in range(0,3):
            print("FOLD",fold+1+10)
            train_div = list(np.arange(0,3))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []

            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(10+fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    if howmany == '10fold':
        divided = ([list(x) for x in mit.divide(10,list_subj)])
        for fold in range(0,10):
            print("FOLD",fold+1)
            train_div = list(np.arange(0,10))
            train_div.remove(fold)

            train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]],*divided[train_div[3]],*divided[train_div[4]],*divided[train_div[5]],*divided[train_div[6]],*divided[train_div[7]],*divided[train_div[8]]] #train
            valid_subj = divided[fold] #validate

            train_dirs = []
            valid_dirs = []


            for dir in all_dirnames:
                for s in train_subj:
                    if s in dir:
                        train = True
                for s in valid_subj:
                    if s in dir:
                        train = False
                if train:
                    train_dirs.append(dir)
                else:
                    valid_dirs.append(dir)

            train_dirs = window_dirs(train_dirs,train_stride,seq_len,True)
            train_dirs = temp_augmentation(train_dirs,train_temp_aug,train_stride,seq_len)

            valid_dirs = window_dirs(valid_dirs,seq_len,seq_len,False)
            valid_dirs = temp_augmentation(valid_dirs,False,seq_len,seq_len)

            #file = open("./folds/"+dataset+"_fold"+str(fold+1)+"_aug2.pkl",'wb')
            file = open("./folds"+str(seq_len)+"/"+dataset+"_fold"+str(fold+1)+".pkl",'wb')
            pickle.dump(train_dirs, file)
            pickle.dump(valid_dirs, file)

            print(len(train_dirs))
            print(len(valid_dirs))

    #NORMALIZE_MEAN = (0.5, 0.5, 0.5)
    #NORMALIZE_STD = (0.5, 0.5, 0.5)
    transforms = [   #add  data Augmentation
                  #T.Resize((192,192)),
                  T.ToTensor()#,
                  #T.Normalize((0.5823, 0.4994, 0.5634), (0.1492, 0.1859, 0.1953))
                  ]
    transforms = T.Compose(transforms)

    train_dataset = mst(data=train_dirs,stride=train_stride,shuffle=True, Training = True, transform=transforms,seq_len=seq_len)
    valid_dataset = mst(data=valid_dirs,stride=seq_len,shuffle=False, Training = False, transform=transforms)
    return train_dataset, valid_dataset



def temp_augmentation(train_dirs,enabled,train_stride,seq_len):
    output_dirs = []
    for tr_dir in train_dirs:
        shift = tr_dir[1]
        dataset=""
        windowed_dirs = []
        if "VIPL" in tr_dir[0]:
            dataset = "vipl"
        if "OBF" in tr_dir[0]:
            dataset = "obf"
        output_dirs.append((tr_dir[0],tr_dir[1],1))
        if enabled:
            #mstmap = np.load(os.path.join(tr_dir[0],"mstmap.npy"))[:,:,:]
            if dataset == "vipl":
                wave = np.load(os.path.join(tr_dir[0],"bvm_map.npy"))[0,:,0]
                fps = np.load(os.path.join(tr_dir[0],"fps.npy"))[0]
            if dataset == "obf":
                wave = np.load(os.path.join(tr_dir[0],"bvp.npy"))
                fps = 30

            wave_filtered = butter_bandpass(wave, 0.5, 3, fps) #low pass filter to remove DC component (introduced by normalisation)
            wave_piece = wave_filtered[shift:seq_len+shift]
            hr, _ , _ = hr_fft(wave,fps)

            """#AUG1
            if (hr > 90 or hr < 65) and shift>0:
                output_dirs.append((tr_dir[0],tr_dir[1]-10,1))
                output_dirs.append((tr_dir[0],tr_dir[1]-20,1))"""
            #AUG2
            """if hr >= 60 and hr <= 110:  #because it needs more samples for downsampling
                output_dirs.append((tr_dir[0],tr_dir[1],0.67))
            if hr >= 70 and hr <= 85  and shift>int(seq_len/2):
                output_dirs.append((tr_dir[0],tr_dir[1],1.5))"""
    return output_dirs


def window_dirs(dirs,stride,seq_len,train):
    dataset=""
    windowed_dirs = []
    for dir in dirs:
        if "VIPL" in dir:
            dataset = "vipl"
        if "OBF" in dir:
            dataset = "obf"
        mstmap = np.load(os.path.join(dir,"mstmap.npy"))[:,:,:]
        if dataset == "vipl":
            bvm_map = np.load(os.path.join(dir,"bvm_map.npy"))[:,:,0:6]
            wave = bvm_map[0,:,0]
        if dataset == "obf":
            wave = np.load(os.path.join(dir,"bvp.npy"))
        N = int(mstmap.shape[1])
        if N > seq_len:
            W = int((N-seq_len)/stride)
            for w in range(0,W+1):
                windowed_dirs.append((dir,w*stride))
        #else:                   SHOULD ADD SOMETHING INSTEAD OF THROWING AWAY SHORT CLIPS
    return windowed_dirs
