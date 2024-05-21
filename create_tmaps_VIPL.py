# Loading required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import butter, filtfilt, resample
import math
import time
import more_itertools as mit
from utils_trad import GREEN_rppg, CHROM_rppg, LGI_rppg, PBV_rppg, PCA_rppg, ICA_rppg, POS_rppg

#######
# Calculated Tmap augmenation from MSTmaps
# MSTmaps are calculated as in https://github.com/nxsEdson/CVD-Physiological-Measurement
#####

def norm(arrr):
    return (arrr-np.min(arrr))/(np.max(arrr)-np.min(arrr))

def butter_bandpass(sig, lowcut, highcut, fs, order=3):
    # butterworth bandpass filter
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

def filt_map(mappina):
    for idx in range(0,mappina.shape[0]):
        temp = mappina[idx,:]
        temp = butter_bandpass(temp,0.7,3,30)
        mappina[idx,:] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;
    #mstmap = mstmap.astype(np.uint8())
    return mappina

#creates a list of all files with a certain extension (if video), or a list of all directories containing files with extension (if images)
def list_files(dir,extension):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name[-len(extension):] == extension:
                namepath = os.path.join(root, name)
                if "source4" not in namepath: #skips NIR videos from VIPL-HR
                    r.append(namepath)
    return r

current_part = int(sys.argv[1]) #start from 1 and can go till total_parts
total_parts = int(sys.argv[2])

dir = "./VIPL-HR/"
outdir = "./tmaps_VIPL-HR/"

all_filenames = list_files(dir,"mstmap.npy")
all_filenames.sort()

split_filenames = [list(x) for x in mit.divide(total_parts, all_filenames)]  #split in parts
filenames = split_filenames[current_part-1] #pick part, index-1 because starting from 1 and not 0


for filename in filenames:
    outfolderino = filename.replace("mstmap.npy","").replace(dir,outdir)
    print("Working on clip: "+filename)
    start = time.time()
    temp = np.load(filename)[:,:,0:3]
    mstmap = np.load(filename.replace("temp","mstmap"))[:,:,0:3]
    #mstmap = mstmap.astype(np.float32)
    #print(mstmap.dtype)
    #exit()
    chrom_map = np.zeros((temp.shape[0],temp.shape[1]))
    pos_map = np.zeros((temp.shape[0],temp.shape[1]))
    #lgi_map = np.zeros((temp.shape[0],temp.shape[1]))
    #pca_map = np.zeros((temp.shape[0],temp.shape[1]))
    #ica_map = np.zeros((temp.shape[0],temp.shape[1]))
    #pbv_map = np.zeros((temp.shape[0],temp.shape[1]))

    R = temp.shape[0]
    for r in range(0,R):
        chrom_map[r,:] = CHROM_rppg(temp[r,:,:].astype(np.float32))
        pos_map[r,:] = POS_rppg(temp[r,:,:].astype(np.float32))
        #lgi_map[r,:] = LGI_rppg(temp[r,:,:].astype(np.float32))
        #pca_map[r,:] = PCA_rppg(temp[r,:,:].astype(np.float32))
        #ica_map[r,:] = ICA_rppg(temp[r,:,:].astype(np.float32))
        #pbv_map[r,:] = PBV_rppg(temp[r,:,:].astype(np.float32))

    green_map = mstmap[:,:,1]
    chrom_map = filt_map(chrom_map)
    pos_map = filt_map(pos_map)
    #lgi_map = filt_map(lgi_map)
    #pca_map = filt_map(pca_map)
    #ica_map = filt_map(ica_map)
    #pbv_map = filt_map(pbv_map)

    tmap = np.stack((green_map,chrom_map,pos_map))#,lgi_map,pca_map,ica_map,pbv_map),axis=2)


    out_name = filename.replace("mstmap","tmap").replace(dir,outdir)

    if not os.path.exists(outfolderino):
        os.makedirs(outfolderino)

    np.save(out_name,tmap)
    dt = time.time()-start
    print(dt,"s")

    """fig,ax = plt.subplots(2,1)
    ax[0].imshow(mstmap)
    ax[1].imshow(tmap.astype(np.uint8()))
    plt.show()
    plt.plot(chrom_map[0,:200],color="blue")
    plt.plot(green_map[0,:200],color="green")
    plt.plot(pos_map[0,:200],color="red")
    plt.show()
    exit()"""
    with open("tmap_completed_vipl.txt", "a") as file_object:
        file_object.write(filename+" "+str(tmap.shape)+" "+str(dt)+"\n")
