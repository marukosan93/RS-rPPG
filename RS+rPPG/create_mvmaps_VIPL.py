# Loading required libraries
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from skimage import io
import sys
import cv2
import os
import time
from tqdm import tqdm
from PIL import Image, ImageDraw
from feat import Detector
from matplotlib.patches import Rectangle
import more_itertools as mit
import itertools
from scipy.ndimage import binary_erosion
from scipy.signal import detrend

def comb():
    X = [0,1]
    result = np.zeros((63,6))
    number=-1
    for combination in itertools.product(X,X,X,X,X,X):
        number+=1
        if number > 0:
            result[number-1,:] = np.array(combination)
    return result.astype(int)

def running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(0,dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b],axis=0)
    return out

#creates a list of all files with a certain extension (if video), or a list of all directories containing files with extension (if images)
def list_files(dir,extension):
    r = []

    if extension == ".avi":
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name[-len(extension):] == extension:
                    namepath = os.path.join(root, name)
                    if "source4" not in namepath: #skips NIR videos from VIPL-HR
                        r.append(namepath)

    if extension == ".jpg":
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                for file in os.listdir(dirpath):
                    if file[-len(extension):] == extension:
                        r.append(dirpath)
                        break

    return r

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

def poly2mask(polyarray,m,n):
    img = Image.new('L', (n, m), 0)
    ImageDraw.Draw(img).polygon(polyarray.flatten().tolist(), outline=1, fill=1)
    mask = np.array(img)
    return mask

def get_combined_mv_signal_map(SignalMap):
    All_idx = comb()
    SignalMapOut = np.zeros((len(All_idx),1,3))
    for index in range(0,63):
        tmp_idx = np.where(All_idx[index]==1)
        tmp_signal = SignalMap[tmp_idx,:]
        SignalMapOut[index,:,:] = np.sum(tmp_signal,axis=1)
    return SignalMapOut

def mult_image(img1,roi):
    img = img1.copy()
    img[:,:,0] = img[:,:,0]*roi
    img[:,:,1] = img[:,:,1]*roi
    img[:,:,2] = img[:,:,2]*roi
    return img

def center_area(mask):
    area = (mask == 1).sum()
    if area > 0:
        x_center, y_center = np.argwhere(mask==1).sum(0)/area
    else:
        x_center = 0
        y_center = 0
    return x_center,y_center,area


def generate_mv_signal_map(image_shape,lmks,signal_map,idx):
    m,n,c = image_shape
    ROI_cheek_left1 = np.array([0,1,2,31,41,0])
    ROI_cheek_left2 = np.array([2,3,4,5,48,31,2])
    ROI_cheek_right1 = np.array([16,15,14,35,46,16])
    ROI_cheek_right2 = np.array([14,13,12,11,54,35,14])
    ROI_mouth = [5,6,7,8,9,10,11,54,55,56,57,58,59,48,5]
    ROI_forehead = [17,18,19,20,21,22,23,24,25,26]
    forehead = lmks[ROI_forehead]
    left_eye = np.mean(lmks[36:42],axis=0)
    right_eye = np.mean(lmks[42:48],axis=0)
    eye_distance = np.linalg.norm(left_eye-right_eye)

    tmp = (np.mean(lmks[17:22],axis=0)+ np.mean(lmks[22:27],axis=0))/2 - (left_eye + right_eye)/2;
    tmp = (eye_distance/np.linalg.norm(tmp))*0.6*tmp;

    ROI_forehead=(np.vstack((forehead,forehead[-1].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)))).round(0).astype(int)

    mask_ROI_cheek_left1 = poly2mask(lmks[ROI_cheek_left1],m,n);
    mask_ROI_cheek_left2 = poly2mask(lmks[ROI_cheek_left2],m,n);
    mask_ROI_cheek_right1 = poly2mask(lmks[ROI_cheek_right1],m,n);
    mask_ROI_cheek_right2 = poly2mask(lmks[ROI_cheek_right2],m,n);
    mask_ROI_mouth  = poly2mask(lmks[ROI_mouth],m,n);
    mask_ROI_forehead = poly2mask(ROI_forehead,m,n);

    #print(area)
    #plt.imshow(mask_ROI_forehead)
    #plt.scatter(y_center,x_center)
    #plt.show()
    #exit()
    """total_mask = mask_ROI_forehead+mask_ROI_cheek_left1+mask_ROI_cheek_left2+mask_ROI_cheek_right1+mask_ROI_cheek_right2+mask_ROI_mouth
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(mult_image(original,total_mask))
    ax[1].imshow(original)
    plt.show()
    exit()
    fig,ax = plt.subplots(1,6)
    ax[0].imshow(mult_image(original,mask_ROI_cheek_left1))
    ax[1].imshow(mult_image(original,mask_ROI_cheek_left2))
    ax[2].imshow(mult_image(original,mask_ROI_cheek_right1))
    ax[3].imshow(mult_image(original,mask_ROI_cheek_right2))
    ax[4].imshow(mult_image(original,mask_ROI_mouth))
    ax[5].imshow(mult_image(original,mask_ROI_forehead))
    plt.show()"""
    """plt.imshow(original)
    plt.scatter(lmks[42:48,0],lmks[42:48,1],s=10)
    plt.scatter(right_eye[0],right_eye[1],s=10)
    plt.scatter(ROI_forehead[:,0],ROI_forehead[:,1],s=10)
    plt.show()
    exit()"""
    Signal_tmp = np.zeros((6,3))
    ROI_num = np.zeros((6,1))

    #Get ROI calculated
    Signal_tmp[0,:] = center_area(mask_ROI_cheek_left1) #Was wrong, but they all look similar anyway
    Signal_tmp[1,:] = center_area(mask_ROI_cheek_left2)
    Signal_tmp[2,:] = center_area(mask_ROI_cheek_right1)
    Signal_tmp[3,:] = center_area(mask_ROI_cheek_right2)
    Signal_tmp[4,:] = center_area(mask_ROI_mouth)
    Signal_tmp[5,:] = center_area(mask_ROI_forehead)

    signal_map[:,idx,:] = get_combined_mv_signal_map(Signal_tmp).squeeze()
    return signal_map


downsample = 1  #Factor that downsamples framerate

folder_path_in = "VIPL-HR"
folder_path_out = "BGmaps"  #the output mstmaps directory will create the same directory structure as input directory
landmark_path_in = "VIPL_landmarks"
#image_shape = "none" #change for VIPL
#...
#folder_path_in = "MMSE_HR/All40_images"
#folder_path_out = "MSTmaps"  #the output mstmaps directory will create the same directory structure as input directory

#Splits the dataset in total_parts, of which only current_part will be processed. Processing is relatively slow, so this way the computation can be split by running the script on different parts of the dataset at the same time on different processes/machines
current_part = int(sys.argv[1]) #start from 1 and can go till total_parts
total_parts = int(sys.argv[2])

extension = ".avi"
#extension = ".jpg"

all_filenames = list_files(folder_path_in,extension)   #gets all filepaths that contain extension
all_filenames.sort()

split_filenames = [list(x) for x in mit.divide(total_parts, all_filenames)]  #split in parts
filenames = split_filenames[current_part-1] #pick part, index-1 because starting from 1 and not 0

for filename in filenames:
    if extension == ".avi":
        cap = cv2.VideoCapture(filename)
        frame_width = int(cap.get(3))
        frame_heigth = int(cap.get(4))
        image_shape =(frame_heigth,frame_width,3)

    start = time.time()
    print("Working on clip: "+filename)

    nomefile = filename[8:-10].replace("/","_")
    landmarks_array = np.load(os.path.join(landmark_path_in,nomefile+'_lnd.npy'))
    landmarks_array = running_mean(landmarks_array,5)
    total_frames = landmarks_array.shape[0]
    MV_map_whole_video = np.zeros((63,total_frames,3))
    frames = range(0,total_frames)
    for frame_no in tqdm(frames):
        idx = int(frame_no)
        start_loop = time.time()

        if np.sum(landmarks_array[idx,:,:]) > 0:
            MV_map_whole_video = generate_mv_signal_map(image_shape,landmarks_array[idx,:,:],MV_map_whole_video,idx)

        #print(MV_map_whole_video[:,idx,:])
    end = time.time()
    dt = end - start
    print("Processed in: ",dt,"s")

    # frames where no face is detected get padded with the previous non-zero value, as to not introduce high frequncy components
    for idx in range(0,MV_map_whole_video.shape[0]):
        for c in range(0,3):
            if MV_map_whole_video[idx,0,c] == 0:         #in case the first frame is not detected
                for i in range(1,MV_map_whole_video.shape[1]):
                    if MV_map_whole_video[idx,i,c] > 0:
                        MV_map_whole_video[idx,0,c] = MV_map_whole_video[idx,i,c]
                        break
            for i in range(1,MV_map_whole_video.shape[1]):
                if MV_map_whole_video[idx,i,c] == 0:
                    MV_map_whole_video[idx,i,c] = MV_map_whole_video[idx,i-1,c]

    #just to be safe, zeros are exchanged with min non zero values
    #nonzeromin = np.min(MV_map_whole_video[np.where(MV_map_whole_video != 0)])
    #MV_map_whole_video[np.where(MV_map_whole_video == 0)] = nonzeromin

    #Min-max normalization along time axis
    for idx in range(0,MV_map_whole_video.shape[0]):
        for c in range(0,3):
            temp = MV_map_whole_video[idx,:,c]
            MV_map_whole_video[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255



    MV_map_whole_video = MV_map_whole_video.astype(np.uint8)

    """fig,ax = plt.subplots(3,1)
    ax[0].plot(MV_map_whole_video[12,:,0])
    ax[1].plot(MV_map_whole_video[12,:,1])
    ax[2].plot(MV_map_whole_video[12,:,2])
    plt.show()

    plt.imshow(MV_map_whole_video)
    plt.show()
    exit()"""
    out_name = folder_path_out+"/"+filename[:-len(extension)]+"/mvmap.npy"
    out_name = out_name.replace('\\','/')
    out_name = out_name.replace('_RGB_','_')
    direc_name =out_name.rsplit('/', 1)[0]


    out_name_temp = out_name.replace("mvmap","temp")
    if not os.path.exists(direc_name):
        os.makedirs(direc_name)

    plt.imsave(direc_name+"/"+'video_mvmap.jpeg', MV_map_whole_video)
    np.save(out_name, MV_map_whole_video)
    with open(folder_path_out+"/vipl_mv_completed.txt", "a") as file_object:
        file_object.write(filename+" "+str(MV_map_whole_video.shape)+" "+str(dt)+"\n")
