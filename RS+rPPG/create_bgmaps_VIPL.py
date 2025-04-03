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

def get_ROI_signal(img,mask):
    m,n,c = img.shape
    signal = np.zeros((1,1,c))
    signal2 = np.zeros((1,1,c))
    denom = np.sum(mask)
    for i in range(0,c):
        tmp = img[:,:,i]
        if denom > 0:
            signal[0,0,i] = np.sum(tmp*mask)/denom
        else:
            signal[0,0,i] = 0
    return signal

def get_combined_bg_signal_map(SignalMap):
    All_idx = comb()
    SignalMapOut = np.zeros((len(All_idx),1,6))
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

def non_nan_empty_mean(arrr,axis):
    if len(arrr) > 0:
        return np.mean(arrr,axis=axis)
    else:
        return 0


def generate_bg_signal_map(img,lmks,signal_map,idx):
    original = img.copy() #DEBUG
    m,n,c = img.shape
    ##R G B Y U V
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    yuv = RGB2YUV(img)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]

    img = np.array([R,G,B,Y,U,V])
    img = np.moveaxis(img, [1, 2, 0], [0, 1, 2])

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

    forehead = (np.vstack((forehead,forehead[-1].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)))).round(0).astype(int)#shrinkroi(forehead,scala)
    cheek_left1 = lmks[ROI_cheek_left1]
    cheek_left2 = lmks[ROI_cheek_left2]
    cheek_right1 = lmks[ROI_cheek_right1]
    cheek_right2 = lmks[ROI_cheek_right2]
    mouth = lmks[ROI_mouth]

    all_points = np.vstack([forehead,cheek_left1,cheek_left2,cheek_right1,cheek_right2,mouth])
    min_x = int(np.min(all_points[:,0]))
    min_y = int(np.min(all_points[:,1]))
    max_x = int(np.max(all_points[:,0]))
    max_y = int(np.max(all_points[:,1]))

    mlato_l = int(min_x/1.5/4)
    mlato_r = int((n-max_x)/1.5/4)

    maxlato = int(np.sqrt((max_x-min_x)*(max_y-min_y))/10)
    if mlato_l > maxlato:
        mlato_l = maxlato
    if mlato_r > maxlato:
        mlato_r = maxlato

    cc = (int(min_x+(max_x-min_x)/2),int(min_y+(max_y-min_y)/2))
    rect = Rectangle((min_x, min_y), max_x-min_x,max_y-min_y, linewidth=1, edgecolor='r', facecolor='none')

    #tl,br
    cc1 = (int(min_x/2),min_y+mlato_l*2)
    cc2 = (int(min_x/2),max_y-mlato_l*2)
    cc3 = (int(min_x/2),int(cc2[1]+(cc1[1]-cc2[1])/2))
    cc4 = (int(max_x+(n-max_x)/2),int(min_y+mlato_r*2))
    cc5 = (int(max_x+(n-max_x)/2),int(max_y-mlato_r*2))
    cc6 = (int(max_x+(n-max_x)/2),int(cc2[1]+(cc1[1]-cc2[1])/2))
# Add the patch to the Axes
    """fig,ax = plt.subplots(2,1)
    rect1 = Rectangle((cc1[0]-mlato_l,cc1[1]-mlato_l), mlato_l*2, mlato_l*2, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = Rectangle((cc2[0]-mlato_l,cc2[1]-mlato_l), mlato_l*2, mlato_l*2, linewidth=1, edgecolor='r', facecolor='none')
    rect3 = Rectangle((cc3[0]-mlato_l,cc3[1]-mlato_l), mlato_l*2, mlato_l*2, linewidth=1, edgecolor='r', facecolor='none')
    rect4 = Rectangle((cc4[0]-mlato_r,cc4[1]-mlato_r), mlato_r*2, mlato_r*2, linewidth=1, edgecolor='r', facecolor='none')
    rect5 = Rectangle((cc5[0]-mlato_r,cc5[1]-mlato_r), mlato_r*2, mlato_r*2, linewidth=1, edgecolor='r', facecolor='none')
    rect6 = Rectangle((cc6[0]-mlato_r,cc6[1]-mlato_r), mlato_r*2, mlato_r*2, linewidth=1, edgecolor='r', facecolor='none')

    ax[0].imshow(img[:,:,0:3].astype(np.uint8()))
    ax[0].scatter(lmks[:,0],lmks[:,1])
    ax[0].add_patch(rect)
    ax[0].add_patch(rect1)
    ax[0].add_patch(rect2)
    ax[0].add_patch(rect3)
    ax[0].add_patch(rect4)
    ax[0].add_patch(rect5)
    ax[0].add_patch(rect6)
    plt.show()"""


    r1 = img[cc1[1]-mlato_l:cc1[1]+mlato_l,cc1[0]-mlato_l:cc1[0]+mlato_l,:]
    r2 = img[cc2[1]-mlato_l:cc2[1]+mlato_l,cc2[0]-mlato_l:cc2[0]+mlato_l,:]
    r3 = img[cc3[1]-mlato_l:cc3[1]+mlato_l,cc3[0]-mlato_l:cc3[0]+mlato_l,:]
    r4 = img[cc4[1]-mlato_r:cc4[1]+mlato_r,cc4[0]-mlato_r:cc4[0]+mlato_r,:]
    r5 = img[cc5[1]-mlato_r:cc5[1]+mlato_r,cc5[0]-mlato_r:cc5[0]+mlato_r,:]
    r6 = img[cc6[1]-mlato_r:cc6[1]+mlato_r,cc6[0]-mlato_r:cc6[0]+mlato_r,:]

    """ax[1].imshow(r1)
    ax[2].imshow(r2)
    ax[3].imshow(r3)
    ax[4].imshow(r4)
    ax[5].imshow(r5)
    ax[6].imshow(r6)
    plt.show()"""

    Signal_tmp = np.zeros((6,6))

    #Get ROI calculated
    Signal_tmp[0,:] = non_nan_empty_mean(r1,(0,1))
    Signal_tmp[1,:] = non_nan_empty_mean(r2,(0,1))
    Signal_tmp[2,:] = non_nan_empty_mean(r3,(0,1))
    Signal_tmp[3,:] = non_nan_empty_mean(r4,(0,1))
    Signal_tmp[4,:] = non_nan_empty_mean(r5,(0,1))
    Signal_tmp[5,:] = non_nan_empty_mean(r6,(0,1))
    signal_map[:,idx,:] = get_combined_bg_signal_map(Signal_tmp).squeeze()
    return signal_map

#Models used for face and landmark detection, different ones can be found at https://py-feat.org/content/intro.html
downsample = 1  #Factor that downsamples framerate

folder_path_in = "VIPL-HR"
folder_path_out = "BGmaps"  #the output BGmaps directory will create the same directory structure as input directory
landmark_path_in = "VIPL_landmarks"

#folder_path_in = "MMSE_HR/All40_images"
#folder_path_out = "BGmaps"  #the output BGmaps directory will create the same directory structure as input directory

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
        fps = int(cap.get(5))
        fps_down = int(fps/downsample)
        size = (frame_width, frame_heigth)
        h = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #0 to total-1
        if downsample > 1:   #when dowsampling, the total frames need to be divisible by the downsample rate
            if total_frames%downsample!=0:
                total_frames+=-(total_frames%downsample)
        frames = range(0,total_frames,downsample)

    if extension == ".jpg":
        frames = os.listdir(filename)
        total_frames = len(frames)

    start = time.time()
    print("Working on clip: "+filename)
    BG_map_whole_video = np.zeros((63,int(total_frames/downsample),6))

    nomefile = filename[8:-10].replace("/","_")
    landmarks_array = np.load(os.path.join(landmark_path_in,nomefile+'_lnd.npy'))
    landmarks_array = running_mean(landmarks_array,5)


    for frame_no in tqdm(frames):
        start_loop = time.time()
        if extension == ".avi":
            cap.set(1,frame_no)
            ret, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if extension == ".jpg":
            rgb_frame = np.array(Image.open(os.path.join(filename,frame_no)))


        #Save and load points
        if extension == ".avi":
            idx = int(frame_no/downsample)
        if extension == ".jpg":
            idx = int(frame_no[:-len(extension)])
        if np.sum(landmarks_array[idx,:,:]) > 0:
            BG_map_whole_video = generate_bg_signal_map(rgb_frame,landmarks_array[idx,:,:],BG_map_whole_video,idx)


        #print(BG[:,idx,:])

    end = time.time()
    dt = end - start
    print("Processed in: ",dt,"s")
    if extension == ".avi":
        cap.release()

    # frames where no face is detected get padded with the previous non-zero value, as to not introduce high frequncy components
    for idx in range(0,BG_map_whole_video.shape[0]):
        for c in range(0,6):
            if BG_map_whole_video[idx,0,c] == 0:         #in case the first frame is not detected
                for i in range(1,BG_map_whole_video.shape[1]):
                    if BG_map_whole_video[idx,i,c] > 0:
                        BG_map_whole_video[idx,0,c] = BG_map_whole_video[idx,i,c]
                        break
            for i in range(1,BG_map_whole_video.shape[1]):
                if BG_map_whole_video[idx,i,c] == 0:
                    BG_map_whole_video[idx,i,c] = BG_map_whole_video[idx,i-1,c]

    #Min-max normalization along time axis
    for idx in range(0,BG_map_whole_video.shape[0]):
        for c in range(0,6):
            temp = BG_map_whole_video[idx,:,c]
            BG_map_whole_video[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255

    BG_map_whole_video = BG_map_whole_video.astype(np.uint8)

    """fig,ax = plt.subplots(3,1)
    ax[0].plot(BG_map_whole_video[12,:,0])
    ax[1].plot(BG_map_whole_video[12,:,1])
    ax[2].plot(BG_map_whole_video[12,:,2])
    plt.show()

    plt.imshow(BG_map_whole_video[:,:,0:3])
    plt.show()
    exit()"""
    out_name = folder_path_out+"/"+filename[:-len(extension)]+"/bgmap.npy"
    out_name = out_name.replace('\\','/')
    out_name = out_name.replace('_RGB_','_')
    direc_name =out_name.rsplit('/', 1)[0]

    if not os.path.exists(direc_name):
        os.makedirs(direc_name)

    plt.imsave(direc_name+"/"+'video_bgmap.jpeg', BG_map_whole_video[:,:,0:3])
    np.save(out_name, BG_map_whole_video)
    with open(folder_path_out+"/vipl_bg_completed.txt", "a") as file_object:
        file_object.write(filename+" "+str(BG_map_whole_video.shape)+" "+str(dt)+"\n")
