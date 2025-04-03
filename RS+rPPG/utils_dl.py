import random
import torch
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import torchvision.transforms as T
import more_itertools as mit
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

def TSCAN_prepro(input_tscan):
    INapp = input_tscan
    INmot = (input_tscan[:,:,1:,:,:]-input_tscan[:,:,:-1,:,:])/(input_tscan[:,:,1:,:,:]+input_tscan[:,:,:-1,:,:])
    INmot = INmot / torch.std(INmot,dim=(1,2,3,4),keepdim=True)
    INapp = input_tscan[:,:,:-1,:,:]
    INapp = INapp - torch.mean(INapp,dim=(1,2,3,4),keepdim=True)
    INapp = INapp  / torch.std(INapp,dim=(1,2,3,4),keepdim=True)
    IN = torch.concatenate((INapp,INmot),dim=1)
    return IN


#basically cuts the whole videos in adequate sized clips (during training they overlap, during testing they don't)
def split_clips(subj, scen, stride, seq_len):
    root = './subjects'
    windowed_files = []
    for s in subj:
        for sc in scen:
            pathname = os.path.join(root, s, "maps", str(sc) + "_mstmap.npy")
            N = np.load(pathname).shape[1]
            if stride != seq_len: #usually during training
                if N > seq_len:
                    W = int((N - seq_len) / stride)
                    for w in range(0, W + 1):
                        windowed_files.append((pathname, w * stride))
            else: #during validing we want to evalutare -30s and +30s from the midpoint
                num_clips = int(N/seq_len)
                start_ind = int((N - num_clips * seq_len)/2)
                for w in range(0, num_clips):
                    windowed_files.append((pathname, start_ind + w * stride))
    return windowed_files


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

class NegativeMaxCrossCov(nn.Module):
    def __init__(self, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels, Fs):
        # Normalize
        preds_norm = preds - torch.mean(preds,dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels,dim=-1, keepdim=True)
        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])

        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2

        preds_pad = F.pad(preds_norm, (0,padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = torch.view_as_real(X)

        Fn = Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])

        # Determine ratio of energy between relevant and non-relevant regions

        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)

        use_energy = torch.sum(torch.linalg.norm(X_real[:,:,:,use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:,:,:,zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)

        #for (i,j,k) in [(i,j,k) for i in range(0,denom.size()[0]) for j in range(0,denom.size()[1]) for k in range(0,denom.size()[2])]:
        #    if denom[i,j,k] < 0:
        #        energy_ratio[i,j,k] = use_energy[i,j,k] / denom[i,j,k]
        energy_ratio = use_energy/denom

        # Zero out irrelevant freqs
        X[:,:,:,zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0]/energy_ratio

        return -max_cc

class SNR_loss(nn.Module):
    def __init__(self, clip_length = 300, delta = 3, loss_type = 1, use_wave = False):
        super(SNR_loss, self).__init__()

        self.clip_length = clip_length;
        self.time_length = 300;
        self.delta = delta;
        self.delta_distribution = [0.4, 0.25, 0.05];
        self.low_bound = 30;
        self.high_bound = 180;

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype = torch.float).cuda()
        self.bpm_range = self.bpm_range / 60.0;

        self.pi = 3.14159265;
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype = torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.cuda();
        self.hanning = hanning.cuda();

        self.cross_entropy = nn.CrossEntropyLoss();
        self.nll = nn.NLLLoss();
        self.l1 = nn.L1Loss();

        self.loss_type = loss_type;
        self.eps = 0.0001;

        self.lambda_l1 = 0.1;
        self.use_wave = use_wave;

    def forward(self, wave, gt, fps, pred = None, flag = None):  # all variable operation

        if flag is not None:
            idx = flag.eq(1);
            wave = wave[idx,:];
            gt = gt[idx,:];
            fps = fps[idx,:];
            pred = pred[idx,:];

            if(gt.shape[0] == 0):
                loss = 0.0;
                return loss, 0;

        hr = torch.mul(gt, fps);
        hr = hr*60/self.clip_length;
        hr[hr.ge(self.high_bound)] = self.high_bound-1;
        hr[hr.le(self.low_bound)] = self.low_bound;

        if pred is not None:
            pred = torch.mul(pred, fps);
            pred = pred * 60 / self.clip_length;

        batch_size = wave.shape[0];

        f_t = self.bpm_range / fps;
        preds = wave * self.hanning;

        preds = preds.view(batch_size, 1, -1);
        f_t = f_t.view(batch_size, -1, 1);

        tmp = self.two_pi_n.repeat(batch_size, 1);
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t*tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t*tmp), dim=-1) ** 2

        target = hr - self.low_bound;
        target = target.type(torch.long).view(batch_size);

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound;

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target);

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).cuda() / torch.sum(complex_absolute, dim = 1));
            norm_t = norm_t.view(-1,1);
            complex_absolute = complex_absolute * norm_t;

            loss = self.cross_entropy(complex_absolute, target);

            idx_l = target - self.delta;
            idx_l[idx_l.le(0)] = 0;
            idx_r = target + self.delta;
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0;
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]]);

            loss_snr = loss_snr / batch_size;

            loss = loss + loss_snr;

        return loss, whole_max_idx

class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, high_pass, low_pass):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(high_pass, low_pass)

    def forward(self, preds, labels):
        Fs = 30
        cov = self.cross_cov(preds, labels, Fs)

        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)

        output = torch.zeros_like(cov)
        output = cov/denom
        output = torch.mean(output)
        return 1+output

def norm_max_min(tensorino):
    tens_max =torch.amax(tensorino,dim=-1,keepdim=True)
    tens_min =torch.amin(tensorino,dim=-1,keepdim=True)
    return (tensorino-tens_min)/(tens_max-tens_min)#/(torch.max(tensorino,-1,keepdim=True)-torch.min(tensorino,-1,keepdim=True))

class MapPSD(nn.Module): #Actually it's the PSD but I don't want to change all the names yet
    def __init__(self,norm):
        super(MapPSD, self).__init__()
        self.norm = norm

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        if self.norm == "mse":
            crit_fft = nn.MSELoss()
        if self.norm == "l1":
            crit_fft = nn.L1Loss()
        #fig ,ax = plt.subplots(6,1)
        preds_fft = torch.fft.rfft(preds,dim=-1)
        preds_psd = torch.real(preds_fft)*torch.real(preds_fft)+torch.imag(preds_fft)*torch.imag(preds_fft)
        labels_fft = torch.fft.rfft(labels,dim=-1)
        labels_psd = torch.real(labels_fft)*torch.real(labels_fft)+torch.imag(labels_fft)*torch.imag(labels_fft)

        f = torch.fft.rfftfreq(labels.size(-1),1/fps)
        indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]

        preds_psd = preds_psd[:,:,:,indices]
        labels_psd = labels_psd[:,:,:,indices]

        preds_psd = norm_max_min(preds_psd)#torch.div(preds_psd,torch.sum(preds_psd,-1,keepdim=True)) #normalise
        labels_psd = norm_max_min(labels_psd)#torch.div(labels_psd,torch.sum(labels_psd,-1,keepdim=True)) #normalise


        loss = crit_fft(preds_psd,labels_psd)

        #power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        #loss = loss/power_labels
        return loss

class MapPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(MapPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        mode = 0
        if mode == 1:
            loss = 0
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    for k in range(preds.shape[3]):
                        pearson = torch.stack((preds[i,j,k,:],labels[i,j,k,:]),axis=0)
                        loss += 1 - torch.corrcoef(pearson)[0,1]
            loss = loss / (preds.shape[0]*preds.shape[1]*preds.shape[3])
            return loss
        else:   #Slightly different value but much faster
            preds_flat = torch.flatten(preds)
            labels_flat = torch.flatten(labels)
            pearson = torch.stack((preds_flat,labels_flat),axis=0)
            loss = 1 - torch.corrcoef(pearson)[0,1]
            return loss

class MapFFTMSE(nn.Module):
    def __init__(self):
        super(MapFFTMSE, self).__init__()
        return

    def forward(self, preds, labels,fps,f_min,f_max):  # tensor [Batch, Temporal]
        crit_fft = nn.MSELoss()
        preds_fft = torch.abs(torch.fft.rfft(preds,dim=3))
        labels_fft = torch.abs(torch.fft.rfft(labels,dim=3))
        f = torch.fft.rfftfreq(labels.size(3),1/fps)
        indices = np.arange(len(f))[(f >= f_min)*(f <= f_max)]
        loss = crit_fft(preds_fft[:,:,:,indices],labels_fft[:,:,:,indices])
        power_labels = torch.sum(labels_fft[:,:,:,indices])/(labels_fft.size(0)*labels_fft.size(1)*labels_fft.size(2))
        loss = loss/power_labels
        return loss

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_scenario(input_set):
    still = ["S1", "S2", "S3"]
    illumination = ["I1", "I2", "I3", "I4", "I5", "I6"]
    movement = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11"]
    conceal = ["C1", "C2", "C3", "C4", "C5", "C6"]
    illumination50 = ["I1", "I2", "I3"]
    illumination100 = ["I4", "I5", "I6"]
    illuminationlow = ["I1", "I4"]
    illuminationmedium = ["I2", "I5"]
    illuminationhigh = ["I3", "I6"]
    movement50 = ["M1", "M2", "M3", "M4", "M5"]
    movement100 = ["M6", "M7", "M8", "M9", "M10"]
    movementvertical = ["M1","M2","M6","M7"]
    movementhorizontal = ["M3","M4","M8","M9"]
    movementsmall = ["M1","M3","M6","M8"]
    movementlarge = ["M2","M4","M7","M9"]
    movementmouth = ["M5","M10"]
    concealobject = ["C1","C2","C3"]
    concealmakeup = ["C4","C5","C6"]
    all = still+illumination+movement+conceal
    C1 = ["C1"]
    C2 = ["C2"]
    C3 = ["C3"]
    C4 = ["C4"]
    C5 = ["C5"]
    C6 = ["C6"]
    M1 = ["M1"]
    M2 = ["M2"]
    M3 = ["M3"]
    M4 = ["M4"]
    M5 = ["M5"]
    M6 = ["M6"]
    M7 = ["M7"]
    M8 = ["M8"]
    M9 = ["M9"]
    M10 = ["M10"]
    M11 = ["M11"]
    I1 = ["I1"]
    I2 = ["I2"]
    I3 = ["I3"]
    I4 = ["I4"]
    I5 = ["I5"]
    I6 = ["I6"]
    S1 = ["S1"]
    S2 = ["S2"]
    S3 = ["S3"]
    all50 = still+illumination50+movement50+M11+conceal
    all100 = still+illumination100+movement100+M11+conceal
    picked_name = input_set

    picked_scenarios = []

    vars = locals()

    if picked_name.__contains__("+"):
        picked_list = picked_name.split("+")
        for i in range(len(picked_list)):
            scenarios = vars[picked_list[i]]
            picked_scenarios = picked_scenarios + scenarios
    else:
        scenarios = vars[picked_name]
        picked_scenarios = scenarios

    return picked_scenarios

def concatenate_output(list):
    # [[8, 3, 64, 576], ...]
    tmp = torch.stack(list[:len(list) - 1], dim=0)
    tmp = torch.flatten(tmp, start_dim=0, end_dim=1)
    tmp = torch.cat([tmp, list[-1]], dim=0) # combine all sample along the first dimension

    return tmp


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
        self.std = 0#math.sqrt(self.sqr_sum / self.count-(self.sum / self.count)**2)


if __name__ == '__main__':
    a, b = set_scenario()
    print(a)
    print(b)
