# RS+rPPG

This is the official code repository of our IEEE TCSVT paper "RS+rPPG: Robust Strongly Self-Supervised Learning for rPPG". RS+rPPG leverages eleven rPPG priors and contrastive learning to robustly handle noisy video data with head movement and illumination variations. The method outperforms state-of-the-art supervised techniques across seven datasets, demonstrating strong generalization, demographic fairness, and stability in mixed-data scenarios, all without requiring labeled training data.

You can find the paper at [Here](https://brosdocs.net/fg2024/013.pdf](https://oulurepo.oulu.fi/handle/10024/54352))

![mainfig](rsrppg_mainfig.png)
Our method relies on the following priors motivated by observations about rPPG:<br>
P1) Spatial-temporal maps are less subject to noise.​ -> STmap input<br>
P2) Self-attention based transformers can lead to better temporal modelling​ -> SwinU-Net model<br>
P3) Signals extracted using traditional methods contain more physiological information than raw averaged signals.​ -> Tmap positive sampling<br>
P4) Different facial videos most likely contain different rPPG signals​ -> Instance wise sampling for negative sampling<br>
P5) STmap signals are spatially and channel redundant​ -> Spatial and channel consistency in positive sampling<br>
P6) The rPPG signal is band limited [0.5, 3]Hz​ -> BP filtering and losses ignore frequencies outside the HR band<br>
P7) The spectrum of rPPG signals is sparse -> Sparsity can be used for more optimal sampling as it estimates level of noise<br>
P8) HR does not vary rapidly, due to the rPPG signal's strong autocorrelation -> Close-by in time segments should have high similarity<br>
P9) Resampling recordings at different rates can be used as an augmentation -> Negative samples with resampling to learn frequency features<br>
P10) Background pixels can be use to estimate noise -> Negative samples from background can help disentagle noise<br>
P11) Landmark positions over time can estimate motion noise -> Negative samples from motion estionamtion can help disentagle motion noise<br>

## Dataset Preprocessing

The original videos are firstly preprocessed by extracting the MSTmaps following https://github.com/nxsEdson/CVD-Physiological-Measurement. Both the MSTmaps and groundtruth bvp are resampled to 30 fps. To get the TMap positive augmentation maps use create_tmaps_VIPL.sh on your dataset. In the example code we assume the data used in pre-processed from PURE, OBF, MMSE or VIPL-HR datasets, but can't provide the actual data or preprocessed files. The structure of the data that can be used with our dataloader is: <br>
Dataset1: <br>
├── Sample1 (signal maps and ground truth are long N frames which is the total length of each video)  <br>
├──├── mstmap.npy (containts [63,N,3] multi-scale spatial-temporal maps calculated from videos) <br>
├──├── tmap.npy (containts [63,N,3] traditional augmenation calculated from the mstmaps by using create_tmaps_VIPL.sh) <br>
├──├── bvp.npy ( array with grountruth bvp signal [N]) <br>
<br>
If your dataset is processed in this way, with some minimal code changes you can use our dataloader as described by the following steps.

## Training
Please make sure your dataset is processed as described above. Firstly you need to pre-train the SwinU-Net network to predict tmaps from mstmaps, this can be done with the pretrain_mstmap2tmap.py script. Put the pretrained models into a folder named "./Trained/" and then you can finally train using the rsrppg method by using train_rsrppg.py.

## Citation
@inproceedings{savic2024rs,
  title={RS-rPPG: Robust Self-Supervised Learning for rPPG},
  author={Savic, Marko and Zhao, Guoying},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}


