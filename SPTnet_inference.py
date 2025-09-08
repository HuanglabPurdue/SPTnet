import argparse
import sys
import os
from SPTnet_toolbox import *
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import tifffile
import torch, torch.nn as nn
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from os.path import dirname, basename
from transformer import *
from transformer3d import *

torch.cuda.empty_cache()
device = torch.device('cuda:0')
current_folder = os.path.dirname(os.path.abspath(__file__))
Tk().withdraw() # keep the root window from appearing
selected_directory = askopenfilename(initialdir=current_folder, title='#######Please select the trained model########')


class TiffDataset(Dataset):
    def __init__(self, tif_path):
        # load the full multi‐page TIFF as a (T, H, W) array
        video = tifffile.imread(tif_path)
        if video.ndim == 2:
            raise ValueError(f"{tif_path} contains only one frame; need a time series.")
        self.video = video.astype(np.float32)
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return {'video': self.video}



spt = SPTnet_toolbox(
    path_saved_model=selected_directory,
    momentum=0.9,
    learning_rate=0.0002,
    batch_size=96,
    use_gpu=True,
    image_size=64,
    number_of_frame=30,
    num_queries= 20
)

filename_test = askopenfilename(multiple=True,initialdir=current_folder, title='#######Please select the Test Data Files########') # show an "Open" dialog box and return the path to the selected file
transformer3d = Transformer3d(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
transformer = Transformer(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
model = spt.SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=30, spatial_t=transformer,  temporal_t=transformer3d, input_channel = 512).to(device)
##############################################
for file in filename_test:
    ext = os.path.splitext(file)[1].lower()
    base = os.path.splitext(basename(file))[0] + '.mat'
    if ext in ['.tif', '.tiff']:
        # use TIFF loader
        data_test = TiffDataset(file)
    else:
        data_test = spt.Transformer_mat2python(SPTnet_toolbox=spt, dataset_path=file)
    test_dataloader = torch.utils.data.DataLoader(data_test,
                                                       batch_size=1, shuffle=False, num_workers=0)
    spt.inference_with_SPTnet(model, test_dataloader)
    estimation_obj = np.vstack(spt.total_obj_est)
    estimation_xy = np.vstack(spt.total_xy_est)
    estimation_H = np.vstack(spt.total_H_est)
    estimation_C = np.vstack(spt.total_C_est)
    if not os.path.exists(dirname(spt.path_saved_model) + '/inference_results'):
        os.makedirs(dirname(spt.path_saved_model) + '/inference_results')
    sio.savemat(dirname(spt.path_saved_model) + '/inference_results/result' + basename(base),
                mdict={'obj_estimation': estimation_obj, 'estimation_xy': estimation_xy, 'estimation_H': estimation_H,
                       'estimation_C': estimation_C})

