import argparse
import sys
import os
from SPTnet_toolbox import *
import scipy.io as sio
import torch, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from contextlib import closing
import multiprocessing
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.multiprocessing import start_processes
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from matplotlib.animation import FuncAnimation
from os.path import dirname, basename
import math
from scipy.optimize import linear_sum_assignment
import torch.profiler
from transformer import *
from transformer3d import *
torch.cuda.empty_cache()
device = torch.device('cuda:0')
import cProfile

current_folder = os.path.dirname(os.path.abspath(__file__))
selected_directory = askdirectory(initialdir=current_folder, title='#######Please select the folder to save the trained model########')
model_name = "trained_model"
# Combine the directory and name
full_path = os.path.join(selected_directory, model_name)

spt = SPTnet_toolbox(
    path_saved_model=full_path,
    momentum=0.9,
    learning_rate=0.0002,
    batch_size=4,
    use_gpu=True,
    image_size=64,
    number_of_frame=30,
    num_queries= 30
)
#/media/chengbi/SSD1/2-24-2022 lowphotontrainingdata/lowSNRtrainigdata_1.mat
# /media/chengbi/SSD1/Trainingdata1-19-2022/single precision/20000_100steps_1.mat

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename_train = askopenfilename(multiple=True, initialdir='/media/chengbi/SSD1/Training 2023/AttentionSPT', title='#######Please select all the Training Data File########') # show an "Open" dialog box and return the path to the selected file
data_train = []
for file in filename_train:
    datafile = spt.Transformer_mat2python(SPTnet_toolbox=spt, dataset_path=file, num_queries=spt.num_queries)
    data_train = torch.utils.data.ConcatDataset([data_train,datafile])
spt.data_loader(data_train, [int(len(data_train)*0.8), int(len(data_train)*0.2)])  # train_set, data_test, split training data for validation [train, val].
file_path = os.path.join(os.path.dirname(__file__), 'CRLB_H_D_frame.mat')
CRLB_matrix = np.array(h5py.File(file_path, 'r')['CRLB_matrix_HD_frame'])

def hungarian_matched_loss(pred_classes, pred_positions, pred_H, pred_C, gt_classes, gt_positions, gt_H, gt_C):
    num_batches, num_queries, num_frames = pred_classes.shape
    loss_pb = 0
    fullindex = np.arange(len(pred_classes[0,:,0]))
    gt_positions = gt_positions.permute(0,2,1,3)
    zeros_pd = torch.zeros(spt.batch_size, 10).cuda()
    gt_H = torch.cat((gt_H, zeros_pd), dim=1)
    gt_C = torch.cat((gt_C, zeros_pd), dim=1)
    for b in range(num_batches):
        total_loss = 0
        non_obj_loss_all = 0
        track_flag = sum(gt_classes[b,:])>=2
        num_tracks = int(sum(track_flag))
        if num_tracks != 0:
                # Calculate the cost matrix for hungarian matching
            gt_pos_track = gt_positions[b,:, :, :][track_flag,:,:].unsqueeze(0).repeat(num_queries,1,1,1)
            gt_classes_pm = gt_classes[b,:][:,track_flag].permute(1,0)
            class_loss_matrix = F.binary_cross_entropy(pred_classes[b,:,:].view(num_queries,1,num_frames).repeat(1, num_tracks,1), gt_classes_pm.view(1,num_tracks,num_frames).repeat(num_queries, 1,1),reduction='none')
            nan_mask = torch.isnan(gt_pos_track)
            gt_pos_track[nan_mask] = 0
            pred_masked = pred_positions[b, :, :, :].unsqueeze(1).repeat(1,num_tracks,1,1)
            pred_masked[nan_mask] = 0
            pos_loss_matrix = pdist(pred_masked, gt_pos_track)
            pos_loss_matrix = torch.nansum(pos_loss_matrix,dim=2)
            cost_matrix_class_pf = torch.mean(class_loss_matrix,dim=2)
            duration = sum(gt_classes[b, :])[track_flag]
            pos_loss_matrix_allfrm_pf = pos_loss_matrix/duration
            gt_H_nonzero = gt_H[b][track_flag]
            gt_C_nonzero = gt_C[b][track_flag]
            H_idx = torch.clamp((gt_H_nonzero*100).round()-1,min=0,max=98).cpu().numpy().astype(int)
            C_idx = torch.clamp((gt_C_nonzero*0.5*100).round() - 1, min=0, max=49).cpu().numpy().astype(int)
            stepidx = duration.cpu().numpy().astype(int)-1
            CRLBweight_H = CRLB_matrix[0,0,C_idx,H_idx,stepidx] / CRLB_matrix[0, 0, C_idx, H_idx, 29]
            CRLBweight_C = CRLB_matrix[1, 1, C_idx, H_idx, stepidx] / CRLB_matrix[1, 1, C_idx, H_idx, 29]
            H_loss_matrix = criterion_mae(pred_H[b].view(-1,1).repeat(1, gt_H_nonzero.shape[-1]),gt_H_nonzero.view(1,-1).repeat(pred_H.shape[-1],1)) / torch.tensor(CRLBweight_H).repeat(pred_H.shape[-1],1).cuda()
            C_loss_matrix = criterion_mae(pred_C[b].view(-1, 1).repeat(1, gt_C_nonzero.shape[-1]),gt_C_nonzero.view(1, -1).repeat(pred_C.shape[-1], 1)) /torch.tensor(CRLBweight_C).repeat(pred_H.shape[-1],1).cuda()
            cost_matrix_all_pf = (cost_matrix_class_pf + 2*pos_loss_matrix_allfrm_pf + 0.5*H_loss_matrix + 0.5*C_loss_matrix).t()
            # Compute the optimal assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix_all_pf.cpu().detach().numpy())
            # Calculate the losses for the assigned pairs
            cost_matrix_all_pf = cost_matrix_all_pf[row_indices, col_indices].sum()

            total_class = (cost_matrix_class_pf.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
            total_coordi = (2*pos_loss_matrix_allfrm_pf.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
            total_hurst = (0.5*H_loss_matrix.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks
            total_diffusion = (0.5*C_loss_matrix.t().cpu().detach().numpy()[row_indices,col_indices].sum()) / num_tracks

            # Not matched trajectory loss
            non_obj_pre = pred_classes[b,:,:][np.setdiff1d(fullindex, col_indices),:]
            non_obj_loss = F.binary_cross_entropy(non_obj_pre, torch.zeros(non_obj_pre.shape).cuda(),reduction='mean')
            loss_pv = (cost_matrix_all_pf/num_tracks) + non_obj_loss
            loss_pb += loss_pv
        else:
            non_obj_loss = F.binary_cross_entropy(gt_classes[b,:,:], torch.zeros(gt_classes[b,:,:].shape).cuda(),reduction='mean')
            loss_pb += non_obj_loss
    return loss_pb / num_batches, total_class / num_batches,  total_coordi / num_batches, total_hurst / num_batches, total_diffusion / num_batches


def train_step(batch_idx, data):
    model.train()
    inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
    inputs = torch.unsqueeze(inputs, 1).float().cuda() # float64 is actually "double"
    for i in range(0, spt.batch_size):
        image_max = inputs[i][0].max()
        image_min = inputs[i][0].min()
        inputs[i][0] = ((inputs[i][0])-image_min) / (image_max-image_min)

    class_out, center_out, H_out, C_out = model(inputs)  # class out [batch, frames, queries, 1]  center out [batch, frames,queries, 2]
    class_out, H_out, C_out = class_out.squeeze(), H_out.squeeze(), C_out.squeeze()
    class_label, position_label, Hlabel, Clabel = class_label.float().cuda(), (position_label/32).float().cuda(), Hlabel.float().cuda(), (Clabel/0.5).float().cuda()
    t_loss, cl_ls, coor_ls, h_ls, diff_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)

    # pos_loss = criterion_xy(x_est, grid_label[:,:,:,:,3],grid_label[:, :, :, :, 0]) + criterion_xy(y_est, grid_label[:,:,:,:,4],grid_label[:, :, :, :, 0])
    optimizer.zero_grad()
    t_loss.backward()
    optimizer.step()
    t_loss, cl_ls, coor_ls, h_ls, diff_ls = t_loss.item(), cl_ls.item(), coor_ls.item(), h_ls.item(), diff_ls.item()
    return t_loss, cl_ls, coor_ls, h_ls, diff_ls

def val_step(batch_idx, data):
    model.eval()
    inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
    inputs = torch.unsqueeze(inputs, 1).float().cuda() # float64 is actually "double"
    for i in range(0, spt.batch_size):
        image_max = inputs[i][0].max()
        image_min = inputs[i][0].min()
        inputs[i][0] = ((inputs[i][0])-image_min) / (image_max-image_min)

    inputs = Variable(inputs, requires_grad=False)
    class_out, center_out, H_out, C_out = model(inputs)
    class_out, H_out, C_out = class_out.squeeze(), H_out.squeeze(), C_out.squeeze()
    class_label, position_label, Hlabel, Clabel = class_label.float().cuda(), (position_label/32).float().cuda(), Hlabel.float().cuda(), (Clabel/0.5).float().cuda()
    v_loss, cl_ls, coor_ls, h_ls, diff_ls = hungarian_matched_loss(class_out, center_out, H_out, C_out, class_label, position_label, Hlabel, Clabel)
    v_loss, cl_ls, coor_ls, h_ls, diff_ls = v_loss.item(), cl_ls.item(), coor_ls.item(), h_ls.item(), diff_ls.item()
    return v_loss, cl_ls, coor_ls, h_ls, diff_ls

#torch.backends.cudnn.benchmark = True  # use the fastest convolution methods when the inputs size are fixed improves performance
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
criterion_mae = nn.L1Loss(reduction='none').to(device)
pdist = nn.PairwiseDistance(p=2)
transformer3d = Transformer3d(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
transformer = Transformer(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
model = spt.SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=spt.number_of_frame, spatial_t=transformer,
                   temporal_t=transformer3d, input_channel=512).to(device)
torch.autograd.set_detect_anomaly(True)

optimizer_SGD = optim.SGD(model.parameters(), lr=spt.learning_rate, momentum=spt.momentum)
optimizer_Adam = optim.Adam(model.parameters(), lr=spt.learning_rate, weight_decay=1e-5)
optimizer_AdamW = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
optimizer = optimizer_AdamW
# scheduler_rdpl = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=5, verbose=True,
#                                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                  eps=1e-08)
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
##############################################
t_loss_append = []
t_loss_epoch_cls_append = []
t_loss_epoch_coor_append = []
t_loss_epoch_hurst_append = []
t_loss_epoch_diff_append = []

v_loss_append = []
v_loss_epoch_cls_append = []
v_loss_epoch_coor_append = []
v_loss_epoch_hurst_append = []
v_loss_epoch_diff_append = []

epoch_list = []

no_improvement = 0
min_v_loss = 99999999999
max_num_of_epoch_without_improving = 6
epoch = 1
#
start = time.time()
lr = []

modelrecord = open(spt.path_saved_model + 'training_log.txt', 'a')
fig, ax = plt.subplots(nrows=2,ncols=3)
while no_improvement < max_num_of_epoch_without_improving:
# for epoch in range(n_epochs):
    epoch_list.append(epoch)
    t_loss_total = 0
    t_loss_total_cls = 0
    t_loss_total_coor = 0
    t_loss_total_hurst = 0
    t_loss_total_diff = 0
    v_loss_total = 0
    v_loss_total_cls = 0
    v_loss_total_coor = 0
    v_loss_total_hurst = 0
    v_loss_total_diff = 0
    pbar = tqdm(spt.train_dataloader)
    for batch_idx, data in enumerate(spt.train_dataloader):
        t_loss, cl_ls, coor_ls, h_ls, diff_ls = train_step(batch_idx, data)
        t_loss_total+= t_loss
        t_loss_total_cls += cl_ls
        t_loss_total_coor += coor_ls
        t_loss_total_hurst += h_ls
        t_loss_total_diff += diff_ls
        pbar.set_description(f"Epoch {epoch}")
        pbar.update(1)
    pbar.close()
            # pbar.set_postfix(str(vallossepo))
    t_loss_epoch = t_loss_total/(batch_idx+1)
    t_loss_epoch_cls  = t_loss_total_cls/(batch_idx+1)
    t_loss_epoch_coor = t_loss_total_coor/(batch_idx+1)
    t_loss_epoch_hurst = t_loss_total_hurst/(batch_idx+1)
    t_loss_epoch_diff = t_loss_total_diff/(batch_idx+1)
        # lr.append(scheduler_rdpl.get_lr()[0])

    for batch_idx, data in enumerate(spt.val_dataloader):
        v_loss, cl_ls, coor_ls, h_ls, diff_ls = val_step(batch_idx, data)
        v_loss_total+=v_loss
        v_loss_total_cls += cl_ls
        v_loss_total_coor += coor_ls
        v_loss_total_hurst += h_ls
        v_loss_total_diff += diff_ls
    v_loss_epoch = v_loss_total / (batch_idx + 1)
    v_loss_epoch_cls = v_loss_total_cls / (batch_idx + 1)
    v_loss_epoch_coor = v_loss_total_coor / (batch_idx + 1)
    v_loss_epoch_hurst = v_loss_total_hurst / (batch_idx + 1)
    v_loss_epoch_diff = v_loss_total_diff / (batch_idx + 1)


    if v_loss_epoch < min_v_loss:
        min_v_loss = v_loss_epoch
        no_improvement = 0
        torch.save(model.state_dict(), spt.path_saved_model) #Save model.module.state_dict() in DDP case!!!
        #torch.save(DP_model.module.state_dict(), spt.path_saved_model)
        torch.save(optimizer.state_dict(), spt.path_saved_model+'optimizer_stat')
        print('==> Saving a new best model')
    else:
        no_improvement+=1
    lr.append(optimizer.param_groups[0]['lr'])
    # scheduler_rdpl.step(v_loss_epoch)
    print('learning rate is: %f' %lr[-1])

    # total loss
    t_loss_append.append(t_loss_epoch)
    v_loss_append.append(v_loss_epoch)
    try:
        t_loss_line.remove(t_loss_line[0])
        v_loss_line.remove(v_loss_line[0])
    except Exception:
        pass
    t_loss_line = ax[0,0].plot(epoch_list, t_loss_append, 'r', lw=2)
    v_loss_line = ax[0,0].plot(epoch_list, v_loss_append, 'b', lw=2)
    ax[0,0].set_title('Total loss')
    modelrecord.write('\nepoch %d, t_loss: %s, v_loss: %s' % (epoch, t_loss_epoch,v_loss_epoch))

    # cls_loss
    t_loss_epoch_cls_append.append(t_loss_epoch_cls)
    v_loss_epoch_cls_append.append(v_loss_epoch_cls)
    try:
        t_cls_loss_line.remove(t_cls_loss_line[0])
        v_cls_loss_line.remove(v_cls_loss_line[0])
    except Exception:
        pass
    t_cls_loss_line = ax[0,1].plot(epoch_list, t_loss_epoch_cls_append, 'r', lw=2)
    v_cls_loss_line = ax[0,1].plot(epoch_list, v_loss_epoch_cls_append, 'b', lw=2)
    ax[0,1].set_title('cls loss')
    modelrecord.write(', t_cls_loss: %s, v_cls_loss: %s' % (t_loss_epoch_cls,v_loss_epoch_cls))

    # coor_loss
    t_loss_epoch_coor_append.append(t_loss_epoch_coor)
    v_loss_epoch_coor_append.append(v_loss_epoch_coor)
    try:
        t_coor_loss_line.remove(t_coor_loss_line[0])
        v_coor_loss_line.remove(v_coor_loss_line[0])
    except Exception:
        pass
    t_coor_loss_line = ax[0,2].plot(epoch_list, t_loss_epoch_coor_append, 'r', lw=2)
    v_coor_loss_line = ax[0,2].plot(epoch_list, v_loss_epoch_coor_append, 'b', lw=2)
    ax[0,2].set_title('coordinate loss')
    modelrecord.write(', t_coor_loss: %s, v_coor_loss: %s' % (t_loss_epoch_coor,v_loss_epoch_coor))

    # Hurst_loss
    t_loss_epoch_hurst_append.append(t_loss_epoch_hurst)
    v_loss_epoch_hurst_append.append(v_loss_epoch_hurst)
    try:
        t_hurst_loss_line.remove(t_hurst_loss_line[0])
        v_hurst_loss_line.remove(v_hurst_loss_line[0])
    except Exception:
        pass
    t_hurst_loss_line = ax[1,0].plot(epoch_list, t_loss_epoch_hurst_append, 'r', lw=2)
    v_hurst_loss_line = ax[1,0].plot(epoch_list, v_loss_epoch_hurst_append, 'b', lw=2)
    ax[1,0].set_title('hurst loss')
    modelrecord.write(', t_hurst_loss: %s, v_hurst_loss: %s' % (t_loss_epoch_hurst, v_loss_epoch_hurst))

    # Hurst_loss
    t_loss_epoch_diff_append.append(t_loss_epoch_diff)
    v_loss_epoch_diff_append.append(v_loss_epoch_diff)
    try:
        t_diff_loss_line.remove(t_diff_loss_line[0])
        v_diff_loss_line.remove(v_diff_loss_line[0])
    except Exception:
        pass
    t_diff_loss_line = ax[1,1].plot(epoch_list, t_loss_epoch_diff_append, 'r', lw=2)
    v_diff_loss_line = ax[1,1].plot(epoch_list, v_loss_epoch_diff_append, 'b', lw=2)
    ax[1,1].set_title('diffusion loss')
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(spt.path_saved_model+'learning curve')
    modelrecord.write(', t_diff_loss: %s, v_diff_loss: %s' % (t_loss_epoch_diff, v_loss_epoch_diff))
    print("(""epoch", epoch, ")", "Training Loss", t_loss_epoch, "Validation Loss", v_loss_epoch)
    epoch+=1
end = time.time()
print("...Done Training...")
print("...Training takes %d s..." % (end - start))

modelrecord.write('\n...Training for %d epoch...\nThe minimal validation loss is %s\n' % (
epoch, min_v_loss))
modelrecord.close()