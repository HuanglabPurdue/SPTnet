## STPnet_toolbox
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D, PositionalEncodingPermute3D, Summer


class SPTnet_toolbox(object):
    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError('''Please call with keys: learning_rate, batch_size, momentum,
                      image_size, path_saved_model, use_gpu, number_of_frame''')
        learning_rate = batch_size = momentum = None
        image_size =  path_saved_model = use_gpu = number_of_frame = num_queries = diff_max = None
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'number_of_frame' in kwargs               :   number_of_frame = kwargs.pop('number_of_frame')
        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model') 
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu')
        if 'num_queries' in kwargs                   :   num_queries = kwargs.pop('num_queries')
        if 'diff_max' in kwargs                      :   diff_max = kwargs.pop('diff_max')
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if number_of_frame:
            self.number_of_frame = number_of_frame
        if image_size:
            self.image_size = image_size
        if  path_saved_model:
            self.path_saved_model = path_saved_model
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-4
        if momentum:
            self.momentum = momentum
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    #self.device = torch.device("cuda:0")
                    self.device = torch.device("cuda")
                else:
                    raise Exception("Please check GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if num_queries:
            self.num_queries = num_queries
        if diff_max:
            self.diff_max = diff_max

    # Transfer Matlab simulated brownian motion movie into suitable datasets
    class Transformer_mat2python(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.Transformer_mat2python, self).__init__()
            self.spt = SPTnet_toolbox
            self.validation = []
            dataset = h5py.File(dataset_path, 'r')
            self.dataset = dataset

            # Detect whether labels exist; cache timelapsedata
            self.has_labels = all(k in self.dataset for k in ['Hlabel', 'Clabel', 'traceposition'])
            if 'timelapsedata' not in self.dataset:
                raise KeyError("Missing variable 'timelapsedata' in dataset.")
            self.td = self.dataset['timelapsedata']  # 3D: (T,H,W) or 4D: (N,T,H,W)

        def __len__(self):
            if self.has_labels:
                return len(self.dataset['Hlabel'][1])
            # video-only
            if self.td.ndim == 3:  # (T,H,W)
                return 1
            elif self.td.ndim == 4:  # (N,T,H,W)
                return self.td.shape[0]
            else:
                raise ValueError(f"'timelapsedata' must be 3D or 4D, got {self.td.shape}.")

        def __getitem__(self, idx):
            # ---------------- Video loading (works for both labeled and unlabeled) ----------------
            if self.td.ndim == 3:
                if idx != 0:
                    raise IndexError("Index out of range for single 3D movie.")
                video = np.array(self.td)  # (T,H,W)
            else:
                video = np.array(self.td[idx])  # (T,H,W)

            # ---------------- Unlabeled path ----------------
            if not self.has_labels:
                # Return just the video (no extra channel here; add channel once in your toolbox)
                return {'video': video}

            # ---------------- Labeled path (your original logic) ----------------
            Hlabel_ref = np.array(self.dataset['Hlabel'][:, idx])  # references
            Clabel_ref = np.array(self.dataset['Clabel'][:, idx])
            position_ref = np.array(self.dataset['traceposition'][:, idx])

            Hlabel = np.zeros(len(Hlabel_ref))
            Clabel = np.zeros(len(Hlabel_ref))
            # video.shape[0] == T
            position = np.full([video.shape[0], len(Hlabel_ref), 2], np.nan)
            class_label = np.full([video.shape[0], len(Hlabel_ref)], 0)

            j = 0
            for i in range(len(Hlabel_ref)):
                if np.array(self.dataset[Hlabel_ref[i]][0]) != 0:
                    Hlabel[j] = float(np.array(self.dataset[Hlabel_ref[i]][0]))
                    Clabel[j] = float(np.array(self.dataset[Clabel_ref[i]][0]))
                    pos_arr = np.array(self.dataset[position_ref[i]]).T  # expect (T,2)
                    if pos_arr.size == video.shape[0] * 2:
                        position[:, j, :] = pos_arr
                        class_label[:, j] = np.multiply(~np.isnan(position[:, j, 0]), 1)
                        j += 1

            class_label_pd = np.pad(
                class_label, [(0, 0), (0, self.spt.num_queries - Hlabel.shape[0])],
                'constant', constant_values=0
            )
            padding_config = ((0, 0), (0, self.spt.num_queries - Hlabel.shape[0]), (0, 0))
            position_pd = np.pad(position, padding_config, 'constant', constant_values=np.nan)

            outfov_mask = np.any(
                (position_pd < -self.spt.image_size / 2) | (position_pd > self.spt.image_size / 2),
                axis=2
            )
            class_label_pd[outfov_mask] = 0

            sample = {
                'video': video,  # (T,H,W)
                'position': position_pd,  # (T, num_queries, 2)
                'Hlabel': Hlabel,
                'Clabel': Clabel,
                'class_label': class_label_pd
            }
            return sample
            
    class inference_simulation_data(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.inference_simulation_data, self).__init__()
            dataset = h5py.File(dataset_path, 'r')
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset['Hlabel'][0])

        def __getitem__(self, idx):
            video = np.array(self.dataset['timelapsedata'][idx])  # timelapsedata is variable name in matlab
            sample = {'video': video}
            return sample

    class runningwindow_simulationdata(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.runningwindow_simulationdata, self).__init__()
            dataset = h5py.File(dataset_path, 'r')
            self.dataset = dataset

        def __len__(self):
            return self.dataset['timelapsedata'].shape[0]-30+1

        def __getitem__(self, idx):
            video = np.array(self.dataset['timelapsedata'])# timelapsedata is variable name in matlab
            sample_rw = np.zeros((video.shape[0] - 30 + 1, 30, video.shape[1], video.shape[2]))
            for i in range(0,video.shape[0]-30+1):
                sample_rw[i,:,:,:] = video[i:(30 + i), :, :]
            sample = {'video': sample_rw[idx,:,:,:]}
            return sample


    #For beads data
    class beadsdata(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.beadsdata,self).__init__() 
            dataset = h5py.File(dataset_path,'r')
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset['beadsdata'])

        def __getitem__ (self,idx):
            video = np.array(self.dataset['beadsdata'])  #timelapsedata is variable name in matlab 
            sample = {'video' : video}
            return sample

    # For beads data
    class experimental_data(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.experimental_data, self).__init__()
            dataset = h5py.File(dataset_path, 'r')
            self.dataset = dataset

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            video = np.array(self.dataset['ims'])
            sample = {'video': video}
            return sample

    # For ER data
    class ER_data(torch.utils.data.Dataset):
        def __init__(self, SPTnet_toolbox, dataset_path):
            super(SPTnet_toolbox.ER_data, self).__init__()
            dataset = h5py.File(dataset_path, 'r')
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset['ims'])

        def __getitem__(self, idx):
            video = np.array(self.dataset['ims'][idx])
            sample = {'video': video}
            return sample



    #Dataloader, load number of data based on the batch size. Also shuffle training data, and data transfer can be performed here.
    def data_loader(self, dataserver_train, split_chunks_for_validation):
        dataserver_train, val_set = torch.utils.data.random_split(dataserver_train, split_chunks_for_validation)
        # mean_t, std_t = dataserver_train.mean(), dataserver_train.std()
        # dataserver_train = transforms.Normalize(mean_t, std_t)
        self.dataserver_train = dataserver_train
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                           batch_size= self.batch_size,shuffle=True, num_workers=0, drop_last=True) #transforms.Normalize
        self.val_dataloader = torch.utils.data.DataLoader(val_set,
                           batch_size=self.batch_size,shuffle=True, num_workers=0, drop_last=True)

    def testdata_loader(self, dataserver_test):
        self.test_data = torch.utils.data.DataLoader(dataserver_test,batch_size=1,
                                                     shuffle=True, num_workers=0,drop_last=True)
    #Testdata_loader, load test data only
    def testdata_loader(self, dataserver_test):   
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                           batch_size=1,shuffle=False, num_workers=0)


    ####################################### SPTnet ##################################################
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(SPTnet_toolbox.ResidualBlock, self).__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3, 3], stride=(1, 1, 1),
                                   padding=(1, 1, 1), bias=True)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3, 3], stride=(1, 1, 1),
                                   padding=(1, 1, 1), bias=True)
            self.shortcut = nn.Sequential()
            if stride != (1, 1, 1) or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=[1, 1, 1], stride=(1, 1, 1), padding=(0, 0, 0),
                              bias=True)
                )

        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class BackBone(nn.Module):
        def __init__(self):
            super(SPTnet_toolbox.BackBone, self).__init__()
            self.in_channels = 16
            self.conv1 = nn.Conv3d(1, 16, kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
            self.layer1 = self.make_layer(32, 2, stride=(1, 1, 1))
            self.layer2 = self.make_layer(64, 2, stride=(1, 1, 1))
            self.layer3 = self.make_layer(128, 2, stride=(1, 1, 1))
            self.layer4 = self.make_layer(256, 2, stride=(1, 1, 1))
            self.avg_pool = nn.AdaptiveAvgPool3d((30, 2, 2))
            self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
            self.adaptive_pool = nn.AdaptiveAvgPool3d((30, 4, 4))

        def make_layer(self, out_channels, num_blocks, stride):
            layers = []
            layers.append(SPTnet_toolbox.ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(SPTnet_toolbox.ResidualBlock(out_channels, out_channels, stride))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.layer1(x)
            x = self.pool1(x)
            x = self.layer2(x)
            x = self.pool1(x)
            x = self.layer3(x)
            x = self.pool1(x)
            x = self.layer4(x)
            x = self.pool1(x)
            # x = x.view(x.size(0), -1)
            return x

    class SPTnet(nn.Module):
        def __init__(self, num_classes, num_queries, num_frames, spatial_t, temporal_t, input_channel):
            super(SPTnet_toolbox.SPTnet,self).__init__()
            self.input_channel = input_channel
            # CNN backbone
            self.backbone = SPTnet_toolbox.BackBone()
            self.conv_temp = nn.Conv1d(1, num_frames, kernel_size=1, stride=1, padding=0)
            # position encoding
            d_model = temporal_t.d_model
            # Transformer
            # Query embedding
            self.num_queries = num_queries
            self.transformer = spatial_t
            self.transformer3d = temporal_t
            self.query_embed = nn.Embedding(num_queries, d_model)
            #self.sp_query_embed = nn.Embedding(num_queries, d_model)
            # FC layer
            self.fc1 = nn.Linear(256, 32)
            self.fc1_1 = nn.Linear(32, 2)
            self.fc2 = nn.Linear(256, 1)
            self.fc3 = nn.Linear(256, 64)
            self.fc4 = nn.Linear(64, 2)

        def forward(self,x):
            # Extract features using CNN
            x1 = self.backbone(x)
            features = F.relu(x1)
            batch_size, channels, num_frames, H, W = features.shape
            # Positional encoding
            pos = PositionalEncodingPermute3D(channels).cuda()
            pos = pos(features)
            sp_pos = PositionalEncodingPermute2D(channels).cuda()
            # Queries and mask
            queries = self.query_embed.weight
            #sp_queries = self.sp_query_embed.weight

            mask = torch.zeros((batch_size, num_frames, H, W), dtype=torch.bool, device=torch.device('cuda'))
            sp_mask = torch.zeros((batch_size*num_frames, H, W), dtype=torch.bool, device=torch.device('cuda'))
            # Transformer input
            sp_features = features.permute(0,2,1,3,4).flatten(0,1)
            sp_pos = sp_pos(sp_features)
            sp_hs = self.transformer(sp_features,sp_mask,queries,sp_pos)[0]
            sp_hs = sp_hs.view(batch_size,num_frames,self.num_queries,channels)

            hs1 = self.transformer3d(features, mask, queries, pos)[0]#hs[1]: batch,channel,frm,H,W  , hs[0]: 1,batch,queries,channel
            ts_hs = hs1.permute(1, 2, 0, 3).flatten(0, 1)
            ts_hs = self.conv_temp(ts_hs)
            ts_hs = ts_hs.view(batch_size,num_frames,self.num_queries,-1)
            deco_comb = ts_hs+sp_hs
            # coordinate loss
            center_pred = self.fc1(deco_comb)
            center_pred = F.relu(center_pred)
            center_pred = torch.tanh(self.fc1_1(center_pred))
            center_pred = center_pred.permute(0,2,1,3)
            # obj detection loss
            xf_obj = self.fc2(deco_comb)
            class_logits = torch.sigmoid(xf_obj).squeeze(-1)
            class_logits = class_logits.permute(0, 2, 1)
            xf_hd = self.fc3(hs1.squeeze(0))
            xf_hd = F.relu(xf_hd)
            xf_hd = torch.sigmoid(self.fc4(xf_hd))
            H_est = xf_hd[:, :, 0].unsqueeze(-1)
            D_est = xf_hd[:, :, 1].unsqueeze(-1)
        # prediction heads
            return class_logits,center_pred,H_est,D_est


############## test_attentionSPT ################################################
    def inference_with_SPTnet(self, model, testdata):
        model.load_state_dict(torch.load(self.path_saved_model),strict=False)
        model.eval()
        total = 0
        total_obj_est = []
        total_xy_est = []
        total_H_est = []
        total_C_est = []
        with torch.no_grad():
            for jj, data in enumerate(testdata):
                inputs = data['video']
                inputs = torch.unsqueeze(inputs, 1).float().cuda()   # float64 is actually "double"
                image_max = inputs.max()
                image_min = inputs.min()
                inputs = ((inputs) - image_min) / (image_max - image_min)
                class_out, center_out, H_out, C_out = model(inputs)
                class_out, H_out, C_out = class_out.unsqueeze(0), H_out.squeeze(), C_out.squeeze()
                total_obj_est.append(np.array(class_out.cpu()))
                total_xy_est.append(np.array(center_out.cpu()))
                total_H_est.append(np.array(H_out.cpu()))
                total_C_est.append(np.array(C_out.cpu()))
                total += 1
        self.total_obj_est = total_obj_est
        self.total_xy_est = total_xy_est
        self.total_H_est = total_H_est
        self.total_C_est = total_C_est
