import torch
import torch.nn as nn
from torch.nn import functional as F

import spconv.pytorch as spconv
from einops.layers.torch import Rearrange
from spconv.pytorch import functional as Fsp
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numba import jit


@jit(nopython=True)
def type_assign(batch_3D_val, batch_2D_val, batch_3Dtype_idf, batch_2Dtype_idf):

    N, M = batch_3D_val.shape[-1], batch_2D_val.shape[-1]
    ii, jj = 0, 0

    # 0, 1 within batch_3D/2Dtype_idf to indicate only 3D/2D and 2D+3D mixed
    while (ii < N) and (jj < M):
        if batch_3D_val[ii] < batch_2D_val[jj]:
            ii += 1
        elif batch_3D_val[ii] == batch_2D_val[jj]:
            batch_3Dtype_idf[ii] = 1
            batch_2Dtype_idf[jj] = 1
            ii += 1
            jj += 1
        else:
            jj += 1

    return batch_3Dtype_idf, batch_2Dtype_idf


def voxel_modality_split(voxel_3D, voxel_2D, B):
    """Split voxels into 3 groups: only 3D, only 2D, 3D+2D
    
        Input: SparseConvTensor, SparseConvTensor
        
        Return: SparseConvTensor, SparseConvTensor (with mix identifier saved in the 1st dim of indices)
    
    """
    coord_3D = voxel_3D.indices
    coord_2D = voxel_2D.indices
    coord_3D_mix, coord_2D_mix = [], []
    syn_mix_3D, syn_mix_2D = [], []
    last_batch_3D_len, last_batch_2D_len = 0, 0
    for i in range(B):
        batch_3D_mask = coord_3D[:, 0] == i
        batch_2D_mask = coord_2D[:, 0] == i
        batch_coord_3D = coord_3D[batch_3D_mask][:, 1:]
        batch_coord_2D = coord_2D[batch_2D_mask][:, 1:]
        
        batch_coord3D_xyz = batch_coord_3D[:, 0]*1e6 + batch_coord_3D[:, 1]*1e3 + batch_coord_3D[:, 2]
        batch_coord2D_xyz = batch_coord_2D[:, 0]*1e6 + batch_coord_2D[:, 1]*1e3 + batch_coord_2D[:, 2]

        batch_3D_val, batch_3D_ind = torch.sort(batch_coord3D_xyz, dim=-1)
        batch_2D_val, batch_2D_ind = torch.sort(batch_coord2D_xyz, dim=-1)

        batch_3Dtype_idf = torch.zeros_like(batch_3D_val)
        batch_2Dtype_idf = torch.zeros_like(batch_2D_val)
        batch_3Dmix_idf = torch.zeros_like(batch_3D_val)
        batch_2Dmix_idf = torch.zeros_like(batch_2D_val)

        device = batch_3D_val.device
        # 0, 1 within batch_3D/2Dtype_idf to indicate only 3D/2D and 2D+3D mixed
        batch_3Dtype_idf, batch_2Dtype_idf = type_assign(
            batch_3D_val.cpu().numpy(), batch_2D_val.cpu().numpy(), 
            batch_3Dtype_idf.cpu().numpy(), batch_2Dtype_idf.cpu().numpy()
        )
        batch_3Dtype_idf = torch.from_numpy(batch_3Dtype_idf).to(device)
        batch_2Dtype_idf = torch.from_numpy(batch_2Dtype_idf).to(device)

        # save mix voxel indices of 3D and 2D voxels for synchronization & remake index with non-empty voxels from previous batch padded 
        batch_orign_voxel3D_inds = batch_3D_ind[torch.where(batch_3Dtype_idf)[0]]
        batch_orign_voxel2D_inds = batch_2D_ind[torch.where(batch_2Dtype_idf)[0]]
        batch_orign_voxel3D_inds_padded = batch_orign_voxel3D_inds + last_batch_3D_len
        batch_orign_voxel2D_inds_padded = batch_orign_voxel2D_inds + last_batch_2D_len

        syn_mix_3D.append(batch_orign_voxel3D_inds_padded)
        syn_mix_2D.append(batch_orign_voxel2D_inds_padded)

        batch_3Dmix_idf = batch_3Dmix_idf.scatter(dim=-1, index=batch_3D_ind, src=batch_3Dtype_idf)
        batch_2Dmix_idf = batch_2Dmix_idf.scatter(dim=-1, index=batch_2D_ind, src=batch_2Dtype_idf)

        # add mix identifier before xyz, after batch identifier i.e., (mix_id, x, y, z)
        batch_coord_3D_mix = torch.cat([batch_3Dmix_idf.unsqueeze(1), batch_coord_3D], dim=-1)
        batch_coord_2D_mix = torch.cat([batch_2Dmix_idf.unsqueeze(1), batch_coord_2D], dim=-1)
        
        # add batch identifier at the very begining i.e., (batch_id, mix_id, x, y, z)
        batch_coord_3D_mix_pad = F.pad(batch_coord_3D_mix, (1, 0), mode='constant', value=i)
        batch_coord_2D_mix_pad = F.pad(batch_coord_2D_mix, (1, 0), mode='constant', value=i)
        coord_3D_mix.append(batch_coord_3D_mix_pad)
        coord_2D_mix.append(batch_coord_2D_mix_pad)

        last_batch_3D_len = batch_coord_3D.shape[0]
        last_batch_2D_len = batch_coord_2D.shape[0]


    coord_3D_mix = torch.cat(coord_3D_mix, dim=0)
    coord_2D_mix = torch.cat(coord_2D_mix, dim=0)
    syn_mix_3D = torch.cat(syn_mix_3D, dim=0)
    syn_mix_2D = torch.cat(syn_mix_2D, dim=0)

    voxel_3D.indices = coord_3D_mix.int()
    voxel_2D.indices = coord_2D_mix.int()

    return voxel_3D, voxel_2D, syn_mix_3D, syn_mix_2D


class RL3DFBackbone_T_split(nn.Module):
    def __init__(self, cfg):
        super(RL3DFBackbone_T_split, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]
        
        load_teacher = cfg.OPTIMIZER.get('LOAD_TEACHER', None)
        if load_teacher is not None:
            cfg_model = self.cfg.MODEL_T
        else:
            cfg_model = self.cfg.MODEL
        
        input_dim = cfg_model.PRE_PROCESSOR.INPUT_DIM # 4

        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.input_conv_new = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        self.input_conv = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] # in [64, 128, 256]
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] # in [64, 128, 256]
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'spconv{idx_enc}_new', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}_new', nn.BatchNorm1d(temp_ch))
            
            
            setattr(self, f'spconv{idx_enc}', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}a', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}a', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}b', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}b', nn.BatchNorm1d(temp_ch))
            
            setattr(self, f'value_layer{idx_enc}', nn.Linear(temp_ch, 1))

        # to BEV
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING
        if cfg_model.BACKBONE.TO_BEV.IS_Z_EMBED:
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat{idx_bev}', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        else:
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256] / ENCODING.CHANNEL
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))
                setattr(self, f'toBEV{idx_bev}', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}', nn.BatchNorm2d(temp_out_channel))
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l3_downsample = spconv.SparseMaxPool3d(kernel_size=2)

    def forward(self, dict_item):
        sparse_featuresR, sparse_indicesR = dict_item['sp_features'], dict_item['sp_indices']
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l'], dict_item['sp_indices_l']

        input_sp_tensorR = spconv.SparseConvTensor(
            features=sparse_featuresR,
            indices=sparse_indicesR.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xR = self.input_conv_new(input_sp_tensorR) 
        
        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xL = self.input_conv(input_sp_tensorL) 

        list_bev_featuresR = []
        list_bev_featuresL = []

        for idx_layer in range(self.num_layer):            
            batch = dict_item['batch_size']
            xL_mix, xR_mix, xL_dup_id, xR_dup_id = voxel_modality_split(xL, xR, batch) 
            
            # xL attention
            xL_feat, xL_indices = xL.features, xL.indices
            for batch_idx in range(batch):
                lidar_feat = xL_feat[xL_indices[:, 0] == batch_idx] 
                xL_attn_predict = torch.abs(lidar_feat).sum(1) / lidar_feat.shape[1] 
                voxel_importance = self.sigmoid(xL_attn_predict.view(-1, 1)).squeeze() 
                lidar_feat = torch.einsum('ab, a -> ab', lidar_feat, voxel_importance)
                if batch_idx == 0:
                    lidar_feat_attn = lidar_feat
                else:
                    lidar_feat_attn = torch.cat((lidar_feat_attn, lidar_feat), 0)

            xL_dup_id_inv = torch.arange(len(xL_mix.indices))[xL_mix.indices[:, 1] == 0]
            
            # enhance radar prompt with aggregation
            if idx_layer == 0:
                new_spatial_shape = self.spatial_shape
            else:
                new_spatial_shape = [int(self.spatial_shape[0] / (2**(idx_layer-1))), int(self.spatial_shape[1] / (2**(idx_layer-1))), int(self.spatial_shape[2] / (2**(idx_layer-1)))]
            
            indices = torch.cat((xL_indices[xL_dup_id_inv], xR.indices), 0)
            xR = spconv.SparseConvTensor(
                features=torch.cat((lidar_feat_attn[xL_dup_id_inv], xR.features), 0),
                indices=torch.cat((indices[:,:1], indices[:,2:]), 1),
                spatial_shape=new_spatial_shape,
                batch_size=dict_item['batch_size']
            )
            
            # update lidar voxel
            for batch_idx in range(batch):
                radar_indices = np.array(xR.indices[xR.indices[:, 0] == batch_idx][:, -3:].cpu())
                lidar_indices = np.array(xL.indices[xL.indices[:, 0] == batch_idx][:, -3:].cpu())
                lidar_feat = xL.features[xL.indices[:, 0] == batch_idx] 
                radar_feat = xR.features[xR.indices[:, 0] == batch_idx] 
                
                nbrs_rl = NearestNeighbors(n_neighbors=int(64 / (2**(idx_layer))), radius=int(8 / (2**(idx_layer)))).fit(radar_indices)
                distances_rl, knn_indices_rl = nbrs_rl.kneighbors(lidar_indices) 
                knn_indices_rl = torch.from_numpy(knn_indices_rl).to(device=lidar_feat.device)                
                near_radar_feat = radar_feat[knn_indices_rl] 
                near_radar_feat_mean = torch.mean(near_radar_feat, 1).squeeze()
                new_xL_feat = lidar_feat + near_radar_feat_mean
                if batch_idx == 0:
                    new_xL_feat_layer = new_xL_feat   
                else:
                    new_xL_feat_layer = torch.cat((new_xL_feat_layer, new_xL_feat), 0)

            xL = spconv.SparseConvTensor(
                features=new_xL_feat_layer,
                indices=torch.cat((xL_indices[:, :1], xL_indices[:, 2:]), 1),
                spatial_shape=new_spatial_shape,
                batch_size=dict_item['batch_size']
            )
            
            if idx_layer == 2:
                xR2 = xR
                dict_item['voxel_feat_radar_final_nodownsample'] = xR2
            
            # feature extraction
            xR = getattr(self, f'spconv{idx_layer}_new')(xR)
            xR = xR.replace_feature(getattr(self, f'bn{idx_layer}_new')(xR.features))
            xR = xR.replace_feature(self.relu(xR.features))
            
            xL = getattr(self, f'spconv{idx_layer}')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}a')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}a')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}b')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}b')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
    
            if self.is_z_embed:
                bev_denseL = getattr(self, f'chzcat{idx_layer}L')(xL)
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_denseL)
            else: 
                bev_spL = getattr(self, f'toBEV{idx_layer}')(xL)
                bev_spL = bev_spL.replace_feature(getattr(self, f'bnBEV{idx_layer}')(bev_spL.features))
                bev_spL = bev_spL.replace_feature(self.relu(bev_spL.features))
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}')(bev_spL.dense().squeeze(2))

            bev_denseL = getattr(self, f'bnt{idx_layer}')(bev_denseL)
            bev_denseL = self.relu(bev_denseL)
            list_bev_featuresL.append(bev_denseL)
            
            dict_item['voxel_feat_lidar_' + str(idx_layer)] = xL
            dict_item['voxel_feat_radar_' + str(idx_layer)] = xR
            dict_item['bev_feat_lidar_' + str(idx_layer)] = bev_denseL


        bev_featuresL = torch.cat(list_bev_featuresL, dim = 1)
        dict_item['bev_feat'] = bev_featuresL # B, C, Y, X

        return dict_item
    

class RL3DFBackbone_S(nn.Module):
    def __init__(self, cfg):
        super(RL3DFBackbone_S, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI
        grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']

        z_shape = int((z_max-z_min) / grid_size)
        y_shape = int((y_max-y_min) / grid_size)
        x_shape = int((x_max-x_min) / grid_size)

        self.spatial_shape = [z_shape, y_shape, x_shape]

        cfg_model = self.cfg.MODEL
        input_dim = cfg_model.PRE_PROCESSOR2.INPUT_DIM 

        list_enc_channel = cfg_model.BACKBONE.ENCODING.CHANNEL
        list_enc_padding = cfg_model.BACKBONE.ENCODING.PADDING
        list_enc_stride  = cfg_model.BACKBONE.ENCODING.STRIDE
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.input_convL = spconv.SparseConv3d(
            in_channels=input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] # in [64, 128, 256]
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] # in [64, 128, 256]
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'spconv{idx_enc}L', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'bn{idx_enc}L', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}aL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}aL', nn.BatchNorm1d(temp_ch))
            setattr(self, f'subm{idx_enc}bL', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'bn{idx_enc}bL', nn.BatchNorm1d(temp_ch))

        # to BEV
        list_bev_channel = cfg_model.BACKBONE.TO_BEV.CHANNEL
        list_bev_kernel = cfg_model.BACKBONE.TO_BEV.KERNEL_SIZE
        list_bev_stride = cfg_model.BACKBONE.TO_BEV.STRIDE
        list_bev_padding = cfg_model.BACKBONE.TO_BEV.PADDING
        if cfg_model.BACKBONE.TO_BEV.IS_Z_EMBED:
            self.is_z_embed = True
            for idx_bev in range(self.num_layer):
                setattr(self, f'chzcat{idx_bev}L', Rearrange('b c z y x -> b (c z) y x'))
                temp_in_channel = int(list_enc_channel[idx_bev]*z_shape/(2**idx_bev))
                temp_out_channel = list_bev_channel[idx_bev]
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_in_channel, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev], padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        else:
            self.is_z_embed = False
            for idx_bev in range(self.num_layer):
                temp_enc_ch = list_enc_channel[idx_bev] # in [64, 128, 256] / ENCODING.CHANNEL
                temp_out_channel = list_bev_channel[idx_bev]
                z_kernel_size = int(z_shape/(2**idx_bev))
                # print(z_kernel_size)
                setattr(self, f'toBEV{idx_bev}L', \
                    spconv.SparseConv3d(in_channels=temp_enc_ch, \
                        out_channels=temp_enc_ch, kernel_size=(z_kernel_size, 1, 1)))
                setattr(self, f'bnBEV{idx_bev}L', \
                    nn.BatchNorm1d(temp_enc_ch))
                setattr(self, f'convtrans2d{idx_bev}L', \
                    nn.ConvTranspose2d(in_channels=temp_enc_ch, out_channels=temp_out_channel, \
                        kernel_size=list_bev_kernel[idx_bev], stride=list_bev_stride[idx_bev],  padding=list_bev_padding[idx_bev]))
                setattr(self, f'bnt{idx_bev}L', nn.BatchNorm2d(temp_out_channel))
        # activation
        self.relu = nn.ReLU()

    def forward(self, dict_item):
        sparse_featuresL, sparse_indicesL = dict_item['sp_features_l_S'], dict_item['sp_indices_l_S']

        input_sp_tensorL = spconv.SparseConvTensor(
            features=sparse_featuresL,
            indices=sparse_indicesL.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        xL = self.input_convL(input_sp_tensorL)

        list_bev_featuresL = []

        for idx_layer in range(self.num_layer):
            xL = getattr(self, f'spconv{idx_layer}L')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}L')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}aL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}aL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            xL = getattr(self, f'subm{idx_layer}bL')(xL)
            xL = xL.replace_feature(getattr(self, f'bn{idx_layer}bL')(xL.features))
            xL = xL.replace_feature(self.relu(xL.features))
            
            if idx_layer == 0:
                dict_item['voxel_s'] = xL
    
            if self.is_z_embed:
                bev_denseL = getattr(self, f'chzcat{idx_layer}L')(new_XL_feat)
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_denseL)
            else: 
                bev_spL = getattr(self, f'toBEV{idx_layer}L')(xL)
                bev_spL = bev_spL.replace_feature(getattr(self, f'bnBEV{idx_layer}L')(bev_spL.features))
                bev_spL = bev_spL.replace_feature(self.relu(bev_spL.features))
                bev_denseL = getattr(self, f'convtrans2d{idx_layer}L')(bev_spL.dense().squeeze(2))

            bev_denseL = getattr(self, f'bnt{idx_layer}L')(bev_denseL)
            bev_denseL = self.relu(bev_denseL)

            list_bev_featuresL.append(bev_denseL)
            
            dict_item['voxel_feat_lidar_S_' + str(idx_layer)] = xL
            dict_item['bev_feat_lidar_S_' + str(idx_layer)] = bev_denseL

        bev_featuresL = torch.cat(list_bev_featuresL, dim = 1)
        dict_item['bev_feat_S'] = bev_featuresL

        return dict_item
    
