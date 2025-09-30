'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* description: pipeline for 3D object detection
* changed: 2023-01-02
'''

import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import shutil
from torch.utils.data import Subset
import cv2
import matplotlib.pyplot as plt
import time

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file
from utils.util_ui_labeling import *

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result

class PipelineDetection_v1_0():
    def __init__(self, path_cfg=None, mode='train'):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        # self.update_cfg_regarding_mode()

        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)
        
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        if mode == 'train':
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated

        self.network_teacher = build_network_teacher(self).cuda()
        state_dict = torch.load(self.cfg.OPTIMIZER.LOAD_TEACHER_PATH)
        self.network_teacher.load_state_dict(state_dict)
        assert torch.equal(self.network_teacher.backbone.bn1_new.weight, state_dict['backbone.bn1_new.weight']), 'params do not match after copying'

        self.network = build_network(self).cuda()
        self.optimizer = build_optimizer(self, self.network)
        self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0

        # Logging
        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False
        
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()

        # Vis
        self.set_vis()
        
        self.show_pline_description()
        
    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 16
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 16
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.dict_cls_id_to_name = dict()
        for k, v in self.dict_cls_name_to_id.items():
            if v != -1:
                self.dict_cls_id_to_name[v] = k
        self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
        self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB
    
    def show_pline_description(self):
        print('* newtork_T (description start) -------')
        print(self.network_teacher)
        print('* newtork_S (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        else:
            print('* Exception error (Pipeline): same folder exists, try again')
            exit()

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        # code backup (TBD)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch + 1
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        except:
            print('* Exception error (Pipeline): check resume network')
            exit()

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def train_network(self, is_shuffle=True):
        self.network.train()
        self.network_teacher.eval()
        self.network_teacher.training = False

        data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
            batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
            collate_fn = self.dataset_train.collate_fn,
            num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        for epoch in range(epoch_start, epoch_end):
            torch.cuda.empty_cache()
            print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')
            
            self.network.train()
            self.network.training = True
            avg_loss = []
            for idx_iter, dict_datum in enumerate(tqdm(data_loader_train)):
                if (idx_iter % 50) == 49:
                    torch.cuda.empty_cache()
                dict_net = self.network(dict_datum)
                with torch.no_grad():
                    dict_net_teacher = self.network_teacher(dict_datum)
                loss = self.network.head_S.loss(dict_net, dict_net_teacher)
                
                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if loss == 0.:
                    pass
                    # print('loss is 0.') # No objs for all samples
                elif torch.isfinite(loss):
                    loss.backward()
                else:
                    print('* Exception error (pipeline): nan or inf loss happend')
                    print('* Meta: ', dict_datum['meta'])

                self.optimizer.step()
                if not (self.scheduler is None):
                    self.scheduler.step()
                
                self.optimizer.zero_grad()

                if self.is_logging:
                    dict_logging = dict_net['logging']
                    idx_log_iter +=1
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (self.scheduler is None):
                        lr = self.scheduler.get_last_lr()
                        self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)

            if self.is_save_model:
                # epoch: indexing from 0
                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch}.pt')

                if (epoch+1) % self.interval_epoch_model == 0:
                    torch.save(self.network.state_dict(), path_dict_model)
                if (epoch+1) % self.interval_epoch_util == 0:
                    dict_util = {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'idx_log_iter': idx_log_iter, 
                    }
                    if not (self.scheduler is None):
                        dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                    torch.save(dict_util, path_dict_util)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', np.mean(avg_loss), epoch)

            if self.is_validate:
                self.network.training=False
                if self.is_consider_subset:
                    if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                        self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                if ((epoch + 1) % self.val_per_epoch_full) == 0:
                    self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr)

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    # V2
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        self.network.eval()
        
        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)
            ### Vis data ###
            pc_lidar = dict_datum['ldr_pc_64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)
            
            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred
    
    def save_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        self.network.eval()
        with torch.no_grad():
            if is_train:
                dataset_loaded = self.dataset_train
            else:
                dataset_loaded = self.dataset_test
            subset = Subset(dataset_loaded, sample_indices)
            data_loader = torch.utils.data.DataLoader(subset,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
            
            for idx, dict_datum in enumerate(data_loader):
                print(sample_indices[idx], end=" ")
                dict_out = self.network(dict_datum)
                seq = dict_out['meta'][0]['path']['cam_front_img'].split('/')[3]
                img_num = dict_out['meta'][0]['path']['cam_front_img'].split('/')[5].split('_')[-1]

                dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)
                rl_bev = self.func_show_rlbev_bbox(dict_out)
                cl_range = self.func_show_clrange_bbox(dict_out)
                
                h1, w1, _ = rl_bev.shape
                h2, w2, _ = cl_range.shape
                rl_bev = np.array((F.interpolate(torch.from_numpy(rl_bev/255).permute(2, 0, 1).unsqueeze(0), (h2, int(w1*h2/h1)), mode='bilinear')).squeeze(0).permute(1, 2, 0))
                final = np.concatenate((cl_range, rl_bev*255), 1)
                cv2.imwrite('./vis/' + str(sample_indices[idx]).zfill(5) + '_' + seq + '_' + img_num, final)


    def func_show_clrange_bbox(self, dict_out):  
        img_size = (1280, 720)
        tr_rotation_default = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        list_lc_calib_keys = [
            'fx', 'fy', 'px', 'py', \
            'k1', 'k2', 'k3', 'k4', 'k5', \
            'roll_c', 'pitch_c', 'yaw_c', \
            'roll_l', 'pitch_l', 'yaw_l', 'x_l', 'y_l', 'z_l']
        
        list_lc_calib_init_values = [
            557.720776478944, 567.2136917114258, 636.720776478944, 369.3068656921387, \
            -0.028873818023371287, 0.0006023302214797655, 0.0039573086622276855, -0.005047176298643093, 0.0, \
            0.0, 0.0, 0.0, \
            0.0, 0.7, -0.5, 0.1, 0.0, -0.7]
        list_lc_calib_init_offsets = [
            2.0, 2.0, 2.0, 2.0, \
            0.01, 0.0001, 0.001, 0.001, 0.001, \
            1.0, 1.0, 1.0, \
            0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
        dict_lc_calib_now_values = dict()
        dict_lc_calib_offsets = dict()
        for idx1, k in enumerate(list_lc_calib_keys):
            dict_lc_calib_now_values.update({k:list_lc_calib_init_values[idx1]})
            dict_lc_calib_offsets.update({k:list_lc_calib_init_offsets[idx1]})

        
        intrinsics, distortion, r_cam, tr_lid_cam = get_matrices_from_dict_lc_calib(dict_lc_calib_now_values)
        cv_img = cv2.imread(dict_out['meta'][0]['path']['cam_front_img'])
        cv_img = cv_img[:,:1280,:] # lc
        cv_img_ori = cv_img.copy()
        img_process = cv_img_ori # self.img_lc
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics[j,i] = ncm[j, i]
        
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, r_cam, ncm, img_size, cv2.CV_32FC1)
        img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)
        
        ### Get additional attribute for point cloud ###
        path_pcd_lc = dict_out['meta'][0]['path']['ldr_pc_64']
        pcd = o3d.io.read_point_cloud(path_pcd_lc)
        list_attributes = ['intensity', 'reflectivity', 'ring']
        dict_values = dict()
        for attribute in list_attributes:
            dict_values.update({attribute: read_attribute_from_pcd(attribute, path_pcd_lc)})
            
        ### Get additional attribute for point cloud ###
        points_roi = np.array(np.asarray(pcd.points))
        
        # if self.checkBox_pc_roi.isChecked():
        # x_min, x_max, y_min, y_max, z_min, z_max = get_pc_roi_from_txt(self.textEditLcCalib_1.toPlainText())
        x_min, x_max, y_min, y_max, z_min, z_max = 0, 98.8, -40, 39.6, -2, 5.6

        points = np.array(np.asarray(pcd.points))
        list_for_concatenation = [points]
        for attribute in list_attributes:
            list_for_concatenation.append(dict_values[attribute])
        points_roi = np.concatenate(list_for_concatenation, axis=1)
        points_roi = np.array(list(filter(lambda x: (x[0]>x_min) and (x[0]<x_max) \
            and (x[1]>y_min) and (x[1]<y_max) and (x[2]>z_min) and (x[2]<z_max), points_roi.tolist())))
        pcd.points = o3d.utility.Vector3dVector(points_roi[:,:3])
        values_roi = points_roi[:,3:]
        for idx_temp, attribute in enumerate(list_attributes):
            dict_values.update({attribute: values_roi[:,idx_temp:idx_temp+1]})
        for idx_temp, attribute in enumerate(['x', 'y', 'z']):
            dict_values.update({attribute: points_roi[:,idx_temp:idx_temp+1]})
        
        P2 = np.insert(intrinsics, 3, values=[0,0,0], axis=1) # K = intrinsics
        R0_hom = r_cam
        R0_hom = np.insert(R0_hom,3,values=[0,0,0],axis=0)
        R0_hom = np.insert(R0_hom,3,values=[0,0,0,1],axis=1) # 4x4
        points_hom = np.transpose(points_roi[:,:3].copy(), (1,0))
        points_hom = np.insert(points_hom, 3, 1, axis=0) # 4xN
        LidarToCamera = np.insert(tr_lid_cam, 3, values=[0,0,0,1], axis=0)
        
        # Adding default rotation
        cam = np.matmul(np.matmul(np.matmul(P2,R0_hom),np.matmul(LidarToCamera,tr_rotation_default)),points_hom)
        cam[:2] /= cam[2,:]
        png = np.flip(img_process, axis=2) # bgr to rgb

        plt.figure(figsize=(6.705, 12.7), dpi=100, tight_layout=True)
        IMG_H,IMG_W,_ = png.shape
        plt.axis([0,IMG_W,IMG_H,0])
        plt.imshow(png)
        u,v,z = cam
        u_out = np.logical_or(u<0, u>IMG_W)
        v_out = np.logical_or(v<0, v>IMG_H)
        outlier = np.logical_or(u_out, v_out)
        cam = np.delete(cam,np.where(outlier),axis=1)
        u,v,z = cam
        plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.2,s=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./vis/lidar_bev/cam_lidar_range.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        cl_range = cv2.imread('./vis/lidar_bev/cam_lidar_range.png')
        alpha = 0.5
        lthick = 2
        labels = dict_out['label'][0]
        calib = dict_out['calib'][0]
        list_obj_label = []
        for label_obj in labels:
            cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
            obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
            list_obj_label.append(obj)
            
        preds = dict_out['pp_bbox']
        list_obj_pred = []
        if preds is None:
            pass
        else: 
            for pred in preds:
                conf_score, x, y, z, xl, yl, zl, theta = pred
                obj3d = Object3D(x, y, z, xl, yl, zl, theta)
                list_obj_pred.append(obj3d)            
            
        list_line_order = [[0, 1], [2, 3], #[0, 3], [1, 2],
                            [4, 5], [6, 7], #[5, 6], [4, 7],
                            [0, 4], [1, 5], [2, 6], [3, 7],
                            [0, 2], [1, 3], [4, 6], [5, 7]]
        for obj_ in list_obj_label:
            corners = obj_.corners
            corners[:,0] -= calib[0]
            corners[:,1] -= calib[1]
            corners[:,2] -= calib[2]

            points_hom = np.transpose(corners.copy(), (1,0))
            points_hom = np.insert(points_hom, 3, 1, axis=0) # 4xN
            cam = np.matmul(np.matmul(np.matmul(P2,R0_hom),np.matmul(LidarToCamera,tr_rotation_default)),points_hom)
            cam[:2] /= cam[2,:]
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = cam[:2, idx_1]
                p2_x, p2_y = cam[:2, idx_2]
                p1_x, p1_y, p2_x, p2_y = int(p1_x*0.5), int(p1_y*0.5), int(p2_x*0.5), int(p2_y*0.5)
                color = (0, 0, 255)
                cl_range = cv2.line(cl_range, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)
        
        for obj_ in list_obj_pred:
            corners = obj_.corners
            corners[:,0] -= calib[0]
            corners[:,1] -= calib[1]
            corners[:,2] -= calib[2]

            points_hom = np.transpose(corners.copy(), (1,0))
            points_hom = np.insert(points_hom, 3, 1, axis=0) # 4xN
            cam = np.matmul(np.matmul(np.matmul(P2,R0_hom),np.matmul(LidarToCamera,tr_rotation_default)),points_hom)
            cam[:2] /= cam[2,:]
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = cam[:2, idx_1]
                p2_x, p2_y = cam[:2, idx_2]
                p1_x, p1_y, p2_x, p2_y = int(p1_x*0.5), int(p1_y*0.5), int(p2_x*0.5), int(p2_y*0.5)
                color = (0, 255, 255)
                cl_range = cv2.line(cl_range, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)
        
        return cl_range

            
    def func_show_rlbev_bbox(self, dict_item):
        # radar bev
        rdr_cube, rdr_cube_cnt = dict_item['rdr_cube'][0].numpy(), dict_item['rdr_cube_cnt'][0].numpy()
        arr_x, arr_y, arr_z = np.arange(0, 99.2, 0.4), np.arange(-40, 40, 0.4), np.arange(-2, 6.0, 0.4)
        num_z, num_y, num_x = rdr_cube.shape
        rdr_cube_bev = np.sum(rdr_cube, axis=0) # 200, 248
        rdr_cube_bev = rdr_cube_bev/rdr_cube_cnt
        rdr_cube_bev = np.maximum(rdr_cube_bev, 1.)
        rdr_cube_bev = 10*np.log10(rdr_cube_bev)
        rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
        arr_0, arr_1 = np.meshgrid(arr_x, arr_y)
              
        rdr_cube_bev_vis = cv2.cvtColor((rdr_cube_bev*255.).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bboxes = dict_item['label'][0]
        list_line_order = [[0,1], [0,2], [1,3], [2,3]]
        lthick = 1
        alpha = 0.5

        rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
        plt.figure(figsize=(2.48, 2))
        plt.pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig('vis/lidar_bev/radar_bev.png', pad_inches=0, dpi=100) 
        plt.close()
        
        # lidar bev9
        pc_lidar = dict_item['ldr_pc_64']
        lpc_roi = [0,100,-40,40,-10,60] 
        lpc_roi[1] = 99
        pc_lidar = pc_lidar[
            np.where(
                (pc_lidar[:, 0] > lpc_roi[0]) & (pc_lidar[:, 0] < lpc_roi[1]) &
                (pc_lidar[:, 1] > lpc_roi[2]) & (pc_lidar[:, 1] < lpc_roi[3]) &
                (pc_lidar[:, 2] > lpc_roi[4]) & (pc_lidar[:, 2] < lpc_roi[5])
            )]

        # 200:248 -> 80:100
        rdr_cube = cv2.imread('vis/lidar_bev/radar_bev.png')
        lidar_cube = (np.ones((200, 248, 3)) * 255).astype(np.uint8)
        for i in range(len(pc_lidar)):
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 0] = 0
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 1] = 0
            lidar_cube[int((pc_lidar[i, 1]+40)*2.5), int(pc_lidar[i, 0]*2.5), 2] = 0
        lidar_cube = cv2.flip(lidar_cube, 0)
        lr_bev = cv2.addWeighted(rdr_cube, alpha, lidar_cube, 1-alpha, 0)
        lr_bev = cv2.flip(lr_bev, 0) 

        for bbox in bboxes:
            _, _, [x, y, z, theta, xl, yl, zl], _ = bbox
            obj3d = Object3D(x, y, z, xl, yl, zl, theta)
            idx_x = np.argmin(np.abs(arr_x-x))
            idx_y = np.argmin(np.abs(arr_y-y))
            pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
            pt_list = []
            for pt in pts:
                idx_x = np.argmin(np.abs(arr_x-pt[0]))
                idx_y = np.argmin(np.abs(arr_y-pt[1]))
                pt_list.append([idx_x, idx_y])
            for idx_1, idx_2 in list_line_order:
                p1_x, p1_y = pt_list[idx_1]
                p2_x, p2_y = pt_list[idx_2]
                color = (0, 0, 255)
                lr_bev = cv2.line(lr_bev, (p1_x,p1_y), (p2_x,p2_y), color, thickness=2)
                    
        preds = dict_item['pp_bbox']
        if preds is None:
            pass
        else:
            for pred in preds:
                conf_score, x, y, z, xl, yl, zl, theta = pred
                obj3d = Object3D(x, y, z, xl, yl, zl, theta)
                idx_x = np.argmin(np.abs(arr_x-x))
                idx_y = np.argmin(np.abs(arr_y-y))
                pts = [obj3d.corners[0,:], obj3d.corners[2,:], obj3d.corners[4,:], obj3d.corners[6,:]]
                pt_list = []
                for pt in pts:
                    idx_x = np.argmin(np.abs(arr_x-pt[0]))
                    idx_y = np.argmin(np.abs(arr_y-pt[1]))
                    pt_list.append([idx_x, idx_y])
                for idx_1, idx_2 in list_line_order:
                    p1_x, p1_y = pt_list[idx_1]
                    p2_x, p2_y = pt_list[idx_2]
                    color = (0, 0, 0)
                    lr_bev = cv2.line(lr_bev, (p1_x,p1_y), (p2_x,p2_y), color, thickness=lthick)

        lr_bev = cv2.flip(lr_bev, 0)
        lr_bev = lr_bev.transpose((1, 0, 2))
        lr_bev = cv2.flip(lr_bev, 0)
        return lr_bev


    def func_show_radar_tensor_bev(self, dict_item, bboxes=None, \
            roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_return_bbox_bev_tensor=False, alpha=0.9, lthick=1, infer=None, infer_gt=None, norm_img=None):
        
        rdr_tensor = get_tesseract(dict_item['meta']['path_rdr_tesseract'])
        rdr_bev = np.mean(np.mean(rdr_tensor, axis=0), axis=2)
        arr_range, arr_azimuth, arr_elevation = self.load_physical_values()
        
        arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)
        height, width = np.shape(rdr_bev)

        plt.clf()
        plt.cla()
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.savefig('./resources/imgs/img_tes_ra.png', bbox_inches='tight', pad_inches=0, dpi=300)
        
        temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
        temp_row, temp_col, _ = temp_img.shape
        if not (temp_row == height and temp_col == width):
            temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

        plt.close()
        plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
        plt.colorbar()
        plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

        # Polar to Cartesian (Should flip image)
        ra = cv2.imread('./resources/imgs/img_tes_ra.png')
        ra = np.flip(ra, axis=0)

        arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, arr_range, arr_azimuth, \
            roi_x = roi_x, roi_y = roi_y, is_in_deg=False)

        ### Image processing ###
        if norm_img is None:
            pass
        elif norm_img.split('_')[0] == 'hist': # histogram stretching
            arr_yx = cv2.normalize(arr_yx, None, 0, 255, cv2.NORM_MINMAX)
        elif norm_img.split('_')[0] == 'alp':
            alp = float(norm_img.split('_')[1])
            arr_yx = np.clip((1+alp)*arr_yx - 128*alp, 0, 255).astype(np.uint8)

        if not (bboxes is None):
            arr_yx_bbox = arr_yx.copy()

            ### inference ###
            if infer_gt is not None:
                bboxes_gt = []
                for idx_obj, obj in enumerate(infer_gt):
                    bboxes_gt.append(['Gt Sedan', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
                arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_gt, lthick=lthick)
            
            if infer is not None:
                # print(infer)
                bboxes_infer = []
                for idx_obj, obj in enumerate(infer):
                    bboxes_infer.append(['Infer', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
                arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_infer, lthick=lthick)

        # alpha = 0.9
        arr_yx_bbox = cv2.addWeighted(arr_yx_bbox, alpha, arr_yx, 1 - alpha, 0)

        # flip before show
        # arr_yx = arr_yx.transpose((1,0,2))
        # arr_yx = np.flip(arr_yx, axis=(0,1))

        # print(dict_item['meta'])
        # cv2.imshow('Cartesian', arr_yx)

        if not (bboxes is None):
            arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
            arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
            cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx_bbox,(0,0),fx=4,fy=4))
        else:
            arr_yx = arr_yx.transpose((1,0,2))
            arr_yx = np.flip(arr_yx_bbox, axis=(0,1))
            cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx,(0,0),fx=2,fy=2))

        # cv2.imshow('Front image', cv2.imread(dict_item['meta']['path_cam_front'])[:,:1280,:])
        # plt.show()
        cv2.waitKey(0)
        
    def get_tesseract(self, path_tesseract, is_in_DRAE=True, is_in_3d=False, is_in_log=False):
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        if is_in_DRAE:
            arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2))
        if is_in_3d:
            arr_tesseract = np.mean(arr_tesseract, axis=3) # reduce elevation
        if is_in_log:
            arr_tesseract = 10*np.log10(arr_tesseract)
        return arr_tesseract
    
    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.eval()
        # self.network.training=False
        with torch.no_grad():
            ### Check is_validate with small dataset ###
            if is_subset:
                is_shuffle = False
                minival_id = list(range(0, 17500, 3)) # num:5834
                tqdm_bar = tqdm(total=len(minival_id), desc='* MiniVal (Subset): ')
                log_header = 'minival'
                dataset_test1 = Subset(self.dataset_test, minival_id)
                data_loader = torch.utils.data.DataLoader(dataset_test1, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS) # self.cfg.OPTIMIZER.NUM_WORKERS
                print(self.cfg.OPTIMIZER.NUM_WORKERS)
            else:
                is_shuffle = False
                tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
                log_header = 'val_tot'
                data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = 4)
            
            if epoch is None:
                dir_epoch = 'none'
            else:
                dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

            # initialize via VAL.LIST_VAL_CONF_THR
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
            # print(path_dir)
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                    f.write('')
                f.close()

            for idx_datum, dict_datum in enumerate(data_loader):
                if (idx_datum % 50) == 49:
                    torch.cuda.empty_cache()
                if is_subset & (idx_datum >= self.val_num_subset):
                    break
                
                try:
                    dict_out = self.network(dict_datum) # inference
                    is_feature_inferenced = True
                except:
                    print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                    # print('* Meta info: ', dict_out['meta'])
                    is_feature_inferenced = False

                idx_name = str(idx_datum).zfill(6)

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    if is_feature_inferenced:
                        dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                    else:
                        dict_out = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                    if dict_out is None:
                        print('* Exception error (Pipeline): dict_item is None in validation')
                        continue

                    dict_out = dict_datum_to_kitti(self, dict_out)

                    if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                        pass
                    else:
                        ### Gt ###
                        for idx_label, label in enumerate(dict_out['kitti_gt']):
                            open_mode = 'w' if idx_label == 0 else 'a'
                            with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                                f.write(label+'\n')
                        ### Gt ###

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out['kitti_desc'])
                        ### Process description ###

                        ### Pred: do not care len 0 with if else: already care as dummy ###
                        for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                            open_mode = 'w' if idx_pred == 0 else 'a'
                            with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                                f.write(pred+'\n')
                        ### Pred: do not care len 0 with if else: already care as dummy ###

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                
                tqdm_bar.update(1)
            tqdm_bar.close()

            ### Validate per conf ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

                dt_annos = kitti.get_label_annos(preds_dir)
                val_ids = read_imageset_file(split_path)
                gt_annos = kitti.get_label_annos(labels_dir, val_ids)

                list_metrics = []
                for idx_cls_val in self.list_val_care_idx:
                    dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                    print(f'-----conf{conf_thr}-----')
                    print(result_log)
                    list_metrics.append(dict_metrics)

                for dict_metrics in list_metrics:
                    cls_name = dict_metrics['cls']
                    ious = dict_metrics['iou']
                    bevs = dict_metrics['bev']
                    ap3ds = dict_metrics['3d']
                    self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': bevs[0],
                        f'iou_{ious[1]}_{cls_name}': bevs[1],
                        f'iou_{ious[2]}_{cls_name}': bevs[2],
                    }, epoch)
                    self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                        f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                        f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                        f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                    }, epoch)
            ### Validate per conf ###
            

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
        self.network.eval()
        # self.network.training=False
        with torch.no_grad():
            
            road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
            time_cond_list = ['day', 'night']
            weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

            ### Check is_validate with small dataset ###
            if is_subset:
                is_shuffle = False
                minival_id = list(range(0, 17500, 10)) # num:1750
                tqdm_bar = tqdm(total=len(minival_id), desc='* MiniVal (Subset): ')
                log_header = 'minival'
                dataset_test1 = Subset(self.dataset_test, minival_id)
                data_loader = torch.utils.data.DataLoader(dataset_test1, \
                    batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
            else:
                is_shuffle = False
                tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
                data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                        batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
                        num_workers = 16)
            
            if epoch is None:
                dir_epoch = 'none'
            else:
                dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

            # initialize via VAL.LIST_VAL_CONF_THR
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

                os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                    f.write('')

                for road_cond in road_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                        f.write('')

                for time_cond in time_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                        f.write('')

                for weather_cond in weather_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                        f.write('')

                pred_dir_list = []
                label_dir_list = []
                desc_dir_list = []
                split_path_list = []

                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
                                
                ### For Specific Conditions ###
                for road_cond in road_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for time_cond in time_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for weather_cond in weather_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)

            # Creating gts and preds txt files for evaluation
            for idx_datum, dict_datum in enumerate(data_loader):
                if is_subset & (idx_datum >= self.val_num_subset):
                    break

                if (idx_datum % 1000) == 0:
                    torch.cuda.empty_cache()

                try:
                    dict_out = self.network(dict_datum) # inference
                    is_feature_inferenced = True
                except:
                    print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                    print('* Meta info: ', dict_out['meta'])
                    is_feature_inferenced = False

                if is_print_memory:
                    print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                    
                idx_name = str(idx_datum).zfill(6)
                
                road_cond_tag, time_cond_tag, weather_cond_tag = \
                    dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                # print(dict_out['desc'][0])

                ### for every conf in list_conf_thr ###
                for conf_thr in list_conf_thr:
                    ### For All Conditions ###
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                    preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                    labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                    desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                    split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                    preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                    labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                    desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                    split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                    preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                    labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                    desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                    split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                    os.makedirs(labels_dir_road, exist_ok=True)
                    os.makedirs(labels_dir_time, exist_ok=True)
                    os.makedirs(labels_dir_weather, exist_ok=True)
                    os.makedirs(desc_dir_road, exist_ok=True)
                    os.makedirs(desc_dir_time, exist_ok=True)
                    os.makedirs(desc_dir_weather, exist_ok=True)
                    os.makedirs(preds_dir_road, exist_ok=True)
                    os.makedirs(preds_dir_time, exist_ok=True)
                    os.makedirs(preds_dir_weather, exist_ok=True)

                    if is_feature_inferenced:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                    else:
                        dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                    if dict_out_current is None:
                        print('* Exception error (Pipeline): dict_item is None in validation')
                        continue

                    dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                    if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                        pass
                    else:
                        ### Gt ###
                        for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                            if idx_label == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')
                            with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(label+'\n')

                        ### Process description ###
                        with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                            f.write(dict_out_current['kitti_desc'])

                        ### Process description ###
                        if len(dict_out_current['kitti_pred']) == 0:
                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write('\n')
                        else:
                            for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                                if idx_pred == 0:
                                    mode = 'w'
                                else:
                                    mode = 'a'

                                with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')
                                with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                    f.write(pred+'\n')

                        str_log = idx_name + '\n'
                        with open(split_path, 'a') as f:
                            f.write(str_log)
                        with open(split_path_road, 'a') as f:
                            f.write(str_log)
                        with open(split_path_time, 'a') as f:
                            f.write(str_log)
                        with open(split_path_weather, 'a') as f:
                            f.write(str_log)
                tqdm_bar.update(1)
            tqdm_bar.close()

            ### Validate per conf ###
            all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
            for conf_thr in list_conf_thr:
                for condition in all_condition_list:
                    try:
                        preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                        labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                        desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                        split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                        dt_annos = kitti.get_label_annos(preds_dir)
                        val_ids = read_imageset_file(split_path)
                        gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                        list_metrics = []
                        list_results = []
                        for idx_cls_val in self.list_val_care_idx:
                            dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                            list_metrics.append(dict_metrics)
                            list_results.append(result)
                        print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                        with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                            for dic_metric in list_metrics:
                                print('='*50)
                                print('Cls: ', dic_metric['cls'])
                                print('IoU:', dic_metric['iou'])
                                print('BEV: ', dic_metric['bev'])
                                print('3D: ', dic_metric['3d'])
                                print('-'*50)
                                
                                f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                                f.write('cls: ' + dic_metric['cls'] + '\n')
                                f.write('iou: ')
                                for iou in dic_metric['iou']:
                                    f.write(str(iou) + ' ')
                                f.write('\n')
                                f.write('bev: ')
                                for bev in dic_metric['bev']:
                                    f.write(str(bev) + ' ')
                                f.write('\n')
                                f.write('3d  :')
                                for det3d in dic_metric['3d']:
                                    f.write(str(det3d) + ' ')
                                f.write('\n\n')
                        print('\n')
                    except:
                        print('* Exception error (Pipeline): Samples for the codition are not found')

            path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
            print(f'* Check {path_check}')
            ### Validate per conf ###
