import mmcv
import numpy as np
import os
import tempfile
import torch
import pickle
from functools import reduce
import time
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet3d.datasets import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import copy
from mmcv.parallel import DataContainer as DC
import random
from mmdet3d.core.bbox import get_box_type

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)

from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from ..core.evaluation.depth_metric import Metric_Depth
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader
from .ego_pose_dataset import EgoPoseDataset
from .waymo_dataset import CustomWaymoDataset
# from .occ_metrics import Metric_FScore,Metric_mIoU
from tqdm import tqdm

# 需要包含的功能，读取历史帧， 生成深度图， 变换矩阵， 读取occ， 留下高精地图接口
@DATASETS.register_module()
class CustomWaymoDataset_T(CustomWaymoDataset):
    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                 history_len=1,
                 skip_len=0,
                 withimage=True,
                 pose_file=None,
                 stereo=False,
                 filter_empty_gt=False,
                 img_info_prototype='bevdet',
                 use_valid_flag=True,
                 multi_adj_frame_id_cfg=None,
                 cams_name=None,
                 imgs_scales=None,
                 revised_label=False,
                 **kwargs):
        print('------------------------Init waymoOCC dataset!=======================')
        if pose_file is not None:
            with open(pose_file, 'rb') as f:
                pose_all = pickle.load(f)
                self.pose_all = pose_all
        #

        self.length = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.skip_len = skip_len
        self.withimage = withimage
        self.load_interval = load_interval
        super().__init__(*args, **kwargs)
        self.load_interval = load_interval
        # print()
        # self.length = len(self.data_infos)
        assert self.length == len(self.data_infos_full)
        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.use_valid_flag = use_valid_flag
        self.filter_empty_gt = filter_empty_gt
        self.stereo = stereo
        self.cams_name = cams_name
        self.imgs_scales = imgs_scales
        self.revised_label = revised_label



    def __len__(self):
        return self.length // self.load_interval

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # print('here after get data info input dict is: {}'.format(input_dict.keys()))
        # print('cams info is: ', input_dict['curr'].keys(), input_dict['curr']['image'].keys())
        # print(input_dict['curr']['image']['image_path'], input_dict['curr']['image']['image_shape'])
        # print(input_dict['curr']['calib'].keys(), input_dict['curr']['pose'].shape)

        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)

        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def union2one(self, queue):
        """
        input: queue: dict of [T-len+1, T], containing data_info
        convert sample queue into one single sample.
        calculate transformation from ego_now to image_old
        note that we dont gather gt objects of previous frames
        """

        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}

        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['sample_idx'] // 1000 != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['sample_idx'] // 1000
                metas_map[i]['scene_token'] = prev_scene_token
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0

            else:
                metas_map[i]['scene_token'] = prev_scene_token
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

    def refine_useful_info(self, info, index):
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []
            for idx_img in range(self.num_views):
                # cam2img @ ego2sensor ==> lidar2img
                pose = self.pose_all[info['curr_scene_idx']][info['curr_frame_idx']][idx_img]
                lidar2img = pose['intrinsics'] @ np.linalg.inv(pose['sensor2ego'])
                intrinsics = pose['intrinsics']
                sensor2ego = pose['sensor2ego']
                img_filename = os.path.join(self.data_root,
                                            info['image']['image_path'])
                # waymo 在转换时, 2-3的位置是刚好相反的
                # if idx_img == 2:
                #     image_paths.append(img_filename.replace('image_0', f'image_3'))
                # elif idx_img == 3:
                #     image_paths.append(img_filename.replace('image_0', f'image_2'))
                # else:
                #     image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))
                if idx_img == 2:
                    image_paths.append(img_filename.replace('image_0', f'image_3'))
                elif idx_img == 3:
                    image_paths.append(img_filename.replace('image_0', f'image_2'))
                else:
                    image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))

                lidar2img_rts.append(lidar2img)
                intrinsics_rts.append(intrinsics)
                sensor2ego_rts.append(sensor2ego)
        pts_filename = self._get_pts_filename(info['curr_sample_idx'])
        annos = self.get_ann_info(index)

        info.update(dict(
            pts_filename=pts_filename,
            # pts_filename=pts_filename,
            # img_prefix=None,
            # img_filename=image_paths,
            # lidar2img=lidar2img_rts,
            # cam2img=intrinsics_rts,
            # cam2ego=sensor2ego_rts,
            ann_infos=annos
        ))
        # print('info keys is: ', info.keys())
        for idx, keys_name in enumerate(self.cams_name):
            info['image'].update({keys_name: dict(
                img_prefix=None,
                img_filename=image_paths[idx],
                lidar2img=lidar2img_rts[idx],
                cam2img=intrinsics_rts[idx],
                cam2ego=sensor2ego_rts[idx],
            )})

        # print('the file should be used is: ', len(image_paths), len(lidar2img_rts), len(lidar2img_rts), len(intrinsics_rts), len(sensor2ego_rts))
        return info

    def get_data_info(self, index):
        info = self.data_infos_full[index]
        curr_sample_idx = info['image']['image_idx']
        curr_scene_idx = curr_sample_idx % 1000000 // 1000
        curr_frame_idx = curr_sample_idx % 1000000 % 1000
        info.update(dict(curr_scene_idx=curr_scene_idx, curr_frame_idx=curr_frame_idx, curr_sample_idx=curr_sample_idx))
        input_dict = dict(
            timestamp=info['timestamp'],
            curr_scene_idx=curr_scene_idx,
            curr_frame_idx=curr_frame_idx
        )
        info = self.refine_useful_info(info, index)

        # print('info curr is: ', info.keys())
        # print('curring 00sdwa: ', info['image'].keys(), info['image']['CAM_FRONT'].keys())
        if self.img_info_prototype == 'mmcv':
            pass
        else:
            input_dict.update(dict(curr=info))
            if '4d' in self.img_info_prototype:  # 需要再读取历史帧的信息
                info_adj_list = self.get_adj_info(info, index)
                input_dict.update(dict(adjacent=info_adj_list))

        # print('final step++++++++++++', input_dict['curr'].keys(), input_dict['adjacent'][0].keys())
        # print('image file path is: ', input_dict['curr']['img_filename'])

        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))     # bevdet4d: [1, ]  只利用前一帧.
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            # 如果使用stereo4d, 不仅当前帧需要利用前一帧图像计算stereo depth, 前一帧也需要利用它的前一帧计算stereo depth.
            # 因此, 我们需要额外读取一帧(也就是前一帧的前一帧).
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])

        # print('adj_id_list is: ', adj_id_list)
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)   # 越靠后的时间簇越靠前
            if not self.data_infos_full[select_id]['image']['image_idx'] % 1000000 // 1000 == \
                   info['image']['image_idx'] % 1000000 // 1000:
                info_adj_list.append(info)
            else:
                app_infos = self.data_infos_full[select_id]
                curr_sample_idx = app_infos['image']['image_idx']
                curr_scene_idx = curr_sample_idx % 1000000 // 1000
                curr_frame_idx = curr_sample_idx % 1000000 % 1000
                app_infos.update(dict(curr_scene_idx=curr_scene_idx, curr_frame_idx=curr_frame_idx, curr_sample_idx=curr_sample_idx))
                info = self.refine_useful_info(app_infos, select_id)
                info_adj_list.append(app_infos)

                # print('yes yes yes')

        return info_adj_list


    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        if self.test_mode == True:
            info = self.data_infos[index]
        else:
            info = self.data_infos_full[index]

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # print('in the most init process: ', gt_bboxes_3d.shape)
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))

        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        # print('+++++++++++++++: gt_bboxes_3d {}, gt_labels_3d {}, gt_bboxes {}, gt_labels {}, gt_names {}'.format(
        #     gt_bboxes_3d.shape, gt_labels_3d.shape, gt_bboxes.shape, gt_labels.shape, gt_names
        # ))
        return anns_results

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        print("metric = ", metric)
        if metric == 'ray-iou':
            occ_gts = []
            occ_preds = []
            lidar_origins = []

            print('\nStarting Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            sample_tokens = [info['token'] for info in self.data_infos]

            for i, batch in enumerate(data_loader):
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                occ_pred = occ_results[data_id]     # (Dx, Dy, Dz)

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

            eval_results = calc_rayiou(occ_preds, occ_gts, lidar_origins)

        elif metric == 'metirc_depth':
            print('+++++++++++++++++++++++++++++++metric is depth_metric')
            self.depth_eval_metrics = Metric_Depth(
                save_dir='.',
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=False,
                camera_id=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK',
                           'CAM_BACK_RIGHT'],
                depth_scale=(0.5, 50),
                image_mask_path=None)
            print('\nStarting Evaluation Depth Metric of NuScenes...')
            print('testing the length of pred is: {}'.format(len(occ_results)))
            for index, depth_pred_gt_depth in enumerate(tqdm(occ_results)):
                # occ_pred: (Dx, Dy, Dz)

                pred_depths = depth_pred_gt_depth[0]
                gt_depths = depth_pred_gt_depth[1]
                # print('pred_depth {}, gt_depth {}'.format(pred_depths.shape, gt_depths.shape))
                self.depth_eval_metrics.add_batch(
                    pred_depths,
                    gt_depths,
                    mask=None
                )
            print('start eval!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            eval_results = self.depth_eval_metrics.cac_depth_metric(
                    # pred_depths,
                    # gt_depths,
                    # mask=None
                )
                #
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                # depth_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                # gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                # mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                # mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                # # occ_pred = occ_pred
                # self.occ_eval_metrics.add_batch(
                #     occ_pred,   # (Dx, Dy, Dz)
                #     gt_semantics,   # (Dx, Dy, Dz)
                #     mask_lidar,     # (Dx, Dy, Dz)
                #     mask_camera     # (Dx, Dy, Dz)
                # )
                #
                # # print('type of occ_pred: {}, gt_semantics {}, mask_lidar: {}, mask_camera is: {}'.format(
                # #     occ_pred.shape, gt_semantics.shape, mask_lidar.shape, mask_camera.shape
                # # ))
                # # if index % 100 == 0 and show_dir is not None:
                # #     gt_vis = self.vis_occ(gt_semantics)
                # #     pred_vis = self.vis_occ(occ_pred)
                # #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                # #                  os.path.join(show_dir + "%d.jpg"%index))
                #
                # if show_dir is not None:
                #     mmcv.mkdir_or_exist(show_dir)
                #     # scene_name = info['scene_name']
                #     scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                #     sample_token = info['token']
                #     mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                #     save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                #     np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
        else:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True)

            print('\nStarting Evaluation...')
            print('testing the length of pred is: {}'.format(len(occ_results)))
            for index, occ_pred in enumerate(tqdm(occ_results)):
                # occ_pred: (Dx, Dy, Dz)
                info = self.data_infos[index]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                # occ_pred = occ_pred
                self.occ_eval_metrics.add_batch(
                    occ_pred,   # (Dx, Dy, Dz)
                    gt_semantics,   # (Dx, Dy, Dz)
                    mask_lidar,     # (Dx, Dy, Dz)
                    mask_camera     # (Dx, Dy, Dz)
                )

                # print('type of occ_pred: {}, gt_semantics {}, mask_lidar: {}, mask_camera is: {}'.format(
                #     occ_pred.shape, gt_semantics.shape, mask_lidar.shape, mask_camera.shape
                # ))
                # if index % 100 == 0 and show_dir is not None:
                #     gt_vis = self.vis_occ(gt_semantics)
                #     pred_vis = self.vis_occ(occ_pred)
                #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                #                  os.path.join(show_dir + "%d.jpg"%index))

                if show_dir is not None:
                    mmcv.mkdir_or_exist(show_dir)
                    # scene_name = info['scene_name']
                    scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                    sample_token = info['token']
                    mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                    np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)

            eval_results = self.occ_eval_metrics.count_miou()

        return eval_results
    # def evaluate(self, occ_results, metric='mIoU', runner=None, **eval_kwargs):
    #     def eval(occ_eval_metrics):
    #         print('\nStarting Evaluation...')
    #         for index, occ_result in enumerate(tqdm(occ_results)):
    #             voxel_semantics = occ_result['voxel_semantics']
    #             voxel_semantics_preds = occ_result['voxel_semantics_preds']
    #             mask_infov = occ_result.get("mask_infov", None)
    #             mask_lidar = occ_result.get("mask_lidar", None)
    #             mask_camera = occ_result.get("mask_camera", None)
    #
    #             occ_eval_metrics.add_batch(voxel_semantics_preds, voxel_semantics, mask_infov=mask_infov,
    #                                        mask_lidar=mask_lidar, mask_camera=mask_camera)
    #         occ_eval_metrics.print()
    #
    #     if "mIoU" in metric:
    #         occ_eval_metrics = Metric_mIoU()
    #         eval(occ_eval_metrics)
    #     elif "FScore" in metric:
    #         occ_eval_metrics = Metric_FScore()
    #         eval(occ_eval_metrics)
    #     else:
    #         raise NotImplementedError
