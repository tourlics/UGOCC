# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from ..core.evaluation.depth_metric import Metric_Depth


from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader

nuscenes_to_revise = {
    0: 6,
    1: 6,
    2: 2,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 7,
    9: 1,
    10: 1,
    11: 0,
    12: 4,
    13: 5,
    14: 4,
    15: 3,
    16: 4,
    17: 8,
}

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])

vis_names = [
    'imgs', 'sensor2keyegos', 'ego2globals', 'intrins', 'post_rots',
    'post_trans', 'bda', 'gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera'
]

@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT',
                 stereo=False,
                 revised_label=None,
                 occ_class_nums=18,
                 occ_label_to_revise=None,
                 revised_occ_label=None
                 ):
        # 调用父类的构造函数
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
            img_info_prototype=img_info_prototype,
            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
            ego_cam=ego_cam,
            stereo=stereo)


        # 子类的额外参数
        self.load_interval = load_interval
        self.revised_label = revised_label
        self.occ_class_nums = occ_class_nums
        self.occ_label_to_revise = occ_label_to_revise
        self.revised_occ_label = revised_occ_label

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def save_tensors_to_npy(self, tensor_list, tensor_names, output_dir='outputs_debug/idx/'):
        """
        保存一个包含张量的列表到指定目录中的 .npy 文件。

        参数:
        tensor_list (list): 包含要保存的 PyTorch 张量的列表。
        tensor_names (list): 每个张量对应的名称列表。
        output_dir (str): 保存 .npy 文件的输出目录。

        返回:
        None
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个Tensor为.npy文件
        for tensor, name in zip(tensor_list, tensor_names):
            np_array = tensor.numpy()  # 转换为NumPy数组
            file_path = os.path.join(output_dir, f'{name}.npy')
            np.save(file_path, np_array)

        print(f'Tensors have been saved to {output_dir}')

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # print('***************------------------------{}-------------------------***************'.format(
        #     input_dict.keys()
        # ))

        # if 'adjacent' in input_dict.keys():
        #     print('curr keys is: {}'.format(input_dict['curr'].keys()), input_dict['curr']['cams']['CAM_FRONT'].keys())
        #     print('ego2global: ', input_dict['curr']['ego2global_translation'], input_dict['curr']['ego2global_rotation'])
        #     print('adjacent keys is: {}'.format(input_dict['adjacent'][0].keys()), len(input_dict['adjacent']))

        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)

        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        # print('is normal?: ')
        # print('type of example is: {}'.format(type(example)), example.keys())
        # print('img_metas {}, img_inputs {}, gt_depth {}, voxel_semantics {}, mask_lidar {}, mask_camera {}'.format(
        #     example['img_metas'].data.keys(), [i.size() for i in example['img_inputs']], example['gt_depth'].size(), example['voxel_semantics'].size()
        #     , example['mask_lidar'].size(), example['mask_camera'].size()
        # ))
        # vis_list = [i for i in example['img_inputs']] + [example['gt_depth'], example['voxel_semantics'], example['mask_lidar'], example['mask_camera']]
        # self.save_tensors_to_npy(tensor_list=vis_list, tensor_names=vis_names, output_dir='outputs_debug/{}/'.format(index))
        # print(example['mask_lidar'].device)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """

        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        # print('here we debug the nuscenes det dataset, before adj_list is: {}, {}'.format(type(input_dict), input_dict.keys()))
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:     # 需要再读取历史帧的信息
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']

        # print('outside isL ', self.data_infos[index]['occ_path'])
        #
        # print('input dict occ gt path is: ', input_dict['occ_gt_path'])
        return input_dict

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
                if self.occ_label_to_revise is not None:
                    gt_semantics += 100
                    for key in self.occ_label_to_revise.keys():
                        gt_semantics[gt_semantics == key + 100] = self.occ_label_to_revise[key]
                    # print('sem first is: ', torch.unique(semantics, return_counts=True))
                else:
                    pass
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

        elif metric == 'map_ten_class':
            print('start ten class test: ++++=======++++++, num_occ_numL----{}'.format(self.occ_class_nums))
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=self.occ_class_nums,
                use_lidar_mask=False,
                use_image_mask=True,
                class_names=self.revised_occ_label
            )
            print('\nStarting Evaluation...')
            print('testing the length of pred is: {}'.format(len(occ_results)))
            for index, occ_pred in enumerate(tqdm(occ_results)):
                info = self.data_infos[index]
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)

                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)


                if self.occ_label_to_revise is not None:
                    gt_semantics += 100
                    for key in self.occ_label_to_revise.keys():
                        gt_semantics[gt_semantics == key + 100] = self.occ_label_to_revise[key]
                    # print('sem first is: ', torch.unique(semantics, return_counts=True))
                else:
                    pass
                # print('gt_semantics: ', np.unique(gt_semantics, return_counts=True))
                occ_pred = occ_pred
                self.occ_eval_metrics.add_batch(
                    occ_pred,   # (Dx, Dy, Dz)
                    gt_semantics,   # (Dx, Dy, Dz)
                    mask_lidar,     # (Dx, Dy, Dz)
                    mask_camera     # (Dx, Dy, Dz)
                )
                if show_dir is not None:
                    mmcv.mkdir_or_exist(show_dir)
                    # scene_name = info['scene_name']
                    scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                    sample_token = info['token']
                    mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                    np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)

            eval_results = self.occ_eval_metrics.count_miou()
        else:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=self.occ_class_nums,
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
                # if self.occ_label_to_revise is not None:
                #     for key in self.occ_label_to_revise.keys():
                #         gt_semantics[gt_semantics == key] = self.occ_label_to_revise[key]
                # else:
                #     pass
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


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis
