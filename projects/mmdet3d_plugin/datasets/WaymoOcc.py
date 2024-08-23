import copy
import tempfile
from os import path as osp
import os
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from torch.utils.data import Dataset
import warnings
from mmdet3d.core import get_box_type
import re
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.depth_metric import Metric_Depth
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_number(file_name):
    # 使用正则表达式匹配前面的数字部分
    match = re.match(r"(\d+)", file_name)
    if match:
        # 将匹配到的数字部分转为int类型
        number = int(match.group(1))
        return number
    else:
        raise ValueError(f"No leading number found in the file name: {file_name}")

@DATASETS.register_module()
class WaymoOccMultiFrame(Dataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 load_interval=5,
                 num_views=5,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 data_prefix: dict = dict(
                     pts='training/velodyne',
                     CAM_FRONT='training/image_0',
                     CAM_FRONT_LEFT='training/image_1',
                     CAM_FRONT_RIGHT='training/image_2',
                     CAM_SIDE_LEFT='training/image_3',
                     CAM_SIDE_RIGHT='training/image_4',
                     OCC='waymo/voxel04/'
                 ),
                 multi_adj_frame_id_cfg=None,
                 stereo=False,
                 img_info_prototype='bevdet',
                 file_client_args=dict(backend='disk'),
                 revised_label=None,
                 revise_dict=None,
                 revised_occ_label=None,
                 occ_label_to_revise=None,
                 occ_class_nums=None,
                 load_multi_frame=False,
                 **kwargs):
        super().__init__()
        self.num_views = num_views
        assert self.num_views <= 5
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.load_interval = load_interval
        # load annotations

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers

        self.data_prefix = data_prefix

        self.stereo = stereo
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.img_info_prototype = img_info_prototype
        self.load_annotations(self.ann_file)
        if not self.test_mode:
            self._set_group_flag()
        print('waymo build over!!!!!!!!!!!!!!!!!!!!!!!')

        # 子类的额外参数
        self.revised_label = revised_label
        self.occ_class_nums = occ_class_nums
        self.occ_label_to_revise = occ_label_to_revise
        self.revised_occ_label = revised_occ_label

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        data_pkl = mmcv.load(ann_file, file_format='pkl')
        self.data_infos_full = data_pkl['data_list']
        data_infos = data_pkl['data_list'][::self.load_interval]
        print('data infos origin length: ', len(data_infos))
        self.meta_info = data_pkl['metainfo']
        self.data_infos = data_infos
        # self.data_infos = self.init_prepare_data_info(data_infos)
        print('data infos after length: ', len(self.data_infos))

    def init_prepare_data_info(self, data_infos):
        print('++++++++++++++++++++++++++++++++++++start refine data_infos+++++++++++++++++++++++++++++++++')
        data_list = []
        for idx, info in enumerate(data_infos):
            camera_info = dict()
            camera_info['sample_idx'] = info['sample_idx']
            camera_info['timestamp'] = info['timestamp']
            camera_info['context_name'] = info['context_name']
            curr_sample_idx = info['sample_idx']
            curr_scene_idx = curr_sample_idx % 1000000 // 1000
            curr_frame_idx = curr_sample_idx % 1000000 % 1000

            camera_info['curr_scene_idx'] = curr_scene_idx
            camera_info['curr_frame_idx'] = curr_frame_idx

            # camera_info['']
            lidar_prefix = self.data_prefix.get('pts', '')
            camera_info['lidar_path'] = osp.join(
                self.data_root, lidar_prefix, info['lidar_points']['lidar_path'])
            camera_info['lidar2ego'] = np.diag([1, 1, 1, 1])

            if 'train' in self.ann_file:
                occ_root_path = os.path.join(self.data_prefix['OCC'], 'training',
                                             "{:03}".format(curr_scene_idx),
                                             '{}_04.npz'.format("{:03}".format(curr_frame_idx)))
            else:
                occ_root_path = os.path.join(self.data_prefix['OCC'], 'validation',
                                             "{:03}".format(curr_scene_idx),
                                             '{}_04.npz'.format("{:03}".format(curr_frame_idx)))
            camera_info['occ_path'] = occ_root_path
            camera_info['images'] = dict()

            if self.modality['use_camera']:
                for (cam_key, img_info) in info['images'].items():
                    camera_info['images'][cam_key] = img_info

                    if 'img_path' in img_info:
                        cam_prefix = self.data_prefix.get(cam_key, '')
                        camera_info['images'][cam_key]['img_path'] = osp.join(
                            self.data_root, cam_prefix, img_info['img_path'])
                    if 'lidar2cam' in img_info:
                        camera_info['images'][cam_key]['lidar2cam'] = np.array(img_info['lidar2cam'])
                    if 'cam2img' in img_info:
                        camera_info['images'][cam_key]['cam2img'] = np.array(img_info['cam2img'])[:3, :3]
                    if 'lidar2img' in img_info:
                        camera_info['images'][cam_key]['lidar2img'] = np.array(img_info['lidar2img'])
                    else:
                        camera_info['images'][cam_key]['lidar2img'] = camera_info[
                                                                          'cam2img'] @ camera_info['lidar2cam']
                    if 'ego2global' in info:
                        camera_info['images'][cam_key]['ego2global'] = np.array(info['ego2global'])

                    camera_info['images'][cam_key]['ego2cam'] = np.array(img_info['lidar2cam'])

                    # camera_info['instances'] = info['cam_instances'][cam_key]
                    # camera_info['ann_info'] = self.parse_ann_info(camera_info)
            else:
                pass
            # print('camera_info keys inner is: {}, lidar_path is: {}, occ_path {}, sample_index is: {}, img_path_front is: {}'.format(camera_info.keys(),
            #     camera_info['lidar_path'], camera_info['occ_path'], camera_info['sample_idx'], camera_info['images']['CAM_FRONT']['img_path']))
            data_list.append(camera_info)
        return data_list



            # print('self.ann_file: ', self.ann_file)


    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
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

    def get_data_info(self, index):
        # print('starting getting datas: {}', len(self))
        info = self.data_infos[index]

        # print('image path is: ', info['images']['CAM_FRONT']['img_path'], info.keys(), type(info['sample_idx']), info['sample_idx'])

        input_dict = dict(
            curr_scene_idx=info['curr_scene_idx'],
            curr_frame_idx=info['curr_frame_idx']
        )
        # print('info curr is: ', info.keys())
        # print('curring 00sdwa: ', info['image'].keys(), info['image']['CAM_FRONT'].keys())
        if self.img_info_prototype == 'mmcv':
            pass
        else:
            input_dict.update(dict(curr=info))
            if '4d' in self.img_info_prototype:  # 需要再读取历史帧的信息
                info_adj_list = self.get_adj_info(info, index)
                input_dict.update(dict(adjacent=info_adj_list))

        # print('info is: {}'.format(info.keys()))
        # # print('final step++++++++++++', input_dict['curr'].keys(), input_dict['adjacent'][0].keys())
        # # print('input_dict is: ', input_dict.keys(), len(input_dict['adjacent']), input_dict['curr'].keys())
        # print('input_dict is: ', input_dict.keys(), input_dict['curr'].keys(), input_dict['curr']['lidar2ego'],  input_dict['curr']['lidar_path'])
        # print('scene idx is: {}, and frame idx is: {}'.format(input_dict['curr_scene_idx'],
        #                                                       input_dict['curr_frame_idx']),
        #       input_dict['curr']['images']['CAM_FRONT']['img_path'], input_dict['curr']['occ_path'])
        # # # 在获取到idx一户，我们需要进行图像的读取，点云的读取，图像的增强，深度图的生成，占据地图的读取
        # print('imgase is: ', input_dict['curr']['images'].keys(), 'self.ann_file is: ', self.ann_file)
        #
        # # if 'val' in self.ann_file:
        # #     print('self.ann_file is: ', 'val')
        # # elif 'train' in self.ann_file:
        # #     print('self.ann_file is: ', 'train')
        #
        # # print('imgase is: ', input_dict['curr']['images']['CAM_FRONT'].keys())
        # print(input_dict['curr']['images']['CAM_FRONT']['img_path'], 'cam2img is: ', input_dict['curr']['images']['CAM_FRONT']['cam2img'])

        return input_dict

    def refine_useful_info(self, info, index):
        '''
        我们在这里应当规整有关相机的变换矩阵全部到对应的名称下，并进行filename补全， lidar path补全，以及occ path补全（在这里进行而不是loading）
        与nuscenes不同的是在这里我们就要把它们直接转化为numpy 矩阵的形式
        Args:
            info:
            index:

        Returns:

        '''
        camera_info = dict()
        camera_info['images'] = dict()
        info_2 = copy.copy(info)
        if self.modality['use_camera']:
            for (cam_key, img_info) in info_2['images'].items():

                camera_info['sample_idx'] = info_2['sample_idx']
                camera_info['context_name'] = info_2['context_name']
                camera_info['images'][cam_key] = img_info
                if 'img_path' in img_info:
                    cam_prefix = self.data_prefix.get(cam_key, '')
                    camera_info['images'][cam_key]['img_path'] = osp.join(
                        self.data_root, cam_prefix, img_info['img_path'])
                if 'lidar2cam' in img_info:
                    camera_info['images'][cam_key]['lidar2cam'] = np.array(img_info['lidar2cam'])
                if 'cam2img' in img_info:
                    camera_info['images'][cam_key]['cam2img'] = np.array(img_info['cam2img'])[:3, :3]
                if 'lidar2img' in img_info:
                    camera_info['images'][cam_key]['lidar2img'] = np.array(img_info['lidar2img'])
                else:
                    camera_info['images'][cam_key]['lidar2img'] = camera_info[
                        'cam2img'] @ camera_info['lidar2cam']
                if 'ego2global' in info_2:
                    camera_info['images'][cam_key]['ego2global'] = np.array(info_2['ego2global'])

                camera_info['images'][cam_key]['ego2cam'] = np.array(img_info['lidar2cam'])

                # camera_info['instances'] = info['cam_instances'][cam_key]
                # camera_info['ann_info'] = self.parse_ann_info(camera_info)
        lidar_prefix = self.data_prefix.get('pts', '')
        camera_info['lidar_path'] = osp.join(
                    self.data_root, lidar_prefix, info_2['lidar_points']['lidar_path'])
        camera_info['lidar2ego'] = np.diag([1, 1, 1, 1])

        # print('self.ann_file: ', self.ann_file)
        if 'train' in self.ann_file:
            occ_root_path = os.path.join(self.data_prefix['OCC'], 'training', "{:03}".format(info_2['curr_scene_idx']),
                                         '{}_04.npz'.format("{:03}".format(info_2['curr_frame_idx'])))
        else:
            occ_root_path = os.path.join(self.data_prefix['OCC'], 'validation', "{:03}".format(info_2['curr_scene_idx']),
                                         '{}_04.npz'.format("{:03}".format(info_2['curr_frame_idx'])))
        camera_info['occ_path'] = occ_root_path
        # Occ_prefix = self.data_prefix.get('OCC', '')
        # camera_info['occ_path'] = osp.join(
        #             self.data_root, Occ_prefix, info['lidar_points']['lidar_path'])


        # print('the file should be used is: ', len(image_paths), len(lidar2img_rts), len(lidar2img_rts), len(intrinsics_rts), len(sensor2ego_rts))
        return camera_info

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

            curr_sample_idx = extract_number(self.data_infos[select_id]['images']['CAM_FRONT']['img_path'])
            curr_scene_idx = curr_sample_idx % 1000000 // 1000
            curr_frame_idx = curr_sample_idx % 1000000 % 1000

            if not curr_scene_idx == \
                   info['curr_scene_idx']:
                info_adj_list.append(info)
            else:
                app_infos = self.data_infos[select_id]
                curr_sample_idx = extract_number(self.data_infos[select_id]['images']['CAM_FRONT']['img_path'])
                curr_scene_idx = curr_sample_idx % 1000000 // 1000
                curr_frame_idx = curr_sample_idx % 1000000 % 1000
                app_infos.update(dict(curr_scene_idx=curr_scene_idx, curr_frame_idx=curr_frame_idx, curr_sample_idx=curr_sample_idx))
                info = self.refine_useful_info(app_infos, select_id)
                info_adj_list.append(info)

                # print('yes yes yes')

        return info_adj_list

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
                occ_gt = np.load(info['occ_path'])
                gt_semantics = occ_gt['voxel_label']      # (Dx, Dy, Dz)
                # print('occ_gt ', occ_gt.keys())
                mask_lidar = occ_gt['origin_voxel_state'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['final_voxel_state'].astype(bool)    # (Dx, Dy, Dz)

                # mask_camera = occ_gt['infov'].astype(bool)

                gt_semantics[gt_semantics == 23] = 15
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

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
