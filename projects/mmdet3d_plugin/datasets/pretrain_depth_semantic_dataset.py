# todo: this dataset is designed to pretrain the depth and 2d semantic, depth is self-supervised,
# and 2d semantic is semi-supervised
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class PretrainWaymoNuscenesDataset(Dataset):
    def __init__(self,
                 ana_file,
                 waymo_full_ana_files,
                 load_interval=1,
                 test_mode=False,
                 nusc_pipeline=None,
                 waymo_pipeline=None,
                 classes=None,
                 img_info_prototype='bevdet',
                 **kwargs
                 ):
        super().__init__()
        self.ana_file = ana_file
        self.load_interval = load_interval
        self.test_mode = test_mode
        self.waymo_full_ana_files = waymo_full_ana_files
        self.img_info_prototype = img_info_prototype
        self.filter_empty_gt = True
        if nusc_pipeline is not None:
            self.nusc_pipeline = Compose(nusc_pipeline)
        if waymo_pipeline is not None:
            self.waymo_pipeline = Compose(waymo_pipeline)
        self.load_annotations(ana_file, waymo_full_ana_files)
        self.CLASSES = self.get_classes(classes)
        if not self.test_mode:
            self._set_group_flag()

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

    def __len__(self):
        return len(self.data_infos)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file, waymo_full_ana_files):
        """Load annotations from ann_file.

        Args:
            waymo_full_ana_files:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        data_pkl = mmcv.load(ann_file, file_format='pkl')

        if self.waymo_full_ana_files is not None:
            self.waymo_align_files = mmcv.load(waymo_full_ana_files, file_format='pkl')
        else:
            self.waymo_align_files = None

        self.data_infos_full = data_pkl['data_list']
        data_infos = data_pkl['data_list'][::self.load_interval]
        print('data infos origin length: ', len(data_infos))
        self.meta_info = data_pkl['metainfo']
        self.data_infos = data_infos
        # self.data_infos = self.init_prepare_data_info(data_infos)
        print('data infos after length: ', len(self.data_infos))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def waymo_get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            curr_scene_idx=info['curr_scene_idx'],
            curr_frame_idx=info['curr_frame_idx']
        )
        if self.img_info_prototype == 'mmcv':
            pass
        else:
            input_dict.update(dict(curr=info))
            if '4d' in self.img_info_prototype:  # 需要再读取历史帧的信息
                info_adj_list = self.get_adj_info(info, index)
                input_dict.update(dict(adjacent=info_adj_list))
        return input_dict

    def nusc_get_data_info(self, index):
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
        assert 'bevdet' in self.img_info_prototype
        input_dict.update(dict(curr=info))
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """

        info = self.data_infos[index]
        dataset_type = info['dataset_type']
        # print('dataset_type is: ', dataset_type)
        if dataset_type == 'waymo':
            input_dict = self.waymo_get_data_info(index)
            input_dict['dataset_type'] = 'waymo'
            example = self.waymo_pipeline(input_dict)

        else:
            input_dict = self.nusc_get_data_info(index)
            input_dict['dataset_type'] = 'nuscenes'
            example = self.nusc_pipeline(input_dict)
            # example['dataset_type'] = 'nuscenes'
        # print('example is: ', example.keys())
        # print('here after get data info input dict is: {}'.format(input_dict.keys()))
        # print('cams info is: ', input_dict['curr'].keys(), input_dict['curr']['image'].keys())
        # print(input_dict['curr']['image']['image_path'], input_dict['curr']['image']['image_shape'])
        # print(input_dict['curr']['calib'].keys(), input_dict['curr']['pose'].shape)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        info = self.data_infos[index]
        dataset_type = info['dataset_type']
        if dataset_type == 'waymo':
            input_dict = self.waymo_get_data_info(index)
            example = self.waymo_pipeline(input_dict)
        else:
            input_dict = self.nusc_get_data_info(index)
            example = self.nusc_pipeline(input_dict)
        return example
