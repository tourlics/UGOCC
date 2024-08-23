# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate

aug_colors = np.array(
    [
        [255, 0, 255],
        [100, 150, 245],
        [255, 30, 30],
        [255, 200, 0],
        [0, 175, 0],
        [75, 0, 75],
        [112, 128, 144],
        [47, 79, 79],
        [30, 60, 150],
        [230, 230, 250]
    ]
).astype(np.uint8)

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


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            test_cam_num_ids=None
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.test_cam_num_ids = test_cam_num_ids


    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            if self.test_cam_num_ids is not None:
                cam_names = np.random.choice(
                    self.data_config['cams'],
                    self.test_cam_num_ids,
                    replace=False)
            else:
                cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            #print('cop is: ', crop)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        # print('post_rot, post_tran, resize, resize_dims, crop, flip, rotate',
        #       post_rot, post_tran, resize, resize_dims,
        #       crop, flip, rotate
        #       )
        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        # print('results keys is: ', results['curr']['ego2global_rotation'], results['curr']['cams']['CAM_FRONT']['ego2global_rotation'],
        #       results['curr']['cams']['CAM_BACK']['ego2global_rotation'], results['curr']['cams']['CAM_FRONT_LEFT']['ego2global_rotation'])

        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.  如果我已经获得了img2
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # print('in the dara_preapre is: ', img.height, img.width, flip, scale)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            # print('img aug is: ',img_augs)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        if self.sequential:
            for adj_info in results['adjacent']:
                # adjacent与current使用相同的图像增广, 相机内参也相同.
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

@PIPELINES.register_module()
class NuscRandomInPrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            test_cam_num_ids=None,
            load_semantic=False,
            convert_label=None
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.test_cam_num_ids = test_cam_num_ids
        self.load_semantic = load_semantic
        self.convert_label = convert_label

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] <= len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
            # print('loading cam_name is: ', cam_names)
        else:
            if self.test_cam_num_ids is not None:
                cam_names = np.random.choice(
                    self.data_config['cams'],
                    self.test_cam_num_ids,
                    replace=False)
            else:
                cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            #print('cop is: ', crop)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def inverse_transform(self, post_rot, post_tran, original_dims):
        """
        Args:
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
            original_dims: Tuple(W, H), 原始图像的尺寸

        Returns:
            resize: float
            resize_dims: Tuple(W, H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        """
        # 逆向计算旋转角度
        theta = np.arctan2(post_rot[1, 0], post_rot[0, 0]) * 180 / np.pi

        # 逆向计算是否翻转
        flip = torch.det(post_rot) < 0

        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            post_rot = A.matmul(post_rot)

        # 逆向计算调整大小比例
        resize = post_rot[0, 0]

        # 计算调整后的图像尺寸
        resize_dims = (int(original_dims[0] * resize), int(original_dims[1] * resize))

        # 逆向计算裁剪参数
        crop_w = -post_tran[0]
        crop_h = -post_tran[1]
        crop = (crop_w, crop_h, crop_w + resize_dims[0], crop_h + resize_dims[1])

        return resize, resize_dims, crop, flip, theta

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image

        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        # print('in loading is ', img.height, img.width)
        # print('post_rot, post_tran, resize, resize_dims, crop, flip, rotate',
        #       post_rot, post_tran, resize, resize_dims,
        #       crop, flip, rotate
        #       )
        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        # print('++++++++++++++ in loading is: ', img.height, img.width, resize_dims, crop, flip, rotate,
        #       type(resize_dims), type(crop), type(flip), type(rotate)
        #       )
        # print(self.inverse_transform(post_rot, post_tran, (img.width, img.height)))
        # print()
        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        # print('results keys is: ', results['curr']['ego2global_rotation'], results['curr']['cams']['CAM_FRONT']['ego2global_rotation'],
        #       results['curr']['cams']['CAM_BACK']['ego2global_rotation'], results['curr']['cams']['CAM_FRONT_LEFT']['ego2global_rotation'])

        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        if self.load_semantic:
            semantic_list = []

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.  如果我已经获得了img2
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            # print('in the dara_preapre is: ', img.height, img.width, flip, scale)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            # print('in loading is is is: ', flip, scale)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            # print('img aug is: ',img_augs)
            resize, resize_dims, crop, flip, rotate = img_augs

            if self.load_semantic:
                sem_path = results['curr']['sem_images'][cam_name]['sem_path']
                semantic = np.fromfile(sem_path,
                                       dtype=np.int8).reshape(900, 1600).astype(np.int16)
                semantic += 100
                for key in list(self.convert_label.keys()):
                    semantic[semantic == key + 100] = self.convert_label[key]
                semantic = torch.from_numpy(self.sem_transform_core(semantic, (resize, resize_dims, crop, flip, rotate)))
                # print('waymo_semantic size is: ', semantic.shape, torch.unique(semantic))
                semantic_list.append(semantic)

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        if self.sequential:
            for adj_info in results['adjacent']:
                # adjacent与current使用相同的图像增广, 相机内参也相同.
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                for cam_name in cam_names:
                    # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        if self.load_semantic:
            semantic_map = torch.stack(semantic_list)
            results['semantic_map'] = semantic_map
        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans
    def sem_transform_core(self, sem_img, aug_factors):
        '''
        :param sem_img: (h,w),语义label
        :param resize_h,resize_w: augement size
        :return: aug_label: (h,w)
        '''

        resize, resize_dims, crop, flip, rotate = aug_factors

        rgb_image = aug_colors[sem_img]
        image = Image.fromarray(rgb_image, 'RGB')
        # img_save_name = f'{save_npy_path}/{cam_name}/{sem_name}.png'
        # image.save(img_save_name)
        # print(f'image: {image.size}')
        aug_image = self.img_transform_core(image, resize_dims, crop, flip, rotate)
        # img_save_aug_name = f'{save_npy_path}/{cam_name}/{sem_name}_aug.png'
        # aug_image.save(img_save_aug_name)
        np_aug_image = np.array(aug_image)

        # 使用颜色查表
        # 创建一个查找表，假设最大颜色值不超过255
        lookup_table = np.full((256, 256, 256), 255, dtype=np.uint8)  # 默认类别为255

        # 填充已知颜色的类别
        lookup_table[0, 0, 0] = 255
        lookup_table[255, 255, 255] = 255
        lookup_table[255, 0, 255] = 0
        lookup_table[100, 150, 245] = 1
        lookup_table[255, 30, 30] = 2
        lookup_table[255, 200, 0] = 3
        lookup_table[0, 175, 0] = 4
        lookup_table[75, 0, 75] = 5
        lookup_table[112, 128, 144] = 6
        lookup_table[47, 79, 79] = 7
        lookup_table[30, 60, 150] = 8
        lookup_table[230, 230, 250] = 9

        aug_label = lookup_table[np_aug_image[:, :, 0], np_aug_image[:, :, 1], np_aug_image[:, :, 2]]

        return aug_label
    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results

@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
            gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
            gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
            # 翻转也会进一步改变方位角
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']      # (N_gt, 9),  (N_gt, )
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        # if 'voxel_semantics' in results:
        #     if flip_dx:
        #         results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
        #     if flip_dy:
        #         results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
        #         results['mask_lidar'] = results['mask_lidar'][:, ::-1, ...].copy()
        #         results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()

        return results

@PIPELINES.register_module()
class PointToMultiViewDepthPoints(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        # print('in the loading coor : {}'.format(coor.size()), coor.max(), coor.min())
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )

        depth_map = torch.cat((coor, depth.unsqueeze(-1)), dim=1)


        # print('in the loading after choose: ', coor.size(), depth.size())
        # ranks = coor[:, 0] + coor[:, 1] * width
        # sort = (ranks + depth / 100.).argsort()
        # coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        # kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        # kept2[1:] = (ranks[1:] != ranks[:-1])
        # coor, depth = coor[kept2], depth[kept2]
        # coor = coor.to(torch.long)
        # depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)      # (N_points, 3):  3: (u, v, d)

            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            # print('in loading points_img is: {}'.format(type(points_img)), points_img.size(), points_img.max(), points_img.min())
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results

@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        # print('in the loading coor : {}'.format(coor.size()), coor.max(), coor.min())
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        # print('in the loading after choose: ', coor.size(), depth.size())
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        # print('in loading kept2 is: ', kept2.size())
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth

        # 待删除
        # mask = depth_map > 0
        # # print('mask size is: ', type(mask), mask.size(), torch.sum(mask))
        # depth_map_filled = self.interpolate_depth_map(depth_map, mask)
        # # print('depth_map_filled is: {}, depth_map is: {}'.format(depth_map_filled.size(), depth.size()))
        # depth_map = depth_map_filled
        return depth_map

    def interpolate_depth_map(self, depth_map, mask):
        from scipy.interpolate import griddata
        # 转换为 numpy 数组以使用 scipy 的 griddata 插值
        depth_map_np = depth_map.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # 获取非零深度值及其对应的坐标
        y, x = np.nonzero(mask_np)
        z = depth_map_np[y, x]

        # 创建完整的网格
        grid_x, grid_y = np.meshgrid(np.arange(depth_map_np.shape[1]), np.arange(depth_map_np.shape[0]))

        # 使用 griddata 进行插值，填充原始深度图中的零值部分
        depth_map_filled_np = griddata((x, y), z, (grid_x, grid_y), method='linear', fill_value=0)

        # 将插值结果转换回 torch 张量
        depth_map_filled = torch.tensor(depth_map_filled_np, dtype=torch.float32).to(depth_map.device)

        return depth_map_filled

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        lidarego2globals = []

        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            lidarego2globals.append(torch.matmul(lidarego2global, lidar2lidarego))

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)      # (N_points, 3):  3: (u, v, d)

            # 再考虑图像增广
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
            #print('in loading points_img is: {}'.format(type(points_img)), points_img.size(), points_img.max(), points_img.min())
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        lidarego2globals_ = torch.stack(lidarego2globals)
        results['lidarego2global'] = lidarego2globals_
        # print('lidarego2globals is is is is is sis', [i.size() for i in lidarego2globals])
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(self, revised_label=False, revise_dict=nuscenes_to_revise):
        self.revised_label = revised_label
        self.revise_dict = revise_dict

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']

        # print('occ_gt_path  isisisisisisisisi: ', occ_gt_path )
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        semantics = torch.from_numpy(semantics)
        mask_lidar = torch.from_numpy(mask_lidar)
        mask_camera = torch.from_numpy(mask_camera)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        # print('self.revise_dict: ', self.revise_dict)
        #
        # print('keys is ', self.revise_dict.keys())
        # print('self.revise_dictis :', self.revise_dict)
        if self.revised_label:
            semantics += 100
            # print('sem first is: ', torch.unique(semantics, return_counts=True))
            for key in self.revise_dict.keys():
                semantics[semantics == key+100] = self.revise_dict[key]
            # print('sem second is: ', torch.unique(semantics, return_counts=True))
        else:
            pass
        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

