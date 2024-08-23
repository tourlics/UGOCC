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
from scipy.ndimage import zoom
import torchvision.transforms as transforms

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


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img
from scipy.interpolate import griddata
import torch.nn.functional as F

@PIPELINES.register_module()
class WaymoNewPrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            bda_aug_conf=None,
            test_cam_num_ids=None,
            load_semantic=False,
            convert_label=None
    ):
        self.is_train = is_train
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.data_config = data_config
        self.bda_aug_conf = bda_aug_conf
        self.test_cam_num_ids = test_cam_num_ids
        self.load_semantic = load_semantic
        self.convert_label = convert_label

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        # print('========================', self.is_train, self.data_config['Ncams'], len(
        #         self.data_config['cams']))
        if self.is_train and self.data_config['Ncams'] <= len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
            # print('cam_names++++++ ', cam_names)
        else:
            if self.test_cam_num_ids is not None:
                cam_names = np.random.choice(
                    self.data_config['cams'],
                    self.test_cam_num_ids,
                    replace=False)
            else:
                cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):  # 原始的大小直接从图片中读取了，所以这里我们不需要考虑更复杂的变化
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
        fH, fW = self.data_config['input_size']  # 这是增强后的大小
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

        ego2global = torch.Tensor(info[cam_name]['ego2global'])
        ego2cam = torch.Tensor(info[cam_name]['ego2cam'])
        sensor2ego = torch.inverse(ego2cam)
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
        cam_names = self.choose_cams()   # 这里选出来的会随机排列原有的视角，从而适应选择
        results['cam_names'] = cam_names
        canvas = []
        if self.load_semantic:
            semantic_list = []
        for cam_name in cam_names:
            cam_data = results['curr']['images'][cam_name]
            filename = cam_data['img_path']   # img_filename
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam2img'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.  如果我已经获得了img2
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr']['images'], cam_name)

            # print('in the dara_preapre is: ', img.height, img.width, flip, scale)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            # print('img aug is: ',img_augs)
            resize, resize_dims, crop, flip, rotate = img_augs

            if self.load_semantic:
                sem_path = results['curr']['sem_images'][cam_name]['sem_path']
                semantic = np.squeeze(np.load(sem_path).astype(np.int16))
                semantic += 100
                for key in list(self.convert_label.keys()):
                    semantic[semantic == key + 100] = self.convert_label[key]
                # print('1sr waymo_semantic size is: ', semantic.shape, np.unique(semantic))
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

            # if self.sequential:
            #     assert 'adjacent' in results
            #     for adj_info in results['adjacent']:
            #         filename_adj = adj_info['cams'][cam_name]['data_path']
            #         img_adjacent = Image.open(filename_adj)
            #         # 对选择的邻近帧图像也进行增广, 增广参数与当前帧图像相同.
            #         img_adjacent = self.img_transform_core(
            #             img_adjacent,
            #             resize_dims=resize_dims,
            #             crop=crop,
            #             flip=flip,
            #             rotate=rotate)
            #         imgs.append(self.normalize_img(img_adjacent))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        # if self.sequential:
        #     for adj_info in results['adjacent']:
        #         # adjacent与current使用相同的图像增广, 相机内参也相同.
        #         post_trans.extend(post_trans[:len(cam_names)])
        #         post_rots.extend(post_rots[:len(cam_names)])
        #         intrins.extend(intrins[:len(cam_names)])
        #
        #         for cam_name in cam_names:
        #             # 获得adjacent帧对应的camera2ego变换 (4, 4)和ego2global变换 (4, 4).
        #             sensor2ego, ego2global = \
        #                 self.get_sensor_transforms(adj_info, cam_name)
        #             sensor2egos.append(sensor2ego)
        #             ego2globals.append(ego2global)

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

        bda = self.sample_bda_augmentation()
        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda

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
        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_bda, 0, 0], [0, scale_bda, 0],
                                  [0, 0, scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)  # 变换矩阵(3, 3)
        return rot_mat

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
        if self.load_semantic:
            results['img_inputs'] = self.get_inputs(results)
        else:
            results['img_inputs'] = self.get_inputs(results)
        results['curr']['lidar2ego'] = torch.Tensor(results['curr']['lidar2ego'])
        # print(' in waymo_img _loading keys is: ', results.keys())
        return results

@PIPELINES.register_module()
class WaymoNewPrepareImageInputs_AdjFrame(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            bda_aug_conf=None,
            test_cam_num_ids=None
    ):
        self.is_train = is_train
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.data_config = data_config
        self.bda_aug_conf = bda_aug_conf

        self.test_cam_num_ids = test_cam_num_ids

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
            # print('cam_names++++++ ', cam_names)
        else:
            if self.test_cam_num_ids is not None:
                cam_names = np.random.choice(
                    self.data_config['cams'],
                    self.test_cam_num_ids,
                    replace=False)
            else:
                cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):  # 原始的大小直接从图片中读取了，所以这里我们不需要考虑更复杂的变化
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
        fH, fW = self.data_config['input_size']  # 这是增强后的大小
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

        ego2global = torch.Tensor(info[cam_name]['ego2global'])
        ego2cam = torch.Tensor(info[cam_name]['ego2cam'])
        sensor2ego = torch.inverse(ego2cam)
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
        cam_names = results['cam_names']  # 这里选出来的会随机排列原有的视角，从而适应选择
        canvas = []

        load_frames = list(results['adjacent_frame'])
        # print('load frame is: ', load_frames)
        assert 'adjacent_frame' in results

        for adj_info_idx in load_frames:
            for cam_name in cam_names:
                cam_data = results['adjacent_frame'][adj_info_idx]['images'][cam_name]
                # print('cam_data keys is: {} and frame is: {}'.format(
                #     cam_data.keys(), adj_info_idx
                # ))
                filename = cam_data['img_path']   # img_filename
                img = Image.open(filename)

                # 初始化图像增广的旋转和平移矩阵
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                # 当前相机内参
                intrin = torch.Tensor(cam_data['cam2img'])

                # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.  如果我已经获得了img2
                sensor2ego, ego2global = \
                    self.get_sensor_transforms(results['curr']['images'], cam_name)

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
                intrins.append(intrin)      # 相机内参 (3, 3)
                sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
                ego2globals.append(ego2global)      # ego2global变换 (4, 4)
                post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
                post_trans.append(post_tran)        # 图像增广平移 (3, ）

        # canvas = torch.stack(canvas)
        imgs = torch.stack(imgs)  # (N_views, 3, H, W)
        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)
        # print('imgs {}, sensor2egos {}, ego2globals {}, intrins {}, post_rots {}, post_trans {}'.format(
        #     imgs.size(), sensor2egos.size(), ego2globals.size(), intrins.size(), post_rots.size(), post_trans.size()
        # ))
        bda = self.sample_bda_augmentation()
        # print('bda isd: ', bda.size())
        results['canvas_old'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6
        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda

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
        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_bda, 0, 0], [0, scale_bda, 0],
                                  [0, 0, scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)  # 变换矩阵(3, 3)
        return rot_mat

    def __call__(self, results):
        adj_frame_input = self.get_inputs(results)
        results['adj_img_inputs'] = self.get_inputs(results)
        # results['curr']['lidar2ego'] = torch.Tensor(results['curr']['lidar2ego'])
        # print(' in waymo_img _loading keys is: ', results.keys())
        return results

@PIPELINES.register_module()
class WaymoNewLoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            use_larger=True,
            crop_x=False,
            revised_label=False,
            revise_dict=None
    ):
        self.use_larger = use_larger
        self.crop_x = crop_x
        self.revised_label = revised_label
        self.revise_dict = revise_dict

    def __call__(self, results):
        # print('results in waymoliadung gt ovvlabel is: ', results.keys(), results['curr'].keys())
        occ_file_path = results['curr']['occ_path']
        # print('occ_file_path: ', occ_file_path)

        occ_labels = np.load(occ_file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov']
        mask_lidar = occ_labels['origin_voxel_state']
        mask_camera = occ_labels['final_voxel_state']
        if self.crop_x:
            w, h, d = semantics.shape
            semantics = semantics[w // 2:, :, :]
            mask_infov = mask_infov[w // 2:, :, :]
            mask_lidar = mask_lidar[w // 2:, :, :]
            mask_camera = mask_camera[w // 2:, :, :]


        semantics[semantics == 23] = 15
        if self.revised_label:
            semantics += 100
            # print('sem first is: ', torch.unique(semantics, return_counts=True))
            for key in self.revise_dict.keys():
                semantics[semantics == key+100] = self.revise_dict[key]
            # print('sem second is: ', torch.unique(semantics, return_counts=True))
        else:
            pass

        results['voxel_semantics'] = torch.from_numpy(semantics)
        results['mask_infov'] = torch.from_numpy(mask_infov)
        results['mask_lidar'] = torch.from_numpy(mask_lidar)
        results['mask_camera'] = torch.from_numpy(mask_camera)
        # print('in gt loading', dict(zip(*np.unique(semantics, return_counts=True))))

        # print('in waymo occ label loading propose: ', results.keys())
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@PIPELINES.register_module()
class WaymoNewLoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        # print('prs_filename in loading is: {}'.format(results.keys()))
        # print('pts in curr is: {}'.format(results['curr'].keys()))
        # print('pts in adjacent is: {}'.format(len(results['adjacent'])))
        # print(results['pts_filename'], results['curr']['lidar_path'],
        #       results['pts_filename'] == results['curr']['lidar_path'])
        #
        # print('debug the img inputs: ', len(results['img_inputs']), results['img_inputs'][1].size())
        # print('in lidar trans result is: {}'.format(results.keys()), results['curr'].keys(), results['curr']['point_cloud'].keys(),
        #       results['curr']['calib'].keys())
        # print('if we do not use the cam_param++++++++++++++++++++++++===================: ',
        #       results['curr']['calib']['P0'].shape, results['curr']['calib']['R0_rect'].shape
        #       , results['curr']['calib']['Tr_velo_to_cam'].shape)

        pts_filename = results['curr']['lidar_path']

        # print('in loading file of waymo dataset is: ', pts_filename)
        # print("pts_filename",pts_filename)

        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points


        # print('++++++++++++++++++++++ in loading points keys is: ', results.keys())
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class WaymoNewPointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

        # 真的抽象这破数据集
        # self.deal_cam2img = torch.tensor([
        #     [0., 0., -1., 0.],
        #     [-1., 0., 0., 0.],
        #     [0., -1., 0., 0.],
        #     [0., 0., 0., -1.]], dtype=torch.float)


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
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth

        # mask = depth_map > 0
        # # print('mask size is: ', type(mask), mask.size(), torch.sum(mask))
        # depth_map_filled = self.interpolate_depth_map(depth_map, mask)
        # # print('depth_map_filled is: {}, depth_map is: {}'.format(depth_map_filled.size(), depth.size()))
        # depth_map = depth_map_filled
        return depth_map

    def interpolate_depth_map(self, depth_map, mask):
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
        lidar2ego = results['curr']['lidar2ego']


        depth_map_list = []
        # print('cam names is: ', results['cam_names'])
        # print('curr keys has: ', results['curr']['image'].keys())
        #
        # print('imgs {}, sensor2egos {}, ego2globals {}, intrins {}, post_rots {}, post_trans {}, bda {}'.format(
        #     imgs.shape, sensor2egos.shape, ego2globals.shape, intrins.shape, post_rots.shape, post_trans.shape, bda.shape
        # ))
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = results['img_inputs']
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE


            curr_cam2ego = sensor2egos[cid, :, :]
            curr_cam2img = intrins[cid, :, :]
            curr_post_rots = post_rots[cid, :, :]
            curr_post_trans = post_trans[cid, :]
            # print('curr_cam2ego {}, curr_cam2img {}, curr_post_rots {}, curr_post_trans{}'.format(
            #     curr_cam2ego.size(), curr_cam2img.size(),curr_post_rots.size() ,curr_post_trans.size()
            # ))
            lidar2cam = torch.matmul(torch.inverse(curr_cam2ego), lidar2ego)
            cam2img_b = np.eye(4, dtype=np.float32)
            cam2img_b = torch.from_numpy(cam2img_b)
            cam2img_b[:3, :3] = curr_cam2img
            lidar2img = torch.matmul(cam2img_b, lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            # print('point_img size is: ', points_img.size())

            points_img = points_img.matmul(
                curr_post_rots.T) + curr_post_trans
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        # print('=============++++++++++++++ ', results.keys())
        #     depth_map = self.points2depthmap(points_img,
        #                                      imgs.shape[2],     # H
        #                                      imgs.shape[3]      # W
        #                                      )
        #     depth_map_list.append(depth_map)
        # depth_map = torch.stack(depth_map_list)
        # results['gt_depth'] = depth_map
        # print('after generate depth map keys is: ', results.keys())
        return results

@PIPELINES.register_module()
class WaymoGenerateViewSegmentationLabel(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.save_path = '/home/ps/lrh_code_root/OCC_DOMAIN_REF_OLD/save_point_voxel_label'
        self.colormap = np.array([[255,0,255],
                                  [100,150,245],
                                  [255,30, 30],
                                  [255,200,0],
                                  [0,175,0],
                                  [75,0,75],
                                  [112,128,144],
                                  [47,79,79],
                                  [30,60,150],
                                  [0,175,0]])

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
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def point_sem(self, img_points, sem_points):

        return 0


    def point2img(self, points, point_voxel_semantic, height, width):
        """
               Args:
                   points: (N_points, 3):  3: (u, v, d)
                   height: int
                   width: int

               Returns:
                   depth_map：(H, W)
               """
        height, width = height // self.downsample, width // self.downsample
        semantic_map_2d = torch.full((height, width), 255, dtype=torch.uint8)
        semantic_map_rgb = torch.zeros((height,width, 3), dtype=torch.uint8)
        coor = torch.round(points[:, :2] / self.downsample)  # (N_points, 2)  2: (u, v)
        # print('in the loading coor : {}'.format(coor.size()), coor.max(), coor.min())
        depth = points[:, 2]  # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < self.grid_config['depth'][1]) & (
                        depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth, semantic = coor[kept1], depth[kept1], point_voxel_semantic[kept1]  # (N, 2), (N, )
        # print('in the loading after choose: ', coor.size(), depth.size())
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth, semantic = coor[kept2], depth[kept2], semantic[kept2]
        coor = coor.to(torch.long)
        colors = self.colormap[semantic]
        semantic_map_rgb[coor[:, 1], coor[:, 0]] = torch.from_numpy(self.colormap[semantic]).to(torch.uint8)
        semantic_map_2d[coor[:, 1], coor[:, 0]] = torch.from_numpy(semantic).to(torch.uint8)

        return semantic_map_rgb, semantic_map_2d

    def compute_voxel_indices(self, points):
        x_min, y_min, z_min = self.grid_config['x'][0], self.grid_config['y'][0], self.grid_config['z'][0]
        delta_x, delta_y, delta_z = 0.4, 0.4, 0.4
        voxel_idx = np.floor((points - np.array([x_min, y_min, z_min])) / np.array([delta_x, delta_y, delta_z])).astype(int)

        return voxel_idx

    def __call__(self, results):
        print('WaymoGenerateViewSegmentationLabel: ', results.keys())

        lidar2ego = results['curr']['lidar2ego']
        points_lidar = results['points'].tensor[:, :3]
        # lidar 2 ego
        print(f'lidar2ego shape: {lidar2ego.shape}')
        points_lidar = points_lidar.matmul(
            lidar2ego[:3, :3].T) + lidar2ego[:3, 3].unsqueeze(0)
        points_lidar = points_lidar.numpy()
        x_min, x_max = self.grid_config['x'][0], self.grid_config['x'][1]
        y_min, y_max = self.grid_config['y'][0], self.grid_config['y'][1]
        z_min, z_max = self.grid_config['z'][0], self.grid_config['z'][1]
        valid_points = points_lidar[(points_lidar[:, 0] >= x_min) & (points_lidar[:, 0] < x_max) &
                                    (points_lidar[:, 1] >= y_min) & (points_lidar[:, 1] < y_max) &
                                    (points_lidar[:, 2] >= z_min) & (points_lidar[:, 2] < z_max)]

        point_voxel_idx = self.compute_voxel_indices(valid_points)

        point_voxel_semantic = results['voxel_semantics'][point_voxel_idx[:,0], point_voxel_idx[:, 1], point_voxel_idx[:,2]]

        point_w_semantic = np.column_stack((valid_points, point_voxel_semantic))
        # ego 2 lidar


        results['semantic_points'] = point_w_semantic
        print('point_w_semantic is: ', point_w_semantic.shape)
        # transform the valid_points into image
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = results['img_inputs']
        # print(f'imgs shape: {imgs.shape}')
        #
        # point_w_semantic = torch.from_numpy(np.swapaxes(point_w_semantic, 0, 1))
        # print('ego semantic poitnshape is: ', point_w_semantic.shape)
        # print('sensor2egos size is: ', sensor2egos.size())
        # print('intrins: ', intrins.size())
        # print('post_rots, post_trans: ', post_rots.size(), post_trans.size())
        #
        mul_semantic_map = []
        # for idx, name in enumerate(results['cam_names']):
        #     curr_cam2ego = sensor2egos[idx, :, :]
        #     curr_cam2img = intrins[idx, :, :]
        #     curr_post_rot = post_rots[idx, :, :]
        #     curr_post_trans = post_trans[idx, :]
        #     xyz_points = point_w_semantic[:3, :]
        #     sem_points = point_w_semantic[3:4, :]
        #     one_concat = torch.ones(1, xyz_points.size()[1],  dtype=torch.float)
        #     img_points = torch.matmul(curr_cam2img, torch.matmul(torch.inverse(curr_cam2ego), torch.cat((xyz_points, one_concat), dim=0))[:3, :])
        #
        #     points_img = torch.cat(
        #         [img_points[:2, :] / img_points[2:3, :], img_points[2:3, :]],
        #         1)
        #     # print('point_img size is: ', points_img.size())
        #
        #     points_img = torch.matmul(curr_post_rot, points_img) + curr_post_trans
        #     print('posfdadbfhnuwef is: ', points_img.size(), sem_points.size())


        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]  # CAM_TYPE

            curr_cam2ego = sensor2egos[cid, :, :]
            curr_cam2img = intrins[cid, :, :]
            curr_post_rots = post_rots[cid, :, :]
            curr_post_trans = post_trans[cid, :]
            # print('curr_cam2ego {}, curr_cam2img {}, curr_post_rots {}, curr_post_trans{}'.format(
            #     curr_cam2ego.size(), curr_cam2img.size(),curr_post_rots.size() ,curr_post_trans.size()
            # ))
            #lidar2cam = torch.matmul(torch.inverse(curr_cam2ego), lidar2ego)

            cam2img_b = np.eye(4, dtype=np.float32)
            cam2img_b = torch.from_numpy(cam2img_b)
            cam2img_b[:3, :3] = curr_cam2img
            lidar2img = torch.matmul(cam2img_b, torch.inverse(curr_cam2ego))
            valid_points_ = torch.tensor(valid_points)
            points_img = valid_points_.matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            # print('point_img size is: ', points_img.size())

            points_img = points_img.matmul(
                curr_post_rots.T) + curr_post_trans
            print('sfdwsfsdfwsefwerf: ', points_img.shape, point_voxel_semantic.shape)
            semantic_map_rgb, semantic_map_2d = self.point2img(points_img, point_voxel_semantic, imgs.shape[2], imgs.shape[3])
            mul_semantic_map.append(semantic_map_2d)
            # semantic_map_pil = Image.fromarray(semantic_map.numpy())
            # semantic_map_pil.save(f'{self.save_path}/{cam_name}.png')
            # img_tensor = imgs[cid].permute(1,2,0).to(torch.uint8)
            # img = Image.fromarray(img_tensor.numpy())
            # img.save(f'{self.save_path}/{cam_name}_gt.png')
        results['mul_semantic_map'] = torch.stack(mul_semantic_map, dim=0)
        print('results keys is: ', results['curr']['lidar_path'])

        print('================ ', results['points'][:, :3].shape, results['voxel_semantics'].shape, results['points'][:, :3][0], results['voxel_semantics'][0, 0, 0] )
        return results

@PIPELINES.register_module()
class WaymoNewLoadSemanticLabels(object):
    def __init__(self, input_size, downsample=1, convert_label=None,
                 is_train=False, data_config=None, fill_value=255):
        self.input_size = input_size
        self.convert_label = convert_label
        self.is_train = is_train
        self.data_config = data_config
        self.transform = transforms.ToTensor()
        self.fill_value = fill_value

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
            resize += np.random.uniform(*self.data_config['resize'])  # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))  # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH  # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))  # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # print('cop is: ', crop)
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
        img = img.resize(resize_dims, Image.NEAREST)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, fillcolor=self.fill_value)
        return img

    def apply_affine_transform_and_crop(self, sem_map, post_rot, post_trans):
        """
        对语义图 sem_map 应用仿射变换并裁剪到目标形状。

        参数:
        sem_map (numpy.ndarray): 形状为 (H, W, 1) 的语义图，值范围为 (0, 1, 2, ..., 9)。
        post_rot (numpy.ndarray): 形状为 (3, 3) 的旋转矩阵。
        post_trans (numpy.ndarray): 形状为 (3,) 的平移向量。
        target_shape (tuple): 目标输出形状 (height, width)。

        返回:
        transformed_sem_map (numpy.ndarray): 应用了仿射变换并裁剪后的语义图。
        """
        # 将语义图转换为 PIL 图像
        sem_map_pil = Image.fromarray(sem_map.squeeze(), mode='L')

        # 提取仿射变换的参数
        a, b, c = post_rot[0, 0], post_rot[0, 1], post_trans[0]
        d, e, f = post_rot[1, 0], post_rot[1, 1], post_trans[1]

        # 注意：Image.transform 的仿射参数顺序为 (a, b, c, d, e, f)
        affine_params = (a, b, c, d, e, f)

        # 应用仿射变换
        transformed_sem_map_pil = sem_map_pil.transform(
            sem_map_pil.size, Image.AFFINE, affine_params, resample=Image.NEAREST
        )

        # 转换回 NumPy 数组
        transformed_sem_map = np.array(transformed_sem_map_pil)

        # 获取目标形状
        target_height, target_width = self.data_config['input_size']

        # 获取中间区域的坐标
        center_x, center_y = transformed_sem_map.shape[1] // 2, transformed_sem_map.shape[0] // 2
        left = max(center_x - target_width // 2, 0)
        right = min(center_x + target_width // 2, transformed_sem_map.shape[1])
        top = max(center_y - target_height // 2, 0)
        bottom = min(center_y + target_height // 2, transformed_sem_map.shape[0])

        # 裁剪到目标形状
        transformed_sem_map_cropped = transformed_sem_map[top:bottom, left:right]

        # 如果裁剪后的图像不符合目标形状，进行填充
        if transformed_sem_map_cropped.shape[0] < target_height or transformed_sem_map_cropped.shape[1] < target_width:
            padded_sem_map = np.zeros((target_height, target_width), dtype=np.uint8)
            padded_sem_map[:transformed_sem_map_cropped.shape[0],
            :transformed_sem_map_cropped.shape[1]] = transformed_sem_map_cropped
            transformed_sem_map_cropped = padded_sem_map

        # 添加最后的维度
        transformed_sem_map_cropped = transformed_sem_map_cropped[..., np.newaxis]

        return transformed_sem_map_cropped

    def __call__(self, results, flip=None, scale=None):
        # print('result cam_name is: ',  results['cam_names'], results.keys())
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]

        sem_image_list = []
        for idx, cid in enumerate(results['cam_names']):
            post_rot_c = post_rots[idx]
            post_trans_c = post_trans[idx]
            sem_path = results['curr']['sem_images'][cid]['sem_path']
            # print('waymo_sem path is: ', sem_path)
            # print('datasset: ', results['dataset_type'], cid, sem_path)

            semantic = np.squeeze(np.load(sem_path).astype(np.int16))
            semantic += 100
            for key in list(self.convert_label.keys()):
                semantic[semantic == key + 100] = self.convert_label[key]

            a, b = np.unique(semantic)
            print('semantic first is: ', semantic.shape, a, b)
            semantic = Image.fromarray(semantic, mode='L')

            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=semantic.height, W=semantic.width, flip=flip, scale=scale)
            semantic = self.img_transform_core(semantic, resize_dims, crop, flip, rotate)
            semantic = torch.tensor(np.array(semantic), dtype=torch.uint8).unsqueeze(0)


            a, b = torch.unique(semantic)
            print('semantic second is: ', semantic.size(), a, b)

            sem_image_list.append(semantic)
        #
        #
        results['semantic_map'] = torch.cat(sem_image_list)
        # print('waymo sem map is:', results['semantic_map'].size())
            # print('waymo keys is: ', semantic.shape, np.unique(semantic, return_counts=True))
        return results

@PIPELINES.register_module()
class NuscNewLoadSemanticLabels(object):
    def __init__(self, input_size, downsample=1, convert_label=None,
                 is_train=False, data_config=None, fill_value=255):
        self.input_size = input_size
        self.convert_label = convert_label
        self.is_train = is_train
        self.data_config = data_config
        self.transform = transforms.ToTensor()
        self.fill_value = fill_value

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
            resize += np.random.uniform(*self.data_config['resize'])  # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))  # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH  # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))  # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # print('cop is: ', crop)
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
        img = img.resize(resize_dims, Image.BILINEAR)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, fillcolor=self.fill_value)
        return img


    def __call__(self, results, flip=None, scale=None):
        # print('result cam_name is: ', results['cam_names'], results.keys())
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        sem_image_list = []
        for idx, cid in enumerate(results['cam_names']):
            post_rot_c = post_rots[idx]
            post_trans_c = post_trans[idx]
            sem_path = results['curr']['sem_images'][cid]['sem_path']
            semantic = np.fromfile(sem_path,
                                   dtype=np.int8).reshape(900, 1600).astype(np.int16)
            # semantic = torch.from_numpy(semantic)


            semantic += 100
            for key in list(self.convert_label.keys()):
                semantic[semantic == key + 100] = self.convert_label[key]
            print('semantic first is: ', semantic.shape, np.unique(semantic))
            semantic = Image.fromarray(semantic, mode='L')
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=semantic.height, W=semantic.width, flip=flip, scale=scale)
            semantic = self.img_transform_core(semantic, resize_dims, crop, flip, rotate)
            semantic = torch.tensor(np.array(semantic), dtype=torch.uint8).unsqueeze(0)

            print('semantic is: ', semantic.size(), torch.unique(semantic))
            sem_image_list.append(semantic)
        results['semantic_map'] = torch.cat(sem_image_list)
        #
        # results['semantic_map'] = torch.stack(sem_image_list)
        return results