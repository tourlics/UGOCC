
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

waymo_to_revise = {
    0: 6,
    1: 1,
    2: 2,
    3: 7,
    4: 2,
    5: 7,
    6: 7,
    7: 7,
    8: 2,
    9: 2,
    10: 3,
    11: 4,
    12: 4,
    13: 0,
    14: 5,
    15: 8,
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
class WaymoPrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

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
            cam_names = self.data_config['cams']
        self.src_size = self.data_config['src_size']
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
            imgs:  (N_views, 3, H, W)        # N_views = 5 * (N_history + 1)    5 is waymo dataset vision sensors nums
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
        # print('cam_names is: ', cam_names)
        for cam_name in cam_names:
            cam_data = results['curr']['image'][cam_name]
            filename = cam_data['img_filename']
            img = Image.open(filename)
            # print(cam_name, 'img is: ', type(img), img.size)
            # lidar2img_mat = torch.Tensor(cam_data['lidar2img'])
            # cam2img_mat = torch.Tensor(cam_data['cam2img'])
            sensor2ego = torch.Tensor(cam_data['cam2ego'])
            ego2global = torch.Tensor(results['curr']['pose'])
            # print('mat debug is: ', type(lidar2img_mat), type(cam2img_mat), type(cam2ego_mat))
            # print('mat debug is: ', lidar2img_mat.size(), cam2img_mat.size(), cam2ego_mat.size())
            # print(lidar2img_mat, cam2ego_mat)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam2img'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.  如果我已经获得了img2
            # sensor2ego, ego2global = \
            #     self.get_sensor_transforms(results['curr'], cam_name)

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
            # print('-----------------')
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
        # print('results here keys is: {} and curr keys is: {}'.format(results.keys(), results['curr'].keys()))
        return results

@PIPELINES.register_module()
class WaymoLoadAnnotationsBEVDepth(object):
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
        # if gt_boxes.shape[0] > 0:
        #     gt_boxes[:, :3] = (
        #         rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
        #     gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
        #     gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
        #     # 翻转也会进一步改变方位角
        #     if flip_dx:
        #         gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        #     if flip_dy:
        #         gt_boxes[:, 6] = -gt_boxes[:, 6]
        #     gt_boxes[:, 7:] = (
        #         rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # print('debug is: {}.'.format(type( results['curr']['ann_infos'])))
        gt_boxes, gt_labels = results['curr']['ann_infos']['bboxes'], results['curr']['ann_infos']['labels']      # (N_gt, 9),  (N_gt, )
        # print('type of gt boxes {} and gt label {}'.format(gt_boxes.shape, gt_labels.shape))
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        #
        # if len(gt_boxes) == 0:
        #     gt_boxes = torch.zeros(0, 9)
        # results['gt_bboxes_3d'] = \
        #     LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
        #                          origin=(0.5, 0.5, 0.5))
        # results['gt_labels_3d'] = gt_labels
        #
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
class WaymoLoadOccGTFromFile(object):
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
            data_root,
            use_larger=True,
            crop_x=False,
            revised_label=False
    ):
        self.use_larger = use_larger
        self.data_root = data_root
        self.crop_x = crop_x
        self.revised_label = revised_label

    def __call__(self, results):
        # print('results in waymoliadung gt ovvlabel is: ', results.keys(), results['curr'].keys())
        pts_filename = results['curr']['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path = os.path.join(self.data_root, seq_name, '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        occ_labels = np.load(file_path)
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
            for key in waymo_to_revise.keys():
                semantics[semantics == key] = waymo_to_revise[key]
        else:
            pass
        results['voxel_semantics'] = semantics
        results['mask_infov'] = mask_infov
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        # print('in gt loading', dict(zip(*np.unique(semantics, return_counts=True))))

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

@PIPELINES.register_module()
class WaymoLoadPointsFromFile(object):
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

        pts_filename = results['curr']['pts_filename']

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
class WaymoPointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

        # 真的抽象这破数据集
        self.deal_cam2img = torch.tensor([
            [0., 0., -1., 0.],
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., -1.]], dtype=torch.float)


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

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]

        intrins = torch.matmul(intrins, self.deal_cam2img)
        sensor2egos = torch.matmul(sensor2egos, self.deal_cam2img)
        results['img_inputs'] = imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda

        depth_map_list = []
        # print('cam names is: ', results['cam_names'])
        # print('curr keys has: ', results['curr']['image'].keys())
        #
        # print('imgs {}, sensor2egos {}, ego2globals {}, intrins {}, post_rots {}, post_trans {}, bda {}'.format(
        #     imgs.shape, sensor2egos.shape, ego2globals.shape, intrins.shape, post_rots.shape, post_trans.shape, bda.shape
        # ))
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            # print(results['curr']['image'][cam_name].keys())
            P_trans = torch.Tensor(results['curr']['calib']['P{}'.format(cid)])
            R_trans = torch.Tensor(results['curr']['calib']['R0_rect'])
            T_V_trans = torch.Tensor(results['curr']['calib']['Tr_velo_to_cam'])
            # print('in depth generate ', type(P_trans), type(R_trans), type(T_V_trans), P_trans.size(),
            #       R_trans.size(), T_V_trans.size())
            lidar2img = torch.matmul(torch.matmul(P_trans, R_trans), T_V_trans)
            # print('new lidar2img is: ', lidar2img.size())

            # 这里可能要进行修改
            cam2img = intrins[cid]
            cam2ego = sensor2egos[cid]
            lidar2img = torch.matmul(cam2img, torch.inverse(cam2ego))



            # print('lidar2img is: {}'.format(lidar2img.size()))
            # lidar2lidarego = np.eye(4, dtype=np.float32)
            # lidar2lidarego[:3, :3] = Quaternion(
            #     results['curr']['lidar2ego_rotation']).rotation_matrix
            # lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            # lidar2lidarego = torch.from_numpy(lidar2lidarego)
            #
            # lidarego2global = np.eye(4, dtype=np.float32)
            # lidarego2global[:3, :3] = Quaternion(
            #     results['curr']['ego2global_rotation']).rotation_matrix
            # lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            # lidarego2global = torch.from_numpy(lidarego2global)
            #
            # cam2camego = np.eye(4, dtype=np.float32)
            # cam2camego[:3, :3] = Quaternion(
            #     results['curr']['cams'][cam_name]
            #     ['sensor2ego_rotation']).rotation_matrix
            # cam2camego[:3, 3] = results['curr']['cams'][cam_name][
            #     'sensor2ego_translation']
            # cam2camego = torch.from_numpy(cam2camego)
            #
            # camego2global = np.eye(4, dtype=np.float32)
            # camego2global[:3, :3] = Quaternion(
            #     results['curr']['cams'][cam_name]
            #     ['ego2global_rotation']).rotation_matrix
            # camego2global[:3, 3] = results['curr']['cams'][cam_name][
            #     'ego2global_translation']
            # camego2global = torch.from_numpy(camego2global)
            #
            # cam2img = np.eye(4, dtype=np.float32)
            # cam2img = torch.from_numpy(cam2img)
            # cam2img[:3, :3] = intrins[cid]
            #
            # # lidar --> lidar_ego --> global --> cam_ego --> cam
            # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            #     lidarego2global.matmul(lidar2lidarego))
            # lidar2img = cam2img.matmul(lidar2cam)
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
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results