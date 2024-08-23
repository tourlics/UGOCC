# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.datasets import PIPELINES
from pyquaternion import Quaternion
import torch.nn.functional as F
from scipy.interpolate import griddata

@PIPELINES.register_module()
class LoadPointsFromFileMultiFrame(object):
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

    def load_single_frame_points(self, pts_filename):
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
        num_adjcanet_frame = len(results['adjacent'])
        # print('prs_filename in loading is: {}'.format(results.keys()))
        # print('pts in curr is: {}'.format(results['curr'].keys()))
        # print('pts in adjacent is: {}'.format(len(results['adjacent'])))
        # print(results['pts_filename'], results['curr']['lidar_path'],
        #       results['pts_filename'] == results['curr']['lidar_path'])
        #
        # print('debug the img inputs: ', len(results['img_inputs']), results['img_inputs'][1].size())
        pts_filename = results['curr']['lidar_path']
        # print("pts_filename",pts_filename)
        points = self.load_single_frame_points(pts_filename)
        results['points'] = points
        # load points in adjacent frame
        for i in range(num_adjcanet_frame):
            pts_filename = results['adjacent'][i]['lidar_path']
            points = self.load_single_frame_points(pts_filename)
            results['points_frame_t-{}'.format(str(i))] = points
        # points = self._load_points(pts_filename)
        # points = points.reshape(-1, self.load_dim)
        # points = points[:, self.use_dim]
        # attribute_dims = None
        #
        # if self.shift_height:
        #     floor_height = np.percentile(points[:, 2], 0.99)
        #     height = points[:, 2] - floor_height
        #     points = np.concatenate(
        #         [points[:, :3],
        #          np.expand_dims(height, 1), points[:, 3:]], 1)
        #     attribute_dims = dict(height=3)
        #
        # if self.use_color:
        #     assert len(self.use_dim) >= 6
        #     if attribute_dims is None:
        #         attribute_dims = dict()
        #     attribute_dims.update(
        #         dict(color=[
        #             points.shape[1] - 3,
        #             points.shape[1] - 2,
        #             points.shape[1] - 1,
        #         ]))
        #
        # points_class = get_points_type(self.coord_type)
        # points = points_class(
        #     points, points_dim=points.shape[-1], attribute_dims=attribute_dims)


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
class PointToMultiViewDepthMultiFrame(object):
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
        # print('in the loading process coor is: {}, depth is: {}'.format(coor.size(), depth.size()))
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

    def loading_and_convert_single_frame_depth(self, results, frame_id=0):

        if frame_id == 0:
            points_lidar = results['points']
        else:
            points_lidar = results['points_frame_t-{}'.format(str(frame_id))]
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []

        num_cams = len(results['cam_names'])
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            if frame_id == 0:
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
            else:
                lidar2lidarego = np.eye(4, dtype=np.float32)
                lidar2lidarego[:3, :3] = Quaternion(
                    results['adjacent'][frame_id]['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['adjacent'][frame_id]['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)

                lidarego2global = np.eye(4, dtype=np.float32)
                lidarego2global[:3, :3] = Quaternion(
                    results['adjacent'][frame_id]['ego2global_rotation']).rotation_matrix
                lidarego2global[:3, 3] = results['adjacent'][frame_id]['ego2global_translation']
                lidarego2global = torch.from_numpy(lidarego2global)

                cam2camego = np.eye(4, dtype=np.float32)
                cam2camego[:3, :3] = Quaternion(
                    results['adjacent'][frame_id]['cams'][cam_name]
                    ['sensor2ego_rotation']).rotation_matrix
                cam2camego[:3, 3] = results['adjacent'][frame_id]['cams'][cam_name][
                    'sensor2ego_translation']
                cam2camego = torch.from_numpy(cam2camego)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    results['adjacent'][frame_id]['cams'][cam_name]
                    ['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = results['adjacent'][frame_id]['cams'][cam_name][
                    'ego2global_translation']
                camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid + num_cams*frame_id]

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
                post_rots[cid + num_cams*frame_id].T) + post_trans[cid + num_cams*frame_id:cid + num_cams*frame_id + 1, :]      # (N_points, 3):  3: (u, v, d)
            #print('in loading points_img is: {}'.format(type(points_img)), points_img.size(), points_img.max(), points_img.min())
            depth_map = self.points2depthmap(points_img,
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        return depth_map

    def __call__(self, results):
        depth_map = self.loading_and_convert_single_frame_depth(results, frame_id=0)
        results['gt_depth'] = depth_map
        num_frame = len(results['adjacent'])
        for i in range(num_frame):
            depth_map = self.loading_and_convert_single_frame_depth(results, frame_id=i)
            results['gt_depth_frame_t-{}'.format(str(i+1))] = depth_map
        # points_lidar = results['points']
        # imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        # post_rots, post_trans, bda = results['img_inputs'][4:]
        # depth_map_list = []
        #
        # num_cams = len(results['cam_names'])
        # for cid in range(len(results['cam_names'])):
        #     cam_name = results['cam_names'][cid]    # CAM_TYPE
        #     # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
        #     # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
        #     # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
        #     lidar2lidarego = np.eye(4, dtype=np.float32)
        #     lidar2lidarego[:3, :3] = Quaternion(
        #         results['curr']['lidar2ego_rotation']).rotation_matrix
        #     lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
        #     lidar2lidarego = torch.from_numpy(lidar2lidarego)
        #
        #     lidarego2global = np.eye(4, dtype=np.float32)
        #     lidarego2global[:3, :3] = Quaternion(
        #         results['curr']['ego2global_rotation']).rotation_matrix
        #     lidarego2global[:3, 3] = results['curr']['ego2global_translation']
        #     lidarego2global = torch.from_numpy(lidarego2global)
        #
        #     cam2camego = np.eye(4, dtype=np.float32)
        #     cam2camego[:3, :3] = Quaternion(
        #         results['curr']['cams'][cam_name]
        #         ['sensor2ego_rotation']).rotation_matrix
        #     cam2camego[:3, 3] = results['curr']['cams'][cam_name][
        #         'sensor2ego_translation']
        #     cam2camego = torch.from_numpy(cam2camego)
        #
        #     camego2global = np.eye(4, dtype=np.float32)
        #     camego2global[:3, :3] = Quaternion(
        #         results['curr']['cams'][cam_name]
        #         ['ego2global_rotation']).rotation_matrix
        #     camego2global[:3, 3] = results['curr']['cams'][cam_name][
        #         'ego2global_translation']
        #     camego2global = torch.from_numpy(camego2global)
        #
        #     cam2img = np.eye(4, dtype=np.float32)
        #     cam2img = torch.from_numpy(cam2img)
        #     cam2img[:3, :3] = intrins[cid]
        #
        #     # lidar --> lidar_ego --> global --> cam_ego --> cam
        #     lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
        #         lidarego2global.matmul(lidar2lidarego))
        #     lidar2img = cam2img.matmul(lidar2cam)
        #     points_img = points_lidar.tensor[:, :3].matmul(
        #         lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)
        #     points_img = torch.cat(
        #         [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        #         1)      # (N_points, 3):  3: (u, v, d)
        #
        #     # 再考虑图像增广
        #     points_img = points_img.matmul(
        #         post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 3):  3: (u, v, d)
        #     #print('in loading points_img is: {}'.format(type(points_img)), points_img.size(), points_img.max(), points_img.min())
        #     depth_map = self.points2depthmap(points_img,
        #                                      imgs.shape[2],     # H
        #                                      imgs.shape[3]      # W
        #                                      )
        #     depth_map_list.append(depth_map)
        # depth_map = torch.stack(depth_map_list)

        return results