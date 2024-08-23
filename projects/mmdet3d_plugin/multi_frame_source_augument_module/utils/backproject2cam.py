import torch
import torch.linalg
import torch.nn as nn
import numpy as np
from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class BackprojectDepth2Cam(nn.Module):
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth2Cam, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
        # self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)

    def forward(self, depth, K, post_rot=None, post_trans=None):
        '''

        Args:
            depth:   (b*n_view, 1, h, w)
            K:     (b, n_view, 3, 3)
            post_rot:  (b, n_view, 3, 3)
            post_trans:   (b, n_view, 3)

        Returns:

        '''
        B, N, _, _ = K.size()
        if post_rot is not None and post_trans is not None:
            # print('-----------------------: ', post_rot.size(), post_trans.size(), self.pix_coords.size())
            # post_trans = post_trans.contiguous()
            # post_rot = post_rot.contiguous()

            b, n, s1, s2 = post_rot.size()
            post_rot = post_rot.view(b*n, s1, s2)
            # print('post_rot is: ', post_rot)
            # print('post_trans is: ', post_trans)
            post_rot_inv = torch.inverse(post_rot)
            b, n, s1 = post_trans.size()
            post_trans = post_trans.view(b*n, s1, 1)
            # print('post_tran: ', post_trans.size())


            # here we cna find that
            # here is we used to debug:
            # real_shape_croods = self.pix_coords.view(5, 3, 256, 704)
            # print('self.pix_coords is: ', self.pix_coords.size(), real_shape_croods[0, :, 0, 0],
            #       real_shape_croods[0, :, 0, 1])
            cam_points = self.pix_coords - post_trans

            # print('cam_points size is: ', cam_points.size())
            cam_points = torch.matmul(post_rot_inv, cam_points)
        else:
            cam_points = self.pix_coords

        # K = K.contiguous()
        b, n, s1, s2 =K.size()
        K = K.view(b*n, s1, s2)
        inv_K = torch.inverse(K)
        # cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # print('self.pix_coords: ', self.pix_coords.size(), 'inv_k size is: {}'.format(inv_K.size()))
        # cam_points = torch.matmul(inv_K, self.pix_coords)
        cam_points = torch.matmul(inv_K, cam_points)
        # print('cam points is: ', cam_points.size(), depth.view(self.batch_size, 1, -1).size())
        # print('cam_points after maual invK: ', cam_points.size())
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        # print('cam_poiunts in backproject is: {}'.format(cam_points.size()), cam_points.max(), cam_points.min())
        _, c, n_points = cam_points.size()
        cam_points = cam_points.view(B, N, c, n_points)
        return cam_points


@MODELS.register_module()
class Project2adjView(nn.Module):
    def __init__(self, height, width, eps=1e-7):
        super(Project2adjView, self).__init__()
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, adj_cam_points, adj_cam2img, adj_post_rot, adj_post_trans):
        '''

        :param adj_cam_points:  (b, n, 4, num_points)
        :param adj_cam2img:     (b, n, 3, 3)
        :param adj_post_rot:    (b, n, 3, 3)
        :param adj_post_trans:  (b, n, 3)
        :return:
        '''
        adj_cam_points = adj_cam_points[:, :, :3, :]
        img_points = torch.matmul(adj_cam2img, adj_cam_points)
        z2 = img_points[:, :, 2, :].unsqueeze(2)
        pix_coords = img_points[:, :, :2, :] / (img_points[:, :, 2, :].unsqueeze(2) + self.eps)
        pix_coords = torch.cat([pix_coords, z2], dim=2)

        if adj_post_rot is not None and adj_post_trans is not None:
            pix_coords = torch.matmul(adj_post_rot, pix_coords) + adj_post_trans.unsqueeze(-1)
            z2 = pix_coords[:, :, 2, :].unsqueeze(2)
            pix_coords = pix_coords[:, :, :2, :]
        else:
            pix_coords = pix_coords[:, :, :2, :]
        B, N, coord, num_points = pix_coords.size()
        pix_coords = pix_coords.view(B, N, coord, self.height, self.width)
        pix_coords = pix_coords.permute(0, 1, 3, 4, 2)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2  # 这个自然只会aware相互关联的部分
        return pix_coords


@MODELS.register_module()
class proj_frustrum_ego(nn.Module):
    def __init__(self, batch_size, num_cams, input_size, grid_config, downsample=1, eps=1e-7):
        super(proj_frustrum_ego, self).__init__()
        self.batch_size = batch_size
        self.height, self.width = input_size
        self.grid_config = grid_config
        self.depth_cfg = grid_config['depth']
        self.downsample = downsample
        self.eps = eps
        self.num_cams = num_cams
        self.frustum = nn.Parameter(self.creat_frustrum().unsqueeze(0).unsqueeze(0).
                                    repeat(batch_size, num_cams, 1, 1, 1, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(batch_size, num_cams, self.height, self.width, self.D, 1))

        self.x_range = (-40, 40)
        self.y_range = (-40, 40)
        self.z_range = (-1, 5.4)
        self.grid_size = (0.4, 0.4, 0.4)
        # self.occ_generate = nn.Parameter(torch.full((batch_size, int((grid_config['x'][1] - grid_config['x'][0]) // grid_config['x'][2])
        #                                              , int((grid_config['y'][1] - grid_config['y'][0]) // grid_config['y'][2]),
        #                                              int((grid_config['z'][1] - grid_config['z'][0]) // grid_config['z'][2])), 9).long())

        # self.ones = torch.ones(self.batch_size, 1, self.height * self.width))

    def creat_frustrum(self):
        H_in, W_in = self.height, self.width
        H_feat, W_feat = H_in // self.downsample, W_in // self.downsample
        d = torch.arange(*self.depth_cfg, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat, dtype=torch.float) \
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat, dtype=torch.float) \
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        frustum = torch.stack((x, y, d), -1)
        return frustum.permute(1, 2, 0, 3)

    def create_mask(self, depth):
        depth_range = torch.arange(*self.depth_cfg, dtype=torch.float).to(depth.device)
        # print('depth_range: ', depth_range)
        depth_expanded = depth.unsqueeze(-1)
        mask = depth_expanded <= depth_range
        return mask

    def project_points_to_occ_space(self, occ_space, points, sem, mask):
        """
        将点云数据投影到给定的 occ_space 中，并根据 mask 和 sem 矩阵进行填充。

        Args:
        occ_space (torch.Tensor): 形状为 (b, 200, 200, 16) 的全9 tensor。
        points (torch.Tensor): 形状为 (b, n, 3, n_points) 的点云数据。
        sem (torch.Tensor): 形状为 (b, n, 1, n_points) 的语义类别矩阵。
        mask (torch.Tensor): 形状为 (b, n, 1, n_points) 的布尔掩码矩阵。

        Returns:
        torch.Tensor: 更新后的 occ_space。
        """
        # 定义空间范围和网格大小

        # 计算网格的大小
        x_bins = int((self.x_range[1] - self.x_range[0]) / self.grid_size[0])
        y_bins = int((self.y_range[1] - self.y_range[0]) / self.grid_size[1])
        z_bins = int((self.z_range[1] - self.z_range[0]) / self.grid_size[2])
        print('x_bins {}, y_bins {}, z_bins {}'.format(x_bins, y_bins, z_bins))
        b, n, _, n_points = points.shape

        # 归一化点云数据到 [0, 1] 范围内
        norm_points = torch.stack([
            (points[..., 0, :] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
            (points[..., 1, :] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]),
            (points[..., 2, :] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        ], dim=-2).to(occ_space.device)

        # 将归一化的点云数据转换为网格坐标
        voxel_coords = (norm_points / torch.tensor(self.grid_size).view(1, 1, 3, 1).to(occ_space.device)).long()

        # 保持在有效范围内
        valid_mask = (
                (voxel_coords[..., 0, :] >= 0) & (voxel_coords[..., 0, :] < x_bins) &
                (voxel_coords[..., 1, :] >= 0) & (voxel_coords[..., 1, :] < y_bins) &
                (voxel_coords[..., 2, :] >= 0) & (voxel_coords[..., 2, :] < z_bins)
        )

        # 应用有效掩码和传入的 mask
        combined_mask = valid_mask & mask.squeeze(-2)
        print('combined_mask is: {}, valid_mask {}'.format(combined_mask.size(), valid_mask.size()), voxel_coords.size())
        # 将网格坐标转换为线性索引
        voxel_coords = voxel_coords.permute(0, 1, 3, 2)[combined_mask]
        sem_values = sem.squeeze(-2).permute(0, 1, 2)[combined_mask]

        # 展平坐标，合并批次和相机维度
        batch_idx, camera_idx, point_idx = torch.where(combined_mask)
        flat_batch_camera_idx = batch_idx * n + camera_idx
        voxel_coords_flat = voxel_coords.view(-1, 3)
        sem_values_flat = sem_values.view(-1)

        # 用语义类别填充 occ_space
        occ_space_flat = occ_space.view(b * x_bins * y_bins * z_bins)
        linear_indices = (
                flat_batch_camera_idx * (x_bins * y_bins * z_bins) +
                voxel_coords_flat[:, 0] * (y_bins * z_bins) +
                voxel_coords_flat[:, 1] * z_bins +
                voxel_coords_flat[:, 2]
        )

        # 解决多个点投影到同一个网格的问题
        occ_space_flat.index_put_(
            (linear_indices,),
            sem_values_flat,
            accumulate=False  # 可根据需求设置为 True 或 False
        )

        # 重新整形为原始的形状
        occ_space = occ_space_flat.view(b, x_bins, y_bins, z_bins)

        return occ_space

    def fill_occ_with_semantics(self, occ, points, labels):
        """
        将语义标签填充到 occ 空间内。

        参数:
        - occ: 形状为 (b, X, Y, Z) 的张量，表示 occ 空间。
        - points: 形状为 (B, 3, n_points) 的张量，表示点的坐标。
        - labels: 形状为 (B, 1, n_points) 的张量，表示每个点的语义标签。

        返回:
        - 填充了语义标签的 occ 张量。
        """
        b, X, Y, Z = occ.shape
        B, _, n_points = points.shape

        # 将点的坐标转换为 occ 空间的索引
        grid_size = torch.tensor([0.4, 0.4, 0.4], device=points.device)
        min_bounds = torch.tensor([-40, -40, -1], device=points.device)
        max_bounds = torch.tensor([40, 40, 5.4], device=points.device)

        # 计算网格索引
        print('debug points is: ', torch.unique(points.long(), return_counts=True, dim=2))
        indices = ((points - min_bounds.unsqueeze(1)) / grid_size.unsqueeze(1)).long()
        print('indices: ', indices.size())
        # # 确保索引在 occ 空间的范围内
        # indices = torch.clamp(indices, min=0,
        #                       max=torch.tensor([X - 1, Y - 1, Z - 1], device=points.device).unsqueeze(1))

        # 将语义标签填充到 occ 空间中
        # 移除超出边界的点
        valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < X) & \
                     (indices[:, 1] >= 0) & (indices[:, 1] < Y) & \
                     (indices[:, 2] >= 0) & (indices[:, 2] < Z)

        print('valid_mask size is: ', valid_mask.size(), torch.unique(valid_mask, return_counts=True))
        for b in range(B):
            valid_indices = indices[b][:, valid_mask[b]]
            valid_labels = labels[b][0, valid_mask[b]].to(occ.dtype)

            x_indices = valid_indices[0]
            y_indices = valid_indices[1]
            z_indices = valid_indices[2]

            print('x_indices is: ', torch.unique(x_indices, return_counts=True))
            occ[b, x_indices, y_indices, z_indices] = valid_labels

        return occ

    def forward(self, depth_map, sem_map, post_rot, post_trans, cam2img, cam2egos):
        print('self.ones and self.frustrum: ', self.ones.size(), self.frustum.size())
        B, N, H, W, D, croods = self.frustum.size()
        print('frustum is:', self.frustum.size(), self.frustum.device, self.frustum[0, 0, 20, 30, 40, :])
        frustrum = self.frustum.view(B, N, -1, croods)
        ones = self.ones.view(B, N, -1, 1)
        xy_croods = frustrum[:, :, :, :2]
        print('xy croods is: {}, ones is: {}'.format(xy_croods.size(), ones.size()))
        depth_croods = frustrum[:, :, :, 2:].permute(0, 1, 3, 2)

        pix_croods = torch.cat([xy_croods, ones], 3).permute(0, 1, 3, 2)
        print('pix_croods is: ', pix_croods.size())
        if post_rot is not None and post_trans is not None:
            # print('-----------------------: ', post_rot.size(), post_trans.size(), self.pix_coords.size())
            # post_trans = post_trans.contiguous()
            # post_rot = post_rot.contiguous()

            # b, n, s1, s2 = post_rot.size()
            post_rot_inv = torch.inverse(post_rot)
            b, n, s1 = post_trans.size()
            post_trans = post_trans.view(b, n, s1, 1)
            # print('post_tran: ', post_trans.size())


            # here we cna find that
            # here is we used to debug:
            # real_shape_croods = self.pix_coords.view(5, 3, 256, 704)
            # print('self.pix_coords is: ', self.pix_coords.size(), real_shape_croods[0, :, 0, 0],
            #       real_shape_croods[0, :, 0, 1])
            cam_points = pix_croods - post_trans

            # print('cam_points size is: ', cam_points.size())
            cam_points = torch.matmul(post_rot_inv, cam_points)
        else:
            cam_points = self.pix_coords

        print('cam points size is: ', cam_points.size(), cam_points[0, 0, :, 1000])
        img2cam = torch.inverse(cam2img)

        cam_points = cam_points * depth_croods
        cam_points = torch.matmul(img2cam, cam_points)
        print('depth_croods is: {}, cam_points is: {}'.format(depth_croods.size(), cam_points.size()), torch.unique(depth_croods))
        print('cam2ego is: {}'.format(cam2egos.size()))
        cam_croods = cam_points

        print('before concat is: ', cam_croods.size(), 'one is: ', ones.size())
        cam_croods = torch.cat([cam_points, ones.permute(0, 1, 3, 2)], 2)
        print('after concat is: ', cam_croods.size(), cam_croods[0, 0, :, 0])

        ego_points = torch.matmul(cam2egos, cam_croods)
        print('ego_points is: {}'.format(ego_points.size()))

        depth_mask = self.create_mask(depth_map)
        print('depth_mask: ', depth_mask.size(), depth_mask.dtype, depth_map.size(),)
        print('ego points is: {}, depth_mask is: {}, sem_map is: {}'.format(
            ego_points.size(), depth_mask.size(), sem_map.size()
        )
        )
        B, N, _, n_points = ego_points.size()
        ego_points = ego_points[:, :, :3, :]


        occ_space = torch.full((self.batch_size, 200, 200, 16), 9).long().to(depth_map.device)
        depth_mask = depth_mask.view(B, N, -1).unsqueeze(2)
        sem = sem_map.unsqueeze(-1)
        B, N, H, W, _ = sem.size()
        sem = sem.expand(B, N, H, W, D)
        # print('sem: ', sem.size(), sem[0, 0, 0, 0, :])
        sem = sem.reshape(B, N, -1)
        sem = sem.unsqueeze(2)
        sem = sem.masked_fill_(~depth_mask, 9)

        sem = sem.permute(0, 2, 3, 1)
        ego_points = ego_points.permute(0, 2, 3, 1)
        B, c, n_points, N = ego_points.size()
        ego_points = ego_points.reshape(B, c, n_points*N)
        sem = sem.reshape(B, 1, n_points*N)

        print('occ_space size is: ', occ_space.size(), ego_points.size(), sem.size())

        occ_concat = self.fill_occ_with_semantics(occ_space, ego_points, sem)
        print('occ_concat size is: ', occ_concat.size(), torch.unique(occ_concat))
        cached_memory = torch.cuda.memory_reserved(sem.device)

        print(f"Cached Memory: {cached_memory / (1024 ** 3):.2f} GB")
        # render_occ = self.project_points_to_occ_space(occ_space, ego_points, sem, depth_mask)
        # print('render_occ size is: ', render_occ.size())
        #
        # print('self.occ_generate is: ', self.occ_generate.size())
              # depth_mask[0, 0, 200, 300, :], depth_map[0, 0, 200, 300])
        #todo: 20240801 到这里我们已经将其投影到了ego下，下一步就是进行depth mask的生成和语义的赋予
        return occ_concat