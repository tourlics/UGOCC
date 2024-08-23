import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import MODELS
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
from mmcv.runner import force_fp32, auto_fp16
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt

@MODELS.register_module()
class Pesudo_RenderNet(nn.Module):
    """
        Camera parameters aware depth net
    """

    def __init__(self,
                 num_cams=1,
                 batch_size=1,
                 downsample=1,
                 grid_config=None,
                 input_size=None,
                 grid_size=[0.4, 0.4, 0.4],
                 num_class=1,
                 SemiSupervisor=None,
                 **kwargs
                 ):
        super(Pesudo_RenderNet, self).__init__()
        self.SemiSupervisor = MODELS.build(SemiSupervisor)

        self.batch_size = batch_size
        self.num_class = num_class

        self.img_H = input_size[0] // downsample
        self.img_W = input_size[1] // downsample

        self.x_bound = [grid_config['x'][0]//grid_size[0], (grid_config['x'][1]+0.1)//grid_size[0]]
        self.y_bound = [grid_config['y'][0]//grid_size[1], (grid_config['y'][1]+0.1)//grid_size[1]]
        self.z_bound = [grid_config['z'][0]//grid_size[2], (grid_config['z'][1]+0.1)//grid_size[2]]
        self.depth_bound = grid_config['depth']

        h = torch.arange(0, self.img_H, 1).float() + 0.5
        w = torch.arange(0, self.img_W, 1).float() + 0.5
        d = torch.arange(self.depth_bound[0], self.depth_bound[1], self.depth_bound[2]).float()

        H, W, D = torch.meshgrid(w, h, d, indexing='xy')
        coords = torch.stack((H, W, D), dim=-1)
        coords = torch.unsqueeze(coords, 0).repeat(num_cams, 1, 1, 1, 1)
        coords = torch.unsqueeze(coords, 0).repeat(batch_size, 1, 1, 1, 1, 1)
        self.coords = nn.Parameter(coords, requires_grad=False)
        self.ones = nn.Parameter(torch.ones_like(coords)[:, :, :, :, :, :1], requires_grad=False)

        self.remuse_downsample = nn.Parameter(torch.tensor([downsample, downsample, 1.0]).view(1, 1, 3, 1), requires_grad=False)
        self.occ_grid_ratio = nn.Parameter(torch.tensor([1/grid_size[0], 1/grid_size[1], 1/grid_size[2]]).view(1, 1, 3, 1), requires_grad=False)
        self.xyz_min = nn.Parameter(torch.tensor([self.x_bound[0], self.y_bound[0], self.z_bound[0]]).view(1, 1, 3, 1), requires_grad=False)
        self.xyz_max = nn.Parameter(torch.tensor([self.x_bound[1], self.y_bound[1], self.z_bound[1]]).view(1, 1, 3, 1), requires_grad=False)
        self.sem_loss = nn.CrossEntropyLoss( ignore_index=9)


    def frustrum2ego(self, cam2ego_rot, cam2ego_trans, cam2img, post_rots, post_trans, xy_croods, depth_croods, ones):
        pix_croods = torch.cat([xy_croods, ones], 3).permute(0, 1, 3, 2)
        if post_rots is not None and post_trans is not None:
            post_rot_inv = torch.inverse(post_rots)
            b, n, s1 = post_trans.size()
            post_trans = post_trans.view(b, n, s1, 1)
            cam_points = pix_croods * self.remuse_downsample - post_trans
            cam_points = torch.matmul(post_rot_inv, cam_points)
        else:
            cam_points = self.pix_coords
        # print('cam_points is: ', cam_points.size(), cam_points[0, 0, :, 0], cam_points[0, 0, :, 100])
        # print(cam_points.view(B, N, croods, H, W, D).size(), cam_points.view(B, N, croods, H, W, D)[0, 0, :, 0, 10, 10],
        #       cam_points.view(B, N, croods, H, W, D)[0, 0, :, 15, 43, 70])
        # print([cam_points.view(B, N, croods, H, W, D)[0, i, :, 15, 43, 70] for i in range(5)])

        img2cam = torch.inverse(cam2img)
        cam_points = cam_points * depth_croods
        cam_points = torch.matmul(img2cam, cam_points)
        cam_croods = cam_points
        b, n, s1 = cam2ego_trans.size()
        ego_croods = torch.matmul(cam2ego_rot, cam_croods) + cam2ego_trans.view(b, n, s1, 1)
        return ego_croods

    def norm_grid_sample(self, ego_croods_occ_space):
        norm_occ_croods = (ego_croods_occ_space - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 -1

        mask_x = torch.ge(norm_occ_croods[:, :, 0, :], -1) & torch.le(norm_occ_croods[:, :, 0, :], 1)
        mask_y = torch.ge(norm_occ_croods[:, :, 1, :], -1) & torch.le(norm_occ_croods[:, :, 1, :], 1)
        mask_z = torch.ge(norm_occ_croods[:, :, 2, :], -1) & torch.le(norm_occ_croods[:, :, 2, :], 1)

        # 结合三个维度的掩码，确保所有坐标都在 [-1, 1]
        occ_mask = mask_x & mask_y & mask_z
        return norm_occ_croods, occ_mask.unsqueeze(2)

    def extract_first_non_ignore(self, sem_depth_map, ignore_value=9):
        """
        从给定的四维张量中提取每个 [H, W] 位置上第一个不是忽略值的 D 维索引处的值。

        参数:
            tensor (torch.Tensor): 输入的四维张量，形状应为 [1, H, W, D]
            ignore_value (int): 需要忽略的值，默认为9

        返回:
            torch.Tensor: 一个二维张量，形状为 [H, W]
        """
        # 确保输入是四维
        if sem_depth_map.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional")
        # sem_depth_map[..., 0] = 9
        # 创建mask，其中不等于忽略值的位置为True
        mask = sem_depth_map != ignore_value
        mask = mask.int()
        # 找到每个 [H, W] 位置上第一个 True 的 D 维度索引
        first_true_indices = torch.argmax(mask, dim=3)
        # print(sem_depth_map.size(), 'sem_depth_size')
        # print('first_true_indices size is: ', first_true_indices.size(), torch.unique(first_true_indices, return_counts=True))
        # 使用这些索引从原始数据中提取值
        batch_indices = torch.arange(sem_depth_map.size(0), device=sem_depth_map.device).view(-1, 1, 1, 1)
        h_indices = torch.arange(sem_depth_map.size(1), device=sem_depth_map.device).view(1, -1, 1, 1)
        w_indices = torch.arange(sem_depth_map.size(2), device=sem_depth_map.device).view(1, 1, -1, 1)
        d_indices = first_true_indices.unsqueeze(-1)  # 添加batch维度
        # print('d_indices size is: ', d_indices.size(), 'sem_depth_map is: ', sem_depth_map.size())
        # 使用gather提取值
        result = torch.gather(sem_depth_map, 3, d_indices).squeeze(3)  # 移除最后一个维度
        # print('result is: ', result.size())
        depth_map = self.depth_bound[0] + self.depth_bound[2] * d_indices.squeeze(3)
        return result, depth_map

    def map_coords_to_semantics(self, semantic_map, grid):
        """
        根据给定的坐标网格为每个点映射并提取三维语义图的语义值。

        参数:
            semantic_map (torch.Tensor): 三维语义图，形状为 (B, X, Y, Z)
            grid (torch.Tensor): 坐标网格，形状为 (B, H, W, D, 3)，其中3是xyz坐标
            x_min, y_min, z_min (float): 坐标范围的最小值
            cell_size_x, cell_size_y, cell_size_z (float): 单元格的大小

        返回:
            torch.Tensor: 映射到输入坐标的语义值，形状为 (B, H, W, D)
        """
        # 计算索引

        # print('inner debug: ',  semantic_map.size(), grid.size())
        B, X, Y, Z = semantic_map.shape
        _, H, W, D, _ = grid.shape

        # 计算索引
        indices_x = grid[..., 0].floor().to(torch.long)
        indices_y = grid[..., 1].floor().to(torch.long)
        indices_z = grid[..., 2].floor().to(torch.long)

        # 保证索引在合法范围内
        indices_x = torch.clamp(indices_x, 0, X - 1)
        indices_y = torch.clamp(indices_y, 0, Y - 1)
        indices_z = torch.clamp(indices_z, 0, Z - 1)

        # 使用advanced indexing 直接索引多维度
        batch_indices = torch.arange(B, device=semantic_map.device).view(B, 1, 1, 1).expand(-1, H, W, D)
        result = semantic_map[batch_indices, indices_x, indices_y, indices_z]

        return result

    def render_view(self, occ_map, occ_space, mask_space):
        occ_map = occ_map.long()
        occ_map = torch.nn.functional.one_hot(occ_map, num_classes=self.num_class).permute(0, 4, 1, 2, 3).float()
        # print('occ_map is: ', occ_map.size(), occ_map.dtype)
        # print('occ_space is: ', occ_space.size())
        render_sem = F.grid_sample(occ_map, occ_space[:, 3, :, :, :, :], mode='bilinear', padding_mode='zeros',
                                   align_corners=True)
        # print('first render_sem: ', render_sem.size())
        render_sem_view = render_sem.argmax(dim=1)
        print('render_sem is: ', render_sem_view.size(), torch.unique(render_sem_view, return_counts=True))
        sem_view = self.extract_first_non_ignore(render_sem_view)
        print('sem_view: ', sem_view.size(), torch.unique(sem_view, return_counts=True))
        return sem_view

    def masked_mse_loss(self, depth_pred, depth_render, depth_mask):
        """
        计算带掩码的深度图 MSE 损失。

        参数:
            depth_pred (torch.Tensor): 预测的深度图，形状为 (b, H, W)
            depth_true (torch.Tensor): 真实的深度图，形状为 (b, H, W)
            mask (torch.Tensor): 布尔掩码，形状为 (b, H, W)，指示哪些位置应该计算损失

        返回:
            torch.Tensor: 计算得到的 MSE 损失
        """
        # 应用掩码，仅计算掩码为 True 的位置
        diff = depth_pred - depth_render
        squared_diff = diff ** 2
        masked_squared_diff = squared_diff * depth_mask.float()  # 将掩码转换为浮点数用于乘法
        mse_loss = masked_squared_diff[depth_mask].mean()  # 仅计算掩码区域的均值

        return mse_loss

    def semantic_loss_with_mask(self, sem_map_pred, sem_map_render, sem_mask, ignore_index=9):
        """
        计算两个语义图之间的损失，考虑掩码和忽略特定类别。

        参数:
            sem_map1 (torch.Tensor): 第一个语义图，形状为 (b, n, H, W)，包含类别索引
            sem_map2 (torch.Tensor): 第二个语义图，形状为 (b, n, H, W)，包含类别索引
            mask (torch.Tensor): 布尔掩码，形状为 (b, n, H, W)，指示哪些位置应该计算损失
            ignore_index (int): 要忽略的类别索引，默认为9

        返回:
            torch.Tensor: 计算得到的损失值
        """
        # 应用掩码，选择有效的位置
        sem_map_pred = F.one_hot(sem_map_pred, num_classes=self.num_class)
        sem_map_pred = sem_map_pred.view(-1, self.num_class).float()
        sem_map_render = sem_map_render.view(-1).long()
        sem_mask = sem_mask.view(-1)

        # valid_positions = sem_mask & (sem_map_pred != ignore_index) & (sem_map_render != ignore_index)

        # 在有效的位置上应用交叉熵损失
        # 由于交叉熵需要目标为长张量，而且输入需要是logits，假设sem_map2是预测的logits
        # sem_map1是ground truth，且已经是分类标签的形式
        loss = self.sem_loss(sem_map_pred, sem_map_render)

        return loss

    def render_each_view_sem_depth(self, ego_croods, sem_OCC_map, scale):
        B, N, H, W, D, croods = scale
        ego_croods_occ_space = ego_croods * self.occ_grid_ratio
        occ_space_croods, occ_mask = self.norm_grid_sample(ego_croods_occ_space)
        occ_space_croods, occ_mask = occ_space_croods.view(B, N, croods, H, W, D).permute(0, 1, 3, 4, 5,
                                                                                          2), occ_mask.view(B, N, 1, H,
                                                                                                            W,
                                                                                                            D).permute(
            0, 1, 3, 4, 5, 2)

        # a = self.render_view(sem_OCC_map, occ_space_croods, mask_space=occ_mask)

        idx_type_ego_croods_occ_space = ego_croods_occ_space - self.xyz_min

        grid_sem = idx_type_ego_croods_occ_space.view(B, N, croods, H, W, D).permute(0, 1, 3, 4, 5, 2)
        render_sem_map = []
        render_depth_map = []
        for i in range(5):
            render_map = self.map_coords_to_semantics(sem_OCC_map, grid_sem[:, i, :, :, :, :])
            # print('render_map type is: ', render_map.size(), torch.unique(render_map, return_counts=True))
            render_map = torch.where(occ_mask[:, i, :, :, :, 0], render_map, torch.tensor(9, device=sem_OCC_map.device))
            extract_map, rdepth_map = self.extract_first_non_ignore(render_map, ignore_value=9)
            render_sem_map.append(extract_map)
            render_depth_map.append(rdepth_map)
        render_sem_map = torch.stack(render_sem_map, dim=1)
        render_depth_map = torch.stack(render_depth_map, dim=1)
        return render_sem_map, render_depth_map


    def forward(self, cam_params, sem_OCC_map, sem_view_map, occ_score):

        sem_OCC_map = sem_OCC_map.softmax(-1)    # (B, Dx, Dy, Dz, C)
        sem_OCC_map = sem_OCC_map.argmax(-1)      # (B, Dx, Dy, Dz)

        cam2ego_rot, cam2ego_trans, cam2img, post_rots, post_trans, bda = cam_params
        B, N, H, W, D, croods = self.coords.size()
        frustrum = self.coords.view(B, N, H*W*D, croods)
        ones = self.ones.view(B, N, -1, 1)
        xy_croods = frustrum[:, :, :, :2]
        depth_croods = frustrum[:, :, :, 2:].permute(0, 1, 3, 2)
        ego_croods = self.frustrum2ego(cam2ego_rot, cam2ego_trans, cam2img, post_rots, post_trans, xy_croods, depth_croods, ones)
        render_sem_map, render_depth_map = self.render_each_view_sem_depth(ego_croods, sem_OCC_map, scale=(B, N, H, W, D, croods))

        pseudo_mask, conf_pseudo_labels, loss_pseudo = self.SemiSupervisor(occ_score)
        loss_sem_pseudo = self.semantic_loss_with_mask(sem_view_map, render_sem_map, torch.ones_like(sem_view_map, dtype=torch.bool, device=sem_view_map.device), ignore_index=9)

        loss_all = 0.05 * loss_sem_pseudo + 0.2 * loss_pseudo
        return render_sem_map, render_depth_map, loss_all

