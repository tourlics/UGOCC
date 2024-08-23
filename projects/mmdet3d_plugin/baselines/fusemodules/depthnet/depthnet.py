# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

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
import cv2
from projects.mmdet3d_plugin.baselines.utils.self_supervised_depth_loss import save_reproject_matrix_result
from projects.mmdet3d_plugin.baselines.utils.utils import save_tensors_to_npy, calculate_fov


def add_proportional_noise(tensor, noise_ratio):
    """
    Add proportional random noise to a tensor to introduce uncertainty.

    Args:
        tensor (torch.Tensor): The original tensor.
        noise_ratio (float): The ratio of the noise relative to the tensor values.

    Returns:
        torch.Tensor: The tensor with added proportional noise.
    """
    noise = torch.randn_like(tensor).to(tensor.device) * noise_ratio * tensor
    return tensor + noise

def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0, normalize=False):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    max_ = tensor.flatten(1).max(-1).values[:, None, None]
    min_ = tensor.flatten(1).min(-1).values[:, None, None]
    tensor = (tensor - min_) / (max_ - min_)
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor * 255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=normalize).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


@MODELS.register_module()
class NaiveDepthNet(BaseModule):
    r"""Naive depthnet used in Lift-Splat-Shoot

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_

    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
            self,
            in_channels=512,
            context_channels=64,
            depth_channels=118,
            downsample=16,
            loss_depth_weight=None,
            grid_config=None,
            uniform=False,
            with_cp=False,
            sid=False,
            **kwargs
    ):
        super(NaiveDepthNet, self).__init__()
        self.loss_depth_weight = loss_depth_weight
        self.grid_config = grid_config
        self.uniform = uniform
        self.with_cp = with_cp
        self.context_channels = context_channels
        self.in_channels = in_channels
        self.D = depth_channels
        self.sid = sid
        self.downsample = downsample,
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.context_channels, kernel_size=1, padding=0)

        self.depth_channels = self.D
    @force_fp32()
    def forward(self, x, mlp_input=None):
        """
        """

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.depth_net, x)
        else:
            x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        context = x[:, self.D:self.D + self.context_channels, ...]
        if self.uniform:
            depth_digit = depth_digit * 0
            depth = depth_digit.softmax(dim=1)
        else:
            depth = depth_digit.softmax(dim=1)
        context = context.view(B, N, self.context_channels, H, W)
        depth = depth.view(B, N, self.D, H, W)
        return context, depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample[0]
        # if self.downsample == 8 and self.se_depth_map:
        #    downsample = 16
        B, N, H, W = gt_depths.shape
        # print(B, N, H, W, downsample, self.downsample)
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:,
                    1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # print('depth labels is: {}, depth_pred is: {}'.format(type(depth_labels), type(depth_preds)),
        #       depth_labels.size(), depth_preds.size())
        # print('depth labels max is: {}, min is: {}'.format(depth_labels.max(), depth_labels.min()))

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4,
                                          2).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return dict(loss_depth=self.loss_depth_weight * depth_loss)

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    @force_fp32()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    @force_fp32()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    @force_fp32()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    @force_fp32()
    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@MODELS.register_module()
class CPM_DepthNet(BaseModule):
    """
        Camera parameters aware depth net
    """

    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 depth_channels=118,
                 mid_channels=512,
                 use_dcn=True,
                 downsample=16,
                 grid_config=None,
                 loss_depth_weight=3.0,
                 with_cp=False,
                 se_depth_map=False,
                 sid=False,
                 bias=0.0,
                 input_size=(),
                 use_aspp=True,
                 noise_ratio=None,
                 **kwargs
                 ):
        super(CPM_DepthNet, self).__init__()

        if noise_ratio is not None:
            self.noise_ratio = noise_ratio
        else:
            self.noise_ratio = None

        self.input_size = input_size
        for key, value in kwargs.items():
            # print(key, value)
            setattr(self, key, MODELS.build(value))

        self.fp16_enable = False
        self.sid = sid
        self.with_cp = with_cp
        self.downsample = downsample
        self.grid_config = grid_config
        self.loss_depth_weight = loss_depth_weight
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.se_depth_map = se_depth_map
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels,
                       downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

    @force_fp32()
    def forward(self, x, mlp_input):

        # if not  x.requires_grad:
        x = x.to(torch.float32)  # FIX distill type error
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)

        context = self.context_conv(context)                     # context has most of the feature which was get from single image
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if self.with_cp and depth.requires_grad:
            depth = cp.checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)

        # print('before ssoftmax depth is: ', depth.size())
        # print('in the depthnet depth is: {}'.format(depth.size()))
        depth = depth.softmax(dim=1)
        context = context.view(B, N, self.context_channels, H, W)
        depth = depth.view(B, N, self.depth_channels, H, W)
        # print('after ssoftmax depth is: ', depth.size())
        return context, depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        if self.noise_ratio is not None:
            B, N, _, _ = rot.shape
            bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
            mlp_input = torch.stack([
                add_proportional_noise(intrin[:, :, 0, 0], self.noise_ratio),
                add_proportional_noise(intrin[:, :, 1, 1], self.noise_ratio),
                add_proportional_noise(intrin[:, :, 0, 2], self.noise_ratio),
                add_proportional_noise(intrin[:, :, 1, 2], self.noise_ratio),
                add_proportional_noise(post_rot[:, :, 0, 0], self.noise_ratio),
                add_proportional_noise(post_rot[:, :, 0, 1], self.noise_ratio),
                add_proportional_noise(post_tran[:, :, 0], self.noise_ratio),
                add_proportional_noise(post_rot[:, :, 1, 0], self.noise_ratio),
                add_proportional_noise(post_rot[:, :, 1, 1], self.noise_ratio),
                add_proportional_noise(post_tran[:, :, 1], self.noise_ratio),
                add_proportional_noise(bda[:, :, 0, 0], self.noise_ratio),
                add_proportional_noise(bda[:, :, 0, 1], self.noise_ratio),
                add_proportional_noise(bda[:, :, 1, 0], self.noise_ratio),
                add_proportional_noise(bda[:, :, 1, 1], self.noise_ratio),
                add_proportional_noise(bda[:, :, 2, 2], self.noise_ratio),
            ],
                dim=-1)
            sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                                   dim=-1).reshape(B, N, -1)

            # print('prepare mlp input is: ', mlp_input.size(), sensor2ego.size())
            mlp_input = torch.cat([mlp_input, add_proportional_noise(sensor2ego, self.noise_ratio)], dim=-1)
        else:
            B, N, _, _ = rot.shape
            bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ],
                dim=-1)
            sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                                   dim=-1).reshape(B, N, -1)

            # print('prepare mlp input is: ', mlp_input.size(), sensor2ego.size())
            mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)

        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        输入:
            gt_depths: [B, N, H, W]   形状与 sigmoid 函数的结果相同
        输出:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample  # 获取下采样因子

        B, N, H, W = gt_depths.shape  # 获取输入张量的形状

        # 调整 gt_depths 的形状以便下采样
        gt_depths = gt_depths.view(B * N, H // downsample, downsample, W // downsample, downsample, 1)
        # 形状变化: [B, N, H, W] -> [B*N, H//downsample, downsample, W//downsample, downsample, 1]

        # 调整维度顺序并使其在内存中连续
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # 形状变化: [B*N, H//downsample, downsample, W//downsample, downsample, 1] -> [B*N, H//downsample, W//downsample, 1, downsample, downsample]

        # 调整形状以进行下采样
        gt_depths = gt_depths.view(-1, downsample * downsample)  # 这部分的操作相当于将深度集中于一起
        # 形状变化: [B*N, H//downsample, W//downsample, 1, downsample, downsample] -> [B*N * H//downsample * W//downsample, downsample * downsample]

        # 将深度值为 0 的位置替换为一个大值 (1e5)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)

        # 在下采样区域内取最小值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        # 形状变化: [B*N * H//downsample * W//downsample, downsample * downsample] -> [B*N * H//downsample * W//downsample]

        # 恢复形状
        gt_depths = gt_depths.view(B * N, H // downsample, W // downsample)  # 将选出区域的最小值拿出来构建一个形状经过改变的新的深度图
        # 形状变化: [B*N * H//downsample * W//downsample] -> [B*N, H//downsample, W//downsample]

        # 如果不是使用 SID（对数空间），则执行标准化
        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            # 否则，转换为对数空间
            gt_depths = torch.log(gt_depths) - torch.log(torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() / self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        # 形状不变: [B*N, H//downsample, W//downsample]

        # 将不在有效范围内的深度值替换为 0
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0), gt_depths,
                                torch.zeros_like(gt_depths))

        # 将深度值转换为 one-hot 编码
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[
                    :, 1:]
        # 形状变化: [B*N, H//downsample, W//downsample] -> [-1, self.depth_channels + 1] -> [-1, self.depth_channels]
        # 这一步的目的是去掉类别 0 的独热编码，只保留有效深度通道的编码。
        # 输出张量的形状将变为 [N, self.depth_channels]。
        # 返回 float 类型的 one-hot 编码
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        # print('depth labels is: {}, depth_pred is: {}'.format(type(depth_labels), type(depth_preds)),
        #       depth_labels.size(), depth_preds.size())
        # print('depth labels max is: {}, min is: {}'.format(depth_labels.max(), depth_labels.min()))

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # print('depth_labels is: ', depth_labels.size())
        # print('depth labels is: ', depth_labels.size())
        depth_preds = depth_preds.permute(0, 1, 3, 4,
                                          2).contiguous().view(-1, self.depth_channels)
        # print('depth pred is: ', depth_preds.size())
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            # print(depth_preds.dtype,
            #     depth_labels.dtype,)
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return dict(loss_depth=self.loss_depth_weight * depth_loss)

    def convert_depth_predictions(self, depth_preds):
        '''

        Args:
            depth_preds:  (B, N, depth_channel, H, W)  softmax format depth

        Returns:
            depth_map   : (B, N, H, W)
        '''
        B, N, depth_channel, H, W = depth_preds.shape
        min_depth = self.grid_config['depth'][0]
        max_depth = self.grid_config['depth'][1]
        depth_granularity = self.grid_config['depth'][2]
        depth_values = torch.arange(min_depth, max_depth, depth_granularity).float().to(
            depth_preds.device)
        depth_values = depth_values.view(1, 1, depth_channel, 1, 1)
        depth_map = torch.sum(depth_preds * depth_values, dim=2)
        depth_map = depth_map.unsqueeze(2)
        return depth_map

    def interpolatr_depthmap(self, depth_map):
        B, N, depth_channel, H, W = depth_map.size()
        depth_map = depth_map.view(B * N, depth_channel, H, W)
        depth_map = F.interpolate(
            depth_map, [self.input_size[0], self.input_size[1]], mode="bilinear", align_corners=False)
        depth_map = depth_map.view(B, N, depth_channel, self.input_size[0], self.input_size[1]).squeeze(2)
        return depth_map

    def generate_depth_map(self, depth_preds):
        # step1: convert to standard depth map format
        depth_map = self.convert_depth_predictions(depth_preds)

        # step2: convert depth map to orginal size
        B, N, depth_channel, H, W = depth_map.size()
        depth_map = depth_map.view(B*N, depth_channel, H, W)
        depth_map = F.interpolate(
                    depth_map, [self.input_size[0], self.input_size[1]], mode="bilinear", align_corners=False)
        depth_map = depth_map.view(B, N, depth_channel, self.input_size[0], self.input_size[1]).squeeze(2)
        return depth_map

    def generate_current_frame_images(self, depth_map, frame_train_inputs, cam_params_keys, key_frame_id_keys):
        outputs = {key: None for key in cam_params_keys}
        B, N, depth_channel, H, W = depth_map.size()
        depth_map = depth_map.view(B*N, depth_channel, H, W)
        depth_scales = [0]
        for scales in depth_scales:
            for key_frame in cam_params_keys:
                outputs[key_frame] = {}
                cam_points = getattr(self, f'backproject_scale{scales}')(depth_map,
                                                                        frame_train_inputs[key_frame_id_keys]['cam2img'],
                                                                        post_rot=frame_train_inputs[key_frame_id_keys]['post_rots'],
                                                                        post_trans=frame_train_inputs[key_frame_id_keys]['post_trans'])
                pix_coords, debug_ouput = getattr(self, f'project_scale{scales}')(
                    cam_points, frame_train_inputs[key_frame]['cam2img'], frame_train_inputs[key_frame_id_keys]['camC2cam{}'.format(key_frame)],
                    post_rot=frame_train_inputs[key_frame]['post_rots'], post_trans=frame_train_inputs[key_frame]['post_trans'], debug=True)

                # todo: 20240711 需要进行debug的时候就把这里取消注释即可
                # depth_map_proj = save_reproject_matrix_result(debug_ouput, N, height=256, width=704)
                # print('depth_map_proj: ', depth_map_proj.size())
                #
                # print('inside 1 is: ', frame_train_inputs[key_frame]['imgs'].unsqueeze(0).size(), depth_map_proj.transpose(0, 1).size())
                # save_tensors_to_npy(frame_train_inputs[key_frame]['imgs'].unsqueeze(0),
                #                     './outputs_debug/ssdbaseline/temporal_imgst{}2t{}'.format(key_frame_id_keys, key_frame),
                #                     save_depth=depth_map_proj.transpose(0, 1))

                _, H, W, C = pix_coords.size()
                outputs[key_frame][("sample", scales)] = pix_coords

                B, N, C, H, W = frame_train_inputs[key_frame]['imgs'].size()
                outputs[key_frame][("color", scales)] = F.grid_sample(
                    frame_train_inputs[key_frame]['imgs'].view(B * N, C, H, W),
                    outputs[key_frame][("sample", scales)],
                    padding_mode="border", align_corners=True)

        return outputs

    @force_fp32()
    def get_self_supervised_depth_loss(self, depth_pred, frame_train_inputs=None, depth_map_gt=None):
        key_frame_id_keys = 't-0'
        cam_params_keys = list(frame_train_inputs.keys())

        # print('depth_map_gt is: ', depth_map_gt)
        # step1: generate depth map
        if depth_map_gt is None:
            depth_map = self.generate_depth_map(depth_preds=depth_pred)
        else:
            B, N, H, W = depth_map_gt.size()
            depth_map_gt2 = depth_map_gt.view(B*N, H, W)
            depth_map_gt2 = depth_map_gt2.unsqueeze(1)
            print('depth_gt_map size: ', depth_map_gt.size())
            depth_map = depth_map_gt.unsqueeze(2)
            save_tensors_to_npy(frame_train_inputs[key_frame_id_keys]['imgs'].unsqueeze(0),
                                './outputs_debug/ssdbaseline/temporal_imgst{}2t{}'.format(key_frame_id_keys, key_frame_id_keys),
                                save_depth=depth_map_gt2.transpose(0, 1))

        # step2: convert trans matrix

        for key_frame in cam_params_keys:
            frame_train_inputs[key_frame]['camC2global'] = torch.matmul(frame_train_inputs[key_frame]['ego2global'],
                                                                     frame_train_inputs[key_frame]['cam2ego'])
        cam_params_keys.remove(key_frame_id_keys)
        # print('debug frame_train_inputs', frame_train_inputs.keys(), frame_train_inputs['t-0'].keys())
        for i in cam_params_keys:
            frame_train_inputs[key_frame_id_keys]['camC2cam{}'.format(i)] = \
                torch.matmul(torch.inverse(frame_train_inputs[i]['camC2global']),
                             frame_train_inputs[key_frame_id_keys]['camC2global'])

        # step3: cacluacate loss and digitial
        outputs = self.generate_current_frame_images(depth_map, frame_train_inputs, cam_params_keys, key_frame_id_keys)
        return depth_map, outputs


@MODELS.register_module()
class CM_ContextNet(nn.Module):
    """
        Camera parameters aware depth net
    """

    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 mid_channels=512,
                 with_cp=False,
                 ):
        super(CM_ContextNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

    @force_fp32()
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)
        context = self.context_conv(context)
        context = context.view(B, N, self.context_channels, H, W)
        return context
