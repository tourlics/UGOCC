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
class PseudoLabelSelector(nn.Module):
    def __init__(self, conf_lower_threshold=0.7,
                 conf_upper_threshold=0.95,
                 smooth_lower_threshold=0.7,
                 smooth_upper_threshold=0.95,
                 kernel_size=1,
                 num_classes=1,
                 pseudo_label_weight=0.5):
        """
        初始化伪标签选择器，集成 nn.Module。
        """
        super(PseudoLabelSelector, self).__init__()
        self.conf_lower_threshold = conf_lower_threshold
        self.conf_upper_threshold = conf_upper_threshold
        self.smooth_lower_threshold = smooth_lower_threshold
        self.smooth_upper_threshold = smooth_upper_threshold
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_classes = num_classes
        self.pseudo_label_weight = pseudo_label_weight

        self.CSR_loss = nn.CrossEntropyLoss()

    def custom_masked_cross_entropy(self, logits, targets, mask):
        """
        计算具有空间维度和类别维度的带掩码的交叉熵损失。

        参数:
            logits (torch.Tensor): 预测的类别得分，形状为 (B, X, Y, Z, C)
            targets (torch.Tensor): 真实的类别索引，形状为 (B, X, Y, Z)
            mask (torch.Tensor): 布尔掩码，形状为 (B, X, Y, Z)，True 表示计算损失

        返回:
            torch.Tensor: 计算得到的损失值
        """
        # 展平 logits 和 targets 以便应用掩码
        logits_flat = logits.view(-1, logits.size(-1))  # 形状变为 (B*X*Y*Z, C)
        targets_flat = targets.view(-1)  # 形状变为 (B*X*Y*Z)
        mask_flat = mask.view(-1)  # 形状变为 (B*X*Y*Z)

        # 使用掩码过滤 logits 和 targets
        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]

        # 计算交叉熵损失
        loss = F.cross_entropy(masked_logits, masked_targets)

        return loss

    def apply_spatial_smoothing(self, occ_pred):
        """
        对预测结果应用空间平滑。

        参数:
            predictions (torch.Tensor): 预测结果，形状为 (B, Dx, Dy, Dz, n_cls)

        返回:
            torch.Tensor: 空间平滑后的预测结果
        """
        padding = self.kernel_size // 2
        # 使用均值滤波器进行平滑，这里我们使用一个简单的均值池化
        smoothed_predictions = torch.sqrt(F.avg_pool3d(torch.square(occ_pred), kernel_size=self.kernel_size,
                                                       stride=1, padding=padding))
        return smoothed_predictions

    def select_smooth_pseudo_labels(self, occ_pred):
        """
        根据给定的预测和阈值选取伪标签。

        参数:
            predictions (torch.Tensor): 预测结果，形状为 (B, Dx, Dy, Dz, n_cls)

        返回:
            torch.Tensor: 伪标签 (B, Dx, Dy, Dz)
            torch.Tensor: 是否为伪标签的掩码 (B, Dx, Dy, Dz)
            int: 被选为伪标签的位置数量
        """
        # 应用空间平滑
        smoothed_predictions = self.apply_spatial_smoothing(occ_pred)

        # 取得最大概率及其索引
        max_probs, pseudo_labels = torch.max(smoothed_predictions, dim=-1)

        # 应用阈值
        selected_mask = max_probs >= self.smooth_lower_threshold  # 创建一个掩码，标记超过阈值的位置为True
        pseudo_labels[~selected_mask] = -1  # 未达到阈值的位置标记为-1（表示忽略）

        # 统计被选为伪标签的位置数量
        num_selected_labels = torch.sum(selected_mask).item()  # 将布尔掩码中的True值计数

        return pseudo_labels, selected_mask, num_selected_labels

    def select_confidence_pseudo_labels(self, occ_pred):
        """
        根据给定的预测和阈值选取伪标签。

        参数:
            predictions (torch.Tensor): 预测结果，形状为 (B, Dx, Dy, Dz, n_cls)

        返回:
            torch.Tensor: 伪标签 (B, Dx, Dy, Dz)
            torch.Tensor: 是否为伪标签的掩码 (B, Dx, Dy, Dz)
            int: 被选为伪标签的位置数量
        """
        # 取得最大概率及其索引
        max_probs, pseudo_labels = torch.max(occ_pred, dim=-1)

        # 应用阈值
        selected_mask = max_probs >= self.conf_lower_threshold  # 创建一个掩码，标记超过阈值的位置为True
        pseudo_labels[~selected_mask] = -1  # 未达到阈值的位置标记为-1（表示忽略）

        # 统计被选为伪标签的位置数量
        num_selected_labels = torch.sum(selected_mask).item()  # 将布尔掩码中的True值计数

        return pseudo_labels, selected_mask, num_selected_labels

    def forward(self, occ_pred):
        """
        使用定义的阈值和空间平滑选择伪标签。
        """
        # print(' in the pseudo generator occ_pred is: ', occ_pred.size())
        # 使用均值滤波器进行空间平滑
        occ_score = occ_pred.softmax(-1)
        conf_pseudo_labels, conf_selected_mask, conf_num_selected_labels = self.select_confidence_pseudo_labels(occ_pred)
        smoo_pseudo_labels, smoo_selected_mask, smoo_num_selected_labels = self.select_smooth_pseudo_labels(occ_pred)

        # print('conf_num_selected_labels and smoo_num_selected_labels', conf_num_selected_labels, smoo_num_selected_labels)
        pseudo_mask = conf_selected_mask & smoo_selected_mask
        conf_pseudo_labels[~pseudo_mask] = -1
        num_selected_labels = torch.sum(pseudo_mask).item()
        # print('conf_pseudo_mask is: ', conf_pseudo_labels.size(), pseudo_mask.size(), num_selected_labels, conf_pseudo_labels.dtype)


        loss = self.custom_masked_cross_entropy(occ_score, conf_pseudo_labels, pseudo_mask)
        # print('PSEDUDP',  loss)
        # occ_label = conf_pseudo_labels.argmax()

        # 返回伪标签和掩码
        return pseudo_mask, conf_pseudo_labels, loss