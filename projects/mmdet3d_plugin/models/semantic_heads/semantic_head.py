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



@MODELS.register_module()
class SemanticDecoder(BaseModule):
    def __init__(
            self,
            in_channels=512,
            num_class=10,
            mask_num=255,
            context_channels=64,
            loss_seg_weight=None,
            head_dropout=0.1,
            layer_dropout=0.1,
            ignore_index=255,
            weights=None,
            input_size=None,
            **kwargs
    ):
        super(SemanticDecoder, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.mask_num = mask_num
        self.context_channels = context_channels
        self.loss_seg_weight = loss_seg_weight
        self.ignore_index = ignore_index
        self.sem_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights), ignore_index=ignore_index)
        self.input_size = input_size
        seg_conv_list = [
            nn.Conv2d(in_channels*2, context_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(context_channels*2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        ]

        seg_deconv_list = [
            nn.ConvTranspose2d(context_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            # 第三个反卷积层
            nn.ConvTranspose2d(64, self.num_class, kernel_size=4, stride=2, padding=1),
            nn.Dropout(head_dropout)
        ]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 3, padding=1, bias=False),
            nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity(),
            *seg_conv_list,
            nn.Conv2d(context_channels*2, context_channels, 1),
            *seg_deconv_list
        )

    def loss(self, pred, target):
        B, N, H, W = target.size()
        _, C, Hp, Wp = pred.size()
        if Hp != H or Wp != W:
            pred = F.interpolate(
                pred, [self.input_size[0], self.input_size[1]], mode="bilinear", align_corners=False)

        sem_map = pred.permute(0, 2, 3, 1).contiguous()
        #print('permute size is: ', sem_map.size())

        sem_map = sem_map.contiguous().view(-1, C)  # Ensure sem_map is of type Float
        target = target.view(-1)
        target = target.long()  # Ensure target is of type Long
        #print('target permute is: ', target.size(), sem_map.size(), sem_map.dtype, target.dtype)

        loss_sem = self.loss_seg_weight * self.sem_loss(sem_map, target)
        return loss_sem

    def forward(self, sem_map, sigmoid_depth_map):

        B, N, C, H, W = sem_map.size()

        sem_map = sem_map.view(B*N, C, H, W)

        B, N, C, H, W = sigmoid_depth_map.size()
        sigmoid_depth_map = sigmoid_depth_map.view(B*N, C, H, W)

        sem_feature_map = torch.cat([sem_map, sigmoid_depth_map], dim=1)
      #  print('sem_feature_map is: ', sem_feature_map.size())

        sem_map = self.head(sem_feature_map)
        sem_map = sem_map.softmax(dim=1)
       # print('sem_map is: ', sem_map.size())

        return sem_map

@MODELS.register_module()
class SeMaSemanticDecoder(BaseModule):
    def __init__(
            self,
            in_channels=512,
            num_class=10,
            mask_num=255,
            context_channels=64,
            loss_seg_weight=None,
            head_dropout=0.1,
            layer_dropout=0.1,
            ignore_index=255,
            weights=None,
            input_size=None,
            semantic_attention=None,
            use_windows=True,
            **kwargs
    ):
        super(SeMaSemanticDecoder, self).__init__()
        self.SemanticAttention = MODELS.build(semantic_attention)
        self.in_channels = in_channels
        self.num_class = num_class
        self.mask_num = mask_num
        self.context_channels = context_channels
        self.loss_seg_weight = loss_seg_weight
        self.ignore_index = ignore_index
        self.sem_loss = nn.CrossEntropyLoss(weight=torch.tensor(weights), ignore_index=ignore_index)
        self.input_size = input_size
        self.use_windows = use_windows
        seg_conv_list = [
            nn.Conv2d(in_channels, context_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        ]


        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity(),
            *seg_conv_list,
            nn.Conv2d(context_channels, context_channels, 1),
            nn.Conv2d(context_channels, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Conv2d(64, self.in_channels-1, kernel_size=3, padding=1),
            nn.Dropout(head_dropout)
        )
    def loss(self, pred, target):
        B, N, H, W = target.size()
        _, C, Hp, Wp = pred.size()
        if Hp != H or Wp != W:
            pred = F.interpolate(
                pred, [self.input_size[0], self.input_size[1]], mode="bilinear", align_corners=False)
        sem_map = pred.permute(0, 2, 3, 1).contiguous()
        #print('permute size is: ', sem_map.size())

        sem_map = sem_map.contiguous().view(-1, C)  # Ensure sem_map is of type Float
        target = target.view(-1)
        target = target.long()  # Ensure target is of type Long
        #print('target permute is: ', target.size(), sem_map.size(), sem_map.dtype, target.dtype)

        loss_sem = self.loss_seg_weight * self.sem_loss(sem_map, target)
        return loss_sem

    def forward(self, sem_map, sigmoid_depth_map):

        B, N, C, H, W = sem_map.size()

        sem_map = sem_map.view(B*N, C, H, W)

        B, N, Cd, H, W = sigmoid_depth_map.size()
        sigmoid_depth_map = sigmoid_depth_map.view(B*N, Cd, H, W)

        sem_feature_map = torch.cat([sem_map, sigmoid_depth_map], dim=1)
      #  print('sem_feature_map is: ', sem_feature_map.size())

        sem_map = self.head(sem_feature_map)
        # print('sem_map is: ', sem_map.size())

        if self.use_windows:
            sem_map = sem_map.permute(0, 2, 3, 1)

            sem_map, feat_map = self.SemanticAttention(sem_map)

            # print('sem_map, feat_map ', sem_map.size(), feat_map.size())
            sem_map = sem_map.permute(0, 3, 1, 2)
            feat_map = feat_map.permute(0, 3, 1, 2).view(B, N, C, H, W)
        else:

            _, C, H, W = sem_map.size()
            sem_query = sem_map.flatten(2).permute(0, 2, 1)
            # print('sem_query is: ', sem_query.size())
            sem_map, feat_map = self.SemanticAttention(sem_query)

            bn, n, C = sem_map.size()
            sem_map = sem_map.permute(0, 1, 2).view(bn, C, H, W)

            bn, n, C = feat_map.size()
            feat_map = feat_map.permute(0, 1, 2).view(bn, C, H, W).view(B, N, C, H, W)
        # print('seg_map, feat_map', sem_map.size(), feat_map.size())
        sem_map = sem_map.softmax(dim=1)

       # print('sem_map is: ', sem_map.size())
       #  print('feat_map, sem_map is: ', feat_map.size(), sem_map.size())
        return feat_map, sem_map