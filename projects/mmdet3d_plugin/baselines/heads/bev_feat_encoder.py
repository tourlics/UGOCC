import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import MODELS
import torch.utils.checkpoint as cp
import time


@MODELS.register_module()
class Bev_Encoder(BaseModule):
    def __init__(
            self,
            bev_backbone=None,
            bev_neck=None
    ):
        super(Bev_Encoder, self).__init__()
        self.bev_backbone = MODELS.build(bev_backbone)
        self.bev_neck = MODELS.build(bev_neck)

    def forward(self, bev_feat):

        bev_feat = self.bev_backbone(bev_feat)
        bev_feat = self.bev_neck(bev_feat)
        if type(bev_feat) in [list, tuple]:
            bev_feat = bev_feat[0]
        return bev_feat