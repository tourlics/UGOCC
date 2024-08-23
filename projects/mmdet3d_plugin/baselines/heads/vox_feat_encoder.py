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
class Vox_Encoder(BaseModule):
    def __init__(
            self,
            vox_backbone=None,
            vox_neck=None
    ):
        super(Vox_Encoder, self).__init__()
        self.vox_backbone = MODELS.build(vox_backbone)
        self.vox_neck = MODELS.build(vox_neck)

    def forward(self, vox_feat):
        vox_feat = self.vox_backbone(vox_feat)
        vox_feat = self.vox_neck(vox_feat)
        if type(vox_feat) not in [list, tuple]:
             vox_feat = [vox_feat]
        return vox_feat