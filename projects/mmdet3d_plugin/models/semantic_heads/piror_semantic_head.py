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

import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.swin import SwinTransformer



# class SeMask_Transformer(SwinTransformer):
#     def __init__(self,
#                  pretrain_img_size=224,
#                  in_channels=3,
#                  embed_dims=96,
#                  patch_size=4,
#                  window_size=7,
#                  mlp_ratio=4,
#                  depths=(2, 2, 6, 2),
#                  num_heads=(3, 6, 12, 24),
#                  strides=(4, 2, 2, 2),
#                  out_indices=(0, 1, 2, 3),
#                  qkv_bias=True,
#                  qk_scale=None,
#                  patch_norm=True,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.1,
#                  use_abs_pos_embed=False,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  with_cp=False,
#                  pretrained=None,
#                  convert_weights=False,
#                  frozen_stages=-1,
#                  init_cfg=None):