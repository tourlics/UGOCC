import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable
from mmcv.runner.base_module import BaseModule
from mmdet3d.models.builder import MODELS
try:
    from itertools import ifilterfalse
except ImportError: # py3k
    from itertools import filterfalse as ifilterfalse
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding


@MODELS.register_module()
class Atention_Fusion_Module3D(BaseModule):
    def __init__(self,
                 *args,
                 transformer=None,
                 volume_h=200,
                 volume_w=200,
                 volume_z=16,
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 positional_encoding=None,
                 use_semantic=True,
                 **kwargs):
        super(Atention_Fusion_Module3D, self).__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.img_channels = img_channels
        self.transformer = transformer

        self.embed_dims = self.transformer.embed_dims
        self.transformer = build_transformer(transformer)
        self._init_layers()
        # if positional_encoding is not None:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)

    def _init_layers(self):
        self.vox_embedding = nn.Embedding(
            self.volume_h * self.volume_w * self.volume_z, self.embed_dims)
    # def init_layers(self):
    #     self.transformer = nn.ModuleList()
    def forward(self, multi_view_feats, cam_params, lss_vox=None, vox_mask=None, **kwargs):
        '''

        Args:
            multi_view_feats:  size--> (B, N, C, H, W) and we should get the downsample scale
            cam_params:

        Returns:

        '''
        B, N, _, _, _ = multi_view_feats[0].shape
        print('now we start debuging the 3dformer fusion module: ', multi_view_feats[0].size())
        dtype = multi_view_feats[0].dtype

        vox_queries = self.vox_embedding.weight.to(dtype)
        print('1st is: ', vox_queries.size())
        vox_queries = vox_queries.unsqueeze(1).repeat(1, B, 1)
        print('2st is: ', vox_queries.size())

        volume_h = self.volume_h
        volume_w = self.volume_w
        volume_z = self.volume_z
        _, _, C, H, W = multi_view_feats[0].shape

        if lss_vox is not None:
            lss_vox = lss_vox.flatten(2).permute(2, 0, 1)
            # print('lss bev size is: {}, and the bev queries is: {}'.format(lss_bev.size(), bev_queries.size()))
            vox_queries = vox_queries + lss_vox

        if vox_mask is not None:
            bev_mask = vox_mask.reshape(B, -1)

        vox_embed = self.transformer(
            multi_view_feats,
            vox_queries,
            volume_h=self.volume_h,
            volume_w=self.volume_w,
            volume_z=self.volume_z,
            cam_params=cam_params,
        )
        print('vox_embed is" ', vox_embed.size())
        # vox_pos = self.positional_encoding(B, self.volume_h, self.volume_w, self.volume_z, bev_queries.device).to(dtype)

        vox_embed = vox_embed.reshape(B, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)  # w是x， h是y
        print('embeding fomnal: ', vox_embed.size())
        return vox_embed