import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.base_module import BaseModule
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class SeSemanticDecoder(BaseModule):
    def __init__(self, semantic_attention, def_attention):
        super(SeSemanticDecoder, self).__init__()
        self.multihead_att =MODELS.build(def_attention)
        self.semantic_att = MODELS.build(semantic_attention)
        self.init_weights()

    def forward(self, img_feats):
        B, N, C, H, W = img_feats.size()
        img_feats = img_feats.view(B*N, C, H, W).flatten(2).permute(0, 2, 1)
        print('new_img_feats size is: ', img_feats.size())
        feats = self.multihead_att(img_feats)
        print('feats is: ', feats)
        seg_map, feat_map = self.semantic_att(feats)
        print('seg_map, feat_map is: ', seg_map.size(), feat_map.size())
        return img_feats


@MODELS.register_module()
class Se_DeformableAttention(BaseModule):
    def __init__(self, dim, num_heads, num_points):
        super(Se_DeformableAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.proj = nn.Linear(dim, dim)

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        grid_init = self._generate_initial_offsets()
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def _generate_initial_offsets(self):
        grid_init = torch.meshgrid(torch.linspace(-1, 1, self.num_points),
                                   torch.linspace(-1, 1, self.num_points))
        grid_init = torch.stack(grid_init, -1)
        grid_init = grid_init.view(-1, 2)
        return grid_init

    def forward(self, query, key, value, spatial_shapes, level_start_index):
        B, L, C = query.shape
        _, S, _ = key.shape

        sampling_offsets = self.sampling_offsets(query).view(B, L, self.num_heads, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(B, L, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, -1)

        # Compute the sampling locations
        sampling_locations = self._compute_sampling_locations(sampling_offsets, spatial_shapes, level_start_index)

        # Perform deformable attention
        output = self._deformable_attention(query, key, value, sampling_locations, attention_weights)

        output = self.proj(output)

        return output

    def _compute_sampling_locations(self, sampling_offsets, spatial_shapes, level_start_index):
        B, L, H, P, _ = sampling_offsets.shape
        sampling_locations = []

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            lvl_offsets = sampling_offsets[:, :, :, :, lvl].view(B, L, H, P, 2)
            lvl_offsets = lvl_offsets / torch.tensor([W_, H_], device=lvl_offsets.device) * 2.0
            lvl_offsets = lvl_offsets + 1.0
            lvl_offsets = lvl_offsets.clamp(0, 2)

            lvl_locations = torch.stack(torch.meshgrid(
                torch.linspace(0, H_ - 1, H_),
                torch.linspace(0, W_ - 1, W_)
            ), -1)

            lvl_locations = lvl_locations.view(1, 1, H_, W_, 2).repeat(B, L, 1, 1, 1)
            lvl_locations = lvl_locations + lvl_offsets

            sampling_locations.append(lvl_locations)

        sampling_locations = torch.cat(sampling_locations, -1)
        return sampling_locations

    def _deformable_attention(self, query, key, value, sampling_locations, attention_weights):
        B, L, H, P, _ = sampling_locations.shape
        _, S, _ = key.shape

        sampling_locations = sampling_locations.view(B * H, L, P, 2)
        attention_weights = attention_weights.view(B * H, L, P)

        sampling_values = F.grid_sample(
            value.permute(0, 2, 1).view(B * H, S, 1, 1),
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        sampling_values = sampling_values.view(B, H, L, P, -1)
        attention_weights = attention_weights.view(B, H, L, P, 1)

        output = (sampling_values * attention_weights).sum(dim=3)
        output = output.view(B, L, -1)

        return output