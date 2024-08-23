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
class SEMAtention_Fusion_Module(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 semantic_decoder=None,
                 input_size=None,
                 downsample=1,
                 transformer=None,
                 positional_encoding=None,
                 in_channels=64,
                 out_channels=64,
                 use_zero_embedding=False,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):
        super().__init__()
        if semantic_decoder is not None:
            self.semantic_decoder = MODELS.build(semantic_decoder)
        else:
            self.semantic_decoder = None

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.use_zero_embedding = use_zero_embedding
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        self.input_H, self.input_W = input_size
        self.downsample = downsample
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def align_feat(self, feat_map):
        B, N, C, H, W = feat_map.size()
        if H != self.input_H/self.downsample or W != self.input_W/self.downsample:
            feat_map = F.interpolate(feat_map.view(-1, C, H, W), size=(int(self.input_H/self.downsample), int(self.input_W/self.downsample)), mode='bilinear', align_corners=False)
            _, _, H_new, W_new = feat_map.size()
            feat_map = feat_map.view(B, N, C, H_new, W_new)
        else:
            pass
        return feat_map

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, img_features, cam_params, pred_depth_map, pred_sem_map):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        pred_depth_map = self.align_feat(pred_depth_map)
        pred_sem_map = self.align_feat(pred_sem_map)

        bs, num_cam, _, _, _ = img_features[0].shape
        dtype = img_features[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        # print('orginal bev_queries size is: {}'.format(bev_queries.size()))
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = self.positional_encoding(bs, self.bev_h, self.bev_w, bev_queries.device).to(dtype)

        print('bev_queries is: ', bev_queries.size(), bev_pos.size())
        # print('real is: ', self.real_h, self.real_w, self.bev_h, self.bev_w)
        print('in sem_attention module input is--> img_features: {}, pred_depth_map: {}, pred_sem_map: {}'.format(
            img_features[0].size(), pred_depth_map.size(), pred_sem_map.size()
        ))


        a = self.semantic_decoder(img_features[0])
        # bev = self.transformer(
        #     img_features,
        #     bev_queries,
        #     self.bev_h,
        #     self.bev_w,
        #     bev_pos=bev_pos,
        #     # img_metas=img_metas,
        #     cam_params=cam_params,
        #     pred_img_depth=pred_depth_map,
        #     pred_sem_map=pred_sem_map,
        #     prev_bev=None,
        #     bev_mask=None,
        # )

        # print(self.input_H/self.downsample, img_features.size()[3], img_features.size()[3] == self.input_H/self.downsample)
        return img_features