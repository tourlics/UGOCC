from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])



@ATTENTION.register_module()
class SEM_SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                layer_scale=None,
                dbound=None,
                 **kwargs
                 ):
        super(SEM_SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dbound = dbound
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        if layer_scale is not None:
            self.layer_scale =  nn.Parameter(
                layer_scale * torch.ones(embed_dims),
                requires_grad=True)
        else:
            self.layer_scale = None
        self.init_weight()
        self.count = 0

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                level_start_index=None,
                flag='encoder',
                bev_query_depth=None,
                pred_img_depth=None,
                pred_sem_map=None,
                bev_mask=None,
                per_cam_mask_list=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        print('in spatial sem attention is: query {}, key {}, value {}'.format(
            query.size(), key.size(), value.size()
        ), 'bev_query_depth is: {}'.format(bev_query_depth.size()), 'reference_points_cam: {}'.format(
            reference_points_cam.size()
        )
        )
        N, B, len_query, Z, _ = bev_query_depth.shape  # 这里的len_query 对应bev空间的x,y
        B, N, DC, H, W = pred_img_depth.shape          # 之前在建立映射时已经考虑了downsample
        bev_query_depth = bev_query_depth.permute(1, 0, 2, 3, 4)
        pred_img_depth = pred_img_depth.view(B*N, DC, H, W)  # 这是为了与后面 N, B的形式保持一致
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)

        B, N, CL, H, W = pred_sem_map.shape
        pred_sem_map = pred_sem_map.permute(1, 0, 2, 3, 4)
        pred_sem_map = pred_sem_map.reshape(B*N, CL, H, W)
        pred_sem_map = pred_sem_map.flatten(2).permute(0, 2, 1)

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        Z = reference_points_cam.size(3)  # 这里的D实际上时Z
        indexes = [[] for _ in range(bs)]  # index数量与batch_size 等同

        if bev_mask is not None:
            per_cam_mask_list_ = per_cam_mask_list & bev_mask[None, :, :, None]
        else:
            per_cam_mask_list_ = per_cam_mask_list
        max_len = 0
        print('here we debug the per_cam_mask_list: ', per_cam_mask_list_.size(), torch.unique(per_cam_mask_list_, return_counts=True))

        # cam, bs, x*y, z  False and True, True is the voxel is ref with the cam_points
        for j in range(bs):
            for i, per_cam_mask in enumerate(per_cam_mask_list_):
                index_query_per_img = per_cam_mask[j].sum(-1).nonzero().squeeze(-1)  # 判断在z轴上是否有射到，依然是一个视角
                print('index_query_per_img is: ', type(index_query_per_img), index_query_per_img.size())
                if len(index_query_per_img) == 0:
                    index_query_per_img = per_cam_mask_list[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                max_len = max(max_len, len(index_query_per_img))

                # 这是计算有效投影点，用于进行求平均


        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, Z, 2])
        bev_query_depth_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, Z, 1])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[j][i]

                print('create ref is: ', index_query_per_img.size())  # it is idx
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                bev_query_depth_rebatch[j, i, :len(index_query_per_img)] = bev_query_depth[j, i, index_query_per_img]  # 这个显然是depth的位置

                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]  # 同样
                # 此时我们得到了ref points rebatch， 其中包含 所有能够对应到bev空间的图像特征
                # bev query depth rebatch 是上面的二维图像坐标所对应的合法depth位置
                # 第一个是bev feat的能够对应上的位置
                # 我们在这里已经将其放到了全0形状的tensor前侧

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # 将视角归于batch
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)


        bev_query_depth_rebatch = (bev_query_depth_rebatch- self.dbound[0])/ self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, DC-1).to(torch.long)
        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                   num_classes=DC)


        print('cross-view transformer inputs is: queries_rebatch {}, reference_points_rebatch {}, bev_query_depth_rebatch{} pred_img_depth {}'.format(
            queries_rebatch.size(), reference_points_rebatch.size(), bev_query_depth_rebatch.size(), pred_img_depth.size()
        ), pred_sem_map.size(), 'key key key key', key.size())
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,\
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, Z, 2), spatial_shapes=spatial_shapes,\
                                            level_start_index=level_start_index,\
                                            bev_query_depth=bev_query_depth_rebatch.view(bs*self.num_cams, max_len, Z, DC),\
                                            pred_img_depth=pred_img_depth,\
                                            ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]  # 吧有关的点之间的关系计算完之后，再填回去

        count = per_cam_mask_list_.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]


        slots = self.output_proj(slots)
        if self.layer_scale is None:
            return self.dropout(slots) + inp_residual
        else:
            return self.dropout(self.layer_scale * slots) +  inp_residual