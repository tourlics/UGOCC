# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule



@POSITIONAL_ENCODING.register_module()
class CustormLearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(CustormLearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, bs, h, w, device):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # h, w = mask.shape[-2:]
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        # print('x_embed y_embed ', x_embed.size(), y_embed.size(), x_embed.unsqueeze(0).repeat(h, 1, 1).size(),
        #       y_embed.unsqueeze(1).repeat(
        #           1, w, 1).size()
        #       )
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(bs, 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str



@POSITIONAL_ENCODING.register_module()
class CustormLearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 z_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(CustormLearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.z_embed = nn.Embedding(z_num_embed, num_feats)

        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.z_num_embed = z_num_embed

    def forward(self, bs, h, w, z, device):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
                h is y
                w is x
                z is z
        """
        # h, w = mask.shape[-2:]
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        z = torch.arange(z, device=device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        z_embed = self.z_embed(z)

        # print('x_embed y_embed ', x_embed.size(), y_embed.size(), x_embed.unsqueeze(0).repeat(h, 1, 1).size(),
        #       y_embed.unsqueeze(1).repeat(
        #           1, w, 1).size()
        #       )
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, z, 1), y_embed.unsqueeze(1).repeat(
                1, w, z,  1),  z_embed.unsqueeze(1).repeat(
                h, w, 1, 1)),
            dim=-1).permute(3, 0,
                            1, 2).unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        print('3d pos is: ', pos.size())
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str
