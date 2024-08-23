import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import sys
import timm
# from timm.models.layers.mlp import Mlp
from timm.models.layers import trunc_normal_
import numpy as np
from mmdet3d.models.builder import MODELS
from .mlp import Mlp

@MODELS.register_module()
class Self_Block(nn.Module):
    def __init__(self, dim, Self_Attention=None, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MODELS.build(Self_Attention)
        # self.attn = Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = MODELS.build(mlp)


    def forward(self, x, x_pos):
        x = x + self.drop_path(self.attn(self.norm1(x), x_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@MODELS.register_module()
class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_pos):
        B, N, C = x.shape

        q_vector = k_vector = x + x_pos
        v_vector = x

        q = self.q_linear(q_vector).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(k_vector).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(v_vector).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@MODELS.register_module()
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=4, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

@MODELS.register_module()
class SeparableDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=4, padding=0, dilation=4, bias=False,
                 output_padding=0):
        super(SeparableDeConv2d, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                        groups=in_channels, padding=padding, dilation=dilation,
                                        output_padding=output_padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # pdb.set_trace()
        # print('--------------SeparableDeConv2d---------------')
        # print('1st: ', x.size())
        x = self.conv1(x)
        # print('2st: ', x.size())
        x = self.pointwise(x)
        # print('3st: ', x.size())
        return x