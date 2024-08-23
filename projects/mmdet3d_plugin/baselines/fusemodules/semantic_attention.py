from mmdet3d.models.builder import MODELS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet3d.models.builder import MODELS
import math
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

@MODELS.register_module()
class SeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads):
        super(SeMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = embed_dims
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == embed_dims, "dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dims, embed_dims)
        self.key = nn.Linear(embed_dims, embed_dims)
        self.value = nn.Linear(embed_dims, embed_dims)
        self.fc_out = nn.Linear(embed_dims, embed_dims)
        self.pos_embedding = PositionalEncoding(embed_dims)



        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.query.bias, 0)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        # 加入位置嵌入
        query = self.pos_embedding(x)
        key = self.pos_embedding(x)

        Q = self.query(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, N, C)

        return self.fc_out(output)


@MODELS.register_module()
class SemanticAttention(nn.Module):
    """ ClassMasking
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, emb_dim, n_cls):

        super().__init__()
        self.dim = emb_dim
        self.n_cls = n_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.n_cls)
        self.mlp_cls_k = nn.Linear(self.dim, self.n_cls)

        self.mlp_v = nn.Linear(self.dim, self.dim)

        self.mlp_res = nn.Linear(self.dim, self.dim)

        self.proj_drop = nn.Dropout(0.1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weight()

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, N, C)
        returns:
            class_seg_map: (B, N, K)
            gated feats: (B, N, C)
        """
        seg_map = self.mlp_cls_q(x)
        seg_ft = self.mlp_cls_k(x)

        feats = self.mlp_v(x)

        seg_score = seg_map @ seg_ft.transpose(-2, -1)
        seg_score = self.softmax(seg_score)

        feats = seg_score @ feats
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x

        return seg_map, feat_map

    def _init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Linear):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.LayerNorm):
                nn.init.constant_(ly.bias, 0)
                nn.init.constant_(ly.weight, 1.0)

        nn.init.zeros_(self.mlp_res.weight)
        if not self.mlp_res.bias is None: nn.init.constant_(self.mlp_res.bias, 0)

@MODELS.register_module()
class SemWindowAttention(nn.Module):
    """ Window-based multi-head self-attention (W-MSA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (int): Window size.
        n_cls (int): Number of classes.
    """

    def __init__(self, dim, window_size, n_cls):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_cls = n_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.n_cls)
        self.mlp_cls_k = nn.Linear(self.dim, self.n_cls)
        self.mlp_v = nn.Linear(self.dim, self.dim)
        self.mlp_res = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weight()

    def _init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Linear):
                nn.init.kaiming_normal_(ly.weight)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.LayerNorm):
                nn.init.constant_(ly.bias, 0)
                nn.init.constant_(ly.weight, 1.0)
        nn.init.zeros_(self.mlp_res.weight)
        if self.mlp_res.bias is not None:
            nn.init.constant_(self.mlp_res.bias, 0)

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, H, W, C)
        returns:
            class_seg_map: (B, H, W, K)
            gated feats: (B, H, W, C)
        """
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, "Height and Width must be divisible by window_size"

        # Partition into windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        seg_map = self.mlp_cls_q(x)
        seg_ft = self.mlp_cls_k(x)
        feats = self.mlp_v(x)

        seg_score = torch.bmm(seg_map, seg_ft.transpose(1, 2))
        seg_score = self.softmax(seg_score)

        feats = torch.bmm(seg_score, feats)
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x

        # Merge windows
        feat_map = feat_map.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        feat_map = feat_map.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        # Compute seg_map for the original input size
        seg_map = seg_map.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        seg_map = seg_map.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

        return seg_map, feat_map
