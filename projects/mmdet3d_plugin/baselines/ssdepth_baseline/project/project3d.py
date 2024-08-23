import torch
import torch.nn as nn
import numpy as np
from mmdet3d.models.builder import MODELS
import copy

@MODELS.register_module()
class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, post_rot=None, post_trans=None, debug=False):

        # K = K.contiguous()
        # T = T.contiguous()

        b, n, s0, s1 = K.size()
        K = K.view(-1, s1, s0)
        K_4x4 = torch.eye(4, device=K.device).repeat(b * n, 1, 1)
        K_4x4[:, :3, :3] = K.view(-1, 3, 3)
        K = K_4x4

        b, n, s0, s1 = T.size()
        T = T.view(-1, s1, s0)
        points = torch.matmul(T, points)
        cam_points = torch.matmul(K, points)[:, :3, :]
        # print('project 3d cam points ---------------- is: {}'.format(cam_points.size()))

        z2 = cam_points[:, 2, :].unsqueeze(1)
        # print('z in the project final is: ', z2.size())
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = torch.cat([pix_coords, z2], 1)
        # print('project 3d cam points ---------------- is after cat: {}'.format(pix_coords.size()))
        if post_rot is not None and post_trans is not None:
            b, n, s1, s2 = post_rot.size()
            # post_trans = post_trans.contiguous()
            # post_rot = post_rot.contiguous()
            post_rot = post_rot.view(b*n, s1, s2)
            # print('post_trans: ', post_trans.size())
            b, n, s1 = post_trans.size()
            post_trans = post_trans.view(b*n,  s1, 1)
            # print('post_trans: ', post_trans.size())
            pix_coords = torch.matmul(post_rot, pix_coords) + post_trans
            if debug:
                debug_output = copy.deepcopy(pix_coords.detach())
            else:
                debug_output = None
            z2 = pix_coords[:, 2, :]
            pix_coords = pix_coords[:, :2, :]
            # print('z2: ', z2.size(), z2.max(), z2.min())
        else:
            if debug:
                debug_output = copy.deepcopy(pix_coords)
            else:
                debug_output = None
            z2 = pix_coords[:, 2, :]
            pix_coords = pix_coords[:, :2, :]
        # print('size and wis in project 3d pixcooed: {} '.format(pix_coords.size()), pix_coords.max(), pix_coords.min())
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        # print('final project3d is: {}'.format(pix_coords.size()), pix_coords[0][0][0])
        return pix_coords, debug_output
