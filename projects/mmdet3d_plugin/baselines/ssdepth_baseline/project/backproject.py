import torch
import torch.linalg
import torch.nn as nn
import numpy as np
from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
        # self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
        #                                requires_grad=False)

    def forward(self, depth, K, post_rot=None, post_trans=None):
        '''

        Args:
            depth:   (b*n_view, 1, h, w)
            K:     (b, n_view, 3, 3)
            post_rot:  (b, n_view, 3, 3)
            post_trans:   (b, n_view, 3)

        Returns:

        '''
        if post_rot is not None and post_trans is not None:
            # print('-----------------------: ', post_rot.size(), post_trans.size(), self.pix_coords.size())
            # post_trans = post_trans.contiguous()
            # post_rot = post_rot.contiguous()

            b, n, s1, s2 = post_rot.size()
            post_rot = post_rot.view(b*n, s1, s2)
            # print('post_rot is: ', post_rot)
            # print('post_trans is: ', post_trans)
            post_rot_inv = torch.inverse(post_rot)
            b, n, s1 = post_trans.size()
            post_trans = post_trans.view(b*n, s1, 1)
            # print('post_tran: ', post_trans.size())
            cam_points = self.pix_coords - post_trans

            # print('cam_points size is: ', cam_points.size())
            cam_points = torch.matmul(post_rot_inv, cam_points)
        else:
            cam_points = self.pix_coords

        # K = K.contiguous()
        b, n, s1, s2 =K.size()
        K = K.view(b*n, s1, s2)
        inv_K = torch.inverse(K)
        # cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # print('self.pix_coords: ', self.pix_coords.size(), 'inv_k size is: {}'.format(inv_K.size()))
        # cam_points = torch.matmul(inv_K, self.pix_coords)
        cam_points = torch.matmul(inv_K, cam_points)
        # print('cam_points after maual invK: ', cam_points.size())
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        # print('cam_poiunts in backproject is: {}'.format(cam_points.size()), cam_points.max(), cam_points.min())
        return cam_points


