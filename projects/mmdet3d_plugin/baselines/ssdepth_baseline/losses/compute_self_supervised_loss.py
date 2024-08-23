import torch
import torch.nn as nn
import numpy as np
from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class Depth_Estimation_Self_Supervised_Loss(nn.Module):

    def __init__(self):
        super(Depth_Estimation_Self_Supervised_Loss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
