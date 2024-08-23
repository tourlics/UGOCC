import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from projects.mmdet3d_plugin.models.losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from projects.mmdet3d_plugin.models.losses.lovasz_softmax import lovasz_softmax
from mmdet3d.models.builder import MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
import torch.nn.functional as F


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

occ_list = np.array([175883646,
                     28374731,
                     2074468,
                     105975596,
                     182111448,
                     51393615,
                     2841174,
                     413451,
                     341413,
                     1892500630]
)




@MODELS.register_module()
class OCC3DHead(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channel,
                 num_level=1,
                 soft_weights=False,
                 use_deblock=True,
                 conv_cfg=dict(type='Conv3d', bias=False),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 balance_cls_weight=True,
                 Dz=16,
                 use_mask=True,
                 use_focal_loss=False,
                 use_dice_loss=False,
                 loss_occ=True,
                 loss_weight_cfg=None,
                 class_balance=True
                 ):

        super().__init__()

        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.soft_weights = soft_weights
        self.use_deblock = use_deblock
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.balance_cls_weight = balance_cls_weight
        self.Dz = Dz
        self.use_mask = use_mask
        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss
        self.loss_occ = build_loss(loss_occ)
        self.loss_weight_cfg = loss_weight_cfg
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:self.out_channel] + 0.001))
            self.cls_weights = class_weights
        if self.use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i],
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel,
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2,
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel,
                        out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel//2,
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

    def forward(self, vox_feats):
        """
        Args:
            img_feats: (B, C, Dy=200, Dx=200)
            img_feats: [(B, C, 100, 100), (B, C, 50, 50), (B, C, 25, 25)]   if ms
        Returns:

        """
        output_occs = []
        output = {}

        x0 = self.deblock(vox_feats[0])
        output_occs.append(x0)
        for feats, occ_conv in zip(vox_feats, self.occ_convs):
            x = occ_conv(feats)
            output_occs.append(x)
        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat
        out_voxel_feats = 0
        _, _, H, W, D = output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)

        output['out_voxel_feats'] = [out_voxel_feats]
        out_voxel = self.occ_pred_conv(out_voxel_feats)

        # print('out voxel is : ', out_voxel.size())


        #  b, n_cls, X, Y, Z ------> b, X, Y, Z, n_cls
        output['occ'] = [out_voxel.permute(0, 2, 3, 4, 1)]
        return output

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0
        loss['loss_occ'] = loss_occ
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=self.out_channel-1)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        # print('occ_res: ', type(occ_res), occ_res.shape)
        return list(occ_res)