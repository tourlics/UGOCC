import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import MODELS
from mmcv.runner import BaseModule, auto_fp16
import torch.distributed as dist
from collections import OrderedDict
from threading import Thread
import time
from .cvt.elements.utils import ConvBlock, Conv3x3, upsample, disp_to_depth
import torch.nn as nn
from ..utils.utils import reshape_tensors
from ..ssdepth_baseline.selfsuper_depth_baseline import save_tensors_to_npy

@MODELS.register_module()
class Multi_view_Depthnet(BaseModule):
    def __init__(self,
                 backbone_level=5,
                 volume_depth_vs_metric_depth=False,  # if true, we use view rendering in the for_fusion_process
                 depth_scales=list(),
                 depth_boundary=(0,0),
                 img_size=(900, 1600),
                 use_multi_scale=False,
                 backbone=None,
                 neck=None,
                 depth_decoder=None,
                 **kwargs
                 ):
        super().__init__()
        self.backbone_level = backbone_level
        self.volume_depth_vs_metric_depth = volume_depth_vs_metric_depth
        self.depth_scales = depth_scales
        self.depth_boundary = depth_boundary
        self.use_multi_scale = use_multi_scale
        self.img_size = img_size

        self.img_backbone = MODELS.build(backbone)
        if neck is not None:
            self.img_neck = MODELS.build(neck)
        for key, value in kwargs.items():
            # print(key, value)
            setattr(self, key, MODELS.build(value))
        self.depth_decoder = MODELS.build(depth_decoder)


    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)


        # stereo_feat = None
        # if stereo:
        #     stereo_feat = x[0]
        #     x = x[1:]
        # if self.with_img_neck:
        # x = self.img_neck(x)
        # if type(x) in [list, tuple]:
        #     x = x[0]
        # _, output_dim, ouput_H, output_W = x.shape
        # x = x.view(B, N, output_dim, ouput_H, output_W)
        img_feats = []
        for i in range(len(x)):
            B, C, H, W = x[i].size()
            # print(B, C, H, W)
            img_feats.append(x[i].view(-1, N, C, H, W))
            # x[i] = x[i].view(-1, N, C, H, W)
        return img_feats

    def generate_depth_map(self, outputs):
        for scale in self.depth_scales:
            disp = outputs[("disp", scale)]
            # print('scales is {}, and disp is: {}'.format(scale, disp.size()))
            if self.use_multi_scale:
                source_scale = scale
            else:
                print('disp is: ', disp.size(), [self.img_size[0], self.img_size[1]])
                disp = F.interpolate(
                    disp, [self.img_size[0], self.img_size[1]], mode="bilinear", align_corners=False)
                # print('in the generate depth map use disp first: 1st')
                source_scale = 0

            if not self.volume_depth_vs_metric_depth:
                depth = disp
            else:
                _, depth = disp_to_depth(disp, self.depth_boundary[0], self.depth_boundary[1], abs=False)
                # print('in the generate depth map use disp first: but generate real depth')
                # print('debug depth and scale: {}'.format(depth.size()), depth.max(), depth.min())
            outputs[("depth", 0, scale)] = depth

        return outputs

    def forward(self, imgs, cam_params=None, key_frame_id=1, depthmap=None):
        img_feats = self.image_encoder(imgs)
        B, N, _, _, _ = img_feats[0].size()
        # print('imf_feats is: ', [i.size() for i in img_feats])
        att_feat = []
        for i in range(self.backbone_level):
            x = getattr(self, f'cvt_{i}')(img_feats[i])
            feat = img_feats[i] + x
            B, N, C, H, W = feat.size()
            att_feat.append(feat.view(B*N, C, H, W))
        outputs = self.depth_decoder(att_feat)
        outputs = self.generate_depth_map(outputs)
        key_frame_id_keys = 't-{}'.format(key_frame_id)

        cam_params_keys = list(cam_params.keys())
        for key_frame in cam_params_keys:
            cam_params[key_frame]['camC2global'] = torch.matmul(cam_params[key_frame]['ego2global'],
                                                                     cam_params[key_frame]['cam2ego'])
        cam_params_keys.remove(key_frame_id_keys)

        for i in cam_params_keys:
            cam_params[key_frame_id_keys]['camC2cam{}'.format(i)] = torch.matmul(torch.inverse(cam_params[i]['camC2global']),
                                                                                 cam_params[key_frame_id_keys]['camC2global'])
        print('cam_param keys is: ', cam_params_keys, [(i, outputs[i].size()) for i in outputs.keys()])

        if depthmap is not None:
            # depth_map = save_reproject_matrix_result(depthmap, N, height=256, width=704)
            save_tensors_to_npy(cam_params[key_frame_id_keys]['imgs'].unsqueeze(0),
                                './outputs_debug/temporal_imgst{}2t{}'.format(key_frame_id_keys, key_frame_id_keys),
                                save_depth=depthmap)
        else:
            pass

        for key_frame in cam_params_keys:
            for scale in self.depth_scales:
                if depthmap is not None:
                    cam_points = getattr(self, f'backproject_scale{scale}')(depthmap.view(6, 1, 256, 704), cam_params[key_frame_id_keys]['cam2img'],
                                                         post_rot=cam_params[key_frame_id_keys]['post_rots'],
                                                         post_trans=cam_params[key_frame_id_keys]['post_trans'])
                    pix_coords, debug_ouput = getattr(self, f'project_scale{scale}')(
                        cam_points, cam_params[key_frame]['cam2img'],
                        cam_params[key_frame_id_keys]['camC2cam{}'.format(key_frame)],
                        post_rot=cam_params[key_frame]['post_rots'], post_trans=cam_params[key_frame]['post_trans'],
                        debug=True)
                else:
                    cam_points = getattr(self, f'backproject_scale{scale}')(outputs[('depth', 0, 0)], cam_params[key_frame_id_keys]['cam2img'],
                                                         post_rot=cam_params[key_frame_id_keys]['post_rots'],
                                                         post_trans=cam_params[key_frame_id_keys]['post_trans'])
                    pix_coords, debug_ouput = getattr(self, f'project_scale{scale}')(
                        cam_points, cam_params[key_frame]['cam2img'], cam_params[key_frame_id_keys]['camC2cam{}'.format(key_frame)],
                        post_rot=cam_params[key_frame]['post_rots'], post_trans=cam_params[key_frame]['post_trans'], debug=True)
                # print('pixcorrd size is: ', pix_coords.size(), cam_params[key_frame]['imgs'].size()), 真实训练时要设置为 False


                if depthmap is not None:
                    depth_map = save_reproject_matrix_result(debug_ouput, N, height=256, width=704)
                    save_tensors_to_npy(cam_params[key_frame]['imgs'].unsqueeze(0),
                                        './outputs_debug/temporal_imgst{}2t{}'.format(key_frame_id_keys, key_frame),
                                        save_depth=depth_map)
                else:
                    pass
                    # depth_map = save_reproject_matrix_result(debug_ouput, N, height=256, width=704, metric=False)
                    # save_tensors_to_npy(cam_params[key_frame]['imgs'].unsqueeze(0),
                    #                     './outputs_debug/temporal_imgst{}2t{}'.format(key_frame_id_keys, key_frame),
                    #                     save_depth=depth_map)

                _, H, W, C = pix_coords.size()
                outputs[("sample", key_frame, scale)] = pix_coords
                B, N, C, H, W = cam_params[key_frame]['imgs'].size()
                outputs[("color", key_frame, scale)] = F.grid_sample(
                    cam_params[key_frame]['imgs'].view(B*N, C, H, W),
                    outputs[("sample", key_frame, scale)],
                    padding_mode="border", align_corners=True)


        return outputs


def save_reproject_matrix_result(debug_ouput, num_views, height=256, width=704, metric=True):
    _, c, n_points = debug_ouput.size()
    pix_coords = debug_ouput.view(-1, num_views, 3, n_points)
    depth_map_list = []
    for i in range(num_views):
        depth_map = torch.zeros((height, width), dtype=torch.float32).cuda()
        coor = torch.round(pix_coords[:, i, :2, :])
        depth = pix_coords[:, i, 2, :]

        if metric==True:
            kept1 = (coor[:, 0, :] >= 0) & (coor[:, 0, :] < width) & (
                    coor[:, 1, :] >= 0) & (coor[:, 1, :] < height) & (
                            depth < 100) & (
                            depth >= 0.1)
        else:
            kept1 = (coor[:, 0, :] >= 0) & (coor[:, 0, :] < width) & (
                    coor[:, 1, :] >= 0) & (coor[:, 1, :] < height)
        true_count = torch.sum(kept1)
        print('kept1 size is: {}'.format(kept1.size()), true_count)
        coor = coor[0, :, :].permute(1, 0)
        depth = depth[0, :]
        kept1 = kept1[0, :]
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        # depth_map = torch.cat((coor, depth), dim=1)
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        depth_map_list.append(depth_map)
    depth_map = torch.stack(depth_map_list).unsqueeze(0)
    return depth_map

@MODELS.register_module()
class SS_Depth_decoder(BaseModule):
    def __init__(self,
                 num_ch_enc,
                 num_ch_dec,
                 scales,
                 use_skips,
                 num_output_channels=1,
                 **kwargs
            ):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.scales = scales
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)


            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        print('self.scales: ', self.scales)
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        x = input_features[-1]
        outputs = {}
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        # for key, tensor in outputs.items():
        #     print(f"{key}: {tensor.size()}")
        return outputs

