import copy
import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import MODELS
from mmcv.runner import BaseModule, auto_fp16
import torch.distributed as dist
from collections import OrderedDict
import os
import numpy as np
from projects.mmdet3d_plugin.baselines.utils.utils import save_tensors_to_npy, calculate_fov


@MODELS.register_module()
class OnlySSSdDepthSamBaseline(BaseModule):
    def __init__(self,
                 repeoject_loss_weight=1.0,
                 depth_loss_weight=1.0,
                 load_multi_frame=0,
                 num_cams=5,
                 to_bev=False,
                 upsample=False,
                 backproj=None,
                 projview=None,
                 Backbone=None,
                 Neck=None,
                 DepthModule=None,
                 FusionMoudle=None,
                 BevFeatEncoder=None,
                 Head=None,
                 ssim_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.
        super().__init__()
        self.repeoject_loss_weight = repeoject_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.cam_num_use = num_cams
        self.load_multi_frame = load_multi_frame
        self.to_bev = to_bev
        self.upsample = upsample
        self.backbone = BACKBONES.build(Backbone)
        self.neck = MODELS.build(Neck)
        self.depth_net = MODELS.build(DepthModule)
        if FusionMoudle is not None:
            self.fusion_moudle = MODELS.build(FusionMoudle)
        if BevFeatEncoder is not None:
            self.bev_encoder = MODELS.build(BevFeatEncoder)
        if Head is not None:
            self.head = MODELS.build(Head)

        if backproj is not None:
            self.backproj = MODELS.build(backproj)
        if projview is not None:
            self.projview = MODELS.build(projview)
        self.ssim_loss = MODELS.build(ssim_loss)


    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def reproj_current_img(self, adj_frame_dict, depth_map, img_inputs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs
        frame_ids = list(adj_frame_dict.keys())
        img_backproj = dict()
        B, N, H, W = depth_map.size()
        depth_map = depth_map.view(B * N, H, W).unsqueeze(1)
        for frame in frame_ids:
            adj_imgs, adj_sensor2keyegos, adj_ego2globals, adj_intrins, adj_post_rots, adj_post_trans, adj_bda = \
            adj_frame_dict[frame]

            cam_points = self.backproj(depth_map, intrins, post_rot=post_rots, post_trans=post_trans)
            # 这个部分需要考虑是逐个视角投影还是想办法设计整体投影
            current_global_points = torch.matmul(torch.matmul(ego2globals, sensor2keyegos), cam_points)
            adj_cam_points = torch.matmul(torch.inverse(torch.matmul(adj_ego2globals, adj_sensor2keyegos)),
                                          current_global_points)
            proj_adj_imf_ref_points = self.projview(adj_cam_points, adj_intrins, adj_post_rots, adj_post_trans)

            B, N, H, W, coord = proj_adj_imf_ref_points.size()
            proj_adj_imf_ref_points = proj_adj_imf_ref_points.view(B * N, H, W, coord)
            B, N, C, H, W = adj_imgs.size()
            # print('size is: ', B, N, C, H, W)
            adj_imgs = adj_imgs.reshape(B*N, C, H, W)
            refined_color_map = F.grid_sample(adj_imgs, proj_adj_imf_ref_points, padding_mode="border",
                                              align_corners=True)
            refined_color_map = refined_color_map.view(B, N, C, H, W)
            img_backproj[frame] = refined_color_map

        return img_backproj

    def compute_losses(self, curr_imgs, adj_proj_imgs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        B, N, C, H, W = curr_imgs.size()
        imgs = curr_imgs.view(-1, C, H, W)
        target = imgs
        frame_ids = list(adj_proj_imgs.keys())

        loss_reproj = []
        for frame in frame_ids:
            pred = adj_proj_imgs[frame]
            pred = pred.view(-1, C, H, W)
            losses['loss_ssim_frame{}'.format(frame)] = self.compute_reprojection_loss(pred, target).mean()


        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      c=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs
                      ):
        '''
        Args:
            points:   use point cloud or not
            img_metas:   img size, aug elements
            gt_bboxes_3d:   object detection gt labels
            gt_labels_3d:   the classification og each object
            c:
            gt_bboxes:
            img_inputs:
                    imgs: Tensor   B, N, C, H, W
                    sensor2ego:   Tensor   B, N, 4, 4
                    ego2global:    Tensor   B, N, 4, 4
                    intrins(cam2imgs):   B, N, 3, 3
                    post_rot:   Tensor   B, N, 3, 3
                    post_trans:   Tensor   B, N, 3
                    bda:    Tensor   B, 3, 3
            proposals:
            gt_bboxes_ignore:
            **kwargs:

        Returns:

        '''
        # print('img metas is: ', len(img_metas), img_metas[0].keys(), img_metas[0]['cam_names'])
        gt_depth = kwargs['gt_depth']
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        # print('I want to get the kwargs keys: ', kwargs.keys(), type(kwargs['adj_img_inputs']), len(kwargs['adj_img_inputs']),
        #       kwargs['adj_img_inputs'][0].size())
        losses = dict()
        # init the model's input
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        adj_frame_dict = self.prepare_inputs_adj(kwargs['adj_img_inputs'])





        cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins[:, :, :3, :3], post_rots, post_trans, bda]

        # extract features from images
        img_feats = self.image_encoder(imgs)
        # print('img_feats isL: ', img_feats.size())
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(img_feats, mlp_input)  # 这里的context用来做语义分割
        else:
            context = None
            depth = None
        # print(depth.size(), '-----------')
        depth_convert = self.depth_net.generate_depth_map(depth)
        # print('depth_convert is: ', depth_convert.size())
        # lss depth fusion module
        B, N, H, W = depth_convert.size()
        depth_convert = depth_convert.view(B*N, H, W).unsqueeze(1)
        img_reproj = self.reproj_current_img(adj_frame_dict, depth_convert, img_inputs)
        # print('img_reproj keys is: ', img_reproj.keys(), img_reproj[-1].size(), img_reproj[1].size())
        #
        # save_tensors_to_npy(img_reproj[-1],
        #                     './outputs_debug/adj_adding/waymo_depth_refined-1')
        # save_tensors_to_npy(img_reproj[1],
        #                     './outputs_debug/adj_adding/waymo_depth_refined+1')


        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)

        losses_sel_supervise = self.compute_losses(imgs, img_reproj)

        losses.update(loss_depth)
        losses.update(losses_sel_supervise)
        losses = {}
        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):

        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs[0])


        cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins, post_rots, post_trans, bda]
        img_feats = self.image_encoder(imgs)
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(img_feats, mlp_input)
        else:
            context = None
            depth = None


        if self.with_specific_component('fusion_moudle'):
            bev_feat = self.fusion_moudle(cam_params, context, depth, **kwargs)  #  B, C, X, Y, Z
            if self.to_bev:
                bev_feat = torch.cat(bev_feat.unbind(dim=-1), -1)
        else:
            bev_feat = None

        # use backbone and neck extract feat from bev space
        if self.with_specific_component('bev_encoder'):
            bev_feat = self.bev_encoder(bev_feat)
        # gt_depth = kwargs['gt_depth']
        # voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        # mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        occ_outs = self.head(bev_feat)
        occ_preds = self.head.get_occ(occ_outs)
        # print('occ pred is: ', type(occ_preds), occ_preds[0].shape, len(occ_preds), occ_preds[0].max(), occ_preds[0].min())
        return occ_preds

    def prepare_inputs_adj(self, inputs):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        assert len(inputs) == 7
        assert self.load_multi_frame >= 1
        assert N / (self.cam_num_use*2) == self.load_multi_frame
        pre_load_index = list(range(-self.load_multi_frame, 0))
        next_load_index = list(range(1, self.load_multi_frame+1))
        load_list = pre_load_index + next_load_index
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs
        adj_frame_dict = dict()
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans = torch.chunk(imgs, 2*self.load_multi_frame, dim=1), \
            torch.chunk(sensor2egos, 2 * self.load_multi_frame, dim=1),  torch.chunk(ego2globals, 2*self.load_multi_frame, dim=1), \
            torch.chunk(intrins, 2 * self.load_multi_frame, dim=1),  torch.chunk(post_rots, 2*self.load_multi_frame, dim=1), \
            torch.chunk(post_trans, 2 * self.load_multi_frame, dim=1),
        for idx, i in enumerate(load_list):
            adj_frame_dict[i] = [imgs[idx], sensor2egos[idx], ego2globals[idx], intrins[idx], post_rots[idx],
                                 post_trans[idx], bda]
        return adj_frame_dict
        #
        # sensor2egos = sensor2egos.view(B, N, 4, 4)
        # ego2globals = ego2globals.view(B, N, 4, 4)
        #
        # # calculate the transformation from adj sensor to key ego
        # keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        # global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        # sensor2keyegos = \
        #     global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        # sensor2keyegos = sensor2keyegos.float()
        #
        # # print('in the training process: intrins is: ', intrins.size(), intrins.size()[-2:], len(intrins))
        # if len(intrins) == 4 and intrins.size()[-2:] == (4, 4):
        #     # If the shape is (B, N_views, 4, 4), slice the last two dimensions
        #     intrins = intrins[..., :3, :3]
        # return [imgs, sensor2keyegos, ego2globals, intrins,
        #         post_rots, post_trans, bda]

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        # print('in the training process: intrins is: ', intrins.size(), intrins.size()[-2:], len(intrins))
        if len(intrins) == 4 and intrins.size()[-2:] == (4, 4):
            # If the shape is (B, N_views, 4, 4), slice the last two dimensions
            intrins = intrins[..., :3, :3]
        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

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
        x = self.backbone(imgs)
        x = self.neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        # print('l1 loss is: ', l1_loss.size(), l1_loss.mean())
        # print('pred, target', pred.size(), target.size())
        ssim_loss = self.ssim_loss(pred, target).mean(1, True)
        # print(ssim_loss.size(), 'ssim_losses', ssim_loss.mean())
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

        # gt_depth = kwargs['gt_depth']
        # voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        # mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        #
        # # init moedel's inputs
        # imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        # cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins, post_rots, post_trans, bda]
        # # extract feats from imgs
        # img_feats = self.image_encoder(imgs)
        #
        # # estimation each view's depth
        # if self.with_specific_component('depth_net'):
        #     mlp_input = self.depth_net.get_mlp_input(*cam_params)
        #     context, depth = self.depth_net(img_feats, mlp_input)  # context is the img representation, depth is depth map
        # else:
        #     context = None
        #     depth = None
        #
        # if self.with_specific_component('fuse_net'):
        #     vox_feat = self.fuse_net(cam_params, context, depth, **kwargs)  #  B, C, X, Y, Z
        # else:
        #     vox_feat = None
        # # if we want to use attention deal with the bev_map depth_map and orginal_img_feat_map, the module should be inserted here
        #
        #
        # # in baseline we do not fuse history key frame, if we want to fuse the history key frame, it should insert here
        # if self.with_specific_component('vox_encoder'):
        #     vox_feat = self.vox_encoder(vox_feat)
        # outs = self.head(vox_feat)
        # occ_pred = outs['occ'][0]
        # loss_occ = self.head.loss(
        #     occ_pred,  # (B, Dx, Dy, Dz, n_cls)
        #     voxel_semantics,  # (B, Dx, Dy, Dz)
        #     mask_camera,  # (B, Dx, Dy, Dz)
        # )
        # loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
        # losses.update(loss_depth)
        # losses.update(loss_occ)
        # # time.sleep(0.05)
        # return losses