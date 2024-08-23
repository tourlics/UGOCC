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
import os
import numpy as np

@MODELS.register_module()
class OCC3dDepthWithTransformerBaseline(BaseModule):
    def __init__(self,
                 to_bev=False,
                 Backbone=None,
                 Neck=None,
                 DepthModule=None,
                 FusionMoudle=None,
                 AtentionModule=None,
                 AtentionavModule=None,
                 VoxFeatEncoder=None,
                 Head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super().__init__()
        self.to_bev = to_bev
        self.backbone = BACKBONES.build(Backbone)
        self.neck = MODELS.build(Neck)
        self.depth_net = MODELS.build(DepthModule)
        self.fuse_net = MODELS.build(FusionMoudle)
        if VoxFeatEncoder is not None:
            self.vox_encoder = MODELS.build(VoxFeatEncoder)
        else:
            self.vox_encoder = None

        if AtentionModule is not None:
            self.attention_fusion = MODELS.build(AtentionModule)

        if AtentionavModule is not None:
            self.attention_fusion_av = MODELS.build(AtentionavModule)

        self.head = MODELS.build(Head)

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

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
        losses = dict()
        '''
        inputs: img_inputs list(list--length=7)
                gt_depth:  Tensor [B, N, H, W]
                voxel_semantics:   Tensor [B, X, Y, Z]
                mask_camera:    Tensor [B, X, Y, Z]
        '''
        gt_depth = kwargs['gt_depth']                   # [B, N, H, W]
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']             # (B, Dx, Dy, Dz)

        # init moedel's inputs
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins, post_rots, post_trans, bda]
        # extract feats from imgs
        img_feats = self.image_encoder(imgs)

        # estimation each view's depth
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(img_feats, mlp_input)  # context is the img representation, depth is depth map
        else:
            context = None
            depth = None

        if self.with_specific_component('fuse_net'):
            vox_feat = self.fuse_net(cam_params, context, depth, **kwargs)  #  B, C, X, Y, Z
            # print('3d first feat size is: ', vox_feat.size())
        else:
            vox_feat = None
        # print('after the fuse moudle vox feat size is: ', vox_feat.size(), 'context is: ', context.size())
        # if we want to use attention deal with the bev_map depth_map and orginal_img_feat_map, the module should be inserted here
        bev_feat = vox_feat.mean(-1)
        # print('after the fuse moudle bev feat size is: ', bev_feat.size())
        # print('type of the vox_feat is: ', type(vox_feat), vox_feat.size(), 'bev feat size is: ', bev_feat.size(),
        #       'depth map: ', depth.size())
        if self.with_specific_component('attention_fusion'):
            bev_feat = self.attention_fusion([context],
                                        img_metas,
                                        lss_bev=bev_feat.permute(0, 1, 3, 2),
                                        cam_params=cam_params,
                                        bev_mask=None,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth
                                        )
            bev_feat = bev_feat.permute(0, 1, 3, 2)   # (B, C, Y, X)  --> (B, C, X, Y)
            # print('3d second feat size is: ', bev_feat.size())
            # print('atention fusion bev is: ', bev_feat.size())
            vox_feat = vox_feat + bev_feat[..., None]

        if self.with_specific_component('attention_fusion_av'):
            # print('process AtentionavModule')
            # print(self.attention_fusion_av)
            bev_feat = self.attention_fusion_av([context],
                                        img_metas,
                                        lss_bev=bev_feat,
                                        cam_params=cam_params,
                                        bev_mask=None,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth
                                        )
            bev_feat = bev_feat  # (B, C, Y, X)  --> (B, C, X, Y)
            # print('3d second feat size is: ', bev_feat.size())
            # print('atention fusion bev is: ', bev_feat.size())
            vox_feat = vox_feat + bev_feat[..., None]


        # print('type of the vox_fea222t is: ', type(vox_feat), vox_feat.size(), 'bev feat size is: ', bev_feat.size(),
        #       'depth map: ', depth.size())
        # in baseline we do not fuse history key frame, if we want to fuse the history key frame, it should insert here
        if self.with_specific_component('vox_encoder'):
            vox_feat = self.vox_encoder(vox_feat)
        outs = self.head(vox_feat)
        occ_pred = outs['occ'][0]

        # print('size of occ_pred is: {}, size of oco_gt is: {}'.format(occ_pred.softmax(-1).argmax(-1).size(), voxel_semantics.size()))
        #
        # save_semantic_depth_img_mask_occ(None, None, None, mask_camera,
        #                                  voxel_semantics, gt_depth, sem_pred=None, occ_pred=occ_pred.softmax(-1).argmax(-1),
        #                                  out_dir='outputs_debug/nusc_occ', forward_save=True)

        loss_occ = self.head.loss(
            occ_pred,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
        losses.update(loss_depth)
        losses.update(loss_occ)
        # time.sleep(0.05)
        # losses = {}

        pred_depth = self.depth_net.convert_depth_predictions(depth)
        converted_pred_depth = self.depth_net.interpolatr_depthmap(pred_depth)
        occ_pred_save = occ_pred.softmax(-1).argmax(-1)

        print('imgs {}, converted_pred_depth {}, mask_camera {}, voxel_semantics {}, gt_depth {}, occ_pred_save {}'.format(
            imgs.size(), converted_pred_depth.size(), mask_camera.size(), voxel_semantics.size(), gt_depth.size(), occ_pred_save.size()
        ))

        # save_semantic_depth_img_mask_occ(imgs, converted_pred_depth, mask_camera,
        #                                  voxel_semantics, gt_depth, gt_sem_map=None, sem_pred=None,
        #                                  occ_pred=occ_pred_save,
        #                                  out_dir='outputs_debug/vis_appendix/waymo2nuscenes/FBOCC/', forward_save=True)

        # save_semantic_depth_img_mask_occ(imgs, converted_pred_depth, mask_camera,
        #                          voxel_semantics, gt_depth, gt_sem_map=None, sem_pred=None,
        #                          occ_pred=occ_pred_save,
        #                          out_dir='outputs_debug/vis_appendix/vis_as_video/FBOCC_waymo2nusc/', forward_save=True)

        save_semantic_depth_img_mask_occ(imgs, converted_pred_depth, mask_camera,
                                         voxel_semantics, gt_depth, gt_sem_map=None, sem_pred=None,
                                         occ_pred=occ_pred_save,
                                         out_dir='outputs_debug/vis_appendix/vis_as_video/FBOCC_nusc2waymo/',
                                         forward_save=True)
        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs[0])
        cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins, post_rots, post_trans, bda]
        img_feats = self.image_encoder(imgs)

        # estimation each view's depth
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(img_feats,
                                            mlp_input)  # context is the img representation, depth is depth map
        else:
            context = None
            depth = None

        if self.with_specific_component('fuse_net'):
            vox_feat = self.fuse_net(cam_params, context, depth, **kwargs)  # B, C, X, Y, Z
        else:
            vox_feat = None
        # if we want to use attention deal with the bev_map depth_map and orginal_img_feat_map, the module should be inserted here
        bev_feat = vox_feat.mean(-1)
        if self.with_specific_component('attention_fusion'):
            bev_feat = self.attention_fusion([context],
                                             img_metas,
                                             lss_bev=bev_feat.permute(0, 1, 3, 2),
                                             cam_params=cam_params,
                                             bev_mask=None,
                                             gt_bboxes_3d=None,  # debug
                                             pred_img_depth=depth
                                             )
            bev_feat = bev_feat.permute(0, 1, 3, 2)  # (B, C, Y, X)  --> (B, C, X, Y)
            # print('3d second feat size is: ', bev_feat.size())
            # print('atention fusion bev is: ', bev_feat.size())
            vox_feat = vox_feat + bev_feat[..., None]

        if self.with_specific_component('attention_fusion_av'):

            bev_feat = self.attention_fusion_av([context],
                                        img_metas,
                                        lss_bev=bev_feat,
                                        cam_params=cam_params,
                                        bev_mask=None,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth
                                        )
            bev_feat = bev_feat   # (B, C, Y, X)  --> (B, C, X, Y)
            # print('3d second feat size is: ', bev_feat.size())
            # print('atention fusion bev is: ', bev_feat.size())
            vox_feat = vox_feat + bev_feat[..., None]

        # in baseline we do not fuse history key frame, if we want to fuse the history key frame, it should insert here
        if self.with_specific_component('vox_encoder'):
            vox_feat = self.vox_encoder(vox_feat)
        outs = self.head(vox_feat)
        occ_pred = outs['occ'][0]
        occ_preds = self.head.get_occ(occ_pred)
        # print('occ pred is: ', type(occ_preds), occ_preds[0].shape)
        return occ_preds

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
        # stereo_feat = None
        # if stereo:
        #     stereo_feat = x[0]
        #     x = x[1:]
        # if self.with_img_neck:
        x = self.neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

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

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]


def save_semantic_depth_img_mask_occ(img_map, pred_depth_map, mask_camera,
                                     occ_label, gt_depth, gt_sem_map=None, sem_pred=None, occ_pred=None,
                                     out_dir=None, render_sem_map=None, render_depth_map=None, forward_save=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if forward_save:
        existing_folders = [name for name in os.listdir(out_dir) if
                            os.path.isdir(os.path.join(out_dir, name)) and name.startswith('t')]
        existing_numbers = [int(folder[1:]) for folder in existing_folders if folder[1:].isdigit()]

        if existing_numbers:
            next_number = max(existing_numbers) + 1
        else:
            next_number = 1

        new_folder_name = f't{next_number}'
        new_folder_path = os.path.join(out_dir, new_folder_name)

        os.makedirs(new_folder_path, exist_ok=True)
        out_dir = new_folder_path
    print('save to out_dir', out_dir)

    if img_map is not None:
        img_map = img_map.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'img_map.npy')
        np.save(file_path, img_map)

    if render_sem_map is not None:
        render_sem_map = render_sem_map.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'render_sem_map.npy')
        np.save(file_path, render_sem_map)

    if render_depth_map is not None:
        render_depth_map = render_depth_map.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'render_depth_map.npy')
        np.save(file_path, render_depth_map)

    if pred_depth_map is not None:
        depth_map = pred_depth_map.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'depth_map.npy')
        np.save(file_path, depth_map)

    if gt_depth is not None:
        gt_depth = gt_depth.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'gt_depth_map.npy')
        np.save(file_path, gt_depth)

    if gt_sem_map is not None:
        gt_sem_map = gt_sem_map.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'gt_sem_map.npy')
        np.save(file_path, gt_sem_map)

    if mask_camera is not None:
        mask_camera = mask_camera.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'mask_camera_map.npy')
        np.save(file_path, mask_camera)

    if occ_label is not None:
        occ_label = occ_label.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'occ_label_map.npy')
        np.save(file_path, occ_label)

    if occ_pred is not None:
        occ_pred = occ_pred.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'occ_pred_map.npy')
        np.save(file_path, occ_pred)

    if sem_pred is not None:
        sem_pred = sem_pred.detach().cpu().numpy()
        file_path = os.path.join(out_dir, f'sem_pred_map.npy')
        np.save(file_path, sem_pred)

