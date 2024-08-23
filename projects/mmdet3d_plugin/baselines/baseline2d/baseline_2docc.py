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
class OCC2dDepthBaseline(BaseModule):
    def __init__(self,
                 grid_config=None,
                 to_bev=False,
                 upsample=False,
                 Backbone=None,
                 Neck=None,
                 DepthModule=None,
                 FusionMoudle=None,
                 BevFeatEncoder=None,
                 Head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.
        super().__init__()
        self.grid_config = grid_config
        self.to_bev = to_bev
        self.upsample = upsample
        self.backbone = BACKBONES.build(Backbone)
        self.neck = MODELS.build(Neck)
        self.depth_net = MODELS.build(DepthModule)
        self.fusion_moudle = MODELS.build(FusionMoudle)
        self.bev_encoder = MODELS.build(BevFeatEncoder)
        self.head = MODELS.build(Head)

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

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
        losses = dict()
        # init the model's input
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)

        # a = (1600, 900), (1600, 900), (1600, 900), (1600, 900), (1600, 900), (1600, 900)
        # a = (1920, 1280), (1920, 1280), (1920, 1280), (1920, 886), (1920, 886)
        #
        # print('intrins ', intrins)
        # # print([calculate_fov(a[i], intrins[0, i, :3, :3]) for i in range(intrins.size()[1])])
        # print('imgs {}, sensor2keyegos {}, ego2globals {}, intrins {}, post_rots {}, post_trans {}, bda {}'.format(
        #     imgs.size(), sensor2keyegos.size(), ego2globals.size(), intrins.size(), post_rots.size(), post_trans.size(), bda.size()
        # ), kwargs['voxel_semantics'].size())
        # print('img_metas is: ', img_metas)
        # print('occ_label have unique: ', torch.unique(kwargs['voxel_semantics'], return_counts=True))
        cam_params = [sensor2keyegos[:, :, :3, :3], sensor2keyegos[:, :, :3, 3], intrins[:, :, :3, :3], post_rots, post_trans, bda]

        # extract features from images
        img_feats = self.image_encoder(imgs)

        # print('img_feats is: ', img_feats.size())

        # pred depth map
        if self.with_specific_component('depth_net'):
            # print('use depth net here4 +++++++++++++++++++++++++')
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            # print('++++++++++++++mlp_inpuy: ', type(mlp_input), mlp_input.size())
            context, depth = self.depth_net(img_feats, mlp_input)
            # print('++++++++++++++depthnet output: ', type(context), type(depth), context.size(), depth.size())
            # return_map['depth'] = depth
            # return_map['context'] = context
        else:
            context = None
            depth = None

        # lss depth fusion module
        if self.with_specific_component('fusion_moudle'):
            bev_feat = self.fusion_moudle(cam_params, context, depth, **kwargs)  #  B, C, X, Y, Z
            # print('******************bev feat: ', bev_feat.size())
            if self.to_bev:
                bev_feat = torch.cat(bev_feat.unbind(dim=-1), -1)
                # print('concat bev feat is: 0, ', bev_feat.size())
            # return_map['cam_params'] = cam_params
        else:
            bev_feat = None

        # use backbone and neck extract feat from bev space
        if self.with_specific_component('bev_encoder'):
            bev_feat = self.bev_encoder(bev_feat)
            # print('debug the depth feats type is {}， and size is: {}'.format(type(bev_feat), bev_feat.size()))
        gt_depth = kwargs['gt_depth']
        # print('imgs shape is: {}, and gt depth size is: {}'.format(imgs.size(), gt_depth.size()))
        # print('')


        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        # print('occ_label dtype is: {}, mask_camera dtype is: {}, and value num in occ_label is: {}'.format(
        #     voxel_semantics.dtype, mask_camera.dtype, torch.unique(voxel_semantics, return_counts=True)
        # ), kwargs.keys())
        # save_tensors_to_npy(imgs,
        #                     './outputs_debug/waymo_depth',
        #                     save_depth=gt_depth,
        #                     save_occ=kwargs)
        # print('bev feat is: {}, gt depth is: {}, voxel_semantics is {}, mask_camera is: {}'.format(bev_feat.size(),
        #         gt_depth.size(), voxel_semantics.size(), mask_camera.size()))
        # print(bev_feat.size())
        occ_outs = self.head(bev_feat)
        # print('occ_outs: ', occ_outs.size())

        # print('gt_depth {}, depth {}'.format(gt_depth.size(), depth.size()))
        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)

        # print('gt_depth {}, depth {}'.format(gt_depth.size(), depth.size()))
        # a = self.convert_softmax_depth2sigmoid(depth)
        # print('losses depth is: ', loss_depth)
        # print('occ pred size is: ', occ_outs.size())
        # print('label has: ',  torch.unique(voxel_semantics, return_counts=True))
        loss_occ = self.head.loss(
            occ_outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
            # mask_camera,  # (B, Dx, Dy, Dz)
        )
        losses.update(loss_depth)
        losses.update(loss_occ)

        print('out_pred is: ', occ_outs.size())
        occ_pred = occ_outs
        pred_depth = self.depth_net.convert_depth_predictions(depth)
        converted_pred_depth = self.depth_net.interpolatr_depthmap(pred_depth)
        occ_pred_save = occ_pred.softmax(-1).argmax(-1)
        print('imgs {}, converted_pred_depth {}, mask_camera {}, voxel_semantics {}, gt_depth {}, occ_pred_save {}'.format(
            imgs.size(), converted_pred_depth.size(), mask_camera.size(), voxel_semantics.size(), gt_depth.size(), occ_pred_save.size()
        ))

        # save_semantic_depth_img_mask_occ(imgs, converted_pred_depth, mask_camera,
        #                                  voxel_semantics, gt_depth, gt_sem_map=None, sem_pred=None,
        #                                  occ_pred=occ_pred_save,
        #                                  out_dir='outputs_debug/vis_appendix/waymo2nuscenes/Flashocc/', forward_save=True)

        # save_semantic_depth_img_mask_occ(imgs, converted_pred_depth, mask_camera,
        #                                  voxel_semantics, gt_depth, gt_sem_map=None, sem_pred=None,
        #                                  occ_pred=occ_pred_save,
        #                                  out_dir='outputs_debug/vis_appendix/nuscenes2nwaymo/Flashocc/', forward_save=True)


        # losses = {}
        # losses = {}
        # losses = {}
        # print('loss occ is: {}'.format(loss_occ))
        # save_tensors_to_npy(imgs,
        #                     './outputs_debug/occ_and_depth_vis/1_occ',
        #                     save_depth=gt_depth,
        #                     save_occ=kwargs,
        #                     occ_pred=occ_outs)
        # print('imgs size is: ', imgs.size())
        # print('after all debug is: {} +++++++++++++++++++++++++++', kwargs.keys(), type(img_metas), len(img_metas), img_metas)
        # losses = {}
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

        # print('after backbone is: ', [i.size() for i in x])
        # stereo_feat = None
        # if stereo:
        #     stereo_feat = x[0]
        #     x = x[1:]
        # if self.with_img_neck:
        x = self.neck(x)
        # print('after neck is: {}'.format([i.size() for i in x]))
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        # print(
        #     f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB, "
        #     f"Memory Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB"
        #     if torch.cuda.is_available() else "CUDA is not available")
        return x

    def convert_softmax_depth2sigmoid(self, depthmap):
        '''

        Args:
            depthmap: size--> (B, N, D, H, W)

        Returns:
        '''
        depth_levels = torch.arange(self.grid_config['depth'][0],
                                    self.grid_config['depth'][1],
                                    self.grid_config['depth'][2], device=depthmap.device)
        print('depth_level size is: ', depth_levels.size())
        depth_levels = depth_levels.view(1, 1, -1, 1, 1)
        print('101020`30" ', depth_levels.size())
        expected_depth = (depthmap * depth_levels).sum(dim=2)
        print('expecr_depth is: ', expected_depth.size())
        return expected_depth
#
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
