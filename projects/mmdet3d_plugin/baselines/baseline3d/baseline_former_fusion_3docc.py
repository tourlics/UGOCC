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


@MODELS.register_module()
class OCC3dTransformerFusionBaseline(BaseModule):
    def __init__(self,
                 to_bev=False,
                 Backbone=None,
                 Neck=None,
                 DepthModule=None,
                 FusionMoudle=None,
                 AtentionModule=None,
                 VoxFeatEncoder=None,
                 Head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super().__init__()
        print('=============================', Backbone, DepthModule)
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
        此处我们稍微修改一下结构，让我们能够输出不同层级的融合结果
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
        else:
            vox_feat = None
        print('vox size orgin is: {}'.format(vox_feat.size()))
        # print('after the fuse moudle vox feat size is: ', vox_feat.size(), 'context is: ', context.size())
        # if we want to use attention deal with the bev_map depth_map and orginal_img_feat_map, the module should be inserted here
        # bev_feat = vox_feat.mean(-1)
        # print('after the fuse moudle bev feat size is: ', bev_feat.size())
        # print('type of the vox_feat is: ', type(vox_feat), vox_feat.size(), 'bev feat size is: ', bev_feat.size(),
        #       'depth map: ', depth.size())
        if self.with_specific_component('attention_fusion'):
            bev_feat = self.attention_fusion([context],
                                        # img_metas,
                                        lss_vox=None,
                                        cam_params=cam_params,
                                        bev_mask=None,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth
                                        )
            # print('atention fusion bev is: ', bev_feat.size())
            vox_feat = bev_feat
        print('vox_feat size is: {}, context size is: {}, depth size is: {}'.format(vox_feat.size(), context.size(), depth.size()))
        # print('type of the vox_fea222t is: ', type(vox_feat), vox_feat.size(), 'bev feat size is: ', bev_feat.size(),
        #       'depth map: ', depth.size())
        # in baseline we do not fuse history key frame, if we want to fuse the history key frame, it should insert here
        if self.with_specific_component('vox_encoder'):
            vox_feat = self.vox_encoder(vox_feat)
        outs = self.head(vox_feat)
        occ_pred = outs['occ'][0]
        loss_occ = self.head.loss(
            occ_pred,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        print('depth size is: ', depth.size(), gt_depth.size())
        loss_depth = self.depth_net.get_depth_loss(gt_depth, depth)
        losses.update(loss_depth)
        losses.update(loss_occ)

        losses = {}
        # time.sleep(0.05)
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
        print('img_neck type is: ', type(x), [i.size() for i in x])
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