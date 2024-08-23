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

from projects.mmdet3d_plugin.baselines.utils.utils import save_tensors_to_npy, calculate_fov, generate_depth_map_from_points, generate_depth_map_from_points_inv

# def save_imgs(imgs, directory):
#     '''
#
#     Args:
#         tensor_list:   imgs (B, N, 3, H, Q)
#         directory:     save_path
#         save_depth:    depthmap  (B, N, H, W)
#         save_occ:      dict---->
#
#     Returns:
#
#     '''
#     # 确保目录存在
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     for idx, tensor in enumerate(tensor_list):
#         # 将张量从 GPU 迁移到 CPU
#         tensor_cpu = tensor.detach().cpu().numpy()
#         print('tensor is: ', tensor.size())
#         # 构建文件路径
#         file_path = os.path.join(directory, f'tensor_{idx}.npy')
#
#         # 保存为 .npy 文件
#         np.save(file_path, tensor_cpu)
#         print(f'Saved tensor {idx} to {file_path}')

@MODELS.register_module()
class DGA_OCC_PIPLINE(BaseModule):
    def __init__(self,
                 num_frame=1,
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

        self.num_frame = num_frame

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
        imgs, cam2ego, ego2global, cam2img, post_rots, post_trans, bda, curr2adjsensor = self.prepare_inputs(img_inputs,
                                                                                                             stereo=False)
        # lidar2global = kwargs['lidarego2global']
        # frmae_id = 0
        # # cac_lidar2cam = torch.matmul(torch.inverse(torch.matmul(ego2global[frmae_id], cam2ego[frmae_id])), lidar2global)
        # n = self.num_frame // 2
        num_frame_id = [0, -1, +1]
        # print('num_frame_if is: ', num_frame_id)
        train_inputs = {
            f't-{i}': {
                'imgs': imgs[idx],
                'cam2ego': cam2ego[idx],
                'ego2global': ego2global[idx],
                'cam2img': cam2img[idx],
                'post_rots': post_rots[idx],
                'post_trans': post_trans[idx],
            }
            for idx, i in enumerate(num_frame_id)
        }

        cam_params = [train_inputs['t-0']['cam2ego'][:, :, :3, :3], train_inputs['t-0']['cam2ego'][:, :, :3, 3], train_inputs['t-0']['cam2img'][:, :, :3, :3],
                      train_inputs['t-0']['post_rots'], train_inputs['t-0']['post_trans'], bda]

        gt_depth = kwargs['gt_depth']
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        img_feats = self.image_encoder(train_inputs['t-0']['imgs'])


        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(img_feats, mlp_input)
        else:
            context = None
            depth = None
        depth_map, outputs = self.depth_net.get_self_supervised_depth_loss(depth, frame_train_inputs=train_inputs, depth_map_gt=None)
        print('the genrate depthmap is: ', depth_map.size(), outputs.keys())   # 2, 6, 1, 256, 704
        print('the gt depth is: ', gt_depth.size())
        print('context {} and depth {}'.format(context.size(), depth.size()))



        # 此处我们准备三种不同形式的深度估计网络，
        # （1）首先是完全自监督的形式，不使用gt标签，但是与后面的occ标签建立联系
        # （2）之后是使用与之前类似的考虑cpm的网络
        # （3）在之后是类似于vfdepth的网络
        # 我们首要的创新点是增加融合模块的泛化能力，
        # 我们的次要创新点是进行视角渲染，二者都建立在nerf学习之上，所以我们应该先学习nerf

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

    def prepare_inputs(self, img_inputs, stereo=False):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1) so what's the real sort of this  ---- multi_frame_id_cfg  (which we can find it in configd file)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            stereo: bool

        Returns:
            imgs: List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]       len = N_frames
            sensor2keyegos: List[(B, N_views, 4, 4), (B, N_views, 4, 4), ...]
            ego2globals: List[(B, N_views, 4, 4), (B, N_views, 4, 4), ...]
            intrins: List[(B, N_views, 3, 3), (B, N_views, 3, 3), ...]
            post_rots: List[(B, N_views, 3, 3), (B, N_views, 3, 3), ...]
            post_trans: List[(B, N_views, 3), (B, N_views, 3), ...]
            bda: (B, 3, 3)
        """
        B, N, C, H, W = img_inputs[0].shape
        N = N // self.num_frame     # N_views = 6
        imgs = img_inputs[0].view(B, N, self.num_frame, C, H, W)    # (B, N_views, N_frames, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]     # List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            img_inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sensor to key ego
        # key_ego --> global  (B, 1, 1, 4, 4)
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        # global --> key_ego  (B, 1, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())
        # sensor --> ego --> global --> key_ego
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_frames, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None


        extra = [
            sensor2keyegos,     # (B, N_frames, N_views, 4, 4)
            ego2globals,        # (B, N_frames, N_views, 4, 4)
            intrins.view(B, self.num_frame, N, 3, 3),   # (B, N_frames, N_views, 3, 3)
            post_rots.view(B, self.num_frame, N, 3, 3),     # (B, N_frames, N_views, 3, 3)
            post_trans.view(B, self.num_frame, N, 3)        # (B, N_frames, N_views, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        if len(intrins) == 4 and intrins.size()[-2:] == (4, 4):
            # If the shape is (B, N_views, 4, 4), slice the last two dimensions
            intrins = intrins[..., :3, :3]
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

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
