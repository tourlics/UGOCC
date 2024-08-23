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
import torch.nn as nn



def save_tensors_to_npy(tensor_list, directory, save_depth=None, save_occ=None):
    '''

    Args:
        tensor_list:   imgs (B, N, 3, H, Q)
        directory:     save_path
        save_depth:    depthmap  (B, N, H, W)
        save_occ:      dict---->

    Returns:

    '''
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)

    if save_depth is not None:
        depth_cpu = save_depth.detach().cpu().numpy()
        file_path = os.path.join(directory, f'depth_tensor_{0}.npy')
        np.save(file_path, depth_cpu)

    if save_occ is not None:
        occ_label_cpu = save_occ['voxel_semantics']   #
        mask_camera_cpu = save_occ['mask_camera']
        occ_label_cpu = occ_label_cpu.detach().cpu().numpy()
        file_path = os.path.join(directory, f'occ_label_cpu_tensor_{0}.npy')
        np.save(file_path, occ_label_cpu)
        mask_camera_cpu = mask_camera_cpu.detach().cpu().numpy()
        file_path = os.path.join(directory, f'mask_camera_cpu_cpu_tensor_{0}.npy')
        np.save(file_path, mask_camera_cpu)
    for idx, tensor in enumerate(tensor_list):
        # 将张量从 GPU 迁移到 CPU
        tensor_cpu = tensor.detach().cpu().numpy()

        # 构建文件路径
        file_path = os.path.join(directory, f'tensor_{idx}.npy')

        # 保存为 .npy 文件
        np.save(file_path, tensor_cpu)
        print(f'Saved tensor {idx} to {file_path}')

@MODELS.register_module()
class Self_Supervise_depth_baseline(BaseModule):
    def __init__(self,
                 to_bev=False,
                 num_adj=1,
                 use_depth_gt_loss=None,
                 repeoject_loss_weight=None,
                 smooth_loss_weight=None,
                 gt_depth_loss=None,
                 Backbone=None,
                 Depthnet=None,
                 Posenet=None,
                 ssim_loss=None,
                 predict_scales=None,
                 train_cfg=None,
                 test_cfg=None,
                 self_supervised_train=False,
                 depth_scale=None,
                 **kwargs
                 ):
        super().__init__()
        self.predict_scales = predict_scales
        self.to_bev = to_bev
        self.num_frame = num_adj + 1

        self.depth_scale = depth_scale
        self.use_depth_gt_loss = use_depth_gt_loss
        self.repeoject_loss_weight = repeoject_loss_weight,
        self.smooth_loss_weight = smooth_loss_weight,
        self.gt_depth_loss = gt_depth_loss,
        self.self_supervised_train = self_supervised_train
        if Backbone is not None:
            self.backbone = BACKBONES.build(Backbone)
        if Depthnet is not None:
            self.depthnet = MODELS.build(Depthnet)
        self.ssim_loss = MODELS.build(ssim_loss)

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
        # todo 20240613 ----
        losses = dict()
        '''
        inputs: img_inputs list(list--length=7)
                gt_depth:  Tensor [B, N, H, W]
                voxel_semantics:   Tensor [B, X, Y, Z]
                mask_camera:    Tensor [B, X, Y, Z]
        '''
        # print('type of the img_input is {}, and the length is: {}'.format(type(img_inputs), len(img_inputs)), len(img_inputs[0]))
        imgs, cam2ego, ego2global, cam2img, post_rots, post_trans, bda, curr2adjsensor = self.prepare_inputs(img_inputs, stereo=False)

        if self.self_supervised_train:
            num_frames = len(imgs)
            train_inputs = {
                f't-{i}': {
                    'imgs': imgs[i],
                    'cam2ego': cam2ego[i],
                    'ego2global': ego2global[i],
                    'cam2img': cam2img[i],
                    'post_rots': post_rots[i],
                    'post_trans': post_trans[i],
                }
                for i in range(num_frames)
            }
        else:
            train_inputs = None
        cam_params = train_inputs

        # 对于自监督深度估计的训练，我们选取t-1时刻的原始图像作为输入，原因是samples连续，同时经过可视化后能看出重合范围更加优越
        outputs = self.depthnet(imgs[1], cam_params=cam_params, depthmap=kwargs['gt_depth_frame_t-1'])
        # outputs = self.depthnet(imgs[1], cam_params=cam_params, depthmap=None)
        # save_tensors_to_npy(cam_params['t-1']['imgs'].unsqueeze(0),
        #                     './outputs_debug/temporal_imgst{}2t{}'.format('t-1', 't-1'),
        #                     save_depth=kwargs['gt_depth_frame_t-1'])


        # print('final outputs keys is: {}'.format(outputs.keys()), [(outputs[i].size(), i) for i in outputs.keys()])
        # print('depth is :', outputs[('depth', 0, 0)].size())
        # print(cam_params['t-1']['imgs'].size())
        losses = self.compute_losses(cam_params['t-1']['imgs'], outputs, ref_frame_id=['t-0', 't-2'], gt_depth=kwargs['gt_depth_frame_t-1'])
        print('pred depth size is: {}, and gt depth size is: {}'.format(outputs[('depth', 0, 0)].size(), kwargs['gt_depth_frame_t-1'].size()))
        # print(losses)
        # losses = {}
        # device = torch.device("cuda")
        # total_memory = torch.cuda.get_device_properties(device).total_memory
        # print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印已分配的显存
        # allocated_memory = torch.cuda.memory_allocated(device)
        # print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印已保留的显存
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印显存使用摘要
        # pred_depth = outputs[('depth', 0, 0)]
        # gt_depth = kwargs['gt_depth_frame_t-1']
        # B, N, H, W = gt_depth.size()
        # pred_depth = pred_depth.view(-1, N, H, W)
        # pred_depth = list(pred_depth)
        # gt_depth = list(gt_depth)
        # result = [(pred_depth[i], gt_depth[i]) for i in range(len(pred_depth))]
        # print('results: ', [(i[0].size(), i[1].size()) for i in result])
        # losses = {}
        # todo: 首先可视化深度图，然后将其使用硬编码的方式投影到相邻帧，检查一下深度图对不对
        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        # print('l1 loss is: ', l1_loss.size(), l1_loss.mean())

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

    def forward_test(self,
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
        # todo 20240613 ----
        losses = dict()
        '''
        inputs: img_inputs list(list--length=7)
                gt_depth:  Tensor [B, N, H, W]
                voxel_semantics:   Tensor [B, X, Y, Z]
                mask_camera:    Tensor [B, X, Y, Z]
        '''
        # print('type of the img_input is {}, and the length is: {}'.format(type(img_inputs), len(img_inputs)), len(img_inputs[0]))
        imgs, cam2ego, ego2global, cam2img, post_rots, post_trans, bda, curr2adjsensor = self.prepare_inputs(img_inputs, stereo=False)

        if self.self_supervised_train:
            num_frames = len(imgs)
            train_inputs = {
                f't-{i}': {
                    'imgs': imgs[i],
                    'cam2ego': cam2ego[i],
                    'ego2global': ego2global[i],
                    'cam2img': cam2img[i],
                    'post_rots': post_rots[i],
                    'post_trans': post_trans[i],
                }
                for i in range(num_frames)
            }
        else:
            train_inputs = None
        cam_params = train_inputs

        # 对于自监督深度估计的训练，我们选取t-1时刻的原始图像作为输入，原因是samples连续，同时经过可视化后能看出重合范围更加优越
        # outputs = self.depthnet(imgs[1], cam_params=cam_params, depthmap=kwargs['gt_depth_frame_t-1'])
        outputs = self.depthnet(imgs[1], cam_params=cam_params, depthmap=None)
        # save_tensors_to_npy(cam_params['t-1']['imgs'].unsqueeze(0),
        #                     './outputs_debug/temporal_imgst{}2t{}'.format('t-1', 't-1'),
        #                     save_depth=outputs[('depth', 0, 0)].view(1, 6, 256, 704))


        # print('final outputs keys is: {}'.format(outputs.keys()), [(outputs[i].size(), i) for i in outputs.keys()])
        # print('depth is :', outputs[('depth', 0, 0)].size())
        # print(cam_params['t-1']['imgs'].size())
        # losses = self.compute_losses(cam_params['t-1']['imgs'], outputs, ref_frame_id=['t-0', 't-2'], gt_depth=kwargs['gt_depth_frame_t-1'])
        # print('pred depth size is: {}, and gt depth size is: {}'.format(outputs[('depth', 0, 0)].size(), kwargs['gt_depth_frame_t-1'].size()))
        # print(losses)
        # losses = {}
        # device = torch.device("cuda")
        # total_memory = torch.cuda.get_device_properties(device).total_memory
        # print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印已分配的显存
        # allocated_memory = torch.cuda.memory_allocated(device)
        # print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印已保留的显存
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
        #
        # # 打印显存使用摘要
        # print(torch.cuda.memory_summary(device=device, abbreviated=True))
        #
        # save_tensors_to_npy(cam_params['t-1']['imgs'].unsqueeze(0),
        #                     './outputs_debug/temporal_imgst{}2t{}'.format('t-1', 't-1'),
        #                     save_depth=outputs[('depth', 0, 0)].view(1, 6, 256, 704))

        pred_depth = outputs[('depth', 0, 0)]
        gt_depth = kwargs['gt_depth_frame_t-1']
        B, N, H, W = gt_depth.size()
        pred_depth = pred_depth.view(-1, N, H, W)
        pred_depth = list(pred_depth)
        gt_depth = list(gt_depth)
        result = [[pred_depth[i].cpu().numpy(), gt_depth[i].cpu().numpy()] for i in range(len(pred_depth))]
        # print('start processing test:', 'results: ', [(i[0].shape, i[1].shape) for i in result])
        # todo: 首先可视化深度图，然后将其使用硬编码的方式投影到相邻帧，检查一下深度图对不对
        return result


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

    def prepare_inputs(self, img_inputs, stereo=False):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1) so what's the real sort of this
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

        # --------------------  for stereo --------------------------
        curr2adjsensor = None
        if stereo:
            # (B, N_frames, N_views, 4, 4),  (B, N_frames, N_views, 4, 4)
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()   # (B, N_temporal=2, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()   # (B, N_temporal=2, N_views, 4, 4)
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()    # (B, N_temporal=2, N_views, 4, 4)
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()    # (B, N_temporal=2, N_views, 4, 4)

            # curr_sensor --> curr_ego --> global --> prev_ego --> prev_sensor
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr       # (B, N_temporal=2, N_views, 4, 4)
            curr2adjsensor = curr2adjsensor.float()         # (B, N_temporal=2, N_views, 4, 4)
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            # curr2adjsensor: List[(B, N_views, 4, 4), (B, N_views, 4, 4), None]
            assert len(curr2adjsensor) == self.num_frame
        # --------------------  for stereo --------------------------

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

    def compute_losses(self, inputs, outputs, ref_frame_id=['t-0', 't-2'], gt_depth=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        B, N, C, H, W = inputs.size()
        imgs = inputs.view(-1, C, H, W)
        for scale in self.predict_scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            # print('scales is: {}, color size is: {}'.format(scale, color.size()))
            target = imgs

            for frame_id in ref_frame_id:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            reprojection_loss = reprojection_losses

            combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)



            losses['reprojection_loss{}'.format(scale)] = self.repeoject_loss_weight[0] * to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, imgs)

            losses['get_smooth_loss{}'.format(scale)] = self.smooth_loss_weight[0] * smooth_loss
            # losses = {}
            # loss += 0.01 * smooth_loss / (2 ** scale)
            # total_loss += loss
            # losses["loss/{}".format(scale)] = loss

        if self.use_depth_gt_loss:
            B, N, H, W = gt_depth.size()
            pred_depth = outputs[('depth', 0, 0)].unsqueeze(1).view(B, N, H, W)
            mask = torch.logical_and(gt_depth > self.depth_scale[0], gt_depth < self.depth_scale[1])

            pred_depth = pred_depth[mask]

            gt_depth = gt_depth[mask]
            loss_gt = F.l1_loss(pred_depth, gt_depth, size_average=True)
            losses['gt_depth_loss{}'.format(scale)] = self.gt_depth_loss[0] * loss_gt
            # print('loss gt is: {}'.format(loss_gt))
            # print('in design loss: {}, gt {}'.format(pred_depth.size(), gt_depth.size()))


        # total_loss /= self.num_scales
        # losses["loss"] = total_loss
        return losses

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def project_depth_to_cam2(depth_map, cam2img, cam2global, global2cam2, cam2img2, post_rot1=None, post_trans1=None, post_rot2=None, post_trans2=None):
    """
    将深度图从一个相机投影到另一个相机的坐标系中，并生成对应的深度图。

    参数:
    depth_map (torch.Tensor): 深度图，形状为 (1, 6, 1, 900, 1600)。
    cam2img (torch.Tensor): 原相机的内参矩阵，形状为 (1, 6, 3, 3)。
    cam2global (torch.Tensor): 原相机到全局坐标系的变换矩阵，形状为 (1, 6, 4, 4)。
    global2cam2 (torch.Tensor): 全局坐标系到目标相机的变换矩阵，形状为 (1, 6, 4, 4)。
    cam2img2 (torch.Tensor): 目标相机的内参矩阵，形状为 (1, 6, 3, 3)。

    返回:
    torch.Tensor: 目标相机的像素坐标，形状为 (1, 6, 900, 1600, 2)。
    torch.Tensor: 目标相机的深度图，形状为 (1, 6, 900, 1600)。
    """
    import torch.nn as nn
    # 获取深度图的形状
    batch_size, num_views,  height, width = depth_map.shape
    print('depth map size is: {}'.format(depth_map.size()))
    print('---------------------------------------cam2img {}, cam2global {}, global2cam2 {}, cam2img2 {},  post_rot1 {}, post_trans1 {}, post_rot2 {}, post_trans2 {}'.format(
        cam2img.size(), cam2global.size(), global2cam2.size(), cam2img2.size(),  post_rot1.size(), post_trans1.size(), post_rot2.size(), post_trans2.size()
    ))
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = nn.Parameter(torch.from_numpy(id_coords),
                             requires_grad=False)
    print('id_croods is: ', id_coords.size())
    ones = nn.Parameter(torch.ones(batch_size, 1, height * width),
                             requires_grad=False)

    pix_coords = torch.unsqueeze(torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
    print('pix_croods is: {}'.format(pix_coords.size()))
    pix_coords = pix_coords.repeat(batch_size, 1, 1)


    pix_coords = nn.Parameter(torch.cat([pix_coords, ones], 1),
                                   requires_grad=False)
    pix_coords = pix_coords.unsqueeze(1)
    pix_coords = pix_coords.repeat(1, num_views, 1, 1)
    print('pix_croods2 is: {}'.format(pix_coords.size()), pix_coords[:, 0, :, 0], pix_coords[:, 0, :, 1], pix_coords[:, 0, :, 2], pix_coords[:, 0, :, 900],
          pix_coords[:, 0, :, 901],  pix_coords[:, 0, :, 1599], pix_coords[:, 0, :, 1600], pix_coords[:, 0, :, 3200])
    # 这里我们进行增广矩阵的复原
    # pix_coords = pix_coords.permute(0, 1, 3, 2)
    post_trans1 = post_trans1.unsqueeze(-1)
    pix_coords = pix_coords.cuda()
    pix_coords = pix_coords - post_trans1

    post_rot1_inv = torch.inverse(post_rot1)
    pix_coords = torch.matmul(post_rot1_inv, pix_coords)
    print('pix_croods3 is: {}'.format(pix_coords.size()), pix_coords[:, 0, :, 0], pix_coords[:, 0, :, 1], pix_coords[:, 0, :, 2], pix_coords[:, 0, :, 900],
          pix_coords[:, 0, :, 901],  pix_coords[:, 0, :, 1599], pix_coords[:, 0, :, 1600], pix_coords[:, 0, :, 3200])
    cam2img_inv = torch.inverse(cam2img)
    pix_coords = torch.matmul(cam2img_inv, pix_coords)
    print('pix_crood4: {}'.format(pix_coords.size()))
    depth_map = depth_map.view(batch_size, num_views, -1)
    depth_map = depth_map.unsqueeze(2)
    print('depthmaop size is: {}'.format(depth_map.size()))
    cam_points = pix_coords * depth_map
    print('cam_points is: {}'.format(cam_points.size()))
    _, _ , _, n_points = cam_points.size()
    ones = torch.ones(batch_size, num_views, 1,  n_points).cuda()
    print('ones size is: {}'.format(ones.size()))
    cam_points = torch.cat([cam_points, ones], 2)
    print('cam_points2: {}'.format(cam_points.size()))


    cam12cam2 = torch.matmul(global2cam2, cam2global)
    cam_points2 = torch.matmul(cam12cam2, cam_points)
    print('cam_points2 size is: {}'.format(cam_points2.size()))
    cam_points2 = cam_points2[:, :, :3, :]
    print('final campoints 2 size is: {}'.format(cam_points2.size()))
    print('K 2 is: {}'.format(cam2img2[0, 0, :, :]))
    img_points2 = torch.matmul(cam2img2, cam_points2)
    print('img_points is: {}'.format(img_points2.size()))
    coor = img_points2[:, :, :2, :]
    depth = img_points2[:, :, 2, :].unsqueeze(2)
    print('coor and depth is: {} {}'.format(coor.size(), depth.size()))
    pix_coords = coor / depth
    pix_coords = torch.cat([pix_coords, depth], 2)
    print('pix_croods is: {}'.format(pix_coords.size()), pix_coords.max(), pix_coords.min())
    pix_coords = torch.matmul(post_rot2, pix_coords) + post_trans2.unsqueeze(-1)
    print('depth new is: ', depth.min(), depth.max())

    depth_map_list = []
    for i in range(num_views):
        depth_map = torch.zeros((height, width), dtype=torch.float32).cuda()
        coor = torch.round(pix_coords[:, i, :2, :])
        depth = pix_coords[:, i, 2, :]
        print('idx is: {}, coor size is: {}, depth size is: {}'.format(i, coor.size(), depth.size()))
        kept1 = (coor[:, 0, :] >= 0) & (coor[:, 0, :] < width) & (
                coor[:, 1, :] >= 0) & (coor[:, 1, :] < height) & (
                        depth < 100) & (
                        depth >= 0.1)
        true_count = torch.sum(kept1)
        print('kept1 size is: {}'.format(kept1.size()), true_count)
        coor = coor[0, :, :].permute(1, 0)
        depth = depth[0, :]
        kept1 = kept1[0, :]
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        # depth_map = torch.cat((coor, depth), dim=1)
        print('depth map size is: {}'.format(depth_map.size()), coor.size(), depth.size())
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        print('final depth map is: {}'.format(depth_map.size()))
        depth_map_list.append(depth_map)
        # ranks = coor[:, 0] + coor[:, 1] * width
        # sort = (ranks + depth / 100.).argsort()
        # coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        # kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        # kept2[1:] = (ranks[1:] != ranks[:-1])
        # coor, depth = coor[kept2], depth[kept2]
        # coor = coor.to(torch.long)
        # depth_map[coor[:, 1], coor[:, 0]] = depth
    depth_map = torch.stack(depth_map_list).unsqueeze(0)
    print('depth size is: ', depth_map.size())
    return depth_map
