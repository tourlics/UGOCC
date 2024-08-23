_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    # 'crop_h': (0.0, 0.0),
    # 'resize_test': 0.00,
    'resize': (-0.00, 0.00),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}


grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    # 'depth': [1.0, 45.0, 0.5],
    'depth': [0.5, 200.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+2, 1)

_depth_fpn_dim_ = 256

samples_per_gpu = 1
# predict_scales = [0, 1, 2, 3]
predict_scales = [0]
num_heads = 8
iter_num = 8
backbone_level = 5
img_feat_channels = [64, 256, 512, 1024, 2048]
downsample_ratio = [16, 8, 4, 2, 1]
kernel_size = [3, 3, 3, 1, 1]
dilation = [7, 3, 1, 1, 1]
output_padding = [1, 1, 1, 1, 0]
# Output_size = (I - 1) × S - 2P + K + (K - 1) × (D - 1) + output_padding
# Output_size: 输出的尺寸（可以是高度或宽度）。
# I: 输入的尺寸（可以是高度或宽度）。
# S: 步幅（stride），卷积核每次移动的步长。
# P: 填充（padding），在输入张量的边缘填充的像素数。
# K: 卷积核大小（kernel size），卷积核的尺寸。
# D: 膨胀（dilation），卷积核的元素之间的间距。
# output_padding: 输出填充（output padding），在反卷积操作后额外添加到输出张量右侧和底部的像素数。
cvt_configs = dict(
    (f'cvt_{i}', dict(
        type='CVT',
        iter_num=iter_num,
        downsample_ratio=downsample_ratio[i],
        dim=img_feat_channels[i],
        num_heads=num_heads,
        positionembedding=dict(
            type='PositionEmbeddingSine',
            num_pos_feats=img_feat_channels[i]
        ),
        self_block=dict(
            type='Self_Block',
            dim=img_feat_channels[i],
            num_heads=num_heads,
            Self_Attention=dict(
                type='Self_Attention',
                dim=img_feat_channels[i],
                num_heads=num_heads,
            )
        ),
        sep_conv=dict(
            type='SeparableConv2d',
            in_channels=img_feat_channels[i],
            out_channels=img_feat_channels[i],
            stride=downsample_ratio[i]
        ),
        sep_deconv=dict(
            type='SeparableDeConv2d',
            in_channels=img_feat_channels[i],
            out_channels=img_feat_channels[i],
            stride=downsample_ratio[i],
            kernel_size=kernel_size[i],
            dilation=dilation[i],
            output_padding=output_padding[i]
        )
    )) for i in range(backbone_level)  # 假设有6个CVT实例
)

backproject_configs = dict(
    (f'backproject_scale{i}', dict(
        type='BackprojectDepth',
        batch_size=samples_per_gpu * data_config['Ncams'],
        height=data_config['input_size'][0],
        width=data_config['input_size'][1]
    )) for i in predict_scales  # 假设有6个CVT实例
)

project_configs = dict(
    (f'project_scale{i}', dict(
        type='Project3D',
        batch_size=samples_per_gpu * data_config['Ncams'],
        height=data_config['input_size'][0],
        width=data_config['input_size'][1],
    )) for i in predict_scales  # 假设有6个CVT实例
)

model = dict(
    type='Self_Supervise_depth_baseline',     # single-frame
    num_adj=multi_adj_frame_id_cfg[1] - 1,
    to_bev=False,
    self_supervised_train=True,
    predict_scales=predict_scales,

    depth_scale=[0.1, 80],
    use_depth_gt_loss=True,
    repeoject_loss_weight=1.0,
    smooth_loss_weight=1.0,
    gt_depth_loss=0.03,

    Depthnet=dict(
        type='Multi_view_Depthnet',
        backbone_level=backbone_level,
        depth_scales=predict_scales,
        volume_depth_vs_metric_depth=True,  #
        depth_boundary=(0.1, 80.0),
        use_multi_scale=False,
        img_size=data_config['input_size'],
        backbone=dict(
                type='FiveOutputResnet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                with_cp=True,
                style='pytorch',
                pretrained='./checkpoints/resnet50-0676ba61.pth',
            ),
        neck=None,
        **cvt_configs,
        **backproject_configs,
        **project_configs,
        depth_decoder=dict(
            type='SS_Depth_decoder',
            num_ch_enc=img_feat_channels,
            num_ch_dec=[16, 32, 64, 128, 256],
            scales=predict_scales,
            use_skips=True
        )
    ),
    ssim_loss=dict(
        type='SSIM'
    )
    # Backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=False,
    #     with_cp=False,
    #     style='pytorch',
    #     pretrained='./checkpoints/resnet50-0676ba61.pth',
    # ),
)


# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFileMultiFrame',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepthMultiFrame', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'gt_depth_frame_t-1',
                                'gt_depth_frame_t-2'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFileMultiFrame',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepthMultiFrame', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'gt_depth_frame_t-1',
                                'gt_depth_frame_t-2'])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes-generative_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=1,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes-generative_infos_val.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=30)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
# resume = True
# load_from = "work_dirs/cvt_depth_estimation/epoch_8_ema.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# use_mask = False
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.35
# ===> barrier - IoU = 39.8
# ===> bicycle - IoU = 21.72
# ===> bus - IoU = 39.62
# ===> car - IoU = 40.56
# ===> construction_vehicle - IoU = 21.11
# ===> motorcycle - IoU = 24.66
# ===> pedestrian - IoU = 22.87
# ===> traffic_cone - IoU = 24.22
# ===> trailer - IoU = 25.98
# ===> truck - IoU = 29.65
# ===> driveable_surface - IoU = 58.07
# ===> other_flat - IoU = 31.47
# ===> sidewalk - IoU = 34.08
# ===> terrain - IoU = 31.23
# ===> manmade - IoU = 18.01
# ===> vegetation - IoU = 18.1
# ===> mIoU of 6019 samples: 28.91


# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.089   |  0.100   |  0.103   |
# |       barrier        |  0.378   |  0.436   |  0.459   |
# |       bicycle        |  0.215   |  0.252   |  0.261   |
# |         bus          |  0.510   |  0.617   |  0.681   |
# |         car          |  0.480   |  0.559   |  0.590   |
# | construction_vehicle |  0.182   |  0.260   |  0.289   |
# |      motorcycle      |  0.208   |  0.294   |  0.315   |
# |      pedestrian      |  0.294   |  0.345   |  0.360   |
# |     traffic_cone     |  0.272   |  0.304   |  0.312   |
# |       trailer        |  0.206   |  0.280   |  0.367   |
# |        truck         |  0.386   |  0.490   |  0.546   |
# |  driveable_surface   |  0.531   |  0.615   |  0.705   |
# |      other_flat      |  0.281   |  0.319   |  0.350   |
# |       sidewalk       |  0.233   |  0.279   |  0.327   |
# |       terrain        |  0.228   |  0.296   |  0.360   |
# |       manmade        |  0.278   |  0.353   |  0.406   |
# |      vegetation      |  0.177   |  0.276   |  0.364   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.291   |  0.357   |  0.400   |
# +----------------------+----------+----------+----------+
# {'RayIoU': 0.34939466889746956, 'RayIoU@1': 0.29110391692490867, 'RayIoU@2': 0.3573914069210632, 'RayIoU@4': 0.39968868284643677}