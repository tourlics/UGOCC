_base_ = ['../../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../../mmdetection3d/configs/_base_/default_runtime.py']

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
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 42.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

_dim_ = 256
_bev_dim_ = 256
numC_Trans = 80


model = dict(
    type='OCC2dDepthBaseline',
    to_bev=True,
    Backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='./checkpoints/resnet50-0676ba61.pth'
    ),
    Neck=dict(
        type='OCCCustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        out_ids=[0]
    ),
    DepthModule=dict(
        type='NaiveDepthNet',    # 若是没有真实对应的表征， 模型应该如何切换
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=int((grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]),   # seems like this loss
        with_cp=False,
        loss_depth_weight=1.0,
        use_dcn=False),
    FusionMoudle=dict(
        type='OCC_LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=(256, 704),
        downsample=16),
    BevFeatEncoder=dict(
        type='Bev_Encoder',
        bev_backbone=dict(
            type='OCCCustomResNet',
            numC_input=numC_Trans,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
        bev_neck=dict(type='OCC_FPN_LSS',
                      in_channels=numC_Trans * 8 + numC_Trans * 2,
                      out_channels=_bev_dim_)),
    Head=dict(
        type='OCC2DHead',
        in_dim=_bev_dim_,
        out_dim=_bev_dim_,
        Dz=16,
        use_mask=False,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0)))




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
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
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
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes-generative_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes-generative_infos_train.pkl',
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
runner = dict(type='EpochBasedRunner', max_epochs=40)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# resume = True
# resume_from = 'work_dirs/nusc_2d_head_baseline_img256_704_200_200_16_res101/latest.pth'
# resume_from = 'work_dirs/nusc_2d_head_baseline_img256_704_200_200_16_res101/epoch_57.pth'
# load_from = "work_dirs/nusc_2d_head_baseline_img256_704_200_200_16_res101/epoch_39_ema.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

# ===> per class IoU of 6019 samples:
# ===> others - IoU = 9.49
# ===> barrier - IoU = 35.19
# ===> bicycle - IoU = 22.16
# ===> bus - IoU = 37.2
# ===> car - IoU = 37.79
# ===> construction_vehicle - IoU = 19.0
# ===> motorcycle - IoU = 22.6
# ===> pedestrian - IoU = 21.29
# ===> traffic_cone - IoU = 21.99
# ===> trailer - IoU = 23.5
# ===> truck - IoU = 27.92
# ===> driveable_surface - IoU = 57.61
# ===> other_flat - IoU = 29.91
# ===> sidewalk - IoU = 33.09
# ===> terrain - IoU = 30.06
# ===> manmade - IoU = 15.31
# ===> vegetation - IoU = 16.79
# ===> mIoU of 6019 samples: 27.11
# {'mIoU': array([0.095, 0.352, 0.222, 0.372, 0.378, 0.19 , 0.226, 0.213, 0.22 ,
#        0.235, 0.279, 0.576, 0.299, 0.331, 0.301, 0.153, 0.168, 0.83 ])}
