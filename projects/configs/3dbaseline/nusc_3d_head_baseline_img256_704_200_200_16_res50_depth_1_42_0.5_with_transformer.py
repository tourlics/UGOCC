_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 42.0, 0.5],
}


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
    samples_per_gpu=1,
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
runner = dict(type='EpochBasedRunner', max_epochs=70)

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
# load_from = "work_dirs/nusc_3d_head_baseline_img256_704_200_200_16_res101_no_transformer/epoch_4.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

_dim_ = 256
_bev_dim_ = 256
numC_Trans = 80


model = dict(
    type='OCC3dDepthWithTransformerBaseline',     # single-frame
    to_bev=False,
    Backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch',
        pretrained='./checkpoints/resnet50-0676ba61.pth',
    ),
    Neck=dict(
        type='OCCCustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    DepthModule=dict(
        type='CPM_DepthNet', # camera-aware depth net
        in_channels=256,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=int((grid_config['depth'][1] - grid_config['depth'][0]) / grid_config['depth'][2]),
        with_cp=False,
        loss_depth_weight=1.,
        use_dcn=False,
    ),
    FusionMoudle=dict(
        type='OCC_LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    AtentionModule=dict(
        type='Atention_Fusion_Module',
        bev_h=int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2]),
        bev_w=int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2]),
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=numC_Trans,
            encoder=dict(
                type='bevformer_encoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config={                            # should this config same with input
                            'x': [-40, 40, 0.8],
                            'y': [-40, 40, 0.8],
                            'z': [-1, 5.4, 1.6],
                           },
                data_config=data_config,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=grid_config['depth'],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=numC_Trans,
                                num_points=8,
                                num_levels=1),
                            embed_dims=numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=numC_Trans,
                        feedforward_channels=numC_Trans * 4,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=numC_Trans * 4,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                    # operation_order=('cross_attn', 'norm'))),
           ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=40,
            row_num_embed=int((grid_config['y'][1] - grid_config['y'][0]) / grid_config['y'][2]),
            col_num_embed=int((grid_config['x'][1] - grid_config['x'][0]) / grid_config['x'][2]),
            ),
    ),
    VoxFeatEncoder=dict(
        type='Vox_Encoder',
        vox_backbone=dict(
            type='OCC_CustomResNet3D',
            depth=18,
            with_cp=False,
            block_strides=[1, 2, 2],
            n_input_channels=numC_Trans,
            block_inplanes=[64, 64*2, 64*4],
            out_indices=(0, 1, 2),
            # norm_cfg=dict(type='BN3d', requires_grad=True)
            norm_cfg=dict(type='SyncBN', requires_grad=True),
        ),
        vox_neck=dict(
            type='OCC_FPN3D',
            with_cp=False,
            in_channels=[64, 64*2, 64*4],
            out_channels=256,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
        )
    ),
    Head=dict(
        type='OCC3DHead',
        in_channels=[256] * len((0, 1, 2)),
        out_channel=18,
        num_level=len((0, 1, 2)),
        soft_weights=True,
        use_deblock=True,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        balance_cls_weight=True,
        Dz=16,
        use_mask=True,
        use_focal_loss=True,
        loss_occ=dict(
            type='CustomFocalLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    )
)
find_unused_parameters = True

# ===> car - IoU = 38.32
# ===> construction_vehicle - IoU = 18.44
# ===> motorcycle - IoU = 20.72
# ===> pedestrian - IoU = 21.87
# ===> traffic_cone - IoU = 20.68
# ===> trailer - IoU = 22.
# ===> construction_vehicle - IoU = 18.44
# ===> motorcycle - IoU = 20.72
# ===> pedestrian - IoU = 21.87
# ===> traffic_cone - IoU = 20.68
# ===> trailer - IoU = 22.22
# ===> truck - IoU = 28.9
# ===> driveable_surface - IoU = 57.06
# ===> other_flat - IoU = 31.4
# ===> sidewalk - IoU = 33.74
# ===> terrain - IoU = 31.01
# ===> manmade - IoU = 16.52
# ===> vegetation - IoU = 17.82
# ===> mIoU of 6019 samples: 27.0
# {'mIoU': array([0.102, 0.364, 0.173, 0.365, 0.383, 0.184, 0.207, 0.219, 0.207,
#        0.222, 0.289, 0.571, 0.314, 0.337, 0.31 , 0.165, 0.178, 0.831])}
