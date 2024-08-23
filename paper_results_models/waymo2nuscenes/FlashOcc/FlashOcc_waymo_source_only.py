_base_ = ['./nus-3d.py',
          './default_runtime.py']


plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

revised_occ_label = [
    "Drivable Surface",
    "Vehicle",
    "Pedestrian",
    "Building",
    "Vegetation",
    "Sidewalk",
    "Others",
    "Traffic Object",
    "Two-wheeler",
    "Free"
]

occ_label_to_revise = {
    0: 6,  # TYPE_GENERALOBJECT（通用物体） --> others(其他)
    1: 1,  # TYPE_VEHICLE(车辆)            --> vehicle(车辆)
    2: 2,  # TYPE_PEDESTRIAN(行人)         --> Pedestrian(行人)
    3: 7,  # TYPE_SIGN(标志)               --> Traffic Object(交通对象)
    4: 8,  # TYPE_CYCLIST(骑行者)          --> Two-wheeler(两轮车)
    5: 7,  # TYPE_TRAFFIC_LIGHT(交通灯)    --> Traffic Object(交通对象)
    6: 6,  # TYPE_POLE(杆)                --> others(其他)
    7: 6,  # TYPE_CONSTRUCTION_CONE（施工锥） --> others(其他)
    8: 8,  # TYPE_BICYCLE(自行车)               --> Two-wheeler(两轮车)
    9: 8,  # TYPE_MOTORCYCLE(摩托车)              --> Two-wheeler(两轮车)
    10: 3,  # TYPE_BUILDING(建筑物)           --> Two-wheeler(两轮车)
    11: 4,  # TYPE_VEGETATION(植被)             --> Vegetation(植被)
    12: 4,  # TYPE_TREE_TRUNK(树干)             --> Vegetation(植被)
    13: 0,  # TYPE_ROAD(可行驶区域)                  --> Drivable Surface(可行驶区域)
    14: 5,  # TYPE_WALKABLE(人行道)                 --> Sidewalk(人行道)
    15: 9   # Free                                 --> Free
}



num_classes = len(revised_occ_label)

data_config = {
    'cams': ['CAM_FRONT', 'CAM_FRONT_LEFT',
             'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT',
             'CAM_SIDE_RIGHT'],  # 这里的排序实际上就是我们图片的压缩方向
    'cams_fov': [[(64.30983882854876, 38.94793974307966), (64.56147792492683, 39.12362250950747),
                  (64.78975608144492, 39.28325183985623), (64.95896090122484, 39.401730584762916),
                  (89.34338531063514, 58.15602765195581), (64.84462775846904, 39.321658720242)]]  # width_fov, height_fov
    ,
    'Ncams':
    5,
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
    'depth': [1.0, 42.0, 0.5],   # 这是一个值得调节的参数
}

_dim_ = 256
_bev_dim_ = 256
numC_Trans = 80
downsample = 16

model = dict(
    type='OCC2dDepthBaseline',
    grid_config=grid_config,
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
        type='CPM_DepthNet', # camera-aware depth net
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=downsample,
        grid_config=grid_config,
        depth_channels=int((grid_config['depth'][1] - grid_config['depth'][0]) // grid_config['depth'][2]),
        with_cp=False,
        loss_depth_weight=1.0,
        use_dcn=False,
        input_size=data_config['input_size'],
        noise_ratio=0.00,
    ),
    FusionMoudle=dict(
        type='OCC_LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=downsample),
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
        num_classes=num_classes,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0)
    )
)


dataset_type = 'WaymoOccMultiFrame'
data_root = 'data/waymo_new/kitti_format/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0,
    flip_dy_ratio=0
)

train_pipeline = [
    dict(
        type='WaymoNewPrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False,
        bda_aug_conf=bda_aug_conf
    ),
    dict(type='WaymoNewLoadOccGTFromFile', revised_label=True, revise_dict=occ_label_to_revise),
    dict(
        type='WaymoNewLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,  # waymo 数据集的lidar天然比nuscenes多一个维度
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='WaymoNewPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'],
        meta_keys=['box_mode_3d', 'box_type_3d', 'cam_names']
    )
]

cam_num_use = 5
test_pipeline = [
    dict(
        type='WaymoNewPrepareImageInputs',
        is_train=False,
        data_config=data_config,
        sequential=False,
        bda_aug_conf=bda_aug_conf,
        test_cam_num_ids=cam_num_use
    ),
    dict(
        type='WaymoNewLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,  # waymo 数据集的lidar天然比nuscenes多一个维度
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='WaymoNewPointToMultiViewDepth', downsample=1, grid_config=grid_config),
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
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'],
                 meta_keys=['box_mode_3d', 'box_type_3d', 'cam_names'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
    occ_class_nums=len(revised_occ_label),
    occ_label_to_revise=occ_label_to_revise,
    revised_occ_label=revised_occ_label
)
test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'waymo_infos_val_refined.pkl',
    load_interval=50,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val_refined.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        load_interval=5,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_LEFT='training/image_1',
            CAM_FRONT_RIGHT='training/image_2',
            CAM_SIDE_LEFT='training/image_3',
            CAM_SIDE_RIGHT='training/image_4',
            OCC='data/waymo_new/waymo_occ/voxel04/'
        ),
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        revised_label=True,
        occ_class_nums=True,
    ),
    val=test_data_config,
    test=test_data_config
)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# load_from = './domain_adaptation_work_dirs/waymo_nine_label/2dbaseline/waymo_2dbaseline_depthfusion_2_42_0.5_backbone_res50_img256_704_without_transformer/epoch_17_ema.pth'


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

evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)

load_from = './paper_used_models/nuscenes2waymo/FlashOcc/nuscenes_source_train_only.pth'