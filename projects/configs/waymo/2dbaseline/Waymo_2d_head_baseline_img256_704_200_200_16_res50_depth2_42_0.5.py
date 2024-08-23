_base_ = [
          '../default_runtime.py'
          ]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'


point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size=[0.4, 0.4, 0.4]
num_classes = 16
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

pose_file = 'data/waymo/cam_infos.pkl'
data_root = './data/waymo/kitti_format/'
occ_gt_data_root = './data/waymo/kitti_format/waymo_occ/voxel04/training/'
val_pose_file = 'data/waymo/cam_infos_vali.pkl'
occ_val_gt_data_root = './data/waymo/kitti_format/waymo_occ/voxel04/validation/'


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
num_feats = [_dim_//3, _dim_//3 , _dim_ - _dim_//3 - _dim_//3]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
total_z = 16
# for bev
pillar_h = 4
num_points_in_pillar = 4
# for volume
# bev_z_ = 8
queue_length = 3 # each sequence contains `queue_length` frames.
num_views = 5
FREE_LABEL = 23
load_interval = 5   # 每隔几个frame选取一个值进行训练

class_weight_binary = [5.314075572339673, 1]
class_weight_multiclass = [
    21.996729830048952,
    7.504469780801267, 10.597629961083673, 12.18107968968811, 15.143940258446506, 13.035521328502758,
    9.861234292376812, 13.64431851057796, 15.121236434460473, 21.996729830048952, 6.201671013759701,
    5.7420517938838325, 9.768712859518626, 3.4607400626606317, 4.152268220983671, 1.000000000000000,
]



dataset_type = 'CustomWaymoDataset_T'



file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 42.0, 0.5],
}

data_config = {
    'cams': [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT',
        'CAM_SIDE_RIGHT'
    ],  # 此处决定输入顺序
    'cams_fov': [(50.28334728509245, 34.747818351544495), (50.208248695697826, 34.692181100350766),
                 (50.00769057360646, 34.54368105523629), (50.031348806164345, 24.304478995342503),
                 (49.84960199926778, 24.206927870765)]
    ,
    'Ncams':
    5,
    'input_size': (640, 960),
    'src_size': [(1280, 1920), (1280, 1920), (1280, 1920), (886, 1920), (886, 1920)],

    # Augmentation
    'resize': (-0.00, 0.00),
    'rot': (-0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

train_pipeline = [
    dict(
        type='WaymoPrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='WaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='WaymoLoadOccGTFromFile', data_root=occ_gt_data_root, use_larger=True, crop_x=False),
    dict(
        type='WaymoLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,  # waymo 数据集的lidar天然比nuscenes多一个维度
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='WaymoPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(
        type='WaymoPrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='WaymoLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='WaymoLoadOccGTFromFile', data_root=occ_gt_data_root, use_larger=True, crop_x=False),
    dict(
        type='WaymoLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,  # waymo 数据集的lidar天然比nuscenes多一个维度
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='WaymoPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]


grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 42.0, 0.5],
}


test_data_config = dict(
    pipeline=test_pipeline,
    pose_file=val_pose_file,
    ann_file=data_root + 'waymo_infos_val.pkl')

multi_adj_frame_id_cfg = (1, 1+2, 1)

share_data_config = dict(
    type=dataset_type,
    num_views=num_views,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    load_interval=load_interval,
    img_info_prototype='bevdet',       # 'bevdet4d'   这一项在使用多帧数据时被设置为   bevdet4d
    # multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,   # 这一项在使用多帧数据时被设置， 反之则为空
    cams_name=[
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_SIDE_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_SIDE_RIGHT'
    ],
    imgs_scales=data_config['src_size']
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        history_len=3,
        ann_file=data_root + 'waymo_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        pose_file=pose_file,
        # use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

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
        type='CPM_DepthNet',    # 若是没有真实对应的表征， 模型应该如何切换
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
        num_classes=16,
        use_predicter=True,
        class_balance=True,
        loss_occ=dict(
            type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0)))

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

evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)