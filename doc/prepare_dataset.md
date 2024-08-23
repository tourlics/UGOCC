
## NuScenes
Download nuScenes V1.0 full dataset data.

For Occupancy Prediction Task, you need to download extra annotation from
https://github.com/Tsinghua-MARS-Lab/Occ3D
to download Occ-nuscenes and Occ-waymo

## Waymo
prepare waymo dataset follow:
https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/waymo.html
then set the converted t6o kitti-format waymo dataset into: data/waymo_new

## prepare infos
download *.pkl files in: https://drive.google.com/file/d/1rcjsMEwMDKScqr5ZAbiY9JkcTPxc2lwu/view?usp=drive_link****

## prepare 2d labels
download nuscenes and waymo datasets' 2d semantic labels from:
### coming soon


## Pretrained model weights*
download *.pth files in: https://drive.google.com/file/d/1rcjsMEwMDKScqr5ZAbiY9JkcTPxc2lwu/view?usp=drive_link


## Final folder structure
**Folder structure**
```
data
├── nuscenes/
│       ├── gts
│       ├── bevdetv3-nuscenes-generative_infos_val.pkl
│       ├── bevdetv3-nuscenes-generative_infos_trian.pkl
│       ├── ...
|--- nuscenes_semantic
│   └── samples
│       ├── CAM_BACK
│       ├── CAM_BACK_LEFT
│       ├── CAM_BACK_RIGHT
│       ├── CAM_FRONT
│       ├── CAM_FRONT_LEFT
│       └── CAM_FRONT_RIGHT
├── waymo_new/
│   ├── kitti_format /
│              ├── waymo_infos_val_refined.pkl
│              ├── waymo_infos_train_refined.pkl
│              ├── ...
│   ├── waymo_occ /
├──  pretrain_depth_semantic_label.pkl

pretrained_models
├── nuscenes2waymo/
│   ├── UGOCC_full.pth 
│   ├── UGOCC_no_depth.pth 
│   ├── UGOCC_no_semantic.pth 
│   ├── UGOCC_semitrain_only.pth 
├── waymo2nuscenes/
│   ├── UGOCC_full.pth 
│   ├── UGOCC_no_depth.pth 
│   ├── UGOCC_no_semantic.pth 
│   ├── UGOCC_semitrain_only.pth 
```