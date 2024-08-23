# Enhancing Generalizability via Utilization of Unlabeled Data for Occupancy Perception

3D occupancy perception accurately estimates the volumetric status and semantic labels of a scene, attracting significant attention in the field of autonomous driving.
However, enhancing the model's ability to generalize across different driving scenarios or sensing systems, often requires redesigning the model or extra-expensive annotations.
To this end, following a comprehensive analysis of the occupancy model architecture, we proposed the \textbf{UGOCC} method that utilizes domain adaptation to efficiently harness unlabeled autonomous driving data, thereby enhancing the model’s generalizability.
Specifically, we design the depth fusion module by employing self-supervised depth estimation, and propose a strategy based on semantic attention and domain adversarial learning to improve the generalizability of the learnable fusion module. Additionally, we propose an OCC-specific pseudo-label selection tailored for semi-supervised learning, which optimizes the overall network's generalizability.
Our experiment results on two challenging datasets nuScenes and Waymo, demonstrate that our method not only achieves state-of-the-art  generalizability but also enhances the model's perceptual capabilities within the source domain by utilizing unlabeled data.


# Waymo to nuScenes
![Workflow Example](./doc/figs/waymo_target_video.gif)
# nuScenes to Waymo
![Workflow Example](./doc/figs/nusc_target_video.gif)

## installation
### [Installation](doc/installation.md)
# Datasets and Pretrained Models
### [Datsets&pretrained](doc/prepare_dataset.md)

## train source only model
We offer a variety of different training configurations, all of which are contained within the following path for your selection:

such as train 2d depth_fusion only models in nuscenes domain:
```shell
./tools/dist_train.sh ./projects/configs/nuscenes_waymo_domain_adapation_configs/nuscenes_nine_label/2dbaseline/nusc_2dbaseline_depthfusion_2_42_0.5_backbone_res50_img256_704_without_transformer.py num_gpus
```
train 3d depth_fusion with attention fusion baseline in nuscenes domain:
```shell
./tools/dist_train.sh ./projects/configs/nuscenes_waymo_domain_adapation_configs/nuscenes_nine_label/3dbaseline/nusc_3dbaseline_depthfusion_2_42_0.5_backbone_res50_img256_704_with_transformer.py num_gpus
```

## pretrain 2d model weights
The 2d pre-trained models include the entire nuScenes dataset and one-fifth of the Waymo dataset, employing training methodologies that either do not require manually annotated labels or are entirely unlabeled. For further details, please refer to Chapter 3 of the article.
```shell
./tools/dist_train.sh ./projects/configs/nuscenes_waymo_domain_adapation_configs/pretrain_semantic/pretrained_self_supervised_depth_estimation_1-61-0.5_cvt_based.py num_gpus
```

## train model
To train the full model, it is necessary to set load_from in the configuration file to load a previously trained 2D pre-trained model, and select the source domain, for example: source_dataset='nuscenes'. Training with pseudo-labels is prone to overfitting; it is strongly recommended to first check the quality of the 2D pre-trained model.
```shell
./tools/dist_train.sh ./projects/configs/nuscenes_waymo_domain_adapation_configs/pretrain_semantic/pretrained_self_supervised_depth_estimation_1-61-0.5_cvt_based_with_occ.py num_gpus
```

# test model

Please first refer to the "Datasets & Pretrained Models" section to download our pretrained models. 
Subsequently, we have placed all test models in the 'paper_results_models' directory.

For instance, to evaluate the transferability from Waymo to nuScenes, consider the following example:

```shell
./tools/dist_test.sh ./paper_used_models/nuscenes2waymo/UGOCC/UGOCC_depth3d_attention_test_in_source_nuscenes.py model_path 6 --eval map_ten_class
```

The term 'map_ten_class' is used here because we have performed a transformation on the labels.
