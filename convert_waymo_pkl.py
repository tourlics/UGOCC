import pickle
from os import path as osp
import numpy as np
import os
from tqdm import tqdm

def save_dict_to_pkl(dictionary, filename):
    """
    将字典保存为 .pkl 文件

    参数:
    dictionary (dict): 需要保存的字典
    filename (str): 保存的文件名
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f"字典已保存到 {filename} 文件中")


def convert_pkl(ann_file_name, data_prefix, data_root='data/waymo_new/kitti_format/',
                out_file_name='data/waymo/kitti_format/waymo_infos_train_refined.pkl'):
    with open(ann_file_name, 'rb') as file:
        train_pkl = pickle.load(file)
    data_list_ori = train_pkl['data_list']
    meta_info = train_pkl['metainfo']
    data_list = []

    for idx, info in tqdm(enumerate(data_list_ori), desc="Processing data list"):
        camera_info = dict()
        camera_info['sample_idx'] = info['sample_idx']
        camera_info['timestamp'] = info['timestamp']
        camera_info['context_name'] = info['context_name']

        # 后面需要加上目标检测 就把这个给解开
        # camera_info['instances'] = info['instances']
        # camera_info['cam_sync_instances'] = info['cam_sync_instances']
        # camera_info['cam_instances'] = info['cam_instances']

        curr_sample_idx = info['sample_idx']
        curr_scene_idx = curr_sample_idx % 1000000 // 1000
        curr_frame_idx = curr_sample_idx % 1000000 % 1000

        camera_info['curr_scene_idx'] = curr_scene_idx
        camera_info['curr_frame_idx'] = curr_frame_idx

        lidar_prefix = data_prefix.get('pts', '')
        camera_info['lidar_path'] = osp.join(
            data_root, lidar_prefix, info['lidar_points']['lidar_path'])
        camera_info['lidar_path'] = osp.normpath(camera_info['lidar_path'])
        camera_info['lidar2ego'] = np.diag([1, 1, 1, 1])

        if 'train' in ann_file_name:
            occ_root_path = os.path.join(data_prefix['OCC'], 'training',
                                         "{:03}".format(curr_scene_idx),
                                         '{}_04.npz'.format("{:03}".format(curr_frame_idx)))
        else:
            occ_root_path = os.path.join(data_prefix['OCC'], 'validation',
                                         "{:03}".format(curr_scene_idx),
                                         '{}_04.npz'.format("{:03}".format(curr_frame_idx)))
        camera_info['occ_path'] = osp.normpath(occ_root_path)
        camera_info['images'] = dict()

        for (cam_key, img_info) in info['images'].items():
            camera_info['images'][cam_key] = img_info

            if 'img_path' in img_info:
                cam_prefix = data_prefix.get(cam_key, '')
                camera_info['images'][cam_key]['img_path'] = osp.join(
                    data_root, cam_prefix, img_info['img_path'])
                camera_info['images'][cam_key]['img_path'] = osp.normpath(camera_info['images'][cam_key]['img_path'])
            if 'lidar2cam' in img_info:
                camera_info['images'][cam_key]['lidar2cam'] = np.array(img_info['lidar2cam'])
            if 'cam2img' in img_info:
                camera_info['images'][cam_key]['cam2img'] = np.array(img_info['cam2img'])[:3, :3]
            if 'lidar2img' in img_info:
                camera_info['images'][cam_key]['lidar2img'] = np.array(img_info['lidar2img'])
            else:
                camera_info['images'][cam_key]['lidar2img'] = camera_info['cam2img'] @ camera_info['lidar2cam']
            if 'ego2global' in info:
                camera_info['images'][cam_key]['ego2global'] = np.array(info['ego2global'])

            camera_info['images'][cam_key]['ego2cam'] = np.array(img_info['lidar2cam'])

        data_list.append(camera_info)

    new_dict = {'metainfo': meta_info, 'data_list': data_list}
    save_dict_to_pkl(new_dict, out_file_name)


if __name__ == '__main__':
    waymo_infos_train_file = 'data/waymo_new/kitti_format/waymo_infos_train.pkl'
    waymo_infos_test_file = 'data/waymo_new/kitti_format/waymo_infos_val.pkl'

    data_prefix = dict(
        pts='training/velodyne',
        CAM_FRONT='training/image_0',
        CAM_FRONT_LEFT='training/image_1',
        CAM_FRONT_RIGHT='training/image_2',
        CAM_SIDE_LEFT='training/image_3',
        CAM_SIDE_RIGHT='training/image_4',
        OCC='data/waymo_new/waymo_occ/voxel04/'
    )
    convert_pkl(ann_file_name=waymo_infos_train_file, data_prefix=data_prefix, data_root='data/waymo_new/kitti_format/',
                out_file_name='data/waymo_new/kitti_format/waymo_infos_train_refined.pkl')

    convert_pkl(ann_file_name=waymo_infos_test_file, data_prefix=data_prefix, data_root='data/waymo_new/kitti_format/',
                out_file_name='data/waymo_new/kitti_format/waymo_infos_val_refined.pkl')