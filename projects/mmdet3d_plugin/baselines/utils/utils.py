import torch
import numpy as np
import os
import copy

def reshape_tensors(tensor_dict, B, N):
    reshaped_dict = {}
    for key, tensor in tensor_dict.items():
        # 获取原始形状
        original_shape = tensor.shape
        # 重新调整形状
        reshaped_tensor = tensor.view(B, N, *original_shape[1:])
        # 将重新调整形状后的张量放入新字典
        reshaped_dict[key] = reshaped_tensor
    return reshaped_dict

def calculate_fov(image_size, camera_matrix):
    """
    计算相机的水平视场角 (HFOV) 和垂直视场角 (VFOV)。

    参数:
    - image_size: 图片分辨率 (width, height) 的元组
    - camera_matrix: 形状为 (3, 3) 的相机内参矩阵 (K)

    返回:
    - (hfov, vfov): 水平视场角和垂直视场角的元组，单位为度数
    """
    # 提取图片分辨率
    print('camera_matrix: ', camera_matrix)
    camera_matrix = camera_matrix.detach().cpu().numpy()
    width, height = image_size

    # 提取内参矩阵中的 f_x 和 f_y
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    # 计算水平视场角 (HFOV)
    hfov = 2 * np.arctan(width / (2 * fx)) * (180 / np.pi)  # 转换为度数

    # 计算垂直视场角 (VFOV)
    vfov = 2 * np.arctan(height / (2 * fy)) * (180 / np.pi)  # 转换为度数

    return hfov, vfov


def save_tensors_to_npy(tensor_list, directory, save_depth=None, save_occ=None, occ_pred=None):
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
        mask_lidar_cpu = save_occ['mask_lidar']

        mask_lidar_cpu = mask_lidar_cpu.detach().cpu().numpy()
        file_path = os.path.join(directory, f'mask_lidar_cpu_tensor_{0}.npy')
        np.save(file_path, mask_lidar_cpu)

        occ_label_cpu = occ_label_cpu.detach().cpu().numpy()
        file_path = os.path.join(directory, f'occ_label_cpu_tensor_{0}.npy')
        np.save(file_path, occ_label_cpu)

        mask_camera_cpu = mask_camera_cpu.detach().cpu().numpy()
        file_path = os.path.join(directory, f'mask_camera_cpu_cpu_tensor_{0}.npy')
        np.save(file_path, mask_camera_cpu)

    if occ_pred is not None:
        print('occ pred is not None')
        occ_pred_copy = copy.copy(occ_pred)
        occ_pred_cpu = occ_pred_copy.detach().cpu().numpy()
        file_path = os.path.join(directory, f'occ_pred_tensor_{0}.npy')
        np.save(file_path, occ_pred_cpu)



        # 将张量从 GPU 迁移到 CPU
    tensor_cpu = tensor_list.detach().cpu().numpy()
    print('tensor is: ', tensor_list.size())
    # 构建文件路径
    file_path = os.path.join(directory, f'tensor_{0}.npy')

    # 保存为 .npy 文件
    np.save(file_path, tensor_cpu)
    # print(f'Saved tensor {idx} to {file_path}')


def generate_depth_map_from_points(points, lidar2cam, cam2img, post_rot, post_trans, grid_config):
    print('in generate_depth_map: points is: {}, lidar2cam is: {}, cam2img is: {}, post_rot is {}, post_trans is: {}'.format(
        type(points), type(lidar2cam), type(cam2img), type(post_rot), type(post_trans)
    ))
    print('in generate_depth_map: points is: {}, lidar2cam is: {}, cam2img is: {}, post_rot is {}, post_trans is: {}'.format(
        points.size(), lidar2cam.size(), cam2img.size(), post_rot.size(), post_trans.size()
    ))

    points = points[:, :3]
    n, _ = points.size()
    # ones_tensor = torch.ones(n, 1).to(points.device)
    # points = torch.cat((points, ones_tensor), dim=1)
    print('points 2 is: ', points.size())
    lidar2cam, cam2img, post_rot, post_trans = lidar2cam[0], cam2img[0], post_rot[0], post_trans[0]
    depth_map_list = []
    for cam_idx in range(lidar2cam.size()[0]):
        cam2img4 = np.eye(4, dtype=np.float32)
        cam2img4 = torch.from_numpy(cam2img4).to(points.device)
        cam2img4[:3, :3] = cam2img[cam_idx]
        lidar2img = torch.matmul(cam2img4, lidar2cam[cam_idx])
        # print('lidar2img is: ', lidar2img.size())
        # print('lidar2img is: ', lidar2img)

        point_cam = torch.matmul(points, lidar2cam[cam_idx][:3, :3].T) + lidar2cam[cam_idx][:3, 3]
        points_img = torch.matmul(point_cam, cam2img[cam_idx].T)
        # points_img = torch.matmul(points, lidar2img[:3, :3].T) + lidar2img[:3, 3]
        # points_cam = points.matmul(lidar2cam[cam_idx])
        # # print('points cam is: ', points_cam.size())
        # points_cam = points_cam[:, :3]
        # points_img = points_cam.matmul(cam2img[cam_idx].T)
        # print('point img is: ', points_img.size())

        points_img = torch.cat(
            [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
            1)  # (N_points, 3):  3: (u, v, d)

        # print('post_rot', post_rot)
        points_img = points_img.matmul(
            post_rot[cam_idx]) + post_trans[cam_idx, :]
        depth_map = points2depthmap(points_img, 256,  704, grid_config, downsample=1)
        print('depth map is: ', depth_map.size())
        depth_map_list.append(depth_map)
    depth_map_all = torch.stack(depth_map_list).unsqueeze(0)
        # print('point_final img is: ', points_img.size())
    return depth_map_all

def generate_depth_map_from_points_inv(points, lidar2cam, cam2img, post_rot, post_trans, grid_config):
    # 这里t的问题确实来源于雷达本身点的形式
    print('in generate_depth_map: points is: {}, lidar2cam is: {}, cam2img is: {}, post_rot is {}, post_trans is: {}'.format(
        type(points), type(lidar2cam), type(cam2img), type(post_rot), type(post_trans)
    ))
    print('in generate_depth_map: points is: {}, lidar2cam is: {}, cam2img is: {}, post_rot is {}, post_trans is: {}'.format(
        points.size(), lidar2cam.size(), cam2img.size(), post_rot.size(), post_trans.size()
    ))

    points = points[:, :3]
    n, _ = points.size()
    ones_tensor = torch.ones(n, 1).to(points.device)
    points = torch.cat((points, ones_tensor), dim=1).transpose(0, 1)
    print('points 2 is: ', points.size())
    lidar2cam, cam2img, post_rot, post_trans = lidar2cam[0], cam2img[0], post_rot[0], post_trans[0]
    depth_map_list = []
    for cam_idx in range(lidar2cam.size()[0]):
        point_cam = torch.matmul(lidar2cam[cam_idx], points)
        points_img = torch.matmul(cam2img[cam_idx], point_cam[:3, :]).transpose(0, 1)

        # points_img = torch.matmul(points, lidar2img[:3, :3].T) + lidar2img[:3, 3]
        # points_cam = points.matmul(lidar2cam[cam_idx])
        # # print('points cam is: ', points_cam.size())
        # points_cam = points_cam[:, :3]
        # points_img = points_cam.matmul(cam2img[cam_idx].T)
        # print('point img is: ', points_img.size())

        points_img = torch.cat(
            [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
            1)  # (N_points, 3):  3: (u, v, d)

        # print('post_rot', post_rot)
        points_img = points_img.matmul(
            post_rot[cam_idx]) + post_trans[cam_idx, :]
        depth_map = points2depthmap(points_img, 256,  704, grid_config, downsample=1)
        print('depth map is: ', depth_map.size())
        depth_map_list.append(depth_map)
    depth_map_all = torch.stack(depth_map_list).unsqueeze(0)
        # print('point_final img is: ', points_img.size())
    return depth_map_all

def points2depthmap(points, height, width, grid_config, downsample=1):
    """
    Args:
        points: (N_points, 3):  3: (u, v, d)
        height: int
        width: int

    Returns:
        depth_map：(H, W)
    """
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32).to(points.device)
    coor = torch.round(points[:, :2] / downsample).to(points.device)     # (N_points, 2)  2: (u, v)
    # print('in the loading coor : {}'.format(coor.size()), coor.max(), coor.min())
    depth = points[:, 2]    # (N_points, )哦
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    # 获取有效投影点.
    coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
    # print('in the loading after choose: ', coor.size(), depth.size())
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool).to(points.device)
    print('kept2 is: ', kept2.size())
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # 待删除
    # mask = depth_map > 0
    # # print('mask size is: ', type(mask), mask.size(), torch.sum(mask))
    # depth_map_filled = self.interpolate_depth_map(depth_map, mask)
    # # print('depth_map_filled is: {}, depth_map is: {}'.format(depth_map_filled.size(), depth.size()))
    # depth_map = depth_map_filled
    return depth_map