import torch
import os
import numpy as np

def convert_depth_predictions(predictions, depth_config):
    """
    将预测的深度图（形状为 [B, N, depth_channel, H, W]）转换为实际深度值的深度图（形状为 [B, N, H, W]）。

    参数:
    - predictions: torch.Tensor，形状为 [B, N, depth_channel, H, W]，预测的深度图。
    - depth_config: dict，包含深度值的配置，键为 'depth'，值为 [min_depth, max_depth, depth_granularity]。

    返回值:
    - depth_map: torch.Tensor，形状为 [B, N, H, W]，实际深度值的深度图。
    """
    # 提取形状参数
    B, N, depth_channel, H, W = predictions.shape

    # 生成深度值索引
    min_depth = depth_config['depth'][0]
    max_depth = depth_config['depth'][1]
    depth_granularity = depth_config['depth'][2]

    # 生成 depth_values, 形状为 [depth_channel]
    depth_values = torch.arange(min_depth, max_depth , depth_granularity).float().to(
        predictions.device)

    # 将 depth_values 扩展到 [1, 1, depth_channel, 1, 1] 形状
    depth_values = depth_values.view(1, 1, depth_channel, 1, 1)

    # 对每个通道加权求和以得到实际深度值
    depth_map = torch.sum(predictions * depth_values, dim=2)

    # depth_map 的形状为 [B, N, H, W]
    return depth_map

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

def save_reproject_matrix_result(debug_ouput, num_views, height=256, width=704, metric=True):
    _, c, n_points = debug_ouput.size()
    pix_coords = debug_ouput.view(-1, num_views, 3, n_points)
    print('pix_coords size is ', pix_coords.size())
    depth_map_list = []
    for i in range(num_views):
        depth_map = torch.zeros((height, width), dtype=torch.float32).cuda()
        coor = torch.round(pix_coords[:, i, :2, :])
        depth = pix_coords[:, i, 2, :]

        if metric==True:
            kept1 = (coor[:, 0, :] >= 0) & (coor[:, 0, :] < width) & (
                    coor[:, 1, :] >= 0) & (coor[:, 1, :] < height) & (
                            depth < 100) & (
                            depth >= 2)
        else:
            kept1 = (coor[:, 0, :] >= 0) & (coor[:, 0, :] < width) & (
                    coor[:, 1, :] >= 0) & (coor[:, 1, :] < height)
        true_count = torch.sum(kept1)
        # print('kept1 size is: {}'.format(kept1.size()), true_count)
        coor = coor[0, :, :].permute(1, 0)
        depth = depth[0, :]
        kept1 = kept1[0, :]
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        # depth_map = torch.cat((coor, depth), dim=1)
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        depth_map_list.append(depth_map)

    depth_map = torch.stack(depth_map_list).unsqueeze(1)
    return depth_map