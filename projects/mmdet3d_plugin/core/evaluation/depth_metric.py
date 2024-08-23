import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce
import copy
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Metric_Depth():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 camera_id=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                 depth_scale=(0.1, 40),
                 image_mask_path=None
                 ):
        self.errors = dict()
        self.errors['scale-aware'] = dict()
        self.errors['scale-ambiguous'] = dict()
        self.data_iter = 0
        self.depth_metric = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        self.depth_metirc_dict = dict()
        for metric in self.depth_metric:
            self.depth_metirc_dict[metric] = 0


        self.camera_id = camera_id
        for i in self.camera_id:
            self.errors['scale-aware'][i] = (0, 0, 0, 0, 0, 0, 0)
            self.errors['scale-ambiguous'][i] = (0, 0, 0, 0, 0, 0, 0)
        print('self.error is: ', self.errors)



        self.depth_min = depth_scale[0]
        self.depth_max = depth_scale[1]
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

        self.ratios_median = []


    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label, (N_valid, )
            gt (1-d array): gt_occupancu_label, (N_valid, )

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)     # N_total
        correct = np.sum((pred[k] == gt[k]))    # N_correct

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),    # (N_cls, N_cls),
            correct,    # N_correct
            labeled,    # N_total
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        """
        Args:
            pred: (N_valid, )
            label: (N_valid, )
            n_classes: int=18

        Returns:

        """
        hist = np.zeros((n_classes, n_classes))     # (N_cls, N_cls)
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist    # (N_cls, N_cls)
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, pred_depths, gt_depths, mask=None):
        """
        Args:
            semantics_pred: (Dx, Dy, Dz, n_cls)
            semantics_gt: (Dx, Dy, Dz)
            mask_lidar: (Dx, Dy, Dz)
            mask_camera: (Dx, Dy, Dz)

        Returns:

        """
        for cam_id, cam_name in enumerate(self.camera_id):
            pred_depth = pred_depths[cam_id]
            gt_depth = gt_depths[cam_id]
            mask = np.logical_and(gt_depth > self.depth_min, gt_depth < self.depth_max)
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if self.use_image_mask:
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
            else:
                pass
            ratio_median = np.median(gt_depth) / np.median(pred_depth)
            # print('ratio_median is: ', ratio_median)
            self.ratios_median.append(ratio_median)

            pred_depth_median = pred_depth.copy() * ratio_median

            pred_depth_median[pred_depth_median < self.depth_min] = self.depth_min
            pred_depth_median[pred_depth_median > self.depth_max] = self.depth_max
            gt_depth[gt_depth < self.depth_min] = self.depth_min
            gt_depth[gt_depth > self.depth_max] = self.depth_max

            self.errors['scale-ambiguous'][cam_name] = tuple(a + b for a, b in zip(self.errors['scale-ambiguous'][cam_name],
                                                                                   self.compute_errors(gt_depth, pred_depth_median)))
            pred_depth[pred_depth < self.depth_min] = self.depth_min
            pred_depth[pred_depth > self.depth_max] = self.depth_max
            gt_depth[gt_depth < self.depth_min] = self.depth_min
            gt_depth[gt_depth > self.depth_max] = self.depth_max
            self.errors['scale-aware'][cam_name] = tuple(a + b for a, b in zip( self.errors['scale-aware'][cam_name],
                                                                               self.compute_errors(gt_depth, pred_depth)))
            self.data_iter += 1
            # .append(self.compute_errors(gt_depth, pred_depth_median))

        # print('-------------debug depth metric is: {}'.format(self.errors['scale-ambiguous']))
        # print('-------------debug depth metric is: {}'.format(self.errors['scale-aware']))

    def cac_av_depth_metric(self, cam_dict):
        result_dict = dict()
        # print(type(cam_dict.keys()), cam_dict.keys())
        length = self.data_iter / len(self.camera_id)
        # print('length is: ', length)
        for keys in cam_dict.keys():
            result_dict[keys] = copy.copy(self.depth_metirc_dict)
            # print(result_dict[keys])
            # print(cam_dict[keys][idx])
            for idx, metric_name in enumerate(self.depth_metric):
                # print(cam_dict[keys])
                # print('++++++++++: ', keys, cam_dict[keys])
                result_dict[keys][metric_name] = list(cam_dict[keys])[idx] / length
            for idx, metric_name in enumerate(self.depth_metric):
                result_dict[keys][metric_name] = result_dict[keys][metric_name]

        return result_dict

    def cac_depth_metric(self):
        type_depth_metric = ['scale-ambiguous', 'scale-aware']
        scale_ambiguous_dict = self.errors['scale-ambiguous']
        scale_aware_dict = self.errors['scale-aware']

        scale_ambiguous_metric = self.cac_av_depth_metric(scale_ambiguous_dict)
        scale_aware_metric = self.cac_av_depth_metric(scale_aware_dict)
        results = dict()
        results['scale-ambiguous'] = scale_ambiguous_metric
        results['scale-aware'] = scale_aware_metric
        # print(results)
        # print('ra is: ', ra)
        rows = []
        for scale_type, cameras in results.items():
            for camera, metrics in cameras.items():
                row = {'scale_type': scale_type, 'camera': camera}
                row.update(metrics)
                rows.append(row)

        df = pd.DataFrame(rows)

        # 打印 DataFrame
        print(df)
        return results


    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        eval_res = dict()
        # eval_res['class_name'] = self.class_names
        eval_res['mIoU'] = mIoU
        # eval_res['cnt'] = self.cnt
        return eval_res

    def compute_errors(self, gt, pred):
        # print('gt max {} and gt min {}'.format(gt.max(), gt.min()))
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
