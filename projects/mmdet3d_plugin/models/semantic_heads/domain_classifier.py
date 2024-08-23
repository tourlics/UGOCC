import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import MODELS


@MODELS.register_module()
class DomainClassifier(nn.Module):
    def __init__(self, input_channels=80, dropout_prob=0.5, source_dataset='nuscenes', batch_size=1, dbce_weights=1):
        super(DomainClassifier, self).__init__()
        # 使用卷积层降维
        self.source_dataset = source_dataset
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # 输出 (B, 32, 50, 50)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 输出 (B, 64, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 输出 (B, 128, 13, 13)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, 128, 1, 1)

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(64, 1)
        self.dbce_weights = dbce_weights

    def loss(self, x, dataset_list):
        index = dataset_list.index(self.source_dataset)
        source_preds = x[index, :]
        target_preds = x[index-1, :]

        source_labels = torch.ones_like(source_preds)
        target_labels = torch.zeros_like(target_preds)
        source_loss = F.binary_cross_entropy(source_preds, source_labels)
        target_loss = F.binary_cross_entropy(target_preds, target_labels)
        # print('losses is: ', source_loss ,target_loss)
        return self.dbce_weights * (source_loss + target_loss)

    def forward(self, x):
        # 假设输入 x 的形状为 (B, 80, 100, 100)
        x = F.relu(self.conv1(x))  # 形状 (B, 32, 50, 50)
        x = F.relu(self.conv2(x))  # 形状 (B, 64, 25, 25)
        x = F.relu(self.conv3(x))  # 形状 (B, 128, 13, 13)

        x = self.global_avg_pool(x)  # 形状 (B, 128, 1, 1)
        x = torch.flatten(x, start_dim=1)  # 将输入展平成 (B, 128)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        # print('x size is: ', x.size(), x, x[1, :], x[1-1, :], x[0, :], x[0-1, :])
        return x
