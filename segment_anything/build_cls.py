import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label
from scipy.ndimage import distance_transform_edt
from skimage.measure import inertia_tensor
from torch import nn
from skimage.filters import threshold_otsu
import cv2

from segment_anything.modeling.transformer_decoder import FeatureConcatenator, AbsolutePositionalEncoding3D, \
    TransformerEncoder3D


class classifier(nn.Module):
    def __init__(self, input_dim, num_classes=117, mlp_ratio=2.0):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.concat = FeatureConcatenator()
        self.position_encoding = AbsolutePositionalEncoding3D(input_dim, 8, 8, 8)
        self.transformer_encoder = TransformerEncoder3D(
            embed_dim=input_dim,
            num_heads=8,
            num_layers=6,
        )
        # input_dim = input_dim + 192
        self.fc1 = nn.Linear(input_dim, int(input_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(input_dim * mlp_ratio), num_classes)
        self.gelu = nn.GELU()
        # self.dropout = nn.Dropout(0.0)
        # self.bn1 = nn.BatchNorm1d(27)  # 添加BatchNorm1d层
        # self.bn2 = nn.BatchNorm1d(192)
        # self.fcp1 = nn.Linear(27, 128)  # have point是128
        # self.fcp2 = nn.Linear(128, 192)

    def forward(self, x, mask):
        x = self.concat(x)
        # logit_masks = []
        # # 对batch中的每个mask进行处理
        # for i in range(mask.shape[0]):
        #     # 使用Otsu's method计算阈值
        #     threshold = threshold_otsu(mask[i].cpu().numpy())
        #
        #     # 进行二值化
        #     logit_mask = mask[i] > threshold
        #
        #     # 将二值化后的mask添加到列表中
        #     logit_masks.append(logit_mask)

        # 将列表转换为tensor
        # logit_masks = torch.stack(logit_masks)
        # feature = self.extract_multiscale_geometric_features(logit_masks)
        # feature = feature.float()
        mask = F.interpolate(mask, size=(8, 8, 8), mode='trilinear', align_corners=False)
        #
        # feature = self.bn1(feature)  # 添加BatchNorm1d层
        # feature = self.fcp1(feature)  # 注意到时候改回来
        # feature = self.dropout(feature)
        # feature = self.gelu(feature)
        # feature = self.fcp2(feature)
        # feature = self.bn2(feature)

        x = x * mask  # result 的形状是 [B, C, 8, 8, 8]
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        # x = torch.cat([x, feature], dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

    def max_inscribed_sphere(self, mask):
        # 确保mask是在CPU上，并转换为NumPy数组
        mask_np = mask.cpu().numpy()
        centers = []
        for i in range(mask_np.shape[0]):
            # 计算距离变换
            dist_transform = distance_transform_edt(mask_np[i])

            # 将距离变换结果转换回tensor，并移回GPU（如果需要）
            dist_transform_tensor = torch.tensor(dist_transform, device=mask.device).squeeze(0)

            # 在扁平化的tensor中找到最大距离及其索引（球心坐标）
            max_dist, idx = torch.max(dist_transform_tensor.view(-1), 0)

            # 将一维索引转换为三维索引
            sphere_center = np.unravel_index(idx.cpu().numpy(), dist_transform_tensor.shape)

            # 将中心坐标转换为tensor并移回GPU
            sphere_center_tensor = torch.tensor(sphere_center, device=mask.device)
            centers.append(sphere_center_tensor)

        return torch.stack(centers)

    def extract_multiscale_geometric_features(self, mask, scales=None):
        """
        提取多尺度几何特征。
        :param mask: 输入的三维mask张量，假设在GPU上。
        :param scales: 要考虑的尺度列表。
        :return: 在多个尺度上计算得到的特征的张量，形状为[B, 所有尺度下的所有特征]。
        """
        if scales is None:
            scales = [1, 0.5, 0.25]

        all_features_tensor_list = []

        for scale in scales:
            if scale == 1:
                scaled_mask = mask
            else:
                # 使用torch.nn.functional.interpolate进行下采样
                size = [int(dim * scale) for dim in mask.shape[2:]]  # mask形状为[B, C, D, H, W]
                scaled_mask = F.interpolate(mask.float(), size=size, mode='trilinear')

            # 计算当前尺度下的几何特征
            features = self.extract_geometric_features(scaled_mask)  # 确保此函数返回的是[B, 特征长度]形状的张量

            # 将特征字典转换为张量列表
            features_tensor_list = [features[key] for key in features.keys()]

            # 将当前尺度的特征张量沿特定维度（如最后一维）拼接
            features_tensor = torch.cat(features_tensor_list, dim=1)  # 假设每个features_tensor都是[B, 特征长度]

            # 添加到所有尺度特征的列表中
            all_features_tensor_list.append(features_tensor)

        # 在特征维度上拼接所有尺度的特征
        all_features = torch.cat(all_features_tensor_list, dim=1)

        return all_features

    def extract_geometric_features(self, mask):
        mask_np = mask.cpu().numpy()
        mask_np = mask_np.squeeze(1)
        B = mask_np.shape[0]

        # 初始化特征列表
        volumes = torch.zeros((B, 1), device=mask.device)
        surface_areas = torch.zeros((B, 1), device=mask.device)
        eccentricitys = torch.zeros((B, 1), device=mask.device)
        principal_axis_lengths = torch.zeros((B, 3), device=mask.device)
        centers = torch.zeros((B, 3), device=mask.device)

        for i in range(B):
            # 计算最大内切球的中心
            dist_transform = distance_transform_edt(mask_np[i])
            max_dist_idx = np.argmax(dist_transform)
            sphere_center = np.unravel_index(max_dist_idx, dist_transform.shape)
            centers[i] = torch.tensor(sphere_center, device=mask.device, dtype=torch.float)

            # 计算体积
            volume = np.sum(mask_np[i])
            volumes[i] = torch.tensor([volume], device=mask.device)

            # 计算表面积
            eroded_mask = binary_erosion(mask_np[i])
            surface_area = np.sum(np.logical_xor(mask_np[i], eroded_mask))
            surface_areas[i] = torch.tensor([surface_area], device=mask.device)

            # 计算惯性张量，偏心率和主轴长度
            labeled_mask, num_features = label(mask_np[i])
            if num_features > 1:
                max_feature = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                binary_mask = labeled_mask == max_feature
            else:
                binary_mask = mask_np[i].astype(bool)

            inertia_tensor_mat = inertia_tensor(binary_mask)
            eigenvalues, _ = np.linalg.eigh(inertia_tensor_mat)
            principal_axis_lengths[i] = torch.tensor(np.sqrt(eigenvalues), device=mask.device)
            # print(principal_axis_lengths[i].shape)
            eccentricity = np.sqrt(1 - eigenvalues.min() / eigenvalues.max())
            eccentricitys[i] = torch.tensor([eccentricity], device=mask.device)
        # 对volume和surface_area进行平方根转换并除以10
        volumes = torch.sqrt(volumes) / 5
        surface_areas = torch.sqrt(surface_areas) / 5

        features = {
            'center': centers,  # [B, 3]
            'volume': volumes,  # [B, 1]
            'surface_area': surface_areas,  # [B, 1]
            'eccentricity': eccentricitys,  # [B, 1]
            'principal_axis_length': principal_axis_lengths,  # [B, 3]
        }
        return features
