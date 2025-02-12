import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label
from scipy.ndimage import distance_transform_edt
from skimage.measure import inertia_tensor
from torch import nn

from memorizing_transformers_pytorch.knn_memory_plus import KNNMemoryList, DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
from segment_anything.modeling.transformer_decoder import FeatureConcatenator, AbsolutePositionalEncoding3D, \
    TransformerEncoderKNN3DPlus


class classifier(nn.Module):
    def __init__(self, input_dim, num_classes=117, mlp_ratio=2.0, max_knn_memories=8192,
                 knn_memory_multiprocessing=True):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.concat = FeatureConcatenator()
        self.position_encoding = AbsolutePositionalEncoding3D(input_dim, 8, 8, 8)
        self.transformer_encoder = TransformerEncoderKNN3DPlus(
            embed_dim=input_dim,
            num_heads=8,
            num_layers=6,
        )
        self.max_knn_memories = max_knn_memories
        # input_dim = input_dim + 128
        self.knn_memory_kwargs = dict(
            dim=self.transformer_encoder.head_dim,
            max_memories=self.max_knn_memories,
            multiprocessing=knn_memory_multiprocessing
        )

        self.knn_memory = KNNMemoryList.create_memories(
            batch_size=16,
            num_memory_layers=1,
            memories_directory=DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        )(**self.knn_memory_kwargs)
        self.knn_memory[0].load('<path_to_knn_memories>/state1')
        # self.knn_memory[1].load('./knn/knn.memories/state2')

        self.fc1 = nn.Linear(input_dim, int(input_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(input_dim * mlp_ratio), num_classes)
        self.gelu = nn.GELU()
        # self.bn1 = nn.BatchNorm1d(27)  # 添加BatchNorm1d层
        # self.fcp1 = nn.Linear(3, 128)  # have point是128
        # self.fcp2 = nn.Linear(128, 128)

    def forward(self, x, mask):
        x = self.concat(x)
        mask = F.interpolate(mask, size=(8, 8, 8), mode='trilinear', align_corners=False)

        x = x * mask  # Shape of result is [B, C, 8, 8, 8]
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, self.knn_memory)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        # x = torch.cat([x, feature], dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

    def max_inscribed_sphere(self, mask):
        # Ensure mask is on CPU and convert to NumPy array
        mask_np = mask.cpu().numpy()
        centers = []
        for i in range(mask_np.shape[0]):
            # Calculate distance transform
            dist_transform = distance_transform_edt(mask_np[i])

            # Convert distance transform result back to tensor and move to GPU if needed
            dist_transform_tensor = torch.tensor(dist_transform, device=mask.device).squeeze(0)

            # Find maximum distance and its index (sphere center coordinates) in flattened tensor
            max_dist, idx = torch.max(dist_transform_tensor.view(-1), 0)

            # Convert 1D index to 3D index
            sphere_center = np.unravel_index(idx.cpu().numpy(), dist_transform_tensor.shape)

            # Convert center coordinates to tensor and move back to GPU
            sphere_center_tensor = torch.tensor(sphere_center, device=mask.device)
            centers.append(sphere_center_tensor)

        return torch.stack(centers)

    def extract_multiscale_geometric_features(self, mask, scales=None):
        """
        Extract multi-scale geometric features.
        :param mask: Input 3D mask tensor, assumed to be on GPU.
        :param scales: List of scales to consider.
        :return: Tensor of features computed at multiple scales, shape [B, all_features_across_scales].
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

        # Initialize feature lists
        volumes = torch.zeros((B, 1), device=mask.device)
        surface_areas = torch.zeros((B, 1), device=mask.device)
        eccentricitys = torch.zeros((B, 1), device=mask.device)
        principal_axis_lengths = torch.zeros((B, 3), device=mask.device)
        centers = torch.zeros((B, 3), device=mask.device)

        for i in range(B):
            # Calculate center of maximum inscribed sphere
            dist_transform = distance_transform_edt(mask_np[i])
            max_dist_idx = np.argmax(dist_transform)
            sphere_center = np.unravel_index(max_dist_idx, dist_transform.shape)
            centers[i] = torch.tensor(sphere_center, device=mask.device, dtype=torch.float)

            # Calculate volume
            volume = np.sum(mask_np[i])
            volumes[i] = torch.tensor([volume], device=mask.device)

            # Calculate surface area
            eroded_mask = binary_erosion(mask_np[i])
            surface_area = np.sum(np.logical_xor(mask_np[i], eroded_mask))
            surface_areas[i] = torch.tensor([surface_area], device=mask.device)

            # Calculate inertia tensor, eccentricity and principal axis lengths
            labeled_mask, num_features = label(mask_np[i])
            if num_features > 1:
                max_feature = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                binary_mask = labeled_mask == max_feature
            else:
                binary_mask = mask_np[i].astype(bool)

            inertia_tensor_mat = inertia_tensor(binary_mask)
            eigenvalues, _ = np.linalg.eigh(inertia_tensor_mat)
            principal_axis_lengths[i] = torch.tensor(np.sqrt(eigenvalues), device=mask.device)
            eccentricity = np.sqrt(1 - eigenvalues.min() / eigenvalues.max())
            eccentricitys[i] = torch.tensor([eccentricity], device=mask.device)

        # Apply square root transform and divide by 10 for volume and surface area
        volumes = torch.sqrt(volumes) / 10
        surface_areas = torch.sqrt(surface_areas) / 10

        features = {
            'center': centers,  # [B, 3]
            'volume': volumes,  # [B, 1]
            'surface_area': surface_areas,  # [B, 1]
            'eccentricity': eccentricitys,  # [B, 1]
            'principal_axis_length': principal_axis_lengths,  # [B, 3]
        }
        return features


