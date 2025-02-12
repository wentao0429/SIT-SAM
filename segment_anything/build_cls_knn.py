import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.measure import inertia_tensor
from torch import nn

from memorizing_transformers_pytorch.knn_memory import KNNMemoryList
from segment_anything.modeling.transformer_decoder import FeatureConcatenator, AbsolutePositionalEncoding3D, \
    TransformerEncoderKNN3D


class classifier(nn.Module):
    def __init__(self, input_dim, num_classes=117, mlp_ratio=2.0, max_knn_memories=70000,
                 knn_memory_multiprocessing=True):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.concat = FeatureConcatenator()
        self.position_encoding = AbsolutePositionalEncoding3D(input_dim, 8, 8, 8)
        self.transformer_encoder = TransformerEncoderKNN3D(
            embed_dim=input_dim,
            num_heads=8,
            num_layers=6,
        )
        self.max_knn_memories = max_knn_memories
        self.knn_memory_kwargs = dict(
            dim=self.transformer_encoder.head_dim,
            max_memories=self.max_knn_memories,
            multiprocessing=knn_memory_multiprocessing
        )

        self.knn_memory = KNNMemoryList.create_memories(
            batch_size=16,
            num_memory_layers=1,
            memories_directory='./117_turbo/knn.memories/state1',
        )(**self.knn_memory_kwargs)
        self.knn_memory[0].load('./117_turbo/knn.memories/state1')
        # self.knn_memory[1].load('./knn/knn.memories/state2')
        input_dim = input_dim + 192
        self.fc1 = nn.Linear(input_dim, int(input_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(input_dim * mlp_ratio), num_classes)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.0)
        self.bn1 = nn.BatchNorm1d(27)  # 添加BatchNorm1d层
        self.bn2 = nn.BatchNorm1d(192)
        self.fcp1 = nn.Linear(27, 128)
        self.fcp2 = nn.Linear(128, 192)

    def forward(self, x, mask):
        x = self.concat(x)
        logit_masks = []
        # Process each mask in the batch
        for i in range(mask.shape[0]):
            # Calculate threshold using Otsu's method
            threshold = threshold_otsu(mask[i].cpu().numpy())

            # Binarization
            logit_mask = mask[i] > threshold

            # Add binarized mask to the list
            logit_masks.append(logit_mask)

        # Convert list to tensor
        logit_masks = torch.stack(logit_masks)
        feature = self.extract_multiscale_geometric_features(logit_masks)
        feature = feature.float()

        mask = F.interpolate(mask, size=(8, 8, 8), mode='trilinear', align_corners=False)
        feature = self.bn1(feature)
        feature = self.fcp1(feature)
        feature = self.dropout(feature)
        feature = self.gelu(feature)
        feature = self.fcp2(feature)
        feature = self.bn2(feature)

        x = x * mask  # Shape of result is [B, C, 8, 8, 8]
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, self.knn_memory)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feature], dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

    def max_inscribed_sphere(self, mask):
        mask_np = mask.cpu().numpy()
        centers = []
        for i in range(mask_np.shape[0]):
            dist_transform = distance_transform_edt(mask_np[i])

            dist_transform_tensor = torch.tensor(dist_transform, device=mask.device).squeeze(0)

            max_dist, idx = torch.max(dist_transform_tensor.view(-1), 0)

            sphere_center = np.unravel_index(idx.cpu().numpy(), dist_transform_tensor.shape)

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
                # Downsample using torch.nn.functional.interpolate
                size = [int(dim * scale) for dim in mask.shape[2:]]  # mask shape is [B, C, D, H, W]
                scaled_mask = F.interpolate(mask.float(), size=size, mode='trilinear')

            # Calculate geometric features at current scale
            features = self.extract_geometric_features(scaled_mask)

            # Convert feature dictionary to tensor list
            features_tensor_list = [features[key] for key in features.keys()]

            # Concatenate feature tensors along specified dimension
            features_tensor = torch.cat(features_tensor_list, dim=1)

            # Add to list of features across all scales
            all_features_tensor_list.append(features_tensor)

        # Concatenate features from all scales along feature dimension
        all_features = torch.cat(all_features_tensor_list, dim=1)

        return all_features

    def extract_geometric_features(self, mask):
        mask_np = mask.cpu().numpy()
        mask_np = mask_np.squeeze(1)
        B = mask_np.shape[0]

        volumes = torch.zeros((B, 1), device=mask.device)
        surface_areas = torch.zeros((B, 1), device=mask.device)
        eccentricitys = torch.zeros((B, 1), device=mask.device)
        principal_axis_lengths = torch.zeros((B, 3), device=mask.device)
        centers = torch.zeros((B, 3), device=mask.device)

        for i in range(B):
            dist_transform = distance_transform_edt(mask_np[i])
            max_dist_idx = np.argmax(dist_transform)
            sphere_center = np.unravel_index(max_dist_idx, dist_transform.shape)
            centers[i] = torch.tensor(sphere_center, device=mask.device, dtype=torch.float)

            volume = np.sum(mask_np[i])
            volumes[i] = torch.tensor([volume], device=mask.device)

            eroded_mask = binary_erosion(mask_np[i])
            surface_area = np.sum(np.logical_xor(mask_np[i], eroded_mask))
            surface_areas[i] = torch.tensor([surface_area], device=mask.device)

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
