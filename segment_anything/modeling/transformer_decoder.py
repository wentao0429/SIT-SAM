import os
from typing import Dict, Any, Tuple, Optional, Type

import torch
from torch import nn, einsum
from segment_anything.build_sam3D import sam_model_registry3D
import math
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock
import hashlib
import numpy as np

from memorizing_transformers_pytorch.knn_memory import KNNMemoryList, DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True).detach()
    return F.softmax(t, dim=dim)


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class AbsolutePositionalEncoding3D(nn.Module):
    def __init__(self, num_channels, width, height, depth):
        super().__init__()
        # Create a 3D grid of positional encodings
        # Positional encodings can be learned or static
        self.positional_encoding = nn.Parameter(torch.randn(1, num_channels, width, height, depth))

    def forward(self, x):
        # Add the positional encoding to the input
        return x + self.positional_encoding


class Attention3D(nn.Module):
    def __init__(
            self,
            embed_dim: int = 528,
            num_heads: int = 8,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim  # Must be divisible
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape [B, D, H, W, C]
        B, D, H, W, C = x.shape
        # Adjust shape for linear layer and perform QKV projection
        qkv = self.qkv_proj(x.view(B * D * H * W, C)).view(B, D * H * W, 3, self.num_heads, self.head_dim).permute(2, 0,
                                                                                                                   3, 1,
                                                                                                                   4)
        q, k, v = qkv.unbind(0)  # Separate Q, K, V

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to V and reshape back to [B, D, H, W, C]
        x = (attn @ v).transpose(2, 3).contiguous().view(B, D, H, W, self.num_heads * self.head_dim)

        # Apply output projection
        x = self.proj(x.view(B * D * H * W, self.num_heads * self.head_dim)).view(B, D, H, W, C)
        x = self.attn_drop(x)

        return x


# attn = Attention3D()
# img = torch.randn(2, 8, 8, 8, 528)
# output = attn(img)
class KNNAttention3D(nn.Module):
    def __init__(
            self,
            embed_dim=528,
            num_heads=8,
            qkv_bias=True,
            dropout_rate=0.0,
            num_retrieved_memories=32,
            # xl_max_memories=512,
            num_memory_layers=1,
            knn_memories_directory=DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
            attn_scale_init=20,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5
        # self.scaling = nn.Parameter(torch.ones(num_heads, 1, 1, 1) * math.log(attn_scale_init))
        self.num_retrieved_memories = num_retrieved_memories
        # self.xl_max_memories = xl_max_memories
        self.num_memory_layers = num_memory_layers
        self.knn_memories_directory = knn_memories_directory
        # self.knn_mem_kwargs = dict(dim=self.head_dim,
        #                            max_memories=250000,
        #                            multiprocessing=False
        #                            )

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)

    def forward(self, x, knn_memory, add_knn_memory=False):
        B, D, H, W, C = x.shape
        qkv = self.qkv_proj(x.view(B * D * H * W, C)).view(B, D * H * W, 3, self.num_heads, self.head_dim).permute(2, 0,
                                                                                                                   3, 1,
                                                                                                                   4)
        q, k, v = qkv.unbind(0)  # batch, head, D*H*W, head_dim
        # scaling = self.scaling.exp()
        scaling = self.scaling

        # Normalize q, k for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Perform KNN search using q
        # Retrieve memories (keys and values) based on query q
        knn_keys_knn_values, mask = knn_memory.search(q,
                                                      self.num_retrieved_memories)  # Need to modify and check what shape of Q to use for retrieval
        knn_keys, knn_values = knn_keys_knn_values.unbind(-2)
        

        # Combine retrieved KNN keys and values with current keys and values
        k_combined = torch.cat([knn_keys, k.unsqueeze(-2)], dim=-2)
        v_combined = torch.cat([knn_values, v.unsqueeze(-2)], dim=-2)

        # Compute attention
        attn = (q.unsqueeze(-2) @ k_combined.transpose(-2, -1)) * scaling
        # attn = attn.softmax(dim=-1)
        attn = stable_softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v_combined).transpose(2, 3).contiguous().view(B, D, H, W, self.num_heads * self.head_dim)

        # Project back to original dimension
        x = self.proj(x.view(B * D * H * W, self.num_heads * self.head_dim)).view(B, D, H, W, C)
        x = self.attn_drop(x)

        # If need to add memory to KNN library
        if add_knn_memory:
            # Here we assume a function that can transform k and v into a form suitable for adding to the memory library
            new_kvs = self.prepare_kv_for_memory(k.contiguous().view(B, self.num_heads, D * H * W, self.head_dim),
                                                 v.contiguous().view(B, self.num_heads, D * H * W, self.head_dim))
            knn_memory.add(new_kvs)

        return x

    def prepare_kv_for_memory(self, k, v):
        # This function needs to be written based on your specific knn_memory object implementation
        # Its purpose is to transform k and v into a format that can be added to the memory library
        # Example implementation
        # Requires B, N, KV, D
        B, _, _, _ = k.shape
        k = k.view(B, -1, self.head_dim)
        v = v.view(B, -1, self.head_dim)
        return torch.stack((k, v), dim=-2).detach()

    def create_knn_memories(self, batch_size):
        return KNNMemoryList.create_memories(
            batch_size=batch_size,
            num_memory_layers=self.num_memory_layers,
            memories_directory=self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(
            self,
            **kwargs
    ):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok=True, parents=True)
        lock = FileLock(str(knn_dir / 'mutex'))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            knn_memories.cleanup()


class KNNAttention3DPlus(nn.Module):
    def __init__(
            self,
            embed_dim=528,
            num_heads=8,
            qkv_bias=True,
            dropout_rate=0.0,
            num_retrieved_memories=32,
            num_memory_layers=1,
            knn_memories_directory=DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
            attn_scale_init=20,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # self.scaling = self.head_dim ** -0.5
        self.scaling = nn.Parameter(torch.ones(num_heads, 1, 1, 1) * math.log(attn_scale_init))
        self.num_retrieved_memories = num_retrieved_memories
        self.num_memory_layers = num_memory_layers
        self.knn_memories_directory = knn_memories_directory

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)

    def forward(self, x, knn_memory, add_knn_memory=True):
        B, D, H, W, C = x.shape
        qkv = self.qkv_proj(x.view(B * D * H * W, C)).view(B, D * H * W, 3, self.num_heads, self.head_dim).permute(2, 0,
                                                                                                                   3, 1,
                                                                                                                   4)
        q, k, v = qkv.unbind(0)  # batch, head, D*H*W, head_dim
        scaling = self.scaling.exp()

        # Normalize q, k for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Perform KNN search using q
        # Retrieve memories (keys and values) based on query q
        knn_keys_knn_values, mask = knn_memory.search(q,
                                                      self.num_retrieved_memories)  # Need to modify and check what shape of Q to use for retrieval
        knn_keys, knn_values = knn_keys_knn_values.unbind(-2)

        # Combine retrieved KNN keys and values with current keys and values
        k_combined = torch.cat([knn_keys, k.unsqueeze(-2)], dim=-2)
        v_combined = torch.cat([knn_values, v.unsqueeze(-2)], dim=-2)

        # Compute attention
        attn = (q.unsqueeze(-2) @ k_combined.transpose(-2, -1)) * scaling
        attn = stable_softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v_combined).transpose(2, 3).contiguous().view(B, D, H, W, self.num_heads * self.head_dim)

        # Project back to original dimension
        x = self.proj(x.view(B * D * H * W, self.num_heads * self.head_dim)).view(B, D, H, W, C)
        x = self.attn_drop(x)

        # If need to add memory to KNN library
        if add_knn_memory:
            # Here we assume a function that can transform k and v into a form suitable for adding to the memory library
            new_kvs = self.prepare_kv_for_memory(k.contiguous().view(B, self.num_heads, D * H * W, self.head_dim),
                                                 v.contiguous().view(B, self.num_heads, D * H * W, self.head_dim))
            knn_memory.add(new_kvs)

        return x

    def prepare_kv_for_memory(self, k, v):
        # This function needs to be written based on your specific knn_memory object implementation
        # Its purpose is to transform k and v into a format that can be added to the memory library
        # Example implementation
        # Requires B, N, KV, D
        B, _, _, _ = k.shape
        k = k.view(B, -1, self.head_dim)
        v = v.view(B, -1, self.head_dim)
        return torch.stack((k, v), dim=-2).detach()

    def create_knn_memories(self, batch_size):
        return KNNMemoryList.create_memories(
            batch_size=batch_size,
            num_memory_layers=self.num_memory_layers,
            memories_directory=self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(
            self,
            **kwargs
    ):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok=True, parents=True)
        lock = FileLock(str(knn_dir / 'mutex'))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            knn_memories.cleanup()


class TransformerEncoderBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention3D(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, embed_dim * 4)

    def forward(self, x):
        # Apply multi-head attention to the input
        x = x + self.attn(self.norm1(x))
        # Apply MLP and return the result
        x = x + self.mlp(self.norm2(x))
        return x


# model = TransformerEncoderBlock3D(528, 8)
# image = torch.randn(2, 8, 8, 8, 528)
# output = model(image)

class TransformerEncoderKNNBlock3D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = KNNAttention3D(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, embed_dim * 4)

    def forward(self, x, knn_memory):
        x = x + self.attn(self.norm1(x), knn_memory)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoderKNNBlock3DPlus(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = KNNAttention3DPlus(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, embed_dim * 4)

    def forward(self, x, knn_memory):
        x = x + self.attn(self.norm1(x), knn_memory)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder3D(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock3D(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Convert input from [B, C, H, W, D] to [B, H, W, D, C] for attention
        # B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1)

        # Pass the input through each Transformer block
        for layer in self.layers:
            x = layer(x)

        # Normalize the output of the last layer
        x = self.norm(x)

        # Reshape back to [B, C, H, W, D] from [B, H*W*D, C]
        x = x.permute(0, 4, 1, 2, 3)
        return x


# model = TransformerEncoder3D(528, 8, 6)
# image = torch.randn(2, 528, 8, 8, 8)
# summary(model, input_data=image)

class TransformerEncoderKNN3D(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, **knn_block_kwargs):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.layers = nn.ModuleList([
            TransformerEncoderBlock3D(embed_dim,
                                      num_heads) if i != num_layers - 1 else TransformerEncoderKNNBlock3D(
                embed_dim, num_heads, **knn_block_kwargs)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, knn_memory):
        x = x.permute(0, 2, 3, 4, 1)
        knn_memory_iter = iter(knn_memory)
        for layer in self.layers:
            if isinstance(layer, TransformerEncoderKNNBlock3D):
                x = layer(x, knn_memory=next(knn_memory_iter))
            else:
                x = layer(x)
        # Normalize the output of the last layer
        x = self.norm(x)

        # Reshape back to [B, C, H, W, D] from [B, H*W*D, C]
        x = x.permute(0, 4, 1, 2, 3)
        return x


class TransformerEncoderKNN3DPlus(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, **knn_block_kwargs):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.layers = nn.ModuleList([
            TransformerEncoderBlock3D(embed_dim,
                                      num_heads) if i != num_layers - 1 else TransformerEncoderKNNBlock3DPlus(
                embed_dim, num_heads, **knn_block_kwargs)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, knn_memory):
        x = x.permute(0, 2, 3, 4, 1)
        knn_memory_iter = iter(knn_memory)
        for layer in self.layers:
            if isinstance(layer, TransformerEncoderKNNBlock3DPlus):
                x = layer(x, knn_memory=next(knn_memory_iter))
            else:
                x = layer(x)
        # Normalize the output of the last layer
        x = self.norm(x)

        # Reshape back to [B, C, H, W, D] from [B, H*W*D, C]
        x = x.permute(0, 4, 1, 2, 3)
        return x


class MaskedAttention3D(nn.Module):
    def __init__(
            self,
            embed_dim: int = 528,
            num_heads: int = 8,
            num_queries: int = 100,  # Number of query slots
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        # self.query_embed = nn.Embedding(num_queries, embed_dim)  # Learnable query parameters
        self.query_embed = nn.Parameter(torch.rand(num_queries, embed_dim))
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, D, H, W, C = x.shape

        if mask is not None:
            assert mask.shape == (B, D, H, W), "Mask shape must be (B, D, H, W)"
            mask = mask.view(B, 1, 1, D * H * W).expand(-1, self.num_heads, -1, -1)

        kv = self.kv_proj(x.view(B * D * H * W, C)).view(B, D * H * W, 2, self.num_heads, self.head_dim).permute(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)
        k, v = kv.unbind(0)

        q = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1).view(self.num_heads, B, self.num_queries,
                                                                      self.head_dim).permute(1, 2, 0,
                                                                                             3)  # (B, num_queries, num_heads, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Since queries are fixed, we do a matmul with V and then sum over the query dimension
        x = (attn @ v).sum(dim=1).view(B, D, H, W, C)

        x = self.proj(x.view(B * D * H * W, C)).view(B, D, H, W, C)
        x = self.attn_drop(x)

        return x


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 384,
            num_heads: int = 8,
            mlp_ratio=3.0,
            dropout_rate=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Masked attention layer
        self.masked_attn = MaskedAttention3D(embed_dim, num_heads)
        self.norm1 = LayerNorm3d(embed_dim)

        # Self-attention layer
        self.self_attn = Attention3D(embed_dim, num_heads)
        self.norm2 = LayerNorm3d(embed_dim)

        # Feedforward layer
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            # nn.Dropout(dropout_rate),
        )
        self.norm3 = LayerNorm3d(embed_dim)

    def forward(self, target, source, mask):
        # Apply masked attention
        target2 = self.masked_attn(target, source, mask)
        target = target + target2
        target = self.norm1(target)

        # Apply self-attention
        target2 = self.self_attn(target)
        target = target + target2
        target = self.norm2(target)

        # Apply feed-forward network
        target2 = self.ffn(target)
        target = target + target2
        target = self.norm3(target)

        return target


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Apply global average pooling to reduce each D x H x W feature map to a single number
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Flatten pooled features
        # Apply the classifier to the pooled features
        logits = self.classifier(x)
        return logits


class FeatureConcatenator(nn.Module):
    def __init__(self):
        super(FeatureConcatenator, self).__init__()
        # Define downsampling layers to accommodate different feature sizes
        # self.downsample1 = nn.AdaptiveMaxPool3d((8, 8, 8))
        self.downsample2 = nn.AdaptiveMaxPool3d((8, 8, 8))
        self.downsample3 = nn.AdaptiveMaxPool3d((8, 8, 8))

    def forward(self, feature_dict):
        # Downsample features to match target size [8, 8, 8]
        feature1 = feature_dict['feature1']
        feature2 = self.downsample2(feature_dict['feature2'])
        feature3 = self.downsample3(feature_dict['feature3'])

        # Concatenate features along the first dimension (feature dimension)
        concatenated_features = torch.cat([feature1, feature2, feature3], dim=1)
        return concatenated_features


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (2, 2, 2),
            stride: Tuple[int, int] = (2, 2, 2),
            padding: Tuple[int, int] = (0, 0, 0),
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x
