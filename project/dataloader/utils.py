#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/utils.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Copy from pytorchvideo.

Have a good code time :)
-----
Last Modified: Wednesday June 25th 2025 5:38:56 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, Optional


import torch
import torch
import torch.nn.functional as F
from torch import Tensor

# class UniformTemporalSubsample:

#     def __init__(self, num_samples: int):
#         self.num_samples = num_samples

#     def __call__(self, video: torch.Tensor) -> torch.Tensor:
#         """
#         video: Tensor of shape (T, C, H, W)
#         returns: Subsampled video of shape (num_samples, C, H, W)
#         """
#         # Reference: https://github.com/facebookresearch/pytorchvideo/blob/a0a131e/pytorchvideo/transforms/functional.py#L19
#         t_max = video.shape[-4] - 1
#         indices = torch.linspace(0, t_max, self.num_samples, device=video.device).long()
#         return torch.index_select(video, -4, indices)


class UniformTemporalSubsample:
    """
    等同于 torchvision.transforms.v2.UniformTemporalSubsample，
    但在帧数不足时会 *均匀复制* 最近邻帧进行补齐。
    支持输入形状 (T, C, H, W)   或 (B, T, C, H, W)。
    """

    def __init__(self, num_samples: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.num_samples = num_samples

    def _compute_indices(self, t: int, device) -> Tensor:
        """得到 size=[num_samples] 的 long 索引张量。"""
        # 产生 float 索引，范围 [0, t-1]，共 num_samples 个点
        idx_float = torch.linspace(
            0, max(t - 1, 0), self.num_samples, dtype=torch.float32, device=device
        )
        # 四舍五入到最近帧，再转 long
        return torch.round(idx_float).long()

    def __call__(self, video: Tensor) -> Tensor:
        """
        Args:
            video: (T, C, H, W) **或** (B, T, C, H, W)

        Returns:
            Tensor: 与输入批量/通道一致，但时间维被采样/补齐为 `num_samples`
        """
        is_batched = video.ndim == 5
        if not is_batched and video.ndim != 4:
            raise ValueError("Input must be (T, C, H, W) or (B, T, C, H, W)")

        # 取出时间维长度
        t = video.shape[-4]
        idx = self._compute_indices(t, video.device)

        # 在时间维(-4)上索引
        return torch.index_select(video, -4, idx)


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0
