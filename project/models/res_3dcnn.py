#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/make_model copy.py
Project: /workspace/code/project/models
Created Date: Thursday May 8th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 8th 2025 1:23:28 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""


import torch
import torch.nn as nn


class LateFusionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重参数 α

    def forward(self, main_feat: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main_feat: 主干网络输出 (B, out_dim)
            attn_map:  attention map (B, 1, T, H, W)

        Returns:
            torch.Tensor: 融合后的输出 (B, out_dim)
        """
        attn_feat = attn_map.mean(dim=[2, 3, 4])  # (B, 1)
        attn_proj = self.attn_mlp(attn_feat)  # (B, out_dim)
        out = self.alpha * main_feat + (1 - self.alpha) * attn_proj
        return out


class Res3DCNN(nn.Module):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.fuse_method = hparams.model.fuse_method

        self.model = self.init_resnet(
            self.model_class_num,
            self.fuse_method,
        )

        if self.fuse_method == "late":
            self.late_fusion = LateFusionBlock(
                in_dim=self.model_class_num,  # Input dimension from the main feature
                out_dim=self.model_class_num,  # Output dimension for the attention map
            )

    @staticmethod
    def init_resnet(class_num: int = 3, fuse_method: str = "add") -> nn.Module:

        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

        if fuse_method == "concat":
            input_channel = 3 + 1
        else:
            input_channel = 3

        # for the folw model and rgb model
        slow.blocks[0].conv = nn.Conv3d(
            input_channel,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        # change the knetics-400 output 400 to model class num
        slow.blocks[-1].proj = nn.Linear(2048, class_num)

        return slow

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            attn_map: (B, 1, T, H, W)

        Returns:
            torch.Tensor: (B, C, T, H, W)
        """

        if self.fuse_method == "late":
            features = self.model(video)
            output = self.late_fusion(features, attn_map)

        else:
            if self.fuse_method == "concat":
                _input = torch.cat([video, attn_map], dim=1)
            elif self.fuse_method == "add":
                _input = video + attn_map
            elif self.fuse_method == "mul":
                _input = video * attn_map
            elif self.fuse_method == "avg":
                _input = (video + attn_map) / 2
            elif self.fuse_method == "none":
                _input = video
            else:
                raise KeyError(
                    f"the fuse method {self.fuse_method} is not in the model zoo"
                )

            output = self.model(_input)

        return output


if __name__ == "__main__":
    from omegaconf import OmegaConf

    hparams = OmegaConf.create({
        "model": {
            "model_class_num": 3,
            "fuse_method": "late"
        }
    })

    model = Res3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)
    out = model(video, attn_map)
    print(out.shape)  # Expected: (2, 3)