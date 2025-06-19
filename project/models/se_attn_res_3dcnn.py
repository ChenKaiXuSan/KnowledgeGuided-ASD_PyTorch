#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/se_att_res_3dcnn.py
Project: /workspace/code/project/models
Created Date: Thursday June 19th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 19th 2025 5:36:16 pm
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
import torch.nn.functional as F


class SEFusion(nn.Module):
    def __init__(self, in_channels, context_channels=1, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(context_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        scale = self.fc(self.pool(context))
        return x * scale + x


class SEFusionRes3DCNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.model_class_num = hparams.model.model_class_num
        self.model = self.init_resnet(self.model_class_num)

        self.se_fusions = nn.ModuleList(
            [
                SEFusion(64, context_channels=1),
                SEFusion(256, context_channels=1),
                SEFusion(512, context_channels=1),
                SEFusion(1024, context_channels=1),
                SEFusion(2048, context_channels=1),
            ]
        )

    @staticmethod
    def init_resnet(class_num: int = 3) -> nn.Module:
        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )
        slow.blocks[0].conv = nn.Conv3d(
            3,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        slow.blocks[-1].proj = nn.Linear(2048, class_num)
        return slow

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        x = self.model.blocks[0](video)
        x = self.se_fusions[0](
            x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
        )

        x = self.model.blocks[1](x)
        x = self.se_fusions[1](
            x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
        )

        x = self.model.blocks[2](x)
        x = self.se_fusions[2](
            x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
        )

        x = self.model.blocks[3](x)
        x = self.se_fusions[3](
            x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
        )

        x = self.model.blocks[4](x)
        x = self.se_fusions[4](
            x, F.interpolate(attn_map, size=x.shape[-3:], mode="trilinear")
        )

        x = self.model.blocks[5](x)

        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf

    hparams = OmegaConf.create({"model": {"model_class_num": 3}})
    model = SEFusionRes3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)
    output = model(video, attn_map)
    print(output.shape)  # Expected output shape: [2, 3] or [2, class_num]
