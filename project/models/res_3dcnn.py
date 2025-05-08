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


class Res3DCNN(nn.Module):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.model = self.init_resnet(self.model_class_num)

        self.fuse_method = hparams.model.fuse_method

    def init_resnet(self, input_channel: int = 3) -> nn.Module:

        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

        if self.fuse_method == "concat":
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
        slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        return slow

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            attn_map: (B, 1, T, H, W)

        Returns:
            torch.Tensor: (B, C, T, H, W)
        """

        if self.fuse_method == "concat":
            # video = torch.cat((video, attn_map), dim=1)
            _input = torch.cat([video, attn_map], dim=1)
        elif self.fuse_method == "add":
            _input = video + attn_map
        elif self.fuse_method == "mul":
            _input = video * attn_map
        elif self.fuse_method == "none":
            _input = video
        else:
            raise KeyError(
                f"the fuse method {self.fuse_method} is not in the model zoo"
            )

        output = self.model(_input)

        return output
