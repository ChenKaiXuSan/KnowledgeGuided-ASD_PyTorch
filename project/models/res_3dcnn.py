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
        self.fuse_method = hparams.model.fuse_method

        self.model = self.init_resnet(
            self.model_class_num,
            self.fuse_method,
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

        assert video.shape[0] == attn_map.shape[0], "the batch size is not equal"
        assert video.shape[2] == attn_map.shape[2], "the time size is not equal"
        assert video.shape[3] == attn_map.shape[3], "the height size is not equal"
        assert video.shape[4] == attn_map.shape[4], "the width size is not equal"

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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    hparams = OmegaConf.create({
        "model": {
            "model_class_num": 3,
            "fuse_method": "add",  # can be 'concat', 'add', 'mul', 'none'
        }
    })
    model = Res3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)
    output = model(video, attn_map)
    print(output.shape)
