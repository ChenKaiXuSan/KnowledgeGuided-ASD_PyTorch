#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday April 19th 2025 7:58:58 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
"""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeVideoModule(nn.Module):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_name = hparams.model.backbone
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth
        self.model = self.initialize_walk_resnet(self.model_class_num)

        self.fuse_method = hparams.model.fuse_method

    def initialize_walk_resnet(self, input_channel: int = 3) -> nn.Module:
        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model_name == "3dcnn":
            return self.initialize_walk_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            attn_map: (B, 1, T, H, W)

        Returns:
            torch.Tensor: (B, C, T, H, W)
        """
        # video = self.video_cnn(video)
        # attn_map = self.attn_map(attn_map)

        # return video * attn_map
        return video


class ATN3DCNN(nn.Module):
    """
    make 3D CNN model with Attention Branch Network.
    https://github.com/machine-perception-robotics-group/attention_branch_network

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth

        self.stem, self.stage, self.head = self.load_resnet(
            input_channel=3, model_class_num=self.model_class_num
        )

        # make self layer
        self.relu = nn.ReLU(inplace=True)

        self.bn_att = nn.BatchNorm3d(2048)
        self.attn_conv = nn.Conv3d(
            2048, self.model_class_num, kernel_size=1, padding=0, bias=False
        )
        self.bn_att2 = nn.BatchNorm3d(self.model_class_num)
        self.attn_conv2 = nn.Conv3d(
            self.model_class_num,
            self.model_class_num,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.attn_conv3 = nn.Conv3d(
            self.model_class_num, 1, kernel_size=1, padding=0, bias=False
        )
        self.bn_att3 = nn.BatchNorm3d(1)
        self.att_gap = nn.AdaptiveAvgPool3d((16))  # copy from the original code
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool3d((8))

    @staticmethod
    def load_resnet(input_channel: int = 3, model_class_num: int = 3) -> nn.Module:
        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

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
        slow.blocks[-1].proj = nn.Linear(2048, model_class_num)

        stem = slow.blocks[0]
        stage = slow.blocks[1:5]
        head = slow.blocks[-1]

        return stem, stage, head

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = self.stem(x)  # b, 64, 8, 56, 56
        for resstage in self.stage:
            x = resstage(x)  # output: b, 2048, 8, 7, 7

        ax = self.bn_att(x)
        ax = self.relu(self.bn_att2(self.attn_conv(ax)))
        axb, axc, axt, axh, axw = ax.size()

        att = self.sigmoid(self.bn_att3(self.attn_conv3(ax)))  # b, 1, 8, 7, 7
        #
        ax = self.attn_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)
        # pred score
        rx = x * att
        rx = rx + x
        # rx = self.avgpool(rx) # ? I think this is not necessary

        rx = self.head(rx)

        # attn map
        res_att = []
        for f in range(t):
            _att = F.interpolate(
                att[:, :, f, ...], size=(h, w), mode="bilinear", align_corners=False
            )
            res_att.append(_att)

        return ax, rx, torch.stack(res_att, dim=2)


class MakeImageModule(nn.Module):
    """
    the module zoo from the torchvision lib, to make the different 2D model.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

    def make_resnet(self, input_channel: int = 3) -> nn.Module:
        if self.transfer_learning:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=True
            )
            model.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            model.fc = nn.Linear(2048, self.model_class_num)

        return model

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model_name == "resnet":
            return self.make_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")


class MakeOriginalTwoStream(nn.Module):
    """
    from torchvision make resnet 50 network.
    input is single figure.
    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

    def make_resnet(self, input_channel: int = 3):
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model
        model.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, self.model_class_num)

        return model


class CNNLSTM(nn.Module):
    """
    the cnn lstm network, use the resnet 50 as the cnn part.
    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

        self.cnn = self.make_cnn()
        # LSTM
        self.lstm = nn.LSTM(
            input_size=300, hidden_size=512, num_layers=2, batch_first=True
        )
        self.fc = nn.Linear(512, self.model_class_num)

    def make_cnn(self, input_channel: int = 3):
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model
        model.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, 300)

        return model

    def forward(self, x):
        b, c, t, h, w = x.size()

        res = []

        for i in range(b):
            hidden = None
            out = self.cnn(x[i].permute(1, 0, 2, 3))
            out, hidden = self.lstm(out, hidden)

            out = F.relu(out)
            out = self.fc(out)

            res.append(out)

        return torch.cat(res, dim=0)
