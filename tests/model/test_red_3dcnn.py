#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/tests/model/test_red_3dcnn.py
Project: /home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/tests/model
Created Date: Friday June 27th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday June 27th 2025 10:09:03 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

# tests/test_res3dcnn.py
import pytest
import torch
from omegaconf import OmegaConf

# 根据你的项目结构调整导入位置
from project.models.res_3dcnn import Res3DCNN


@pytest.mark.parametrize(
    "fuse_method",
    ["concat", "add", "mul", "avg", "none", "late"],
)
def test_forward_output_shape(fuse_method):
    """
    确认不同 fuse_method 下，模型能够正常前向并输出正确形状。
    """
    num_classes = 3
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": num_classes,
                "fuse_method": fuse_method,
                # ckpt_path 可按需补全；此处用空字符串代表不用预训练
                "ckpt_path": "",
            }
        }
    )

    model = Res3DCNN(hparams).eval()  # 评估模式避免 dropout 影响

    batch_size, T, H, W = 2, 8, 224, 224
    video = torch.randn(batch_size, 3, T, H, W)

    # 所有模式都需要 attn_map；部分模式内部可能忽略或广播
    attn_map = torch.randn(batch_size, 1, T, H, W)

    with torch.no_grad():
        output = model(video, attn_map)

    # 统一要求输出维度为 (B, num_classes)
    assert output.shape == (batch_size, num_classes), (
        f"fuse_method='{fuse_method}' 输出形状应为 "
        f"({batch_size}, {num_classes})，实际 {tuple(output.shape)}"
    )


def test_invalid_fuse_method_raises():
    """
    当 fuse_method 不在支持列表时，应抛出 KeyError。
    """
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 3,
                "fuse_method": "unsupported_method",
                "ckpt_path": "",
            }
        }
    )

    model = Res3DCNN(hparams)

    video = torch.randn(1, 3, 8, 224, 224)
    attn_map = torch.randn(1, 1, 8, 224, 224)

    with pytest.raises(KeyError):
        model(video, attn_map)
