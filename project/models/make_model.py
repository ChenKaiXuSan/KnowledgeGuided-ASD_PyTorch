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


import torch.nn as nn

from project.models.cross_attnetion_res_3dcnn import CrossAttentionRes3DCNN
from project.models.res_3dcnn import Res3DCNN
from project.models.res_3dcnn_atn import Res3DCNNATN


def select_model(hparams) -> nn.Module:
    """
    Select the model based on the hparams.

    Args:
        hparams: the hyperparameters of the model.

    Returns:
        nn.Module: the selected model.
    """

    model_backbone = hparams.model.backbone
    fuse_method = hparams.model.fuse_method

    if model_backbone == "3dcnn":
        if fuse_method == "cross_atn":
            model = CrossAttentionRes3DCNN(hparams)
        else:
            model = Res3DCNN(hparams)
    elif model_backbone == "3dcnn_atn":
        model = Res3DCNNATN(hparams)
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    return model
