#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/med_attn_map.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 6:11:19 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import os
import torch
from torchvision.utils import save_image

import pandas as pd

COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

region_to_keypoints = {
    "foot": [15, 16],
    "wrist": [9, 10],
    "shoulder": [5, 6],
    "lumbar_pelvis": [11, 12],
    "head": [0, 1, 2, 3, 4],
}


class MedAttnMap:

    def __init__(
        self,
        doctor_res_path: str,
        skeleton_path: str,
    ) -> None:

        self.doctor_res = self.load_doctor_res(doctor_res_path)
        self.skeleton = pd.read_pickle(skeleton_path + "/whole_annotations.pkl")

    def load_doctor_res(self, docker_res_path: str) -> list[pd.DataFrame]:
        """
        Load the doctor result from the given video path.
        """
        doctor_1 = pd.read_csv(docker_res_path + "/doctor1.csv")
        doctor_2 = pd.read_csv(docker_res_path + "/doctor2.csv")

        return doctor_1, doctor_2

    def find_doctor_res(self, video_name: str) -> list[list[str]]:
        """
        Find the doctor result for the given video path.
        """

        doctor_attn = []
        keypoint_num = []

        for one_doctor in self.doctor_res:
            for idx, row in one_doctor.iterrows():
                if row["video file name"] in video_name:
                    doctor_attn.append(row["attention"][2:-6])
                    for i in region_to_keypoints[row["attention"][2:-6]]:
                        keypoint_num.append(i)

        return set(doctor_attn), set(keypoint_num)

    def find_skeleton(self, video_name: str) -> list[dict[str, Any]]:
        """
        Find the skeleton for the given video path.
        """
        res = []

        # Find the skeleton for the given video path
        for one in self.skeleton["annotations"]:

            # keypoint = one["keypoint"]
            # keypoint_score = one["keypoint_score"]
            # total_frame = one["total_frames"]
            _video_name = one["frame_dir"].split("/")[-1]

            if video_name in _video_name:

                res.append(one)

        return res

    def generate_attention_map(
        self,
        vframes: torch.Tensor,
        mapped_keypoint: list,
        keypoint: torch.Tensor,
        confidence_score,
    ) -> torch.Tensor:
        """
        Generate the attention map for the given video path.
        """

        t, c, h, w = vframes.shape

        sigma = 0.1 * min(h, w)  # standard deviation for Gaussian kernel

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )  # shape: [H, W]

        res = []

        for frame in range(t):

            attn_maps = []

            for i in mapped_keypoint:

                x = keypoint[0, frame, i, 0] * w
                y = keypoint[0, frame, i, 1] * h

                # none keypoint
                if x < 0 or y < 0:
                    attn_maps.append(torch.zeros((h, w)))
                    continue

                dist_squared = (x_grid - x) ** 2 + (y_grid - y) ** 2
                heatmap = torch.exp(-dist_squared / (2 * sigma**2))

                curr_confidence = confidence_score[0, frame, i]
                if curr_confidence > 0.8:
                    heatmap *= curr_confidence

                attn_maps.append(heatmap)

            attn_stack = torch.stack(attn_maps, dim=0)  # [K, H, W]
            attn_mean = torch.mean(attn_stack, dim=0).unsqueeze(0)

            res.append(attn_mean)

        return torch.stack(res, dim=0)  # [T, H, W]

    def save_attention_map(
        self, attention_map: torch.Tensor, save_path: str, video_name: str
    ) -> None:
        """
        Save the generated attention map to the specified path.
        """
        # Save the attention map
        t, *_ = attention_map.shape

        save_pth = os.path.join(save_path, "attention_map", video_name)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        for i in range(t):

            save_image(attention_map[i], save_pth + f"/attn_{i}.png", normalize=True)

    def __call__(self, video_path, disease, vframes, video_name) -> torch.Tensor:

        # for one video file
        # * 1 find the doctor result
        doctor_attn, mapped_keypoint = self.find_doctor_res(video_name)

        # * 2 find the skeleton
        # FIXME: 为什么会有两个skeleton被找出来？
        skeleton = self.find_skeleton(video_name)

        # * 3 generate the attention map
        attn_map = self.generate_attention_map(
            vframes,
            mapped_keypoint,
            skeleton[0]["keypoint"],
            confidence_score=skeleton[0]["keypoint_score"],
        )

        # self.save_attention_map(attn_map, "logs", video_name)

        return attn_map
