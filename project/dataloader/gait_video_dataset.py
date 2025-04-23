#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/gait_video_dataset.py
Project: /workspace/code/project/dataloader
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday April 22nd 2025 11:18:09 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

23-04-2025	Kaixu Chen	init the code.
"""

from __future__ import annotations

import logging
import json

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png
from torchvision.transforms.v2.functional import (
    uniform_temporal_subsample_video,
    uniform_temporal_subsample,
)

from project.dataloader.med_attn_map import MedAttnMap

logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        doctor_res_path: str = None,
        skeleton_path: str = None,
    ) -> None:

        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._experiment = experiment

        if "True" in self._experiment:
            self.attn_map = MedAttnMap(doctor_res_path, skeleton_path)

    def move_transform(self, vframes: list[torch.Tensor]) -> None:

        if self._transform is not None:
            video_t_list = []
            for video_t in vframes:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)

            return torch.stack(video_t_list, dim=0)  # c, t, h, w
        else:
            print("no transform")
            return torch.stack(vframes, dim=0)

    def __len__(self):
        return len(self._labeled_videos)

    def __getitem__(self, index) -> Any:

        # load the video tensor from json file
        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        # load video info from json file
        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]

        try:
            vframes, _, _ = read_video(video_path, output_format="TCHW")
        except Exception as e:
            _video_path = video_path.replace("/data/", "/dataset/")
            vframes, _, _ = read_video(_video_path, output_format="TCHW")

            logger.warning(
                f"replace the video path {video_path} to {_video_path}, because of {e}"
            )

        label = file_info_dict["label"]
        disease = file_info_dict["disease"]
        gait_cycle_index = file_info_dict["gait_cycle_index"]
        bbox_none_index = file_info_dict["none_index"]
        bbox = file_info_dict["bbox"]

        # todo: here generate the attn map with skeleton
        attn_map = self.attn_map(
            video_name=video_name,
            video_path=video_path,
            disease=disease,
            vframes=vframes,
        )

        # print(f"video name: {video_name}, gait cycle index: {gait_cycle_index}")
        if "True" in self._experiment:
            # should return the new frame, named temporal mix.
            defined_vframes = self._temporal_mix(vframes, gait_cycle_index, bbox)
            defined_vframes = self.move_transform(defined_vframes)

        elif "late_fusion" in self._experiment:

            stance_vframes, used_gait_idx = split_gait_cycle(
                vframes, gait_cycle_index, 0
            )
            swing_vframes, used_gait_idx = split_gait_cycle(
                vframes, gait_cycle_index, 1
            )

            # * keep shape
            if len(stance_vframes) > len(swing_vframes):
                stance_vframes = stance_vframes[: len(swing_vframes)]
            elif len(stance_vframes) < len(swing_vframes):
                swing_vframes = swing_vframes[: len(stance_vframes)]

            trans_stance_vframes = self.move_transform(stance_vframes)
            trans_swing_vframes = self.move_transform(swing_vframes)

            # * 将不同的phase组合成一个batch返回
            defined_vframes = torch.stack(
                [trans_stance_vframes, trans_swing_vframes], dim=-1
            )

        elif "single" in self._experiment:
            if "stance" in self._experiment:
                defined_vframes, used_gait_idx = split_gait_cycle(
                    vframes, gait_cycle_index, 0
                )
            elif "swing" in self._experiment:
                defined_vframes, used_gait_idx = split_gait_cycle(
                    vframes, gait_cycle_index, 1
                )

            defined_vframes = self.move_transform(defined_vframes)

        else:
            raise ValueError("experiment name is not correct")

        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index,
            "bbox_none_index": bbox_none_index,
        }

        return sample_info_dict


def labeled_gait_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    doctor_res_path: str = None,
    skeleton_path: str = None,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        transform=transform,
        labeled_video_paths=dataset_idx,
        doctor_res_path=doctor_res_path,
        skeleton_path=skeleton_path,
    )

    return dataset
