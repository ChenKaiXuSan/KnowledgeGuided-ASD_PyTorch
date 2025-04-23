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

import torch
from torchvision.io import write_png

import pandas as pd 

COCO_KEYPOINTS = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

region_to_keypoints = {
    "foot": [15, 16],
    "wrist": [9, 10],
    "shoulder": [5, 6],
    "lumbar_pelvis": [11, 12],
    "head": [0, 1, 2, 3, 4]
}

class MedAttnMap:

    def __init__(
        self,
        doctor_res_path: str,
        skeleton_path: str,
    ) -> None:

        self.doctor_res = self.load_doctor_res(doctor_res_path)
        self.skeleton = self.load_skeleton(skeleton_path)

    def load_doctor_res(self, docker_res_path: str) -> Optional[dict]:
        """
        Load the doctor result from the given video path.
        """
        doctor_1 = pd.read_csv(docker_res_path + "/doctor1.csv")
        doctor_2 = pd.read_csv(docker_res_path + "/doctor2.csv")

        return doctor_1, doctor_2  
    
    def find_doctor_res(self, video_name: str) -> Optional[dict]:
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
    

    def load_skeleton(self, skeleton_path: str) -> Optional[dict]:
        """
        Load the skeleton from the given video path.
        """
        # Load the skeleton from the video path
        skeleton = pd.read_pickle(skeleton_path + "/whole_annotations.pkl")
        return skeleton

    def find_skeleton(self, video_name: str) -> Optional[dict]:
        """
        Find the skeleton for the given video path.
        """
        res = [] 

        # Find the skeleton for the given video path
        for one in self.skeleton['annotations']:

            keypoint = one['keypoint']
            keypoint_score = one['keypoint_score']
            total_frame = one['total_frames']
            _video_name = one['frame_dir'].split('/')[-1]

            if video_name in _video_name:

                res.append(one)

        return res
            
    
    def generate_attention_map(self, vframes: torch.Tensor, mapped_keypoint: list) -> None:
        """
        Generate the attention map for the given video path.
        """

        t, c, h, w = vframes.shape


        pass

    def save_attention_map(self, attention_map: Any, save_path: str) -> None:
        """
        Save the generated attention map to the specified path.
        """
        # Save the attention map
        pass

    def __call__(self, video_path, disease, vframes, video_name) -> None:

        attn_map = [] 

        t, c, h, w = vframes.shape

        # for one video file
        # * 1 find the doctor result
        doctor_attn, mapped_keypoint = self.find_doctor_res(video_name)

        # * 2 find the skeleton
        # ? 为什么会有两个skeleton被找出来？
        skeleton = self.find_skeleton(video_name)

        # * 2 generate the attention map
        # todo: here should be the attention map generation
        attn_map = self.generate_attention_map(vframes, mapped_keypoint, skeleton[0])

        return attn_map 
