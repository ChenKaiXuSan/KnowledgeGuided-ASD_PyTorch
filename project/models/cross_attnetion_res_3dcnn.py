#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/cross_attention_res_3dcnn_visual.py
Project: /workspace/code/project/models
Created Date: 2025-06-19
Author: Kaixu Chen
Comment: Res3DCNN with cross-attention fusion and attention map visualization support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, name=None):
        super().__init__()
        self.query_proj = nn.Conv3d(dim_q, dim_out, kernel_size=1)
        self.key_proj = nn.Conv3d(dim_kv, dim_out, kernel_size=1)
        self.value_proj = nn.Conv3d(dim_kv, dim_out, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.name = name or "unnamed"
        self.save_attn = False
        self.attn_map = None  # for storing

    def forward(self, x, context):
        B, C, T, H, W = x.shape
        q = self.query_proj(x).flatten(2).transpose(1, 2)   # [B, THW, C]
        k = self.key_proj(context).flatten(2)               # [B, C, THW]
        v = self.value_proj(context).flatten(2).transpose(1, 2)  # [B, THW, C]
        attn = torch.bmm(q, k) / (k.shape[1] ** 0.5)         # [B, THW, THW]
        attn = self.softmax(attn)

        if self.save_attn:
            self.attn_map = attn.detach().cpu()

        out = torch.bmm(attn, v).transpose(1, 2).view(B, -1, T, H, W)
        return out + x


class Res3DCNN(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.model_class_num = hparams.model.model_class_num
        self.model = self.init_resnet(self.model_class_num)

        self.attn_fusions = nn.ModuleList([
            CrossAttentionFusion(64, 1, 64, name="res1"),
            CrossAttentionFusion(256, 1, 256, name="res2"),
            CrossAttentionFusion(512, 1, 512, name="res3"),
            CrossAttentionFusion(1024, 1, 1024, name="res4"),
            CrossAttentionFusion(2048, 1, 2048, name="res5"),
        ])

        for fusion in self.attn_fusions:
            fusion.save_attn = True

    @staticmethod
    def init_resnet(class_num: int = 3) -> nn.Module:
        slow = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        slow.blocks[0].conv = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        slow.blocks[-1].proj = nn.Linear(2048, class_num)
        return slow

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        x = self.model.blocks[0](video)
        x = self.attn_fusions[0](x, F.interpolate(attn_map, size=x.shape[-3:], mode='trilinear'))

        x = self.model.blocks[1](x)
        x = self.attn_fusions[1](x, F.interpolate(attn_map, size=x.shape[-3:], mode='trilinear'))

        x = self.model.blocks[2](x)
        x = self.attn_fusions[2](x, F.interpolate(attn_map, size=x.shape[-3:], mode='trilinear'))

        x = self.model.blocks[3](x)
        x = self.attn_fusions[3](x, F.interpolate(attn_map, size=x.shape[-3:], mode='trilinear'))

        x = self.model.blocks[4](x)
        x = self.attn_fusions[4](x, F.interpolate(attn_map, size=x.shape[-3:], mode='trilinear'))

        x = self.model.blocks[5](x)
        
        return x

    def save_attention_maps(self, save_dir="attn_vis"):
        os.makedirs(save_dir, exist_ok=True)
        for i, fusion in enumerate(self.attn_fusions):
            if fusion.attn_map is not None:
                B, N, N2 = fusion.attn_map.shape
                for b in range(min(1, B)):
                    plt.imshow(fusion.attn_map[b].mean(0).view(int(N ** 0.5), -1))
                    plt.colorbar()
                    plt.title(f"{fusion.name}_sample{b}_attn")
                    plt.savefig(f"{save_dir}/layer{i}_sample{b}.png")
                    plt.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    hparams = OmegaConf.create({
        "model": {
            "model_class_num": 3
        }
    })
    model = Res3DCNN(hparams)
    video = torch.randn(2, 3, 8, 224, 224)
    attn_map = torch.randn(2, 1, 8, 224, 224)
    output = model(video, attn_map)
    # model.save_attention_maps()  # Save attention maps after forward pass
    print(output.shape)
