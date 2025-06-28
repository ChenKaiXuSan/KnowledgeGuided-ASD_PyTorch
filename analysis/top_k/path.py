#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/analysis/path.py
Project: /workspace/code/analysis
Created Date: Saturday June 28th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday June 28th 2025 1:20:50 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

# dict for title/res path pair
# running on ws3
# 2025-06-27 first run

ws3 = {
    "add": "/workspace/code/logs/3dcnn_attn_map_True_add/2025-06-21/13-28-55/best_preds",
    "mul": "/workspace/code/logs/3dcnn_attn_map_True_mul/2025-06-21/13-28-55/best_preds",
    "concat": "/workspace/code/logs/3dcnn_attn_map_True_concat/2025-06-24/09-12-09/best_preds",
    "avg": "/workspace/code/logs/3dcnn_attn_map_True_avg/2025-06-24/09-12-09/best_preds",
    "late": "/workspace/code/logs/3dcnn_attn_map_True_late/2025-06-24/09-12-09/best_preds",
    "none": "/workspace/code/logs/3dcnn_attn_map_True_none/2025-06-24/09-12-09/best_preds",
}


# dict for title/res path pair
# running on pegasus
# 2025-06027 first run

pegasus = {
    "add": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_add_5/2025-06-27/10-34-53/best_preds",
    "mul": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_mul_5/2025-06-27/10-34-54/best_preds",
    "concat": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_concat_5/2025-06-27/10-34-54/best_preds",
    "avg": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_avg_5/2025-06-27/10-34-54/best_preds",
    "late": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_late_5/2025-06-27/10-34-54/best_preds",
    "none": "/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_none_5/2025-06-27/10-34-54/best_preds",
}

# dict for title/res path pair 
# running on LELAB new 
# wait for ws3 to be ready

LELAB_new = {
	"add":
		"/workspace/code/logs/3dcnn_attn_map_True_add/2025-06-19/08-58-07/best_preds",
	"mul":
		"/workspace/code/logs/3dcnn_attn_map_True_mul/2025-06-19/08-58-07/best_preds",
	"concat":
		"/workspace/code/logs/3dcnn_attn_map_True_concat/2025-06-19/08-58-07/best_preds",
	# "avg":
	# 	"",
	# "late":
	# 	"",
	"none":
		"/workspace/code/logs/3dcnn_attn_map_True_none/2025-06-19/08-58-07/best_preds",
}

