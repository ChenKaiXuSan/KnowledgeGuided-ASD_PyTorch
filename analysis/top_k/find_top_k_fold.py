#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_analysis.py  ‒ 轻量级 K-Fold 结果收集 & 汇总工具
------------------------------------------------------
• 支持多模型（模型名→结果目录）批量评测
• find_top_folds：按指定指标选 Top-k 折
• aggregate_folds ：把若干折并成一个总体并给出最终指标
• 可一键保存 TSV / XLSX

Author : Kaixu Chen   (refactor by ChatGPT)
Date   : 2025-06-28
"""

from __future__ import annotations
import heapq, itertools, sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────────────────────

METRIC_FNS = {
    "accuracy": MulticlassAccuracy,
    "precision": MulticlassPrecision,
    "recall": MulticlassRecall,
    "f1_score": MulticlassF1Score,
    "auroc": MulticlassAUROC,
}


def compute_metrics(
    pred: torch.Tensor, label: torch.Tensor, num_class: int = 3
) -> Tuple[pd.Series, torch.Tensor]:
    """返回 1 行 pd.Series（6 个数值指标）+ 归一化混淆矩阵"""
    metric_vals = {}
    for name, fn_cls in METRIC_FNS.items():
        metric_vals[name] = fn_cls(num_class)(pred, label).item()

    cm = MulticlassConfusionMatrix(num_class, normalize="true")(pred, label)
    return pd.Series(metric_vals), cm.cpu() * 100


def iter_folds(result_dir: Path):
    """yield (fold_idx, pred_tensor, label_tensor)"""
    for label_file in result_dir.glob("*_label.pt"):
        idx = int(label_file.stem.split("_")[0])
        pred_file = result_dir / f"{idx}_pred.pt"
        if pred_file.exists():
            yield idx, torch.load(pred_file, map_location="cpu"), torch.load(
                label_file, map_location="cpu"
            ).to(torch.int)


# ──────────────────────────────────────────────────────────────────────────────
# 主要 API
# ──────────────────────────────────────────────────────────────────────────────


def find_top_folds(
    model_paths: Dict[str, str],
    *,
    top_k: int = 5,
    key_metric: str = "accuracy",
    num_class: int = 3,
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, dict]]]:
    """
    返回：
    • all_folds_df     : 每一折一行的指标 DataFrame
    • best_fold_dict   : {model: {fold_idx: {"metrics":Series,"cm":Tensor}}}
    """
    rows: List[pd.Series] = []
    best_fold_dict: Dict[str, Dict[int, dict]] = {}

    iterable = model_paths.items()
    if use_tqdm:
        iterable = tqdm(iterable, desc="Models")

    for model_name, path_str in iterable:
        path = Path(path_str)
        heap: List[Tuple[float, int]] = []  # (metric, idx)

        for idx, pred, label in iter_folds(path):
            metrics_s, cm = compute_metrics(pred, label, num_class)

            row = pd.Series({"model": model_name, "fold": idx})._append(
                metrics_s, verify_integrity=True
            )
            rows.append(row)

            metric_val = metrics_s[key_metric]
            heapq.heappush(heap, (metric_val, idx))
            if len(heap) > top_k:
                heapq.heappop(heap)

            best_fold_dict.setdefault(model_name, {})[idx] = {
                "metrics": metrics_s,
                "cm": cm,
            }

        # 只保留 top-k
        keep = {idx for _, idx in heap}
        best_fold_dict[model_name] = {
            idx: data for idx, data in best_fold_dict[model_name].items() if idx in keep
        }

    all_folds_df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(False)
    return all_folds_df, best_fold_dict


def aggregate_folds(
    best_fold_dict: Dict[str, Dict[int, dict]],
    model_paths: Dict[str, str],
    num_class: int = 3,
) -> pd.DataFrame:
    """
    把选出来的折在样本维度 concat，重新算一次总指标。
    返回 DataFrame（每模型一行）。
    """
    rows = []
    res_cm = {}
    for model, folds in best_fold_dict.items():
        all_pred, all_label = [], []
        for idx in folds:
            p = Path(model_paths[model])
            all_pred.append(torch.load(p / f"{idx}_pred.pt", map_location="cpu"))
            all_label.append(
                torch.load(p / f"{idx}_label.pt", map_location="cpu").to(torch.int)
            )

        all_pred = torch.cat(all_pred, 0)
        all_label = torch.cat(all_label, 0)
        metrics_s, cm = compute_metrics(all_pred, all_label, num_class)
        rows.append(pd.Series({"model": model})._append(metrics_s))
        res_cm[model] = cm

    return pd.DataFrame(rows), res_cm


def save_df(df: pd.DataFrame, outfile: str):
    ext = Path(outfile).suffix.lower()
    if ext in {".tsv", ".txt"}:
        df.to_csv(outfile, sep="\t", index=False)
    elif ext in {".csv"}:
        df.to_csv(outfile, index=False)
    elif ext in {".xlsx"}:
        df.to_excel(outfile, index=False)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    print(f"✔ Saved → {outfile}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 便利
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from analysis.top_k.path import *  # or import your yaml & safe_load

    server = "ws3"  # 替换为实际的模型路径字典
    TOP_K = 3
    KEY_METRIC = "accuracy"
    NUM_CLASS = 3

    SAVE_PATH = Path("logs/analysis_results")

    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    all_folds_df, best_fold_dict = find_top_folds(
        server, top_k=TOP_K, key_metric=KEY_METRIC, num_class=NUM_CLASS
    )

    print("\n=== Each Fold Metrics (tsv) ===")
    print(all_folds_df.to_csv(sep="\t", index=False))

    summary_df, res_cm = aggregate_folds(best_fold_dict, server, NUM_CLASS)
    print("\n=== Aggregated Final Metrics ===")
    print(summary_df.to_csv(sep="\t", index=False))

    # 按需保存
    save_df(all_folds_df, SAVE_PATH / "all_folds.tsv")
    save_df(summary_df, SAVE_PATH / f"summary_top{TOP_K}.tsv")

    # 保存混淆矩阵为 PNG 图像
    cm_img_dir = SAVE_PATH / Path(f"confusion_matrix_images_top{TOP_K}")
    cm_img_dir.mkdir(exist_ok=True)

    for model, cm in res_cm.items():
        plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})
        axis_labels = ["ASD", "DHS", "LCS_HipOA"]

        # draw confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            xticklabels=axis_labels,
            yticklabels=axis_labels,
            vmin=0,
            vmax=100,
        )
        # plt.title(f'{title} (%)', fontsize=30)
        plt.title(f"{model} (%)", fontsize=30)
        plt.ylabel("Actual Label", fontsize=30)
        plt.xlabel("Predicted Label", fontsize=30)
        plt.savefig(cm_img_dir / f"{model}_cm.png")
        plt.close()
        print(
            f"✔ Saved confusion matrix image for {model} → {cm_img_dir / f'{model}_cm.png'}"
        )

    import shutil

    topk_pt_dir = SAVE_PATH / f"top{TOP_K}_folds_pt"
    topk_pt_dir.mkdir(exist_ok=True)

    for model, folds in best_fold_dict.items():
        model_dir = topk_pt_dir / model
        model_dir.mkdir(exist_ok=True)

        model_path = Path(server[model])  # 替换 ws3 为你当前的模型路径变量

        for idx in folds:
            pred_file = model_path / f"{idx}_pred.pt"
            label_file = model_path / f"{idx}_label.pt"

            if pred_file.exists():
                shutil.copy(pred_file, model_dir / f"{idx}_pred.pt")
            if label_file.exists():
                shutil.copy(label_file, model_dir / f"{idx}_label.pt")

    print(f"✔ Top-{TOP_K} fold .pt files saved to → {topk_pt_dir.resolve()}")
