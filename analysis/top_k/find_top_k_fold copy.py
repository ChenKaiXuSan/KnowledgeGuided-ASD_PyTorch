#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_analysis.py  ‒ 轻量级 K-Fold 结果收集 & 汇总工具
------------------------------------------------------
• 支持多模型（模型名→结果目录）批量评测
• find_top_folds：按指定指标选 Top-k 折
• aggregate_folds ：把若干折并成一个总体并给出最终指标
• 可一键保存 TSV / XLSX / Confusion Matrix / Top-k pt 文件

Author : Kaixu Chen   (refactor by ChatGPT)
Date   : 2025-06-28
"""

from __future__ import annotations
import heapq
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
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
    metric_vals = {
        name: fn_cls(num_class)(pred, label).item() for name, fn_cls in METRIC_FNS.items()
    }
    cm = MulticlassConfusionMatrix(num_class, normalize="true")(pred, label)
    return pd.Series(metric_vals), cm.cpu() * 100

def iter_folds(result_dir: Path):
    for label_file in result_dir.glob("*_label.pt"):
        idx = int(label_file.stem.split("_")[0])
        pred_file = result_dir / f"{idx}_pred.pt"
        if pred_file.exists():
            yield idx, torch.load(pred_file, map_location="cpu"), torch.load(label_file, map_location="cpu").to(torch.int)

# ──────────────────────────────────────────────────────────────────────────────
# 主要 API
# ──────────────────────────────────────────────────────────────────────────────

def find_top_folds(
    model_paths: Dict[str, str], *,
    top_k: int = 5,
    key_metric: str = "accuracy",
    num_class: int = 3,
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, dict]]]:
    rows = []
    best_fold_dict = {}

    iterable = tqdm(model_paths.items(), desc="Models") if use_tqdm else model_paths.items()

    for model_name, path_str in iterable:
        path = Path(path_str)
        heap = []
        for idx, pred, label in iter_folds(path):
            metrics_s, cm = compute_metrics(pred, label, num_class)
            row = pd.Series({"model": model_name, "fold": idx})._append(metrics_s, verify_integrity=True)
            rows.append(row)
            heapq.heappush(heap, (metrics_s[key_metric], idx))
            if len(heap) > top_k:
                heapq.heappop(heap)
            best_fold_dict.setdefault(model_name, {})[idx] = {"metrics": metrics_s, "cm": cm}

        keep = {idx for _, idx in heap}
        best_fold_dict[model_name] = {
            idx: data for idx, data in best_fold_dict[model_name].items() if idx in keep
        }

    return pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(False), best_fold_dict

def aggregate_folds(
    best_fold_dict: Dict[str, Dict[int, dict]],
    model_paths: Dict[str, str],
    num_class: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    rows, res_cm = [], {}
    for model, folds in best_fold_dict.items():
        all_pred, all_label = [], []
        for idx in folds:
            path = Path(model_paths[model])
            all_pred.append(torch.load(path / f"{idx}_pred.pt", map_location="cpu"))
            all_label.append(torch.load(path / f"{idx}_label.pt", map_location="cpu").to(torch.int))
        pred, label = torch.cat(all_pred), torch.cat(all_label)
        metrics_s, cm = compute_metrics(pred, label, num_class)
        rows.append(pd.Series({"model": model})._append(metrics_s))
        res_cm[model] = cm
    return pd.DataFrame(rows), res_cm

def save_df(df: pd.DataFrame, outfile: Path):
    ext = outfile.suffix.lower()
    if ext in {".tsv", ".txt"}:
        df.to_csv(outfile, sep="\t", index=False)
    elif ext == ".csv":
        df.to_csv(outfile, index=False)
    elif ext == ".xlsx":
        df.to_excel(outfile, index=False)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    print(f"✔ Saved → {outfile}")

def save_confusion_matrices(res_cm: Dict[str, torch.Tensor], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for model, cm in res_cm.items():
        plt.rcParams.update({'font.size': 30, 'font.family': 'sans-serif'})
        labels = ['ASD', 'DHS', 'LCS_HipOA']
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
        plt.title(f"{model} (%)")
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_dir / f"{model}_cm.png")
        plt.close()
        print(f"✔ Saved confusion matrix image for {model} → {save_dir / f'{model}_cm.png'}")

def save_topk_pt_files(best_fold_dict: Dict[str, Dict[int, dict]], model_paths: Dict[str, str], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for model, folds in best_fold_dict.items():
        model_dir = save_dir / model
        model_dir.mkdir(exist_ok=True)
        for idx in folds:
            src_path = Path(model_paths[model])
            for kind in ["pred", "label"]:
                src = src_path / f"{idx}_{kind}.pt"
                dst = model_dir / f"{idx}_{kind}.pt"
                if src.exists():
                    shutil.copy(src, dst)
        print(f"✔ Copied Top-{len(folds)} fold .pt files for {model} → {model_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from path import pegasus, ws3, LELAB_new  # or import your yaml & safe_load
    
    server = ws3

    TOP_K = 5
    KEY_METRIC = "accuracy"
    NUM_CLASS = 3
    SAVE_PATH = Path("logs/analysis_results")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    all_folds_df, best_fold_dict = find_top_folds(server, top_k=TOP_K, key_metric=KEY_METRIC, num_class=NUM_CLASS)
    summary_df, res_cm = aggregate_folds(best_fold_dict, server, NUM_CLASS)

    save_df(all_folds_df, SAVE_PATH / "all_folds.tsv")
    save_df(summary_df, SAVE_PATH / f"summary_top{TOP_K}.tsv")
    save_confusion_matrices(res_cm, SAVE_PATH / f"confusion_matrix_images_top{TOP_K}")
    save_topk_pt_files(best_fold_dict, server, SAVE_PATH / f"top{TOP_K}_folds_pt")
