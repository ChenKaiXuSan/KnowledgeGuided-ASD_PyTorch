#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_analysis.py  ‒ 轻量级 K‑Fold 结果收集 & 汇总工具
------------------------------------------------------
• 支持多模型（模型名→结果目录）批量评测
• 两种折选择策略
  1. **top**  ‒ 按指定指标选每个模型的 Top‑k 折（默认逻辑）
  2. **same** ‒ 保留 _所有模型都同时拥有的_ 前 k 个折（例如折 0‑4），方便横向对比
• aggregate_folds ：把若干折并成一个总体并给出最终指标
• 一键保存 TSV / XLSX / Confusion Matrix / 选中折的 .pt 文件

Author : Kaixu Chen   (refactor by ChatGPT)
Updated: 2025‑06‑29
"""

from __future__ import annotations
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Set
import shutil

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
    "accuracy":  MulticlassAccuracy,
    "precision": MulticlassPrecision,
    "recall":    MulticlassRecall,
    "f1_score":  MulticlassF1Score,
    "auroc":     MulticlassAUROC,
}

def compute_metrics(pred: torch.Tensor, label: torch.Tensor, *, num_class: int = 3) -> Tuple[pd.Series, torch.Tensor]:
    """返回 1 行 pd.Series（五个数值指标）+ 归一化混淆矩阵"""
    mvals = {name: fn(num_class)(pred, label).item() for name, fn in METRIC_FNS.items()}
    cm = MulticlassConfusionMatrix(num_class, normalize="true")(pred, label).cpu() * 100
    return pd.Series(mvals), cm

def iter_folds(result_dir: Path):
    """yield (fold_idx, pred_tensor, label_tensor)"""
    for lbl_file in result_dir.glob("*_label.pt"):
        idx = int(lbl_file.stem.split("_")[0])
        pred_file = result_dir / f"{idx}_pred.pt"
        if pred_file.exists():
            yield idx, torch.load(pred_file, map_location="cpu"), torch.load(lbl_file,  map_location="cpu").to(torch.int)

# ──────────────────────────────────────────────────────────────────────────────
# 折选择策略
# ──────────────────────────────────────────────────────────────────────────────

FoldSelectMode = Literal["top", "same"]

def _same_k_indices(model_paths: Dict[str, str], k: int) -> List[int]:
    """返回所有模型共同拥有的最小 k 个折号。如果不足 k，抛出 ValueError"""
    common: Set[int] | None = None
    for path in model_paths.values():
        indices = {int(p.stem.split("_")[0]) for p in Path(path).glob("*_label.pt")}
        common = indices if common is None else common & indices
    if not common:
        raise ValueError("❌ 无共同折可比较！")
    sel = sorted(common)[:k]
    if len(sel) < k:
        raise ValueError(f"❌ 共同折仅 {len(sel)} 个，不足要求的 k={k}")
    return sel

# ──────────────────────────────────────────────────────────────────────────────
# 主要 API
# ──────────────────────────────────────────────────────────────────────────────

def collect_folds(
    model_paths: Dict[str, str],
    *,
    k: int = 5,
    mode: FoldSelectMode = "top",          # "top" or "same"
    key_metric: str = "accuracy",          # 仅当 mode=="top" 使用
    num_class: int = 3,
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, dict]]]:
    """根据选择策略返回 (all_folds_df, selected_fold_dict)"""

    rows: List[pd.Series] = []
    selected: Dict[str, Dict[int, dict]] = {}

    # 若 same‑k 先计算共同折
    same_k_indices: List[int] | None = None
    if mode == "same":
        same_k_indices = _same_k_indices(model_paths, k)
        print(f"✔ 同比模式: 使用折 {same_k_indices}")

    iterable = tqdm(model_paths.items(), desc="Models") if use_tqdm else model_paths.items()

    for model, path_str in iterable:
        path = Path(path_str)
        heap: List[Tuple[float, int]] = []  # (metric, idx) 仅 top 模式用

        for idx, pred, label in iter_folds(path):
            metrics_s, cm = compute_metrics(pred, label, num_class=num_class)
            rows.append(pd.Series({"model": model, "fold": idx})._append(metrics_s))

            # 记录信息
            selected.setdefault(model, {})[idx] = {"metrics": metrics_s, "cm": cm}

            # 如果 top 模式维护小顶堆
            if mode == "top":
                heapq.heappush(heap, (metrics_s[key_metric], idx))
                if len(heap) > k:
                    heapq.heappop(heap)

        # 依据模式筛选该模型最终折
        if mode == "top":
            keep = {idx for _, idx in heap}
        else:  # same
            keep = set(same_k_indices)  # type: ignore[arg-type]
        selected[model] = {idx: data for idx, data in selected[model].items() if idx in keep}

    all_folds_df = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(False)
    return all_folds_df, selected


def aggregate_folds(selected: Dict[str, Dict[int, dict]], model_paths: Dict[str, str], *, num_class: int = 3):
    rows, res_cm = [], {}
    for model, folds in selected.items():
        preds, labels = [], []
        for idx in folds:
            p = Path(model_paths[model])
            preds.append(torch.load(p / f"{idx}_pred.pt", map_location="cpu"))
            labels.append(torch.load(p / f"{idx}_label.pt", map_location="cpu").to(torch.int))
        pred_cat, label_cat = torch.cat(preds), torch.cat(labels)
        metrics_s, cm = compute_metrics(pred_cat, label_cat, num_class=num_class)
        rows.append(pd.Series({"model": model})._append(metrics_s))
        res_cm[model] = cm
    return pd.DataFrame(rows), res_cm

# ──────────────────────────────────────────────────────────────────────────────
# 辅助保存
# ──────────────────────────────────────────────────────────────────────────────

def save_df(df: pd.DataFrame, outfile: Path):
    ext = outfile.suffix.lower()
    if ext in {".tsv", ".txt"}:
        df.to_csv(outfile, sep="\t", index=False)
    elif ext == ".csv":
        df.to_csv(outfile, index=False)
    elif ext == ".xlsx":
        df.to_excel(outfile, index=False)
    else:
        raise ValueError("Unsupported extension: " + ext)
    print("✔ Saved →", outfile)


def save_confusion_matrices(cm_dict: Dict[str, torch.Tensor], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    labels = ["ASD", "DHS", "LCS_HipOA"]
    for model, cm in cm_dict.items():
        plt.rcParams.update({'font.size': 30, 'font.family': 'sans-serif'})

		# draw confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
        # plt.title(f'{title} (%)', fontsize=30)
        plt.title(f"{model} (%)", fontsize=30)
        plt.ylabel('Actual Label', fontsize=30)
        plt.xlabel('Predicted Label', fontsize=30)
        plt.savefig(save_dir / f"{model}_cm.png")
        plt.close()
        print("✔ Saved CM →", save_dir / f"{model}_cm.png")


def save_pt(selected: Dict[str, Dict[int, dict]], model_paths: Dict[str, str], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for model, folds in selected.items():
        mdir = save_dir / model
        mdir.mkdir(exist_ok=True)
        src_base = Path(model_paths[model])
        for idx in folds:
            for kind in ("pred", "label"):
                src = src_base / f"{idx}_{kind}.pt"
                dst = mdir / src.name
                if src.exists():
                    shutil.copy(src, dst)
        print("✔ Copied .pt for", model, "→", mdir)

# ──────────────────────────────────────────────────────────────────────────────
# CLI 示例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from path import pegasus, ws3, LELAB_new # 替换为你的路径字典
    
    server = ws3
    MODE: FoldSelectMode = "top"   # "top" or "same"
    K            = 5                # 保留折数
    KEY_METRIC   = "accuracy"       # 只在 top 模式用
    NUM_CLASS    = 3

    OUT_DIR = Path("logs/analysis_results") / MODE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folds_df, sel_dict = collect_folds(server, k=K, mode=MODE, key_metric=KEY_METRIC, num_class=NUM_CLASS)
    agg_df, cm_dict    = aggregate_folds(sel_dict, server, num_class=NUM_CLASS)

    # 保存
    save_df(folds_df, OUT_DIR / f"all_folds_{MODE}.tsv")
    save_df(agg_df,   OUT_DIR / f"summary_{MODE}_{K}.tsv")
    save_confusion_matrices(cm_dict, OUT_DIR  / f"cm_{MODE}_{K}")
    save_pt(sel_dict, server, OUT_DIR / f"pt_{MODE}_{K}")
