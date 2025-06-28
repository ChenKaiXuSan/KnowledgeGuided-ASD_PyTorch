import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from pathlib import Path
import pandas as pd
import heapq

from torchmetrics.classification import (
MulticlassAccuracy,
MulticlassPrecision,
MulticlassRecall,
MulticlassF1Score,
MulticlassConfusionMatrix,
MulticlassAUROC,
)

# dict for title/res path pair 
# running on pegasus
# 2025-06027 first run

pegasus = {
	"add":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_add_5/2025-06-27/10-34-53/best_preds",
	"mul":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_mul_5/2025-06-27/10-34-54/best_preds",
	"concat":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_concat_5/2025-06-27/10-34-54/best_preds",
	"avg":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_avg_5/2025-06-27/10-34-54/best_preds",
	"late":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_late_5/2025-06-27/10-34-54/best_preds",
	"none":
		"/home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch/logs/3dcnn_attn_map_True_none_5/2025-06-27/10-34-54/best_preds",
}


def metrics(all_pred: torch.Tensor, all_label: torch.Tensor, num_class: int = 3):
    # define metrics
    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    accuracy = _accuracy(all_pred, all_label).item()
    precision = _precision(all_pred, all_label).item()
    recall = _recall(all_pred, all_label).item()
    f1 = _f1_score(all_pred, all_label).item()
    auroc = _auroc(all_pred, all_label).item()
    cm = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1_score: {f1:.4f}")
    print(f"auroc: {auroc:.4f}")
    print(f"confusion_matrix:\n{cm}")

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
    }

    return cm, metrics_dict


# analysis with title/res path pair

# ----------------------------------------------------------------------
# metrics() 必须返回 (confusion_matrix, metrics_dict)
# metrics_dict 至少要包含 key_metric（默认为 "accuracy"）
# ----------------------------------------------------------------------

def find_top_folds(
        pair: dict[str, str],
        *,
        top_k: int = 5,
        key_metric: str = "accuracy",
        num_class: int = 3,
        flag: str = "with_attn_loss",
):
    """
    从给定模型-结果目录映射中，找出每个模型 key_metric 最高的 Top-k fold。

    Parameters
    ----------
    pair : dict
        {"模型名称": "/path/to/results"}，结果目录里应包含
        <fold_idx>_label.pt  和  <fold_idx>_pred.pt  两类文件。
    top_k : int, default=5
        取前多少个 fold。
    key_metric : str, default="accuracy"
        以哪一个指标排序（metric 字典里的 key）。
    num_class : int, default=3
        分类类别数，传给 metrics()。
    flag : str
        仅用于打印，方便区分不同设置。

    Returns
    -------
    summary_records : list[dict]
        每个折一次，行格式示例：
        {"Model": "modelA", "Fold": 0, "accuracy": 0.83, "precision": ..., ...}
    best_fold_records : dict[str, dict[int, dict]]
        {"模型名称": {fold_idx: {"confusion_matrix": ..., "metrics": {...}}, ...}}
        只保留该模型 top-k 的 fold。
    """
    summary_records   = []
    best_fold_records = {}

    for title, res_path in pair.items():
        res_path = Path(res_path)
        print(f"{'*'*80}\n{flag} | {title}")

        # 估计共有多少折：统计 *_label.pt 文件数
        fold_indices = sorted(
            int(p.stem.split('_')[0]) for p in res_path.glob("*_label.pt")
        )
        if not fold_indices:
            print(f"[WARN] {res_path} 里没有 *_label.pt 文件，跳过。")
            continue

        fold_metrics_heap = []  # (metric_value, fold_idx) 组成的小顶堆
        model_best_dict   = {}

        for i in fold_indices:
            label_path = res_path / f"{i}_label.pt"
            pred_path  = res_path / f"{i}_pred.pt"
            if not (label_path.exists() and pred_path.exists()):
                print(f"[Skip] Fold {i} 缺文件。")
                continue

            label = torch.load(label_path, map_location="cpu").to(torch.int)
            pred  = torch.load(pred_path,  map_location="cpu")

            cm, metric_dict = metrics(pred, label, num_class=num_class)

            # -------- 汇总到 summary_records --------
            row = {"Model": title, "Fold": i}
            row.update(metric_dict)
            summary_records.append(row)

            # -------- 用堆维护 top-k --------
            metric_val = metric_dict.get(key_metric)
            if metric_val is None:
                raise KeyError(
                    f"[metrics] 返回的字典里找不到 key_metric='{key_metric}'"
                )
            heapq.heappush(fold_metrics_heap, (metric_val, i))
            if len(fold_metrics_heap) > top_k:
                heapq.heappop(fold_metrics_heap)  # 保证堆大小 ≤ top_k

            # 先把所有 fold 的信息保存，之后再筛选
            model_best_dict[i] = {
                "confusion_matrix": cm,
                "metrics": metric_dict,
            }

        # 把 top-k fold 存入 best_fold_records
        top_fold_indices = [idx for _, idx in heapq.nlargest(
            top_k, fold_metrics_heap)]
        best_fold_records[title] = {idx: model_best_dict[idx]
                                    for idx in top_fold_indices}

        print(f"✔  一共找到 {len(model_best_dict)} folds，"
              f"已保存 top-{top_k}（按 {key_metric} 排序）。")

    return summary_records, best_fold_records

# def analysis_with_title_res_path_pair(pair: dict, flag: str = "with_attn_loss"):

# 	summary_records = []
# 	fold_records = {}
	
# 	for title, res_path in pair.items():
		
# 		fold = 0
# 		print('*' * 100)
# 		print(f"{flag}, {title}")
		
# 		fold = int(len(list(Path(res_path).iterdir())) / 2 )

# 		print(f"fold: {fold}")

# 		for i in range(fold):
# 			label = torch.load(f"{res_path}/{i}_label.pt", map_location="cpu").to(torch.int)
# 			pred = torch.load(f"{res_path}/{i}_pred.pt", map_location="cpu")

#             one_cm_data, one_metric_dict = metrics(pred, label, num_class=3)

#             fold_records[i] = {
#                 "confusion_matrix": one_cm_data,
#                 "metrics": one_metric_dict
#             }

#         summary_records, fold_records = find_5_fold(pair, flag)

def total_analysis(pair: dict, flag: str = "with_attn_loss"):

    summary_records = []
	
    for title, res_path in pair.items():
		
        fold = 0
        print('*' * 100)
        print(f"{flag}, {title}")
        all_label = []
        all_pred = []

        fold = int(len(list(Path(res_path).iterdir())) / 2 )

        print(f"fold: {fold}")

        for i in range(fold):
            label = torch.load(f"{res_path}/{i}_label.pt", map_location="cpu").to(torch.int)
            pred = torch.load(f"{res_path}/{i}_pred.pt", map_location="cpu")
            all_label.append(label)
            all_pred.append(pred)

        all_label = torch.cat(all_label, dim=0)
        all_pred = torch.cat(all_pred, dim=0)
        
        print('*' * 100)
        print(title)
        confusion_matrix_data, metric_dict = metrics(all_pred, all_label, num_class=3)
        print('#' * 100)

        # save summary records
        summary_row = {"Model": title, "Fold": fold}
        summary_row.update(metric_dict)
        summary_records.append(summary_row)

        plt.rcParams.update({'font.size': 30, 'font.family': 'sans-serif'})
        axis_labels = ['ASD', 'DHS', 'LCS_HipOA']

        # draw confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='.2f', cmap='Reds', xticklabels=axis_labels, yticklabels=axis_labels, vmin=0, vmax=100)
        # plt.title(f'{title} (%)', fontsize=30)
        plt.title(f"{title} (%)", fontsize=30)
        plt.ylabel('Actual Label', fontsize=30)
        plt.xlabel('Predicted Label', fontsize=30)
        plt.show()
        
    df_summary = pd.DataFrame(summary_records)
    print("Summary of Metrics:")
    print(df_summary.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    # total analysis 
    total_analysis(pegasus, flag="pegasus")

    # top-k folds analysis
    top_k = 5
    key_metric = "accuracy"
    num_class = 3
    flag = "pegasus"
    summary_records, best_fold_records = find_top_folds(
        pegasus,
        top_k=top_k,
        key_metric=key_metric,
        num_class=num_class,
        flag=flag
    )

    df_summary = pd.DataFrame(summary_records)
    print("Summary of Metrics:")
    print(df_summary.to_csv(sep="\t", index=False)) 
    print(f"Best {top_k} folds for each model:")
    for model, folds in best_fold_records.items():
        print(f"\nModel: {model}")
        for fold_idx, fold_data in folds.items():
            print(f"  Fold {fold_idx}:")
            print(f"    Confusion Matrix:\n{fold_data['confusion_matrix']}")
            print(f"    Metrics: {fold_data['metrics']}")   