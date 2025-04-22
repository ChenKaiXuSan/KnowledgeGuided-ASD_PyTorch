import pandas as pd

# result path 
doctor_1 = '/workspace/doctor_result/Takahiro Sunami_2023-05-15-13-28-03.csv'
doctor_1_name = doctor_1.split('/')[-1].split('_')[0]
doctor_2 = '/workspace/doctor_result/KS_2023-05-15-14-08-08.csv'
doctor_2_name = doctor_2.split('/')[-1].split('_')[0]

df = pd.read_csv(doctor_1)

label_list = []
predict_list = []
predict_attention_list = []

for i in range(len(df)):
    video_file_name = df['video file name'][i]
    for s in video_file_name.split('_'):
        if s in ['ASD']:
            label_list.append(0)
        elif s in ['DHS', 'LCS', 'HipOA']:
            label_list.append(1)

    selected_attention = df['attention'][i]
    predict_attention_list.append(selected_attention)

    if df['disease'][i][2:-2] == 'asd_btn':
        predict_list.append(0)
    else:
        predict_list.append(1)

import torchmetrics

# define the metrics.
_accuracy = torchmetrics.classification.BinaryAccuracy()
_precision = torchmetrics.classification.BinaryPrecision()
_binary_recall = torchmetrics.classification.BinaryRecall()
_binary_f1 = torchmetrics.classification.BinaryF1Score()

_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix(normalize='true')

_aucroc = torchmetrics.classification.BinaryAUROC()

import torch

preds_tensor = torch.tensor(predict_list)    
labels_tensor = torch.tensor(label_list)
print('*' * 100)

# video rgb metrics
print('the result of :', doctor_2_name)
print('accuracy: %s' % _accuracy(preds_tensor, labels_tensor))
print('precision: %s' % _precision(preds_tensor, labels_tensor))
print('_binary_recall: %s' % _binary_recall(preds_tensor, labels_tensor))
print('_binary_f1: %s' % _binary_f1(preds_tensor, labels_tensor))
print('_aurroc: %s' % _aucroc(preds_tensor, labels_tensor))
print('_confusion_matrix: %s' % _confusion_matrix(preds_tensor, labels_tensor))
print('#' * 100)

import seaborn as sns
import matplotlib.pyplot as plt

cm = _confusion_matrix(preds_tensor, labels_tensor).to(float)

total = sum(sum(cm))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm[i][j] = float(cm[i][j] / total)

print(cm)

ax = sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

ax.set_title('Confusion Matrix')
ax.set(xlabel="pred class", ylabel="ground truth")
ax.xaxis.tick_top()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay


fpr, tpr, _ = roc_curve(labels_tensor, preds_tensor, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="doctor").plot()
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC curves")
plt.legend()
plt.show()
PART = ['foot_btn',  'head_btn', 'lumbar_pelvis_btn', 'shoulder_btn', 'wrist_btn']

df1 = pd.read_csv(doctor_1)
df2 = pd.read_csv(doctor_2)

df1.sort_values(by='path', ascending=False)
df2.sort_values(by='path', ascending=False)

same = 0
for i in range(len(df1)):
    if df1['attention'][i] == df2['attention'][i]:
        same += 1
    
not_same = len(df1) - same

for doctor in [df1]:
    part_dict = {p:0 for p in PART}

    for p in PART:
        for i in range(len(doctor)):
            if doctor['attention'][i].split('\'')[1] == p:
                part_dict[p] += 1

    print(part_dict)


# pie plot
import matplotlib.pyplot as plt

# make the pie circular by setting the aspect ratio to 1
# plt.figure(figsize=plt.figaspect(1))

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

plt.pie(x = list(part_dict.values()), labels=list(part_dict.keys()), autopct=make_autopct(list(part_dict.values())))
plt.title('doctor 1')
plt.axis('equal')
# plt.legend()
plt.show()

final_name = set(df1['video file name'])
final_dict = {n:[] for n in final_name}
for i in range(len(df1)):
    for j in final_name:
        if df1['video file name'][i] == j:
            final_dict[j].append(df1['attention'][i][2:-2])

final_name = set(df2['video file name'])
for i in range(len(df2)):
    for j in final_name:
        if df2['video file name'][i] == j:
            final_dict[j].append(df2['attention'][i][2:-2])
# %%
for k in final_dict.keys():
    final_dict[k] = set(final_dict[k])
final_dict
# %%
different_part = ['lumbar_pelvis_btn', 'shoulder_btn', 'head_btn', 'wrist_btn', 'foot_btn']
# %%
disease = [[] for _ in range(len(different_part))]
# %%
for k in final_dict.keys():
    for v in final_dict[k]:
        index = different_part.index(v)
        disease[index].append(k)
# %%
plt.barh(range(len(disease)), [len(i) for i in disease], tick_label=different_part)

# %%
final_dict
for k, v in final_dict.items():
    for i in range(len(different_part)):
        temp = [None for _ in range(5)]
        for j in v:
            if j in different_part:
                temp[different_part.index(j)] = j
        final_dict[k] = temp

final_dict


# %%
dict_save = pd.DataFrame.from_dict(final_dict, orient='index', columns=different_part)
dict_save

# %%
dict_save.to_excel("/workspace/doctor_result/result.xlsx")
# %%
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

# cm = _confusion_matrix(preds_tensor, labels_tensor).to(float)
cm = np.array([[0.6295,1-0.6295], [0.1630,1-0.1630]])
ax = sns.heatmap(cm, annot=True, annot_kws={"fontsize": 20}, vmin=0, vmax=1, fmt=".4f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

# color bar font size 
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

ax.set_title('D. Proposed method.', fontsize=20)
ax.set_xlabel("pred class", fontsize=20)
ax.set_ylabel("ground truth", fontsize=20)
ax.xaxis.tick_top()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('C.png')
plt.show()

# %%
