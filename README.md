<div align="center">

# KnowledgeGuided-ASD\_PyTorch

**Clinical Knowledge-Guided Attention Framework for Gait-Based Adult Spinal Deformity Diagnosis**

</div>

---

## 🧠 Overview

This repository presents a **knowledge-guided gait analysis framework** for the diagnosis of Adult Spinal Deformity (ASD) from monocular walking videos. It introduces **clinician-informed attention maps** that highlight anatomically and diagnostically important regions (e.g., lumbar, pelvis, shoulder), guiding the deep model to focus on **clinically relevant features**.

---

## ✨ Key Contributions

* 🪻 **Medical Prior Integration**: Expert annotations are encoded as attention maps targeting key joints linked to spinal disorders.
* 🎯 **Clinically Explainable Attention**: Gaussian-based spatial priors emphasize pathological body regions.
* 🤀 **Fusion Strategies**: Systematic evaluation of how different attention map fusion mechanisms affect performance.
* 📈 **Superior Performance**: Outperforms baselines like CNN, ST-GCN, and 3D CNN across accuracy, F1-score, and interpretability.

---

## 🗂️ Project Structure

```
KnowledgeGuided-ASD_PyTorch/
👉 configs/                # Configuration YAMLs
👉 data/                   # Dataset interfaces
👉 models/                 # Attention, CNN, 3D CNN, Fusion modules
👉 trainers/               # Training and evaluation logic
👉 utils/                  # Utility scripts
👉 attention_maps/         # Priors and Gaussian heatmaps
👉 scripts/                # Bash automation scripts
👉 requirements.txt        # Python requirements
👉 README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/ChenKaiXuSan/KnowledgeGuided-ASD_PyTorch.git
cd KnowledgeGuided-ASD_PyTorch
pip install -r requirements.txt
```

### 2. Prepare Attention Maps

```bash
python scripts/generate_attention_maps.py --config configs/roi.yaml
```

This uses pose keypoints and clinical priors to generate Gaussian-weighted attention heatmaps.

### 3. Train Model

```bash
python trainers/train.py --config configs/train_fusion.yaml
```

### 4. Evaluate Model

```bash
python trainers/eval.py --config configs/train_fusion.yaml
```

---

## 🧪tase

* 🎥 **Video**: 1920×1080, 30 FPS, walking videos (10m indoor path)
* 👨‍⚕️ **Subjects**: 81 participants (ASD, DHS, LCS, HipOA)
* 🔎 **Annotation**: Radiographic-based diagnosis + ROI annotation from orthopedic surgeons
* ❗ *Dataset is not publicly released due to ethical constraints. Contact authors for collaboration.*

---

## 📊 Performance (from paper)

| Method              | Accuracy (%) | Precision (%) | F1 Score (%) |
| ------------------- | ------------ | ------------- | ------------ |
| CNN (Baseline)      | 52.56        | 81.11         | 54.01        |
| ST-GCN              | 60.42        | 60.22         | 59.80        |
| 3D CNN (RGB only)   | 62.09        | 64.55         | 60.13        |
| + Add Fusion (Ours) | **71.35**    | **75.51**     | **71.12**    |
| + Concat Fusion     | 59.26        | 62.46         | 54.70        |
| + Avg Fusion        | 65.83        | 66.18         | 64.88        |

---

## 📄 Citation

If you find this repository helpful, please cite:

```bibtex
@inproceedings{chen2025clinicalgait,
  title={A Clinical Knowledge-Guided Attention Framework for Gait-Based Adult Spinal Deformity Diagnosis},
  author={Chen, Kaixu and Asada, Tomoyuki and Miura, Kousei and Yamazaki, Masashi and Ienaga, Naoto and Kuroda, Yoshihiro and Kitahara, Itaru},
  booktitle={Proceedings of SPIE Medical Imaging},
  year={2025},
  organization={SPIE}
}
```

---

## 🔗 Related Projects

* [PhaseMix (IEEE Access 2024)](https://ieeexplore.ieee.org/document/10714330)
* [Two-Stage ASD Gait Classifier (Frontiers in Neuroscience 2023)](https://www.frontiersin.org/articles/10.3389/fnins.2023.1278584)
