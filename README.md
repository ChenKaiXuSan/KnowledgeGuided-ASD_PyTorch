<div align="center">    
 
# A Clinical Knowledge-Guided Attention Framework for Gait-Based Adult Spinal Deformity Diagnosis

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!--  
Conference   
-->
</div>
 
## Description   

This repository contains the official implementation of our research on leveraging clinical knowledge to guide attention mechanisms in gait-based classification of Adult Spinal Deformity (ASD) using monocular video data.

🧠 **Key Contributions**:

- Incorporates domain-specific anatomical knowledge from orthopedic experts
- Introduces a clinically-guided attention mechanism focusing on spinal and lower-limb joints
- Applies cross-attentive temporal fusion for periodic motion representation
- Enhances interpretability and diagnostic accuracy of ASD classification models


## 📂 Project Structure

```bash
ClinicalGait-ASD/
├── configs/               # Configuration files
├── data/                  # Dataset loading and preprocessing
├── models/                # Network components (backbone, attention, etc.)
├── trainer/               # Training and evaluation scripts
├── utils/                 # Visualization, metrics, and helpers
├── demo/                  # Inference examples and demo video generation
├── scripts/               # Shell scripts for training and evaluation
├── docs/                  # Figures and documentation
├── requirements.txt       # Python dependencies
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/your_username/ClinicalGait-ASD.git
cd ClinicalGait-ASD
pip install -r requirements.txt
```

## 🧪 Demo (Inference)

You can run the pre-trained model on sample videos to visualize predictions and attention maps.

```bash
python demo/run_demo.py --video_path sample.mp4 --output_path outputs/
```

## 🚀 Training

To train the model on your own data:

```bash
python trainer/train.py --config configs/asd_config.yaml
```

For evaluation:

```bash
python trainer/evaluate.py --config configs/asd_config.yaml
```

## 🧬 Dataset

We use a video-based gait dataset for ASD diagnosis, approved by the University of Tsukuba Hospital Ethics Committee (H30-087). The dataset contains annotated gait clips with clinical labels. See [our paper](#) for details.
_Note: The dataset is not publicly released due to privacy constraints. Please contact us for collaboration._

## 📈 Results

| Method             | Accuracy  | F1-score | AUC      |
| ------------------ | --------- | -------- | -------- |
| Baseline CNN       | 78.5%     | 0.76     | 0.81     |
| **Ours (CK-Attn)** | **85.2%** | **0.83** | **0.89** |

## 📄 Citation

If you find this project helpful, please cite our work:

```bibtex
@article{chen2025clinical,
  title={A Clinical Knowledge-Guided Attention Framework for Gait-Based Adult Spinal Deformity Diagnosis},
  author={Chen, Kaixu and ...},
  journal={TBA},
  year={2025}
}
```
