#!/bin/bash
#PBS -A SKIING                        # ✅ 项目名（必须修改）
#PBS -q gen_S                        # ✅ 队列名（gpu / debug / gen_S）
#PBS -l elapstim_req=24:00:00         # ⏱ 运行时间限制（最多 24 小时）
#PBS -N train_multi                     # 🏷 作业名称
#PBS -t 0-5
#PBS -o logs/pegasus/train_multi_out.log            # 📤 标准输出日志
#PBS -e logs/pegasus/train_multi_err.log            # ❌ 错误输出日志

# === 切换到作业提交目录 ===
cd /home/SKIING/chenkaixu/code/KnowledgeGuided-ASD_PyTorch

mkdir -p logs/pegasus/
mkdir -p checkpoints/

# === 下载预训练模型（如果需要） ===
# wget -O /home/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/checkpoints/SLOW_8x8_R50.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW_8x8_R50.pyth

# === 加载 Python + 激活 Conda 环境 ===
module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate # 确保先退出任何现有的 Conda 环境
source /home/SKIING/chenkaixu/code/med_atn/bin/activate

# === 可选：打印 GPU 状态 ===
nvidia-smi

NUM_WORKERS=$(nproc)
# 输出当前环境信息
echo "Current working directory: $(pwd)"
echo "Total CPU cores: $NUM_WORKERS, use $((NUM_WORKERS / 2)) for data loading"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"
echo "Current Model load path: $(ls checkpoints/SLOW_8x8_R50.pyth)"

# 映射关系：数字 → 融合方式名称
fuse_methods=("add" "mul" "avg" "concat" "none" "late")

# 用数字选择（比如从命令行传入，或固定指定）
fuse_index=${PBS_SUBREQNO}
fuse_method=${fuse_methods[$fuse_index]}

echo "Selected fuse method: $fuse_method"

# params 
root_path=/work/SKIING/chenkaixu/data/asd_dataset

# === 运行你的训练脚本（Hydra 参数可以加在后面）===
python -m project.main data.root_path=${root_path} model.fuse_method=${fuse_method} train.fold=10 data.num_workers=$((NUM_WORKERS / 2))