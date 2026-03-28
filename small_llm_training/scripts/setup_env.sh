#!/bin/bash

#################################################################
# 超小语言模型训练环境设置脚本
# 适配中国大陆网络环境
#################################################################

echo "🚀 设置超小语言模型训练环境（中国大陆优化版）..."

# 检测GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ℹ️  No NVIDIA GPU detected. Will use CPU/MPS"
fi

# ===== 国内镜像配置 =====
echo ""
echo "🌐 配置国内镜像源..."

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "   HuggingFace 镜像: $HF_ENDPOINT"

# pip 镜像
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
echo "   pip 镜像: https://pypi.tuna.tsinghua.edu.cn/simple"

# conda 镜像（如果使用conda）
if command -v conda &> /dev/null; then
    echo "   conda 镜像: 已配置"
fi

# ===== 环境变量 =====
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 设置项目根目录
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_DIR="${PROJECT_ROOT}/data"
export MODEL_DIR="${PROJECT_ROOT}/models"
export EXPERIMENT_DIR="${PROJECT_ROOT}/experiments"

# 创建必要的目录
mkdir -p "${DATA_DIR}"/{raw,processed,datasets,synthetic}
mkdir -p "${MODEL_DIR}"/{pretrained,checkpoints,final}
mkdir -p "${EXPERIMENT_DIR}"/{logs,results,comparisons}

# 设置缓存目录
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${DATA_DIR}/datasets"
export TRANSFORMERS_CACHE="${MODEL_DIR}/pretrained"

# ModelScope 缓存
export MODELSCOPE_CACHE="${MODEL_DIR}/modelscope"

# 设置W&B (可选)
if [ -n "$WANDB_API_KEY" ]; then
    echo "✅ Weights & Biases configured"
    export WANDB_PROJECT="small-llm-training"
    export WANDB_ENTITY="your-entity"
else
    echo "ℹ️  WANDB_API_KEY not set. Disabling W&B logging"
    export WANDB_MODE=disabled
fi

# 设置混合精度训练
export ACCELERATE_MIXED_PRECISION=bf16

echo ""
echo "✅ 环境设置完成!"
echo ""
echo "📋 国内镜像配置:"
echo "   - HuggingFace: hf-mirror.com"
echo "   - pip: 清华大学镜像"
echo "   - ModelScope: 阿里云镜像"
echo ""
echo "📋 项目目录:"
echo "   - 根目录: ${PROJECT_ROOT}"
echo "   - 数据目录: ${DATA_DIR}"
echo "   - 模型目录: ${MODEL_DIR}"
echo ""
echo "🚀 下一步操作:"
echo "   1. 安装依赖: pip install -r requirements.txt"
echo "   2. 下载数据: python3 scripts/data_processing/download_datasets.py"
echo "   3. 下载模型: python3 scripts/models/download_models.py"
echo "   4. 开始训练: python3 scripts/training/train.py --model smollm-135m"
echo ""
echo "💡 提示:"
echo "   - 所有脚本已配置国内镜像"
echo "   - 模型优先从 ModelScope 下载"
echo "   - 数据集优先从国内镜像下载"