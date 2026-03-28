#!/bin/bash

#################################################################
# 快速安装脚本（中国大陆优化版）
# 一键安装所有依赖并测试环境
#################################################################

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        超小语言模型训练 - 快速安装（中国优化版）            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

echo "🌐 使用国内镜像:"
echo "   HuggingFace: $HF_ENDPOINT"
echo "   pip: 清华大学镜像"
echo ""

# 检查Python
echo "1️⃣  检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "   ✅ $PYTHON_VERSION"

# 创建虚拟环境
echo ""
echo "2️⃣  创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ 虚拟环境已创建"
else
    echo "   ℹ️  虚拟环境已存在"
fi

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo ""
echo "3️⃣  升级 pip（清华镜像）..."
python3 -m pip install --upgrade pip setuptools wheel -q -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "   ✅ pip 已升级"

# 安装PyTorch
echo ""
echo "4️⃣  安装 PyTorch（CPU版本）..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
echo "   ✅ PyTorch 已安装"

# 安装核心依赖
echo ""
echo "5️⃣  安装核心依赖（清华镜像）..."
pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers datasets accelerate tokenizers \
    tqdm numpy pandas pyyaml omegaconf \
    matplotlib seaborn
echo "   ✅ 核心依赖已安装"

# 安装ModelScope
echo ""
echo "6️⃣  安装 ModelScope（清华镜像）..."
pip install modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "   ✅ ModelScope 已安装"

# 验证安装
echo ""
echo "7️⃣  验证安装..."
python3 -c "
import torch
import transformers
import modelscope
print('   ✅ PyTorch:', torch.__version__)
print('   ✅ Transformers:', transformers.__version__)
print('   ✅ ModelScope: 已安装')

# 检查MPS
if torch.backends.mps.is_available():
    print('   ✅ MPS (Metal): 可用')
else:
    print('   ℹ️  MPS: 不可用（将使用CPU）')
"

# 创建必要的目录
echo ""
echo "8️⃣  创建项目目录..."
mkdir -p data/{raw,processed,datasets,synthetic}
mkdir -p models/{pretrained,checkpoints,final,modelscope}
mkdir -p experiments/{logs,results,comparisons}
mkdir -p logs
echo "   ✅ 目录结构已创建"

# 完成
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 ✅ 安装完成！                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 下一步操作:"
echo ""
echo "   1. 快速测试（推荐首次运行）:"
echo "      ./scripts/quick_test.sh"
echo ""
echo "   2. 下载推荐模型:"
echo "      python3 scripts/models/download_models.py --model qwen2.5-0.5b"
echo ""
echo "   3. 完整训练流程:"
echo "      ./scripts/run_all.sh"
echo ""
echo "💡 推荐:"
echo "   - Qwen2.5-0.5B: ModelScope下载最快"
echo "   - MiniCPM4-0.5B: ModelScope下载最快"
echo ""
echo "📖 查看完整指南:"
echo "   cat CHINA_GUIDE.md"
echo ""