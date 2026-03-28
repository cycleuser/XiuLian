#!/bin/bash

#################################################################
# 非Transformer架构模型训练与对比 - 一键运行脚本
#################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

# 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 使用conda dev环境的python
PYTHON="/Users/fred/miniconda3/envs/dev/bin/python"
echo "🐍 使用Python: $($PYTHON --version)"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   非Transformer架构模型训练与对比研究                     ║"
echo "║                                                            ║"
echo "║   您的模型：修炼 (XiuLian)                                 ║"
echo "║   对比基线：Qwen3.5-0.8B, MiniCPM4-0.5B                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

log "阶段1: 训练您的非Transformer模型（修炼）..."
$PYTHON train_xiulian.py --data data/training --output models/xiulian

log "阶段2: 下载对比基线模型..."
$PYTHON download_baselines.py --models qwen3.5-0.8b

log "阶段3: 运行性能基准测试..."
$PYTHON benchmark.py \
    --xiulian models/xiulian \
    --baselines "qwen3.5-0.8b=models/baselines/qwen3.5-0.8b" \
    --output results

log "完成！查看结果:"
echo ""
echo "  📊 对比报告: results/report.md"
echo "  📈 详细数据: results/benchmark_results.json"
echo ""

# 显示报告
if [ -f "results/report.md" ]; then
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    性能对比报告                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    cat results/report.md
fi