#!/bin/bash

#################################################################
# 非深度学习AI策略综合研究 - 一键运行脚本
#################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   非深度学习AI策略综合研究                                 ║"
echo "║   Non-Deep Learning AI Strategies Comprehensive Research     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 创建必要目录
mkdir -p RESULTS
mkdir -p REPORTS

# 阶段1: 环境准备
echo "📦 阶段1: 检查环境..."
python3 --version

# 阶段2: 运行基准测试
echo ""
echo "📊 阶段2: 运行全面基准测试..."
python3 BENCHMARKS/benchmark_suite.py

# 阶段3: 生成对比分析
echo ""
echo "📈 阶段3: 生成对比分析..."

# 阶段4: 汇总报告
echo ""
echo "📝 阶段4: 生成汇总报告..."

# 显示结果
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    测试完成                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 查看结果:"
echo "   - 基准测试报告: RESULTS/benchmark_report.md"
echo "   - 详细数据: RESULTS/benchmark_report.json"
echo "   - 参考文献: REFERENCES.md"
echo ""

# 显示报告
if [ -f "RESULTS/benchmark_report.md" ]; then
    echo "═══════════════════════════════════════════════════════════"
    echo "                    基准测试报告"
    echo "═══════════════════════════════════════════════════════════"
    cat RESULTS/benchmark_report.md
fi