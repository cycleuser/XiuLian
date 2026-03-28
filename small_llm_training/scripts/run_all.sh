#!/bin/bash

#################################################################
# 超小语言模型训练 - 一键运行脚本
# 适配系统：macOS, 16GB RAM, CPU-only
# 模型规模：100M - 500M 参数
#################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志文件
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1" | tee -a "$MAIN_LOG"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1" | tee -a "$MAIN_LOG"
}

# 错误处理
error_exit() {
    log_error "$1"
    exit 1
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        error_exit "$1 未安装，请先安装"
    fi
}

#################################################################
# 阶段 1: 系统检查和环境准备
#################################################################
stage_1_system_check() {
    log "========== 阶段 1: 系统检查 =========="
    
    # 检查Python
    check_command python3
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log "Python 版本: $PYTHON_VERSION"
    
    # 检查内存
    if [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024/1024}')
        log "系统内存: ${TOTAL_MEM} GB"
        
        if [ "$TOTAL_MEM" -lt 8 ]; then
            log_warn "内存不足8GB，建议使用更小的模型和数据集"
        fi
    fi
    
    # 检查磁盘空间
    AVAILABLE_SPACE=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    log "可用磁盘空间: $AVAILABLE_SPACE"
    
    # 检查GPU (macOS使用MPS)
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        log "✅ 检测到 Apple Metal Performance Shaders (MPS)"
        export TRAINING_DEVICE="mps"
    else
        log_info "未检测到GPU，将使用CPU训练（速度较慢）"
        export TRAINING_DEVICE="cpu"
    fi
    
    # 创建项目目录结构
    log "创建项目目录结构..."
    mkdir -p "$PROJECT_ROOT"/{data/{raw,processed,datasets,synthetic},models/{pretrained,checkpoints,final},experiments/{logs,results,comparisons},logs}
    
    log "✅ 阶段 1 完成"
}

#################################################################
# 阶段 2: 安装依赖
#################################################################
stage_2_install_dependencies() {
    log "========== 阶段 2: 安装依赖 =========="
    
    # 创建虚拟环境（如果不存在）
    VENV_DIR="$PROJECT_ROOT/venv"
    if [ ! -d "$VENV_DIR" ]; then
        log "创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # 激活虚拟环境
    log "激活虚拟环境..."
    source "$VENV_DIR/bin/activate"
    
    # 升级pip
    log "升级pip..."
    python3 -m pip install --upgrade pip setuptools wheel | tee -a "$MAIN_LOG"
    
    # 安装依赖
    log "安装Python依赖（这可能需要几分钟）..."
    
    # 核心依赖（最小化安装）
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu | tee -a "$MAIN_LOG"
    pip install transformers datasets accelerate tokenizers sentencepiece | tee -a "$MAIN_LOG"
    pip install tqdm numpy pandas pyyaml omegaconf | tee -a "$MAIN_LOG"
    pip install matplotlib seaborn | tee -a "$MAIN_LOG"
    
    log "✅ 阶段 2 完成"
}

#################################################################
# 阶段 3: 数据准备
#################################################################
stage_3_data_preparation() {
    log "========== 阶段 3: 数据准备 =========="
    
    # 下载数据集
    log "下载训练数据集..."
    python3 scripts/data_processing/download_datasets.py 2>&1 | tee -a "$MAIN_LOG" || log_warn "数据下载出现问题，将使用备用数据"
    
    # 处理数据
    log "处理和过滤数据..."
    python3 scripts/data_processing/process_data.py 2>&1 | tee -a "$MAIN_LOG"
    
    # 创建训练/验证/测试集
    log "创建数据集划分..."
    python3 scripts/data_processing/split_dataset.py 2>&1 | tee -a "$MAIN_LOG"
    
    log "✅ 阶段 3 完成"
}

#################################################################
# 阶段 4: 下载预训练模型
#################################################################
stage_4_download_models() {
    log "========== 阶段 4: 下载预训练模型 =========="
    
    log "下载基础模型（可能需要几分钟）..."
    python3 scripts/models/download_models.py 2>&1 | tee -a "$MAIN_LOG"
    
    log "✅ 阶段 4 完成"
}

#################################################################
# 阶段 5: 模型训练
#################################################################
stage_5_training() {
    log "========== 阶段 5: 模型训练 =========="
    
    # 训练配置
    MODELS=(
        "smollm-135m"
        "qwen2.5-0.5b"
        "minicpm4-0.5b"
    )
    
    log "将训练以下模型: ${MODELS[*]}"
    
    for MODEL in "${MODELS[@]}"; do
        log "--- 开始训练: $MODEL ---"
        python3 scripts/training/train.py \
            --model "$MODEL" \
            --config configs/training_configs/cpu_optimized.yaml \
            2>&1 | tee -a "$LOG_DIR/${MODEL}_$(date +%Y%m%d_%H%M%S).log"
        
        if [ $? -eq 0 ]; then
            log "✅ $MODEL 训练完成"
        else
            log_error "$MODEL 训练失败"
        fi
    done
    
    log "✅ 阶段 5 完成"
}

#################################################################
# 阶段 6: 模型评估
#################################################################
stage_6_evaluation() {
    log "========== 阶段 6: 模型评估 =========="
    
    log "运行评估基准测试..."
    python3 scripts/evaluation/evaluate_all.py 2>&1 | tee -a "$MAIN_LOG"
    
    log "✅ 阶段 6 完成"
}

#################################################################
# 阶段 7: 结果分析和报告生成
#################################################################
stage_7_analysis() {
    log "========== 阶段 7: 结果分析 =========="
    
    log "生成对比分析报告..."
    python3 scripts/analysis/generate_report.py 2>&1 | tee -a "$MAIN_LOG"
    
    log "✅ 阶段 7 完成"
}

#################################################################
# 阶段 8: 清理和总结
#################################################################
stage_8_cleanup() {
    log "========== 阶段 8: 清理和总结 =========="
    
    # 显示最终报告
    REPORT_FILE="$PROJECT_ROOT/experiments/results/final_report.md"
    if [ -f "$REPORT_FILE" ]; then
        log "📊 最终报告已生成: $REPORT_FILE"
        echo ""
        cat "$REPORT_FILE"
    fi
    
    # 显示训练摘要
    SUMMARY_FILE="$PROJECT_ROOT/experiments/results/training_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        log "📋 训练摘要:"
        python3 -c "import json; print(json.dumps(json.load(open('$SUMMARY_FILE')), indent=2))"
    fi
    
    log ""
    log "🎉 全部流程完成！"
    log ""
    log "📁 项目结构:"
    tree -L 2 "$PROJECT_ROOT" 2>/dev/null || find "$PROJECT_ROOT" -maxdepth 2 -type d | sed 's|[^/]*/| |g'
    
    log ""
    log "📊 查看结果:"
    log "   - 训练日志: $PROJECT_ROOT/logs/"
    log "   - 评估结果: $PROJECT_ROOT/experiments/results/"
    log "   - 最终报告: $PROJECT_ROOT/experiments/results/final_report.md"
    log ""
}

#################################################################
# 主函数
#################################################################
main() {
    clear
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     超小语言模型本地训练研究 - 一键运行脚本                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    log "开始时间: $(date)"
    log "项目目录: $PROJECT_ROOT"
    log "日志文件: $MAIN_LOG"
    echo ""
    
    # 解析参数
    SKIP_DEPS=false
    SKIP_TRAINING=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  --skip-deps      跳过依赖安装"
                echo "  --skip-training  跳过训练步骤（仅评估）"
                echo "  --help           显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行各个阶段
    stage_1_system_check
    
    if [ "$SKIP_DEPS" = false ]; then
        stage_2_install_dependencies
    else
        log "⏭️  跳过依赖安装"
        source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || error_exit "虚拟环境不存在，请移除 --skip-deps 参数"
    fi
    
    stage_3_data_preparation
    stage_4_download_models
    
    if [ "$SKIP_TRAINING" = false ]; then
        stage_5_training
    else
        log "⏭️  跳过训练步骤"
    fi
    
    stage_6_evaluation
    stage_7_analysis
    stage_8_cleanup
    
    log ""
    log "完成时间: $(date)"
    log "总耗时: $(( ($(date +%s) - $(date -j -f "%a %b %d %H:%M:%S %Y" "$(head -1 $MAIN_LOG | cut -d'[' -f1)" +%s 2>/dev/null || echo 0)) )) 秒"
}

# 运行主函数
main "$@"