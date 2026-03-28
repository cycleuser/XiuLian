# 超小语言模型本地训练研究 - 完整项目

## 🎯 项目简介

本项目旨在研究和对比超小规模语言模型（100M-3B参数）在相同数据规模下的训练效果。项目提供完整的训练流程、评估基准和分析工具，支持一键运行。

## ✨ 核心特性

- 🚀 **一键运行**: 完整自动化流程
- 📊 **多模型对比**: 支持 SmolLM、Qwen、MiniCPM 等多个模型
- 💻 **本地训练**: 优化CPU/MPS训练，无需GPU
- 📈 **完整评估**: 困惑度、问答、生成质量等多维度评估
- 📝 **自动报告**: 自动生成Markdown格式研究报告
- 🔧 **高度可配置**: 灵活的YAML配置系统
- 🇨🇳 **国内优化**: 支持HuggingFace镜像和ModelScope

## 🏗️ 项目结构

```
small_llm_training/
├── README.md                   # 本文件
├── QUICKSTART.md              # 快速开始指南
├── CHINA_GUIDE.md             # 中国用户指南 ⭐
├── START_HERE.md              # 快速开始（中国用户）⭐
├── install.sh                 # 一键安装 ⭐
├── requirements.txt           # Python依赖
├── configs/                   # 配置文件
│   └── training_configs/
│       └── cpu_optimized.yaml
├── scripts/                   # 所有脚本
│   ├── run_all.sh            # 一键运行脚本 ⭐
│   ├── quick_test.sh         # 快速测试脚本 ⭐
│   ├── project_info.py       # 项目信息查看
│   ├── setup_env.sh          # 环境设置
│   ├── data_processing/      # 数据处理脚本
│   ├── models/               # 模型下载脚本
│   ├── training/             # 训练脚本
│   ├── evaluation/           # 评估脚本
│   └── analysis/             # 分析脚本
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   ├── datasets/             # 数据集
│   └── synthetic/            # 合成数据
├── models/                    # 模型目录
│   ├── pretrained/           # 预训练模型
│   ├── checkpoints/          # 训练检查点
│   └── final/                # 最终模型
└── experiments/               # 实验结果
    ├── logs/                 # 训练日志
    ├── results/              # 评估结果
    └── comparisons/          # 对比分析
```

## 🚀 快速开始

### 方式1: 一键安装并运行（推荐）

```bash
# 安装依赖
./install.sh

# 运行快速测试
./scripts/quick_test.sh

# 或运行完整流程
./scripts/run_all.sh
```

### 方式2: 快速测试（5分钟内完成）

```bash
./scripts/quick_test.sh
```

仅测试流程，使用最小配置和合成数据。

### 方式3: 分步执行

详见 [QUICKSTART.md](QUICKSTART.md)

## 🇨🇳 中国大陆用户

如果您在中国大陆，请查看 [CHINA_GUIDE.md](CHINA_GUIDE.md) 获取详细的镜像配置指南。

**快速开始**:
```bash
./install.sh
./scripts/quick_test.sh
```

## 📊 支持的模型

### 超小模型（100M-500M）

| 模型 | 参数 | 大小 | 特点 | 国内下载 |
|------|------|------|------|---------|
| SmolLM-135M | 135M | ~270MB | 最小测试模型 | HF镜像 |
| Qwen2.5-0.5B | 500M | ~1GB | 多语言支持 | ModelScope+HF ⭐ |
| MiniCPM4-0.5B | 500M | ~1GB | 设备端优化 | ModelScope+HF ⭐ |

### 中等模型（1B-3B）

| 模型 | 参数 | 大小 | 特点 | 国内下载 |
|------|------|------|------|---------|
| TinyLlama-1.1B | 1.1B | ~2.2GB | 轻量级对话 | ModelScope+HF |
| Phi-1 | 1.3B | ~2.6GB | 代码能力强 | HF镜像 |
| Qwen2.5-3B | 3B | ~6GB | 知识密度高 | ModelScope+HF |

## 💻 系统要求

### 最低配置
- **CPU**: 任意现代CPU
- **内存**: 8 GB
- **磁盘**: 20 GB
- **Python**: 3.8+

### 推荐配置
- **CPU**: Apple Silicon (M1/M2/M3)
- **内存**: 16 GB
- **磁盘**: 50 GB
- **Python**: 3.10+

## 📈 评估指标

项目提供多维度评估：

1. **困惑度 (Perplexity)**: 衡量模型对文本的预测能力
2. **问答准确率**: 基于关键词匹配的问答性能
3. **生成质量**: 文本生成的长度和多样性
4. **参数效率**: 困惑度/参数量比值

## 📝 生成的报告

运行完成后，会自动生成：

```
experiments/results/
├── final_report.md           # 完整研究报告
├── evaluation_results.json   # 评估结果数据
└── training_summary.json     # 训练摘要
```

报告包含：
- 📊 性能对比表格
- 💡 关键发现和分析
- 📋 参数效率排名
- 🎯 使用建议

## 🛠️ 常见问题

### 1. 内存不足

**解决方案**:
```yaml
# 修改 configs/training_configs/cpu_optimized.yaml
data:
  batch_size: 1
  
training:
  gradient_accumulation_steps: 16
```

### 2. 训练速度慢

**原因**: CPU训练本身较慢

**解决方案**:
- 使用 Apple Silicon 会自动启用 MPS 加速
- 减少训练轮数
- 使用更小的模型

### 3. 模型下载失败

**解决方案**:
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用ModelScope
pip install modelscope
python3 scripts/models/download_models.py --model qwen2.5-0.5b
```

## 📚 相关文档

- [快速开始指南](QUICKSTART.md) - 详细的分步说明
- [中国用户指南](CHINA_GUIDE.md) - 国内镜像配置 ⭐
- [研究方法](docs/methodology.md) - 研究方法论
- [实验设计](docs/experiments.md) - 实验方案
- [结果分析](docs/results.md) - 结果解读

## 🔬 研究重点

1. **数据质量 vs 数据量**
   - 合成数据与真实数据的对比
   - 数据清洗和过滤策略

2. **参数效率**
   - 同等性能下最小参数量
   - 模型规模与性能的权衡

3. **训练效率**
   - 训练时间对比
   - 显存/内存占用分析
   - 能耗对比

4. **推理效率**
   - 生成速度测试
   - 内存占用分析
   - 批处理性能

5. **量化效果**
   - FP16 vs INT8 vs 1.58-bit
   - 性能与精度权衡

## 🎯 预期成果

- ✅ 完整的训练流程和脚本
- ✅ 多模型对比基准测试
- ✅ 数据质量分析报告
- ✅ 最佳实践指南
- ✅ 可复现的实验结果

## 📞 获取帮助

### 查看项目状态
```bash
python3 scripts/project_info.py
```

### 查看脚本帮助
```bash
python3 scripts/training/train.py --help
python3 scripts/evaluation/evaluate_all.py --help
```

### 列出所有可用模型
```bash
python3 scripts/models/download_models.py --list
```

## 📄 许可证

本项目仅供学习和研究使用。

各模型的许可证请参考：
- SmolLM: Apache 2.0
- Qwen: Apache 2.0 (部分模型)
- MiniCPM: Apache 2.0
- Phi: MIT License

## 🙏 致谢

感谢以下开源项目：
- Hugging Face Transformers
- PyTorch
- Apple Metal Performance Shaders
- ModelScope

---

**开始训练**: `./scripts/run_all.sh`

**快速测试**: `./scripts/quick_test.sh`

**查看状态**: `python3 scripts/project_info.py`

**中国用户**: 查看 [CHINA_GUIDE.md](CHINA_GUIDE.md)