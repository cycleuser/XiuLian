# 超小语言模型本地训练研究项目

## 🎯 项目简介

研究和对比**2024-2025最新**超小规模语言模型（100M-14B参数）在相同数据规模下的训练效果。包含最新模型：MiniCPM4、Phi-4、Qwen2.5等。

## ✨ 核心特性

- 🆕 **最新模型**: MiniCPM4 (2025.06), Phi-4 (2024.12), Qwen2.5 (2024.09)
- 🚀 **一键运行**: 完整自动化流程
- 📊 **多模型对比**: 支持 10+ 最新模型
- 💻 **本地训练**: 优化CPU/MPS训练，无需GPU
- 📈 **完整评估**: 困惑度、问答、生成质量等多维度评估
- 📝 **自动报告**: 自动生成Markdown格式研究报告
- 🔧 **高度可配置**: 灵活的YAML配置系统
- 🇨🇳 **国内优化**: 支持HuggingFace镜像和ModelScope

## 🏗️ 项目结构

```
small_llm_training/
├── README.md                   # 本文件
├── LATEST_MODELS.md           # 最新模型列表 ⭐
├── QUICKSTART.md              # 快速开始指南
├── CHINA_GUIDE.md             # 中国用户指南
├── START_HERE.md              # 快速开始（中国用户）
├── install.sh                 # 一键安装
├── scripts/                   # 所有脚本
│   ├── run_all.sh            # 一键运行脚本 ⭐
│   ├── quick_test.sh         # 快速测试脚本
│   ├── models/download_models.py  # 模型下载（最新模型）⭐
│   ├── training/             # 训练脚本
│   └── evaluation/           # 评估脚本
├── data/                      # 数据目录
├── models/                    # 模型目录
└── experiments/               # 实验结果
```

## 🚀 快速开始

### 方式1: 一键安装并运行（推荐）

```bash
./install.sh
./scripts/quick_test.sh
```

### 方式2: 查看最新模型

```bash
# 列出所有2024-2025最新模型
python3 scripts/models/download_models.py --list

# 下载推荐模型
python3 scripts/models/download_models.py --recommended
```

## 📊 支持的最新模型（2024-2025）

### 超小模型（< 1B）⭐ 推荐

| 模型 | 参数 | 发布时间 | 特点 | 论文 |
|------|------|---------|------|------|
| **MiniCPM4-0.5B** | 0.5B | 2025.06 | InfLLM v2稀疏注意力 | [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) |
| **Qwen2.5-0.5B** | 0.5B | 2024.09 | 多语言，18T tokens | [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) |
| **SmolLM-135M** | 135M | 2024 | 最小测试模型 | - |

### 小模型（1B-3B）

| 模型 | 参数 | 发布时间 | 特点 | 论文 |
|------|------|---------|------|------|
| **Phi-4-mini** | ~3.8B | 2024.12 | 推理能力强 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **Qwen2.5-1.5B** | 1.5B | 2024.09 | 平衡性能 | - |
| **Qwen2.5-3B** | 3B | 2024.09 | 高知识密度 | - |

### 中等模型（7B-14B）

| 模型 | 参数 | 发布时间 | 特点 | 论文 |
|------|------|---------|------|------|
| **Phi-4** | 14B | 2024.12 | 合成数据训练，STEM推理 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **MiniCPM4-8B** | 8B | 2025.06 | 混合推理 | [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) |
| **Qwen2.5-7B** | 7B | 2024.09 | 18T tokens训练 | - |

## 🎯 推荐模型

### 用于快速测试
- **SmolLM-135M**: 最小，下载快，适合验证流程
- **MiniCPM4-0.5B**: 最新技术，性能优异

### 用于实际训练（国内）
- **Qwen2.5-0.5B**: ModelScope下载最快，性能稳定 ⭐
- **MiniCPM4-0.5B**: 最新架构，效率最高 ⭐

### 用于深入研究
- **Phi-4-mini**: 推理能力强
- **Qwen2.5-3B**: 知识密度高

## 📚 重要论文引用

### MiniCPM4 (2025)
```bibtex
@article{minicpm4_2025,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM Team},
  journal={arXiv preprint arXiv:2506.07900},
  year={2025}
}
```

### Phi-4 (2024)
```bibtex
@article{phi4_2024,
  title={Phi-4 Technical Report},
  author={Abdin, Marah and others},
  journal={arXiv preprint arXiv:2412.08905},
  year={2024}
}
```

### Qwen2.5 (2024)
```bibtex
@misc{qwen2.5,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  year={2024},
  howpublished={\url{https://qwenlm.github.io/blog/qwen2.5/}}
}
```

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

## 🇨🇳 中国大陆用户

查看 [CHINA_GUIDE.md](CHINA_GUIDE.md) 获取详细的镜像配置指南。

**快速开始**:
```bash
./install.sh
./scripts/quick_test.sh
```

## 📈 评估指标

项目提供多维度评估：

1. **困惑度 (Perplexity)**: 衡量模型对文本的预测能力
2. **问答准确率**: 基于关键词匹配的问答性能
3. **生成质量**: 文本生成的长度和多样性
4. **参数效率**: 困惑度/参数量比值

## 📞 获取帮助

```bash
# 查看最新模型列表
python3 scripts/models/download_models.py --list

# 查看项目状态
python3 scripts/project_info.py

# 查看脚本帮助
python3 scripts/training/train.py --help
```

## 📖 相关文档

- [LATEST_MODELS.md](LATEST_MODELS.md) - 最新模型详细对比 ⭐
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [CHINA_GUIDE.md](CHINA_GUIDE.md) - 中国用户指南

---

**开始训练**: `./scripts/run_all.sh`

**快速测试**: `./scripts/quick_test.sh`

**查看最新模型**: `python3 scripts/models/download_models.py --list`