# 🚀 2025最新小语言模型对比研究

## 🔥 核心模型：Qwen3.5-0.8B（2025最新）

### Qwen3.5系列（2025年最新）⭐⭐⭐⭐⭐

| 模型 | 参数 | 发布时间 | 特点 | 论文 | 国内下载 |
|------|------|---------|------|------|---------|
| **Qwen3.5-0.8B** | 0.8B | 2025 | 🔥最新核心模型 | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B) | ⭐⭐⭐⭐⭐ |
| **Qwen3.5-1.7B** | 1.7B | 2025 | 平衡性能 | - | ⭐⭐⭐⭐⭐ |
| **Qwen3.5-4B** | 4B | 2025 | 高性能 | - | ⭐⭐⭐⭐⭐ |

### 其他最新模型

| 模型 | 参数 | 发布时间 | 特点 | 论文 | 国内下载 |
|------|------|---------|------|------|---------|
| **MiniCPM4-0.5B** | 0.5B | 2025.06 | InfLLM v2稀疏注意力 | [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) | ⭐⭐⭐⭐⭐ |
| **Phi-4-mini** | ~3.8B | 2024.12 | 推理能力强 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) | ⭐⭐⭐⭐ |
| **SmolLM-135M** | 135M | 2024 | 最小测试模型 | - | ⭐⭐⭐ |

### 小模型（1B-3B参数）

| 模型 | 参数 | 发布时间 | 特点 | 论文 | 国内下载 |
|------|------|---------|------|------|---------|
| **Phi-4-mini** | ~3.8B | 2024.12 | 推理能力强 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) | ⭐⭐⭐⭐ |
| **Qwen2.5-1.5B** | 1.5B | 2024.09 | 平衡性能 | [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-3B** | 3B | 2024.09 | 高知识密度 | [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) | ⭐⭐⭐⭐⭐ |
| **MiniCPM4-8B** | 8B | 2025.06 | 混合推理 | [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) | ⭐⭐⭐⭐⭐ |

### 中等模型（7B-14B参数）

| 模型 | 参数 | 发布时间 | 特点 | 论文 |
|------|------|---------|------|------|
| **Phi-4** | 14B | 2024.12 | 合成数据训练，STEM推理 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **MiniCPM4.1-8B** | 8B | 2025.09 | 混合推理模式 | [GitHub](https://github.com/OpenBMB/MiniCPM) |
| **Qwen2.5-7B** | 7B | 2024.09 | 18T tokens训练 | [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) |

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

## 🔬 关键技术特性

### MiniCPM4 (2025最新)
- ✅ **InfLLM v2**: 可训练稀疏注意力机制
- ✅ **UltraClean**: 高效预训练数据过滤
- ✅ **BitCPM**: 数据高效的三值LLM
- ✅ **CPM.cu**: 集成稀疏注意力和量化

### Phi-4 (2024)
- ✅ **合成数据训练**: 大量高质量合成数据
- ✅ **STEM推理**: 超越GPT-4的STEM能力
- ✅ **数据质量优先**: 质量大于数量

### Qwen2.5 (2024)
- ✅ **18万亿tokens**: 大规模预训练
- ✅ **多语言支持**: 29+语言
- ✅ **长文本**: 128K上下文

## 💡 选择建议

### 内存限制（< 8GB）
- SmolLM-135M
- MiniCPM4-0.5B
- Qwen2.5-0.5B

### 平衡性能（8-16GB）
- Qwen2.5-1.5B
- Phi-4-mini
- Qwen2.5-3B

### 高性能需求（16GB+）
- MiniCPM4-8B
- Qwen2.5-7B
- Phi-4

## 📈 性能对比

详见实验结果的评估报告。