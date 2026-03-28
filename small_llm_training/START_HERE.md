# 🚀 开始使用（中国大陆用户）

## 快速开始（3步）

### 第1步：安装依赖（2分钟）

```bash
./install.sh
```

这会自动安装所有依赖，使用清华大学镜像。

### 第2步：查看最新模型（2024-2025）

```bash
# 列出所有最新模型
python3 scripts/models/download_models.py --list

# 下载推荐模型
python3 scripts/models/download_models.py --recommended
```

### 第3步：快速测试（5分钟）

```bash
./scripts/quick_test.sh
```

## 🆕 最新模型（2024-2025）

### 超小模型（推荐）⭐

| 模型 | 参数 | 年份 | 特点 | 国内下载 |
|------|------|------|------|---------|
| **MiniCPM4-0.5B** | 0.5B | 2025.06 | 最新稀疏注意力 | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-0.5B** | 0.5B | 2024.09 | 多语言支持 | ⭐⭐⭐⭐⭐ |
| **SmolLM-135M** | 135M | 2024 | 最小测试模型 | ⭐⭐⭐ |

### 小模型

| 模型 | 参数 | 年份 | 特点 | 国内下载 |
|------|------|------|------|---------|
| **Phi-4-mini** | ~3.8B | 2024.12 | 推理能力强 | ⭐⭐⭐⭐ |
| **Qwen2.5-1.5B** | 1.5B | 2024.09 | 平衡性能 | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-3B** | 3B | 2024.09 | 高知识密度 | ⭐⭐⭐⭐⭐ |

## 📚 重要论文

- **MiniCPM4**: [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) (2025)
- **Phi-4**: [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) (2024.12)

## 🌐 国内优化

- ✅ HuggingFace 镜像：hf-mirror.com
- ✅ pip 镜像：清华大学
- ✅ ModelScope：阿里云（最快）

## 💡 推荐配置

### 内存 < 8GB
- SmolLM-135M
- MiniCPM4-0.5B
- Qwen2.5-0.5B

### 内存 8-16GB
- Qwen2.5-1.5B
- Qwen2.5-3B
- Phi-4-mini

### 内存 16GB+
- MiniCPM4-8B
- Qwen2.5-7B
- Phi-4

## 或一键运行全部

```bash
./scripts/run_all.sh
```

## 📖 详细指南

查看 `CHINA_GUIDE.md` 获取完整说明。

开始训练吧！🚀