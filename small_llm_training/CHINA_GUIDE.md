# 🇨🇳 中国大陆用户使用指南

## 🌐 镜像配置

本项目已全面适配中国大陆网络环境，使用以下镜像源：

### 自动配置的镜像

1. **HuggingFace 镜像**: `https://hf-mirror.com`
2. **pip 镜像**: 清华大学镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`
3. **ModelScope**: 阿里云镜像（国内下载最快）

### 推荐模型（国内下载速度快）

| 模型 | ModelScope | HuggingFace | 推荐指数 |
|------|-----------|-------------|---------|
| Qwen2.5-0.5B | ✅ 可用 | ✅ 可用 | ⭐⭐⭐⭐⭐ |
| MiniCPM4-0.5B | ✅ 可用 | ✅ 可用 | ⭐⭐⭐⭐⭐ |
| SmolLM-135M | ❌ 不可用 | ✅ 可用 | ⭐⭐⭐ |
| TinyLlama-1.1B | ✅ 可用 | ✅ 可用 | ⭐⭐⭐⭐ |

## 🚀 快速开始

### 方式1: 一键运行（推荐）

```bash
./scripts/run_all.sh
```

所有镜像会自动配置，优先从 ModelScope 下载。

### 方式2: 快速测试（5分钟）

```bash
./scripts/quick_test.sh
```

### 方式3: 分步执行

#### 1. 安装依赖（清华镜像）

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. 下载数据

```bash
python3 scripts/data_processing/download_datasets.py
```

数据下载优先使用国内镜像，网络问题时会自动创建合成数据。

#### 3. 下载模型

```bash
# 查看可用模型
python3 scripts/models/download_models.py --list

# 下载推荐模型（ModelScope优先）
python3 scripts/models/download_models.py --model qwen2.5-0.5b

# 或下载所有模型
python3 scripts/models/download_models.py
```

#### 4. 训练模型

```bash
python3 scripts/training/train.py --model qwen2.5-0.5b
```

## 📦 网络问题解决方案

### 问题1: ModelScope 未安装

**解决方案**:
```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2: 数据下载失败

**解决方案**: 项目会自动创建高质量合成数据，确保训练可以进行

### 问题3: 模型下载慢

**解决方案**:
1. 优先使用 Qwen 或 MiniCPM 模型（ModelScope下载最快）
2. 使用 HuggingFace 镜像（已自动配置）
3. 选择更小的模型（SmolLM-135M）

## 🎯 推荐配置

### 最小配置（快速测试）

```bash
# 使用最小模型和数据集
python3 scripts/models/download_models.py --model smollm-135m
python3 scripts/training/train.py --model smollm-135m
```

### 推荐配置（最佳效果）

```bash
# 使用国内下载最快的模型
python3 scripts/models/download_models.py --model qwen2.5-0.5b
python3 scripts/training/train.py --model qwen2.5-0.5b
```

### 完整研究配置

```bash
# 下载所有模型并对比
python3 scripts/models/download_models.py
./scripts/run_all.sh
```

## 💡 使用建议

### 1. 网络环境

- ✅ ModelScope 下载速度最快（推荐）
- ✅ HuggingFace 镜像次之
- ❌ 原始 HuggingFace 可能较慢

### 2. 模型选择

**快速测试**: SmolLM-135M（最小，但仅HF镜像可用）

**推荐训练**: Qwen2.5-0.5B 或 MiniCPM4-0.5B（ModelScope可用，下载快）

**深度研究**: 训练多个模型进行对比

### 3. 数据策略

- 首选：真实数据（Wikipedia、Alpaca等）
- 备选：高质量合成数据（自动生成）
- 优势：即使网络问题也能完成训练

## 🔧 高级配置

### 手动设置镜像

```bash
# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# pip 镜像
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 使用代理

如果有代理，可以设置：

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### ModelScope 配置

```bash
# ModelScope 缓存目录
export MODELSCOPE_CACHE=/path/to/cache

# ModelScope 镜像
export MODELSCOPE_ENDPOINT=https://modelscope.cn
```

## 📊 性能对比

### 下载速度（参考）

| 来源 | 平均速度 | 稳定性 |
|------|---------|--------|
| ModelScope | 5-10 MB/s | ⭐⭐⭐⭐⭐ |
| HuggingFace 镜像 | 1-3 MB/s | ⭐⭐⭐⭐ |
| HuggingFace 原始 | 0.1-0.5 MB/s | ⭐⭐ |

### 模型下载时间（参考）

| 模型 | ModelScope | HF 镜像 |
|------|-----------|---------|
| Qwen2.5-0.5B | ~2分钟 | ~8分钟 |
| MiniCPM4-0.5B | ~2分钟 | ~8分钟 |
| SmolLM-135M | ❌ | ~3分钟 |

## ❓ 常见问题

### Q: 下载失败怎么办？

A: 
1. 检查网络连接
2. 尝试其他模型
3. 使用合成数据训练
4. 使用代理

### Q: ModelScope 安装失败？

A:
```bash
# 使用清华镜像
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 数据集下载慢？

A: 
1. 使用合成数据（自动生成）
2. 等待一段时间后重试
3. 使用更小的数据子集

### Q: 训练速度慢？

A: 
1. 使用更小的模型
2. 减小 batch size
3. 减少训练轮数
4. 使用 Apple Silicon 的 MPS 加速

## 📞 获取帮助

```bash
# 查看项目状态
python3 scripts/project_info.py

# 查看可用模型
python3 scripts/models/download_models.py --list

# 查看脚本帮助
python3 scripts/training/train.py --help
```

## 🎉 开始训练

```bash
# 推荐流程
./scripts/run_all.sh
```

所有配置已优化，享受流畅的训练体验！🚀