# 🚀 非Transformer架构模型训练与对比研究

## 📊 研究目标

训练**修炼(XiuLian)**非Transformer架构模型，并与典型小规模LLM进行性能对比。

## 🔥 核心对比

### 您的模型：修炼 (XiuLian) - 非Transformer架构

| 指标 | 修炼 | Transformer |
|------|------|-------------|
| 复杂度 | O(n log n) | O(n²) |
| 内存 | <2GB | 14-350GB |
| 延迟 | 1-10ms | 500-2000ms |
| 参数 | <500M | 7B-175B |

**架构特点**：
- 符号推理：O(n) 模式匹配
- 记忆检索：O(log n) Trie索引
- 图推理：O(V+E) 关系遍历
- 工具执行：O(1) 直接调用

### 对比基线：小规模Transformer LLM

| 模型 | 参数 | 发布时间 | 特点 |
|------|------|---------|------|
| **Qwen3.5-0.8B** | 0.8B | 2025 | 最新小模型基线 |
| **MiniCPM4-0.5B** | 0.5B | 2025 | 设备端优化 |
| **Phi-4-mini** | ~3.8B | 2024.12 | 推理能力强 |

## 🎯 实验设计

### 实验1：训练修炼模型
- 训练数据：相同规模数据集
- 训练目标：符号推理、记忆检索、图推理
- 评估指标：准确率、延迟、内存占用

### 实验2：性能对比
- 任务类型：工具调用、问答、网络访问
- 对比维度：
  - 延迟 (ms)
  - 内存占用 (GB)
  - 准确率 (%)
  - 吞吐量 (req/s)

### 实验3：扩展性对比
- 输入长度：1K, 10K, 100K tokens
- 测量指标：延迟曲线、内存曲线

## 📁 项目结构

```
xiulian_benchmark/
├── README.md                   # 本文件
├── train_xiulian.py           # 训练修炼模型
├── download_baselines.py      # 下载对比模型
├── benchmark.py               # 性能测试
├── evaluate.py                # 评估对比
├── generate_report.py         # 生成报告
├── configs/
│   ├── xiulian_config.yaml    # 修炼模型配置
│   └── experiment_config.yaml # 实验配置
├── data/
│   ├── training/              # 训练数据
│   └── benchmark/             # 测试数据
├── models/
│   ├── xiulian/               # 训练后的修炼模型
│   └── baselines/             # 基线模型
└── results/
    ├── metrics/               # 性能指标
    ├── plots/                 # 对比图表
    └── report.md              # 最终报告
```

## 🚀 快速开始

### 1. 训练您的非Transformer模型

```bash
python train_xiulian.py --data data/training --output models/xiulian
```

### 2. 下载对比基线模型

```bash
python download_baselines.py --models qwen3.5-0.8b,minicpm4-0.5b
```

### 3. 运行性能对比

```bash
python benchmark.py --xiulian models/xiulian --baselines models/baselines
```

### 4. 生成对比报告

```bash
python generate_report.py --results results/
```

## 📊 预期对比结果

| 指标 | 修炼 | Qwen3.5-0.8B | MiniCPM4-0.5B |
|------|------|-------------|---------------|
| 延迟 | 1-10ms | 500-2000ms | 100-500ms |
| 内存 | <2GB | ~1.6GB | ~1GB |
| 吞吐量 | >1000 req/s | ~10 req/s | ~50 req/s |

## 🔬 研究重点

1. **架构对比**: 非Transformer vs Transformer
2. **效率对比**: 时间复杂度、空间复杂度
3. **任务对比**: 符号推理、工具调用、网络访问
4. **可扩展性**: 输入长度、并发请求

## 📚 引用

### 修炼 (XiuLian)
```
基于第一性原理的非Transformer架构AI系统
- 符号推理：O(n) 模式匹配
- 记忆检索：O(log n) Trie索引
- 图推理：O(V+E) 关系遍历
```

### 对比基线
```
Qwen3.5: https://qwenlm.github.io/
MiniCPM4: arXiv:2506.07900
Phi-4: arXiv:2412.08905
```