# 快速开始指南

## 🚀 一键运行

### 完整流程（推荐首次使用）

```bash
./scripts/run_all.sh
```

### 快速测试（跳过依赖安装）

```bash
./scripts/run_all.sh --skip-deps
```

### 仅评估已有模型

```bash
./scripts/run_all.sh --skip-training
```

## 📋 分步执行

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载数据集
python3 scripts/data_processing/download_datasets.py

# 处理数据
python3 scripts/data_processing/process_data.py

# 划分数据集
python3 scripts/data_processing/split_dataset.py
```

### 3. 下载模型

```bash
# 下载所有模型
python3 scripts/models/download_models.py

# 或下载指定模型
python3 scripts/models/download_models.py --model smollm-135m
```

### 4. 训练模型

```bash
# 训练单个模型
python3 scripts/training/train.py --model smollm-135m

# 或批量训练
bash scripts/training/train_all_models.sh
```

### 5. 评估模型

```bash
# 评估所有模型
python3 scripts/evaluation/evaluate_all.py

# 评估指定模型
python3 scripts/evaluation/evaluate_all.py --model models/final/smollm-135m
```

### 6. 生成报告

```bash
python3 scripts/analysis/generate_report.py
```

## ⚙️ 配置调整

### 内存不足时

编辑 `configs/training_configs/cpu_optimized.yaml`:

```yaml
data:
  batch_size: 1  # 减小batch size

training:
  gradient_accumulation_steps: 16  # 增加梯度累积
```

### 使用MPS加速（Apple Silicon）

系统会自动检测并使用MPS，无需额外配置。

### 自定义训练数据

将数据放在 `data/raw/` 目录，格式为：

```json
{"text": "你的文本内容"}
{"text": "另一条文本"}
```

## 🐛 常见问题

### 1. 下载模型失败

**原因**: 网络问题或HuggingFace访问限制

**解决**: 
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 内存不足

**原因**: 模型或数据集过大

**解决**:
- 减小batch size
- 使用更小的模型
- 减少数据量

### 3. 训练速度慢

**原因**: CPU训练本身较慢

**解决**:
- 使用MPS（自动启用）
- 减少训练轮数
- 使用更小的数据集

## 📊 查看结果

### 训练日志
```bash
ls logs/
```

### 评估结果
```bash
cat experiments/results/evaluation_results.json
```

### 最终报告
```bash
cat experiments/results/final_report.md
```

## 🎯 快速测试（最小配置）

如果只想快速测试流程，可以：

1. 只下载最小的模型（SmolLM-135M）
2. 使用合成数据（自动生成）
3. 只训练1个epoch

```bash
# 1. 下载数据（会自动生成合成数据）
python3 scripts/data_processing/download_datasets.py

# 2. 处理数据
python3 scripts/data_processing/process_data.py --output test_data
python3 scripts/data_processing/split_dataset.py --input test_data.jsonl

# 3. 下载最小模型
python3 scripts/models/download_models.py --model smollm-135m

# 4. 修改配置：num_epochs=1
# 编辑 configs/training_configs/cpu_optimized.yaml

# 5. 训练
python3 scripts/training/train.py --model smollm-135m

# 6. 评估
python3 scripts/evaluation/evaluate_all.py

# 7. 查看报告
python3 scripts/analysis/generate_report.py
```

## 💡 提示

- 首次运行建议使用 `--skip-training` 先测试数据和环境
- 训练时间：SmolLM-135M 约1-2小时（CPU）
- 可以随时中断训练，进度会自动保存
- 最终模型保存在 `models/final/`
- 可以使用量化进一步减小模型体积

## 📞 获取帮助

```bash
# 查看脚本帮助
python3 scripts/training/train.py --help
python3 scripts/evaluation/evaluate_all.py --help

# 列出所有可用模型
python3 scripts/models/download_models.py --list
```