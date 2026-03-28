#!/bin/bash

#################################################################
# 快速测试脚本 - 仅测试流程，不进行完整训练
# 适用于快速验证环境和数据
#################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 快速测试模式"
echo "这将执行最小化测试，仅验证流程"
echo ""

# 1. 环境检查
echo "1️⃣  检查环境..."
python3 --version

# 2. 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "2️⃣  创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

# 3. 安装最小依赖
echo "3️⃣  安装核心依赖..."
pip install -q torch transformers datasets tokenizers tqdm

# 4. 创建测试数据
echo "4️⃣  创建测试数据..."
mkdir -p data/processed

cat > data/processed/train.jsonl << 'EOF'
{"text": "人工智能是计算机科学的一个分支。"}
{"text": "机器学习使用算法从数据中学习。"}
{"text": "自然语言处理是AI的重要领域。"}
EOF

cat > data/processed/val.jsonl << 'EOF'
{"text": "深度学习使用神经网络。"}
{"text": "强化学习通过奖励学习策略。"}
EOF

cat > data/processed/test.jsonl << 'EOF'
{"text": "计算机视觉让机器理解图像。"}
EOF

echo "   ✅ 测试数据创建完成"

# 5. 下载最小模型
echo "5️⃣  下载测试模型（SmolLM-135M）..."
mkdir -p models/pretrained

python3 << 'PYTHON'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceTB/SmolLM-135M"
save_path = "models/pretrained/smollm-135m"

print("下载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("下载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.save_pretrained(save_path)

print("✅ 模型下载完成")
PYTHON

# 6. 快速训练测试
echo "6️⃣  快速训练测试（1个epoch）..."

mkdir -p models/checkpoints models/final

python3 << 'PYTHON'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 加载模型
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    "models/pretrained/smollm-135m",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("models/pretrained/smollm-135m")
tokenizer.pad_token = tokenizer.eos_token

# 简单数据集
class SimpleDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path) as f:
            self.data = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = eval(self.data[idx])['text']
        encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

train_dataset = SimpleDataset("data/processed/train.jsonl")

# 训练参数
training_args = TrainingArguments(
    output_dir="models/checkpoints/test",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=100,
    report_to="none"
)

# 训练
print("开始训练...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# 保存
print("保存模型...")
trainer.save_model("models/final/smollm-135m-test")
tokenizer.save_pretrained("models/final/smollm-135m-test")

print("✅ 训练完成")
PYTHON

# 7. 测试推理
echo "7️⃣  测试推理..."

python3 << 'PYTHON'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("加载训练后的模型...")
model = AutoModelForCausalLM.from_pretrained("models/final/smollm-135m-test")
tokenizer = AutoTokenizer.from_pretrained("models/final/smollm-135m-test")

prompt = "人工智能是"
inputs = tokenizer(prompt, return_tensors='pt')

print(f"\n提示: {prompt}")
print("生成中...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,
        do_sample=True
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n生成结果: {generated}")
print("\n✅ 推理测试完成")
PYTHON

# 完成
echo ""
echo "🎉 快速测试完成！"
echo ""
echo "✅ 环境验证通过"
echo "✅ 数据处理正常"
echo "✅ 模型训练成功"
echo "✅ 推理功能正常"
echo ""
echo "📋 下一步："
echo "   运行完整训练: ./scripts/run_all.sh"
echo ""