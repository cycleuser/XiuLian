#!/usr/bin/env python3
"""
模型训练脚本
支持多种小语言模型的训练
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, file_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        print(f"📂 加载数据: {file_path}")
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"   数据量: {len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def load_config(config_path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)

def setup_device(config: Dict[str, Any]) -> torch.device:
    """设置训练设备"""
    device_str = config.get('training', {}).get('device', 'cpu')
    
    if device_str == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"✅ 使用设备: MPS (Apple Metal)")
        else:
            device = torch.device('cpu')
            print(f"⚠️  MPS 不可用，使用 CPU")
    else:
        device = torch.device('cpu')
        print(f"✅ 使用设备: CPU")
    
    return device

def train_model(args):
    """训练模型"""
    
    print("\n" + "="*60)
    print(f"      训练模型: {args.model}")
    print("="*60 + "\n")
    
    # 加载配置
    config_path = Path(args.config)
    config = load_config(config_path)
    
    # 模型路径
    model_dir = PROJECT_ROOT / "models" / "pretrained" / args.model
    
    if not model_dir.exists():
        print(f"❌ 模型不存在: {model_dir}")
        print("   请先运行: python scripts/models/download_models.py --model {args.model}")
        return False
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "models" / "checkpoints" / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer
    print(f"\n📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   ✅ Tokenizer 加载完成")
    print(f"   词汇表大小: {len(tokenizer)}")
    
    # 加载模型
    print(f"\n📥 加载模型...")
    device = setup_device(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.to(device)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ 模型加载完成")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 加载数据集
    print(f"\n📂 加载数据集...")
    data_config = config.get('data', {})
    
    train_file = PROJECT_ROOT / data_config['train_file']
    val_file = PROJECT_ROOT / data_config['val_file']
    max_length = data_config.get('max_length', 512)
    
    train_dataset = TextDataset(train_file, tokenizer, max_length)
    val_dataset = TextDataset(val_file, tokenizer, max_length)
    
    print(f"   训练集: {len(train_dataset)} 条")
    print(f"   验证集: {len(val_dataset)} 条")
    
    # 训练参数
    train_config = config.get('training', {})
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.get('num_epochs', 3),
        per_device_train_batch_size=data_config.get('batch_size', 2),
        per_device_eval_batch_size=data_config.get('batch_size', 2),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 8),
        learning_rate=train_config.get('learning_rate', 5e-5),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 100),
        
        logging_dir=str(output_dir / 'logs'),
        logging_steps=train_config.get('logging_steps', 10),
        
        evaluation_strategy="steps",
        eval_steps=train_config.get('eval_steps', 100),
        
        save_strategy="steps",
        save_steps=train_config.get('save_steps', 500),
        save_total_limit=train_config.get('save_total_limit', 3),
        
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        
        fp16=train_config.get('fp16', False),
        bf16=train_config.get('bf16', False),
        
        dataloader_num_workers=config.get('dataloader_num_workers', 0),
        
        report_to="none",  # 不使用wandb等
        
        seed=config.get('seed', 42)
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("      开始训练")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        
        # 保存最终模型
        final_dir = PROJECT_ROOT / "models" / "final" / args.model
        final_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        print(f"📁 最终模型: {final_dir}")
        print(f"📁 检查点: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="训练小语言模型")
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--config', type=str, 
                        default='configs/training_configs/cpu_optimized.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    success = train_model(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()