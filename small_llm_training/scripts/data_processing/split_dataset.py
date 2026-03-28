#!/usr/bin/env python3
"""
数据集划分脚本
将数据划分为训练集、验证集和测试集
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: Path):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """划分数据集"""
    
    # 打乱数据
    random.seed(seed)
    random.shuffle(data)
    
    # 计算划分点
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument('--input', type=str, default='train_data.jsonl', help='输入文件')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("      超小语言模型训练 - 数据集划分")
    print("="*60 + "\n")
    
    input_path = PROCESSED_DIR / args.input
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    # 加载数据
    print(f"📂 加载数据: {input_path}")
    data = load_jsonl(input_path)
    print(f"   总数据量: {len(data)} 条")
    
    # 划分数据集
    print(f"\n🔪 划分数据集 (train:{args.train_ratio}, val:{args.val_ratio}, test:{args.test_ratio})")
    train_data, val_data, test_data = split_dataset(
        data, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio,
        args.seed
    )
    
    # 保存
    print("\n💾 保存数据集...")
    
    save_jsonl(train_data, PROCESSED_DIR / "train.jsonl")
    print(f"   ✅ 训练集: {len(train_data)} 条 -> train.jsonl")
    
    save_jsonl(val_data, PROCESSED_DIR / "val.jsonl")
    print(f"   ✅ 验证集: {len(val_data)} 条 -> val.jsonl")
    
    save_jsonl(test_data, PROCESSED_DIR / "test.jsonl")
    print(f"   ✅ 测试集: {len(test_data)} 条 -> test.jsonl")
    
    print("\n" + "="*60)
    print("✅ 数据集划分完成！")
    print("="*60)

if __name__ == "__main__":
    main()