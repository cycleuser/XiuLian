#!/usr/bin/env python3
"""
数据处理脚本
清洗、过滤和预处理训练数据
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'"()\-\[\]{}]', '', text)
    return text.strip()

def filter_quality(text: str, min_length: int = 50, max_length: int = 2048) -> bool:
    """过滤低质量文本"""
    # 长度检查
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # 检查是否有足够的内容（非空白字符）
    if len(text.replace(' ', '')) < min_length * 0.5:
        return False
    
    # 检查重复字符
    if len(set(text)) / len(text) < 0.3:
        return False
    
    return True

def process_wikipedia():
    """处理 Wikipedia 数据"""
    print("\n📖 处理 Wikipedia 数据...")
    
    wiki_path = DATA_DIR / "datasets" / "wikipedia_small"
    if not wiki_path.exists():
        print("   ⚠️  Wikipedia 数据不存在，跳过")
        return None
    
    wiki = load_from_disk(str(wiki_path))
    
    processed = []
    for item in tqdm(wiki, desc="处理 Wikipedia"):
        text = item.get('text', '')
        cleaned = clean_text(text)
        
        if filter_quality(cleaned):
            processed.append({
                'text': cleaned,
                'source': 'wikipedia',
                'title': item.get('title', '')
            })
    
    print(f"   ✅ 处理完成: {len(processed)} 条")
    return processed

def process_alpaca():
    """处理 Alpaca 数据"""
    print("\n🦙 处理 Alpaca 数据...")
    
    alpaca_path = DATA_DIR / "datasets" / "alpaca"
    if not alpaca_path.exists():
        print("   ⚠️  Alpaca 数据不存在，跳过")
        return None
    
    alpaca = load_from_disk(str(alpaca_path))
    
    processed = []
    for item in tqdm(alpaca, desc="处理 Alpaca"):
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        if instruction and output:
            text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
            processed.append({
                'text': text,
                'source': 'alpaca',
                'instruction': instruction,
                'output': output
            })
    
    print(f"   ✅ 处理完成: {len(processed)} 条")
    return processed

def process_code_alpaca():
    """处理代码数据"""
    print("\n💻 处理 CodeAlpaca 数据...")
    
    code_path = DATA_DIR / "datasets" / "code_alpaca"
    if not code_path.exists():
        print("   ⚠️  CodeAlpaca 数据不存在，跳过")
        return None
    
    code_data = load_from_disk(str(code_path))
    
    processed = []
    for item in tqdm(code_data, desc="处理 CodeAlpaca"):
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        if instruction and output:
            text = f"### 指令:\n{instruction}\n\n### 代码:\n{output}"
            processed.append({
                'text': text,
                'source': 'code_alpaca',
                'instruction': instruction,
                'output': output
            })
    
    print(f"   ✅ 处理完成: {len(processed)} 条")
    return processed

def process_synthetic():
    """处理合成数据"""
    print("\n🔬 处理合成数据...")
    
    synthetic_path = DATA_DIR / "datasets" / "synthetic_small.json"
    if not synthetic_path.exists():
        print("   ⚠️  合成数据不存在，跳过")
        return None
    
    with open(synthetic_path, 'r', encoding='utf-8') as f:
        synthetic_data = json.load(f)
    
    processed = []
    for item in synthetic_data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        if instruction and output:
            text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
            processed.append({
                'text': text,
                'source': 'synthetic',
                'instruction': instruction,
                'output': output,
                'type': item.get('type', 'qa')
            })
    
    print(f"   ✅ 处理完成: {len(processed)} 条")
    return processed

def merge_and_save(all_data: List[Dict], output_name: str = "train_data"):
    """合并并保存数据"""
    print(f"\n💾 合并并保存数据...")
    
    if not all_data:
        print("   ⚠️  没有数据可保存")
        return
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 打乱数据
    import random
    random.shuffle(all_data)
    
    # 保存为JSONL格式
    output_path = PROCESSED_DIR / f"{output_name}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✅ 保存完成: {len(all_data)} 条")
    print(f"   📁 文件: {output_path}")
    
    # 统计信息
    print(f"\n📊 数据统计:")
    sources = {}
    for item in all_data:
        source = item.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sources.items():
        print(f"   - {source}: {count} 条")

def main():
    parser = argparse.ArgumentParser(description="处理训练数据")
    parser.add_argument('--output', type=str, default='train_data', help='输出文件名')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("      超小语言模型训练 - 数据处理")
    print("="*60 + "\n")
    
    try:
        all_data = []
        
        # 处理各个数据集
        wiki_data = process_wikipedia()
        if wiki_data:
            all_data.extend(wiki_data)
        
        alpaca_data = process_alpaca()
        if alpaca_data:
            all_data.extend(alpaca_data)
        
        code_data = process_code_alpaca()
        if code_data:
            all_data.extend(code_data)
        
        synthetic_data = process_synthetic()
        if synthetic_data:
            all_data.extend(synthetic_data)
        
        # 合并保存
        merge_and_save(all_data, args.output)
        
        print("\n" + "="*60)
        print("✅ 数据处理完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()