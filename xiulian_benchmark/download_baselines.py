#!/usr/bin/env python3
"""
下载对比基线模型
Qwen3.5-0.8B, MiniCPM4-0.5B等
"""

import os
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BASELINES_DIR = Path(__file__).parent / "models" / "baselines"
MODELSCOPE_DIR = Path(__file__).parent / "models" / "modelscope"

# 对比基线模型
BASELINE_MODELS = {
    "qwen3.5-0.8b": {
        "modelscope": "Qwen/Qwen3.5-0.8B",
        "huggingface": "Qwen/Qwen3.5-0.8B",
        "description": "Qwen3.5 0.8B - 对比基线 ⭐⭐⭐⭐⭐",
        "size": "~1.6GB"
    },
    "minicpm4-0.5b": {
        "modelscope": "OpenBMB/MiniCPM4-0.5B",
        "huggingface": "openbmb/MiniCPM4-0.5B",
        "description": "MiniCPM4 0.5B - 对比基线 ⭐⭐⭐⭐⭐",
        "size": "~1GB"
    },
    "phi-4-mini": {
        "modelscope": None,
        "huggingface": "microsoft/Phi-4-mini-instruct",
        "description": "Phi-4-mini - 对比基线",
        "size": "~8GB"
    }
}

def check_modelscope() -> bool:
    try:
        import modelscope
        return True
    except ImportError:
        return False

def download_from_modelscope(model_id: str, save_path: Path) -> bool:
    try:
        from modelscope import snapshot_download
        
        print(f"📥 从 ModelScope 下载: {model_id}")
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(MODELSCOPE_DIR)
        )
        
        import shutil
        if save_path.exists():
            shutil.rmtree(save_path)
        shutil.copytree(model_dir, save_path)
        
        print(f"   ✅ 下载成功")
        return True
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        return False

def download_from_huggingface(model_id: str, save_path: Path) -> bool:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"📥 从 HuggingFace 镜像下载: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(str(save_path))
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(str(save_path))
        
        print(f"   ✅ 下载成功")
        return True
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        return False

def download_model(model_key: str):
    if model_key not in BASELINE_MODELS:
        print(f"❌ 未知模型: {model_key}")
        return
    
    config = BASELINE_MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"下载对比基线模型: {model_key}")
    print(f"   描述: {config['description']}")
    print(f"   大小: {config['size']}")
    print(f"{'='*60}")
    
    save_path = BASELINES_DIR / model_key
    save_path.mkdir(parents=True, exist_ok=True)
    
    if config['modelscope'] and check_modelscope():
        if download_from_modelscope(config['modelscope'], save_path):
            return
    
    if config['huggingface']:
        download_from_huggingface(config['huggingface'], save_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载对比基线模型")
    parser.add_argument('--models', default='qwen3.5-0.8b,minicpm4-0.5b', 
                       help='要下载的模型，逗号分隔')
    
    args = parser.parse_args()
    
    models = args.models.split(',')
    
    print("="*60)
    print("下载对比基线模型")
    print("="*60)
    
    for model in models:
        download_model(model.strip())
    
    print("\n" + "="*60)
    print("✅ 下载完成")
    print("="*60)

if __name__ == "__main__":
    main()