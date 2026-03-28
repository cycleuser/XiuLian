#!/usr/bin/env python3
"""
模型下载脚本（2025最新版）
核心模型：Qwen3-0.8B（2025年最新）
优先从 ModelScope 下载，备选 HuggingFace 镜像
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "pretrained"
MODELSCOPE_DIR = PROJECT_ROOT / "models" / "modelscope"

# 2025 最新模型配置（Qwen3.5为核心）
MODELS = {
    # ===== 核心：Qwen3.5系列（2025最新）=====
    "qwen3.5-0.8b": {
        "modelscope": "Qwen/Qwen3.5-0.8B",
        "huggingface": "Qwen/Qwen3.5-0.8B",
        "description": "Qwen3.5 0.8B (2025) - 最新核心模型 🔥🔥🔥🔥🔥",
        "size": "~1.6GB",
        "arxiv": None,
        "year": 2025,
        "recommended": True,
        "core": True  # 核心推荐
    },
    "qwen3.5-1.7b": {
        "modelscope": "Qwen/Qwen3.5-1.7B",
        "huggingface": "Qwen/Qwen3.5-1.7B",
        "description": "Qwen3.5 1.7B (2025) ⭐⭐⭐⭐⭐",
        "size": "~3.4GB",
        "arxiv": None,
        "year": 2025,
        "recommended": True
    },
    "qwen3.5-4b": {
        "modelscope": "Qwen/Qwen3.5-4B",
        "huggingface": "Qwen/Qwen3.5-4B",
        "description": "Qwen3.5 4B (2025) ⭐⭐⭐⭐",
        "size": "~8GB",
        "arxiv": None,
        "year": 2025,
        "recommended": True
    },
    
    # ===== MiniCPM4系列（2025）=====
    "minicpm4-0.5b": {
        "modelscope": "OpenBMB/MiniCPM4-0.5B",
        "huggingface": "openbmb/MiniCPM4-0.5B",
        "description": "MiniCPM4 0.5B (2025.06) - 稀疏注意力 ⭐⭐⭐⭐⭐",
        "size": "~1GB",
        "arxiv": "2506.07900",
        "year": 2025,
        "recommended": True
    },
    
    # ===== Qwen2.5系列（稳定版）=====
    "qwen2.5-0.5b": {
        "modelscope": "Qwen/Qwen2.5-0.5B",
        "huggingface": "Qwen/Qwen2.5-0.5B",
        "description": "Qwen2.5 0.5B (2024.09) - 稳定版 ⭐⭐⭐⭐",
        "size": "~1GB",
        "arxiv": None,
        "year": 2024,
        "recommended": False
    },
    "smollm-135m": {
        "modelscope": None,
        "huggingface": "HuggingFaceTB/SmolLM-135M",
        "description": "SmolLM 135M - 最小测试模型",
        "size": "~270MB",
        "arxiv": None,
        "year": 2024,
        "recommended": False
    },
    
    # ===== 小模型（1B-3B）=====
    "qwen2.5-1.5b": {
        "modelscope": "Qwen/Qwen2.5-1.5B",
        "huggingface": "Qwen/Qwen2.5-1.5B",
        "description": "Qwen2.5 1.5B - 平衡性能",
        "size": "~3GB",
        "arxiv": None,
        "year": 2024,
        "recommended": True
    },
    "qwen2.5-3b": {
        "modelscope": "Qwen/Qwen2.5-3B",
        "huggingface": "Qwen/Qwen2.5-3B",
        "description": "Qwen2.5 3B - 高知识密度 ⭐",
        "size": "~6GB",
        "arxiv": None,
        "year": 2024,
        "recommended": True
    },
    "phi-4-mini": {
        "modelscope": None,  # ModelScope可能暂无
        "huggingface": "microsoft/Phi-4-mini-instruct",
        "description": "Phi-4-mini (2024.12) - 推理能力强 ⭐",
        "size": "~8GB",
        "arxiv": "2412.08905",
        "year": 2024,
        "recommended": True
    },
    
    # ===== 中等模型（7B-14B）=====
    "minicpm4-8b": {
        "modelscope": "OpenBMB/MiniCPM4-8B",
        "huggingface": "openbmb/MiniCPM4-8B",
        "description": "MiniCPM4 8B (2025.06) - 混合推理 ⭐",
        "size": "~16GB",
        "arxiv": "2506.07900",
        "year": 2025,
        "recommended": True
    },
    "qwen2.5-7b": {
        "modelscope": "Qwen/Qwen2.5-7B",
        "huggingface": "Qwen/Qwen2.5-7B",
        "description": "Qwen2.5 7B - 18T tokens训练",
        "size": "~14GB",
        "arxiv": None,
        "year": 2024,
        "recommended": False
    },
    "phi-4": {
        "modelscope": None,
        "huggingface": "microsoft/phi-4",
        "description": "Phi-4 14B (2024.12) - 合成数据训练 ⭐",
        "size": "~28GB",
        "arxiv": "2412.08905",
        "year": 2024,
        "recommended": False  # 需要16GB+内存
    }
}

def check_modelscope() -> bool:
    """检查是否安装了 modelscope"""
    try:
        import modelscope
        return True
    except ImportError:
        return False

def install_modelscope():
    """安装 modelscope"""
    print("📦 安装 modelscope...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "modelscope", "-i", 
        "https://pypi.tuna.tsinghua.edu.cn/simple"
    ])

def download_from_modelscope(model_id: str, save_path: Path) -> bool:
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        
        print(f"📥 从 ModelScope 下载: {model_id}")
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(MODELSCOPE_DIR),
            revision='master'
        )
        
        import shutil
        if save_path.exists():
            shutil.rmtree(save_path)
        shutil.copytree(model_dir, save_path)
        
        print(f"   ✅ ModelScope 下载成功")
        return True
        
    except Exception as e:
        print(f"   ❌ ModelScope 下载失败: {e}")
        return False

def download_from_huggingface(model_id: str, save_path: Path) -> bool:
    """从 HuggingFace 镜像下载模型"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"📥 从 HuggingFace 镜像下载: {model_id}")
        print(f"   使用镜像: {os.environ.get('HF_ENDPOINT', 'default')}")
        
        print("   下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(save_path))
        
        print("   下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(str(save_path))
        
        print(f"   ✅ HuggingFace 镜像下载成功")
        return True
        
    except Exception as e:
        print(f"   ❌ HuggingFace 镜像下载失败: {e}")
        return False

def download_model(model_key: str, force: bool = False) -> bool:
    """下载单个模型（优先ModelScope，备选HuggingFace镜像）"""
    
    if model_key not in MODELS:
        print(f"❌ 未知模型: {model_key}")
        return False
    
    config = MODELS[model_key]
    
    print(f"\n{'='*60}")
    print(f"📥 下载模型: {model_key}")
    print(f"   名称: {config['description']}")
    print(f"   大小: {config['size']}")
    print(f"   年份: {config['year']}")
    if config['arxiv']:
        print(f"   论文: arXiv:{config['arxiv']}")
    if config['recommended']:
        print(f"   🌟 推荐模型")
    print(f"{'='*60}")
    
    save_path = MODELS_DIR / model_key
    
    if save_path.exists() and not force:
        print(f"⚠️  模型已存在: {save_path}")
        response = input("是否重新下载? (y/n): ")
        if response.lower() != 'y':
            print("   跳过下载")
            return True
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 优先从 ModelScope 下载
    if config['modelscope']:
        if not check_modelscope():
            print("\n⚠️  ModelScope 未安装")
            install_choice = input("是否安装 ModelScope? (y/n): ")
            if install_choice.lower() == 'y':
                install_modelscope()
            else:
                print("   将使用 HuggingFace 镜像")
        
        if check_modelscope():
            if download_from_modelscope(config['modelscope'], save_path):
                return True
            else:
                print("   尝试使用 HuggingFace 镜像...")
    
    # 备选：从 HuggingFace 镜像下载
    if config['huggingface']:
        if download_from_huggingface(config['huggingface'], save_path):
            return True
    
    print(f"❌ 所有下载方式都失败")
    return False

def download_recommended_models():
    """下载所有推荐模型"""
    
    print("\n" + "="*60)
    print("      下载推荐模型（2024-2025最新）")
    print("="*60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    recommended = [k for k, v in MODELS.items() if v.get('recommended', False)]
    print(f"\n推荐模型: {', '.join(recommended)}")
    
    success_count = 0
    for model_key in recommended:
        if download_model(model_key):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"✅ 下载完成: {success_count}/{len(recommended)}")
    print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载最新预训练模型（2024-2025）")
    parser.add_argument('--model', type=str, help='指定模型名称')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--recommended', action='store_true', help='下载所有推荐模型')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 可用模型列表（2024-2025最新版）:")
        print(f"{'模型名称':<20} {'参数量':<15} {'年份':<8} {'推荐':<6} {'镜像支持'}")
        print("-" * 80)
        
        for key, config in MODELS.items():
            ms_support = "✅" if config['modelscope'] else "❌"
            hf_support = "✅"
            rec = "🌟" if config.get('recommended', False) else ""
            
            print(f"{key:<20} {config['description'][:15]:<15} {config['year']:<8} {rec:<6} MS:{ms_support} HF:{hf_support}")
        
        print("\n💡 推荐: minicpm4-0.5b, qwen2.5-0.5b (国内下载最快)")
        print("🌟 标记为推荐模型")
        return
    
    if args.recommended:
        download_recommended_models()
    elif args.model:
        download_model(args.model, args.force)
    else:
        download_recommended_models()

if __name__ == "__main__":
    main()