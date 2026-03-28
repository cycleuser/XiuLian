#!/usr/bin/env python3
"""
项目信息脚本
显示项目当前状态和信息
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def check_directory(name: str, path: Path) -> tuple:
    """检查目录状态"""
    if not path.exists():
        return (name, "❌ 不存在", 0)
    
    files = list(path.rglob("*"))
    size = sum(f.stat().st_size for f in files if f.is_file())
    size_mb = size / (1024 * 1024)
    
    return (name, "✅ 存在", f"{size_mb:.2f} MB")

def count_files(path: Path, pattern: str) -> int:
    """统计文件数量"""
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))

def show_project_info():
    """显示项目信息"""
    
    print("\n" + "="*70)
    print(" " * 20 + "超小语言模型训练项目")
    print("="*70 + "\n")
    
    # 项目结构
    print("📁 项目结构:\n")
    
    directories = [
        ("数据目录", PROJECT_ROOT / "data"),
        ("模型目录", PROJECT_ROOT / "models"),
        ("实验目录", PROJECT_ROOT / "experiments"),
        ("日志目录", PROJECT_ROOT / "logs"),
    ]
    
    print(f"{'目录':<20} {'状态':<15} {'大小'}")
    print("-" * 70)
    
    for name, path in directories:
        dir_name, status, size = check_directory(name, path)
        print(f"{dir_name:<20} {status:<15} {size}")
    
    # 数据集统计
    print("\n📊 数据集统计:\n")
    
    data_dirs = {
        "原始数据": PROJECT_ROOT / "data" / "raw",
        "处理后数据": PROJECT_ROOT / "data" / "processed",
        "数据集": PROJECT_ROOT / "data" / "datasets",
        "合成数据": PROJECT_ROOT / "data" / "synthetic",
    }
    
    for name, path in data_dirs.items():
        if path.exists():
            jsonl_count = count_files(path, "*.jsonl")
            json_count = count_files(path, "*.json")
            total = jsonl_count + json_count
            if total > 0:
                print(f"   {name}: {total} 个文件")
        else:
            print(f"   {name}: 未创建")
    
    # 模型统计
    print("\n🤖 模型统计:\n")
    
    pretrained_dir = PROJECT_ROOT / "models" / "pretrained"
    final_dir = PROJECT_ROOT / "models" / "final"
    checkpoints_dir = PROJECT_ROOT / "models" / "checkpoints"
    
    if pretrained_dir.exists():
        models = [d.name for d in pretrained_dir.iterdir() if d.is_dir()]
        if models:
            print(f"   预训练模型: {', '.join(models)}")
        else:
            print("   预训练模型: 无")
    else:
        print("   预训练模型: 未下载")
    
    if final_dir.exists():
        models = [d.name for d in final_dir.iterdir() if d.is_dir()]
        if models:
            print(f"   训练后模型: {', '.join(models)}")
        else:
            print("   训练后模型: 无")
    else:
        print("   训练后模型: 未训练")
    
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*"))
        if checkpoints:
            print(f"   检查点数量: {len(checkpoints)}")
    
    # 实验结果
    print("\n📈 实验结果:\n")
    
    results_dir = PROJECT_ROOT / "experiments" / "results"
    if results_dir.exists():
        result_files = {
            "评估结果": "evaluation_results.json",
            "训练摘要": "training_summary.json",
            "最终报告": "final_report.md"
        }
        
        for name, filename in result_files.items():
            file_path = results_dir / filename
            if file_path.exists():
                print(f"   {name}: ✅ 已生成")
            else:
                print(f"   {name}: ❌ 未生成")
    else:
        print("   未开始实验")
    
    # 日志
    print("\n📝 训练日志:\n")
    
    log_dir = PROJECT_ROOT / "logs"
    if log_dir.exists():
        logs = list(log_dir.glob("*.log"))
        if logs:
            print(f"   日志文件数: {len(logs)}")
            latest_log = max(logs, key=lambda x: x.stat().st_mtime)
            print(f"   最新日志: {latest_log.name}")
        else:
            print("   无日志文件")
    else:
        print("   无日志目录")
    
    # 系统信息
    print("\n💻 系统信息:\n")
    
    import sys
    print(f"   Python版本: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("   加速设备: MPS (Apple Metal) ✅")
        else:
            print("   加速设备: CPU")
    except ImportError:
        print("   PyTorch: 未安装")
    
    # 下一步建议
    print("\n" + "="*70)
    print("📋 下一步操作:\n")
    
    # 根据当前状态给出建议
    if not (PROJECT_ROOT / "data" / "processed" / "train.jsonl").exists():
        print("   1. 准备数据:")
        print("      python3 scripts/data_processing/download_datasets.py")
        print("      python3 scripts/data_processing/process_data.py")
        print("      python3 scripts/data_processing/split_dataset.py")
        print()
    
    if not pretrained_dir.exists() or not list(pretrained_dir.glob("*")):
        print("   2. 下载模型:")
        print("      python3 scripts/models/download_models.py")
        print()
    
    if not final_dir.exists() or not list(final_dir.glob("*")):
        print("   3. 训练模型:")
        print("      python3 scripts/training/train.py --model smollm-135m")
        print()
    
    if not (results_dir / "evaluation_results.json").exists():
        print("   4. 评估模型:")
        print("      python3 scripts/evaluation/evaluate_all.py")
        print()
    
    if not (results_dir / "final_report.md").exists():
        print("   5. 生成报告:")
        print("      python3 scripts/analysis/generate_report.py")
        print()
    
    print("   或运行一键脚本:")
    print("      ./scripts/run_all.sh")
    print()
    
    print("   快速测试:")
    print("      ./scripts/quick_test.sh")
    print("="*70 + "\n")

if __name__ == "__main__":
    show_project_info()