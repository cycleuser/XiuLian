#!/usr/bin/env python3
"""
报告生成脚本
生成训练和评估的对比报告
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_results() -> Dict[str, Any]:
    """加载评估结果"""
    results_file = PROJECT_ROOT / "experiments" / "results" / "evaluation_results.json"
    
    if not results_file.exists():
        print(f"❌ 结果文件不存在: {results_file}")
        return {}
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """生成Markdown报告"""
    
    report = []
    report.append("# 超小语言模型训练研究报告\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 研究概述
    report.append("\n## 1. 研究概述\n")
    report.append("本研究对比了多个超小规模语言模型（100M-500M参数）在相同数据规模下的训练效果。\n")
    
    report.append("\n### 1.1 研究模型\n")
    report.append("| 模型 | 参数量 | 类型 |\n")
    report.append("|------|--------|------|\n")
    
    for name, result in results.items():
        params = result.get('params', 0)
        model_type = result.get('type', 'unknown')
        report.append(f"| {name} | {params:,} | {model_type} |\n")
    
    # 2. 实验设置
    report.append("\n## 2. 实验设置\n")
    report.append("\n### 2.1 硬件环境\n")
    report.append("- **CPU**: macOS (Apple Silicon)\n")
    report.append("- **内存**: 16 GB\n")
    report.append("- **训练设备**: CPU / MPS (Metal Performance Shaders)\n")
    
    report.append("\n### 2.2 数据集\n")
    report.append("- **训练数据**: Wikipedia + Alpaca + CodeAlpaca + 合成数据\n")
    report.append("- **验证数据**: 训练数据的10%\n")
    report.append("- **测试数据**: 训练数据的10%\n")
    
    report.append("\n### 2.3 训练配置\n")
    report.append("- **Batch Size**: 2 (梯度累积: 8)\n")
    report.append("- **学习率**: 5e-5\n")
    report.append("- **训练轮数**: 3\n")
    report.append("- **优化器**: AdamW\n")
    
    # 3. 评估结果
    report.append("\n## 3. 评估结果\n")
    
    report.append("\n### 3.1 性能对比\n")
    report.append("| 模型 | 参数量 | 困惑度 | 问答准确率 |\n")
    report.append("|------|--------|--------|------------|\n")
    
    for name, result in results.items():
        params = result.get('params', 0)
        ppl = result.get('perplexity', 0)
        qa_acc = result.get('qa_accuracy', 0)
        report.append(f"| {name} | {params:,} | {ppl:.2f} | {qa_acc:.2%} |\n")
    
    # 3.2 生成示例
    report.append("\n### 3.2 生成示例\n")
    
    for name, result in results.items():
        gen_quality = result.get('generation_quality', {})
        generations = gen_quality.get('generations', [])
        
        if generations:
            report.append(f"\n#### {name}\n")
            for i, gen in enumerate(generations[:3], 1):
                report.append(f"\n**示例 {i}:**\n")
                report.append(f"- **提示**: {gen['prompt'][:100]}...\n")
                report.append(f"- **生成**: {gen['generated'][:200]}...\n")
    
    # 4. 分析与结论
    report.append("\n## 4. 分析与结论\n")
    
    # 找出最佳模型
    if results:
        best_ppl_model = min(results.items(), key=lambda x: x[1].get('perplexity', float('inf')))
        best_qa_model = max(results.items(), key=lambda x: x[1].get('qa_accuracy', 0))
        
        report.append("\n### 4.1 关键发现\n")
        report.append(f"- **最低困惑度**: {best_ppl_model[0]} (困惑度: {best_ppl_model[1].get('perplexity', 0):.2f})\n")
        report.append(f"- **最高问答准确率**: {best_qa_model[0]} (准确率: {best_qa_model[1].get('qa_accuracy', 0):.2%})\n")
        
        # 参数效率分析
        report.append("\n### 4.2 参数效率分析\n")
        report.append("\n| 模型 | 参数量 | 困惑度/参数 | 效率排名 |\n")
        report.append("|------|--------|-------------|----------|\n")
        
        efficiency = []
        for name, result in results.items():
            params = result.get('params', 1)
            ppl = result.get('perplexity', 1)
            ratio = ppl / (params / 1e6)  # 困惑度 / 百万参数
            efficiency.append((name, params, ppl, ratio))
        
        # 按效率排序
        efficiency.sort(key=lambda x: x[3])
        
        for rank, (name, params, ppl, ratio) in enumerate(efficiency, 1):
            report.append(f"| {name} | {params:,} | {ratio:.6f} | {rank} |\n")
    
    # 5. 建议
    report.append("\n### 4.3 建议\n")
    report.append("\n1. **模型选择**:\n")
    report.append("   - 对于计算资源有限的场景，推荐使用参数量较小的模型\n")
    report.append("   - 对于需要高质量生成的场景，优先考虑困惑度最低的模型\n")
    
    report.append("\n2. **训练策略**:\n")
    report.append("   - 使用梯度累积可以有效缓解小batch size的问题\n")
    report.append("   - 数据质量比数据量更重要\n")
    
    report.append("\n3. **部署建议**:\n")
    report.append("   - CPU环境适合推理，训练建议使用GPU\n")
    report.append("   - 可以通过量化进一步减小模型体积\n")
    
    # 6. 未来工作
    report.append("\n## 5. 未来工作\n")
    report.append("\n1. 扩展到更大规模的模型（1B-3B参数）\n")
    report.append("2. 探索更高效的数据筛选方法\n")
    report.append("3. 测试不同的训练技巧（如课程学习）\n")
    report.append("4. 对比量化模型的性能\n")
    
    return ''.join(report)

def generate_training_summary() -> Dict[str, Any]:
    """生成训练摘要"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': [],
        'total_training_time': 'N/A',
        'best_model': 'N/A'
    }
    
    # 查找训练日志
    log_dir = PROJECT_ROOT / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            if 'train' in log_file.name or any(model in log_file.name for model in ['smollm', 'qwen', 'minicpm']):
                summary['models_trained'].append(log_file.stem)
    
    return summary

def main():
    print("\n" + "="*60)
    print("      生成研究报告")
    print("="*60 + "\n")
    
    # 加载结果
    print("📂 加载评估结果...")
    results = load_results()
    
    if not results:
        print("⚠️  没有找到评估结果，生成空报告")
    
    # 生成报告
    print("📝 生成Markdown报告...")
    report = generate_markdown_report(results)
    
    # 保存报告
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = results_dir / "final_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存: {report_file}")
    
    # 生成训练摘要
    print("\n📊 生成训练摘要...")
    summary = generate_training_summary()
    
    summary_file = results_dir / "training_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 摘要已保存: {summary_file}")
    
    print("\n" + "="*60)
    print("✅ 报告生成完成！")
    print("="*60)
    
    # 显示报告
    print("\n" + report)

if __name__ == "__main__":
    main()