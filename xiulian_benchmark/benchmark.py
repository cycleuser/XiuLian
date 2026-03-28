#!/usr/bin/env python3
"""
性能基准测试
对比修炼(XiuLian)与Transformer LLM的性能
"""

import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Callable
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xiulian.core import Engine

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    task: str
    latency_ms: float
    memory_mb: float
    success: bool
    error: str = ""

class XiuLianBenchmark:
    """修炼模型基准测试"""
    
    def __init__(self, model_path: str):
        self.engine = Engine()
        self.name = "XiuLian"
        
        # 加载训练后的模型
        model_dir = Path(model_path)
        if (model_dir / "memory.json").exists():
            with open(model_dir / "memory.json", 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
                self.engine.memory.store = memory_data.get("store", {})
                self.engine.memory.index = memory_data.get("index", {})
    
    def run_task(self, task: str, input_text: str) -> BenchmarkResult:
        """运行单个任务"""
        import tracemalloc
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            result = self.engine.process(input_text)
            success = result.success
            error = "" if success else "Processing failed"
        except Exception as e:
            success = False
            error = str(e)
        
        latency_ms = (time.time() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            model_name=self.name,
            task=task,
            latency_ms=latency_ms,
            memory_mb=peak / 1024 / 1024,
            success=success,
            error=error
        )

class TransformerBenchmark:
    """Transformer模型基准测试"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # 加载模型
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"加载模型: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print(f"   ✅ {model_name} 加载完成")
        except Exception as e:
            print(f"   ❌ {model_name} 加载失败: {e}")
    
    def run_task(self, task: str, input_text: str) -> BenchmarkResult:
        """运行单个任务"""
        if self.model is None:
            return BenchmarkResult(
                model_name=self.model_name,
                task=task,
                latency_ms=0,
                memory_mb=0,
                success=False,
                error="Model not loaded"
            )
        
        import tracemalloc
        import torch
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(input_text, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1
                )
            
            success = True
            error = ""
        except Exception as e:
            success = False
            error = str(e)
        
        latency_ms = (time.time() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return BenchmarkResult(
            model_name=self.model_name,
            task=task,
            latency_ms=latency_ms,
            memory_mb=peak / 1024 / 1024,
            success=success,
            error=error
        )

def run_benchmark_suite(xiulian_path: str, baseline_paths: Dict[str, str], iterations: int = 10):
    """运行完整基准测试套件"""
    
    print("="*60)
    print("🚀 开始性能基准测试")
    print("="*60)
    
    # 测试任务
    test_tasks = [
        ("tool_call", "调用echo msg=hello"),
        ("search", "搜索人工智能"),
        ("question", "什么是机器学习？"),
        ("web", "打开 https://example.com"),
    ]
    
    # 初始化模型
    xiulian = XiuLianBenchmark(xiulian_path)
    
    baselines = {}
    for name, path in baseline_paths.items():
        if Path(path).exists():
            baselines[name] = TransformerBenchmark(path, name)
    
    results = []
    
    # 运行测试
    for task_name, task_input in test_tasks:
        print(f"\n📊 测试任务: {task_name}")
        print(f"   输入: {task_input}")
        
        # 测试修炼
        print(f"\n   测试 XiuLian...")
        for _ in range(iterations):
            result = xiulian.run_task(task_name, task_input)
            results.append(result)
        print(f"      ✅ 完成")
        
        # 测试基线模型
        for name, benchmark in baselines.items():
            print(f"   测试 {name}...")
            result = benchmark.run_task(task_name, task_input)
            results.append(result)
            if result.success:
                print(f"      ✅ 完成")
            else:
                print(f"      ❌ 失败: {result.error}")
    
    return results

def generate_comparison_report(results: List[BenchmarkResult], output_path: str):
    """生成对比报告"""
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按模型分组
    by_model = {}
    for r in results:
        if r.model_name not in by_model:
            by_model[r.model_name] = []
        by_model[r.model_name].append(r)
    
    # 计算统计数据
    stats = {}
    for model, model_results in by_model.items():
        latencies = [r.latency_ms for r in model_results if r.success]
        memories = [r.memory_mb for r in model_results if r.success]
        success_rate = sum(1 for r in model_results if r.success) / len(model_results)
        
        stats[model] = {
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_memory_mb": statistics.mean(memories) if memories else 0,
            "peak_memory_mb": max(memories) if memories else 0,
            "success_rate": success_rate
        }
    
    # 保存结果
    with open(output_dir / "benchmark_results.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    report = ["# 性能基准测试报告\n"]
    report.append("## 测试结果对比\n")
    report.append("| 模型 | 平均延迟(ms) | 最小延迟(ms) | 最大延迟(ms) | 平均内存(MB) | 峰值内存(MB) | 成功率 |")
    report.append("|------|-------------|-------------|-------------|-------------|-------------|--------|")
    
    for model, s in stats.items():
        report.append(f"| {model} | {s['avg_latency_ms']:.2f} | {s['min_latency_ms']:.2f} | {s['max_latency_ms']:.2f} | {s['avg_memory_mb']:.2f} | {s['peak_memory_mb']:.2f} | {s['success_rate']:.2%} |")
    
    report.append("\n## 结论\n")
    
    # 找出最佳模型
    best_latency = min(stats.items(), key=lambda x: x[1]['avg_latency_ms'])
    best_memory = min(stats.items(), key=lambda x: x[1]['peak_memory_mb'])
    
    report.append(f"- **最快模型**: {best_latency[0]} ({best_latency[1]['avg_latency_ms']:.2f}ms)")
    report.append(f"- **最省内存**: {best_memory[0]} ({best_memory[1]['peak_memory_mb']:.2f}MB)")
    
    with open(output_dir / "report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 报告已生成: {output_dir / 'report.md'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="性能基准测试")
    parser.add_argument('--xiulian', default='models/xiulian_trained', help='修炼模型路径')
    parser.add_argument('--baselines', default='qwen3.5-0.8b=models/baselines/qwen3.5-0.8b,minicpm4-0.5b=models/baselines/minicpm4-0.5b',
                       help='基线模型，格式: name=path,name=path')
    parser.add_argument('--iterations', type=int, default=10, help='每任务迭代次数')
    parser.add_argument('--output', default='results', help='结果输出路径')
    
    args = parser.parse_args()
    
    # 解析基线模型
    baselines = {}
    if args.baselines:
        for item in args.baselines.split(','):
            if '=' in item:
                name, path = item.split('=', 1)
                baselines[name] = path
    
    # 运行测试
    results = run_benchmark_suite(args.xiulian, baselines, args.iterations)
    
    # 生成报告
    generate_comparison_report(results, args.output)

if __name__ == "__main__":
    main()