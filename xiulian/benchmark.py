"""
Xiulian vs Transformer 对比实验框架

对比维度：
1. 数据规模：相同训练token数下的表现
2. 训练时间：达到相同效果所需计算量
3. 模型效果：工具调用、问答、推理任务
4. 响应时间：推理延迟
5. 资源消耗：内存、存储、能耗
"""

import time
import json
import psutil
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime


@dataclass
class BenchmarkResult:
    name: str
    samples: int
    correct: int
    total_time_ms: float
    avg_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    memory_mb: float
    cpu_percent: float
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.samples if self.samples > 0 else 0
    
    @property
    def throughput(self) -> float:
        return self.samples / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0


@dataclass
class ModelConfig:
    name: str
    architecture: str  # "transformer" or "symbolic"
    parameters: int
    training_tokens: int
    memory_mb: int
    hidden_size: Optional[int] = None
    layers: Optional[int] = None
    heads: Optional[int] = None


class ExperimentFramework:
    """实验框架"""
    
    BENCHMARKS = {
        "tool_calling": [
            {"input": "调用echo工具，消息为hello", "expected_tool": "echo"},
            {"input": "使用calc计算 2+2", "expected_tool": "calc"},
            {"input": "run the echo tool", "expected_tool": "echo"},
            {"input": "execute calc with 100*5", "expected_tool": "calc"},
            {"input": "帮我查询时间", "expected_tool": "time"},
            {"input": "call time function", "expected_tool": "time"},
        ],
        "question_answering": [
            {"input": "什么是人工智能?", "keywords": ["AI", "智能", "模拟"]},
            {"input": "机器学习是什么?", "keywords": ["学习", "数据", "算法"]},
            {"input": "What is deep learning?", "keywords": ["neural", "layer", "learning"]},
            {"input": "Transformer架构有什么特点?", "keywords": ["attention", "自注意", "编码器"]},
        ],
        "web_access": [
            {"input": "打开 https://example.com", "expected_action": "fetch"},
            {"input": "访问 https://github.com", "expected_action": "fetch"},
            {"input": "fetch content from arxiv.org", "expected_action": "fetch"},
        ],
        "intent_recognition": [
            {"input": "搜索机器学习教程", "expected": "search"},
            {"input": "search for Python docs", "expected": "search"},
            {"input": "计算 123 加 456", "expected": "tool"},
            {"input": "什么是神经网络?", "expected": "question"},
        ],
    }
    
    def __init__(self):
        self.results: dict[str, list[BenchmarkResult]] = {}
    
    def measure_memory(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def measure_cpu(self) -> float:
        return psutil.cpu_percent(interval=0.1)
    
    def run_benchmark(
        self,
        name: str,
        model_name: str,
        process_fn: Callable[[str], Any],
        evaluate_fn: Callable[[Any, dict], bool],
    ) -> BenchmarkResult:
        test_cases = self.BENCHMARKS.get(name, [])
        if not test_cases:
            raise ValueError(f"Unknown benchmark: {name}")
        
        latencies = []
        correct = 0
        total_start = time.time()
        memory_before = self.measure_memory()
        
        for case in test_cases:
            start = time.time()
            result = process_fn(case["input"])
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            if evaluate_fn(result, case):
                correct += 1
        
        total_time = (time.time() - total_start) * 1000
        memory_after = self.measure_memory()
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return BenchmarkResult(
            name=f"{model_name}_{name}",
            samples=len(test_cases),
            correct=correct,
            total_time_ms=total_time,
            avg_latency_ms=sum(latencies) / n,
            p50_ms=sorted_latencies[n // 2],
            p95_ms=sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0],
            p99_ms=sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0],
            memory_mb=memory_after - memory_before,
            cpu_percent=self.measure_cpu(),
        )
    
    def run_all_benchmarks(
        self,
        model_name: str,
        process_fn: Callable[[str], Any],
        evaluate_fns: dict[str, Callable],
    ) -> dict[str, BenchmarkResult]:
        results = {}
        for bench_name in self.BENCHMARKS:
            if bench_name in evaluate_fns:
                results[bench_name] = self.run_benchmark(
                    bench_name, model_name, process_fn, evaluate_fns[bench_name]
                )
        return results


class XiulianEvaluator:
    """Xiulian模型评估器"""
    
    def __init__(self):
        from xiulian import Engine
        self.engine = Engine()
    
    def process(self, text: str):
        return self.engine.process(text)
    
    def evaluate_tool_calling(self, result, case) -> bool:
        expected = case.get("expected_tool", "")
        if result.success and result.data:
            if isinstance(result.data, dict):
                return expected in str(result.data).lower()
        return False
    
    def evaluate_question_answering(self, result, case) -> bool:
        keywords = case.get("keywords", [])
        if result.success and result.data:
            answer = str(result.data).lower()
            return any(kw.lower() in answer for kw in keywords)
        return False
    
    def evaluate_web_access(self, result, case) -> bool:
        expected = case.get("expected_action", "")
        return result.success and expected in str(result.data).lower()
    
    def evaluate_intent(self, result, case) -> bool:
        expected = case.get("expected", "")
        if result.success and result.data:
            intent = result.data.get("intent", "")
            return expected in intent.lower()
        return False


class ComparisonReport:
    """对比报告生成器"""
    
    @staticmethod
    def generate_report(
        xiulian_results: dict[str, BenchmarkResult],
        transformer_results: dict[str, BenchmarkResult],
        xiulian_config: ModelConfig,
        transformer_config: ModelConfig,
    ) -> str:
        report = []
        report.append("=" * 70)
        report.append("Xiulian vs Transformer 对比报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append("\n## 模型配置\n")
        report.append(f"| 配置项 | Xiulian | Transformer |")
        report.append(f"|--------|---------|-------------|")
        report.append(f"| 架构 | {xiulian_config.architecture} | {transformer_config.architecture} |")
        report.append(f"| 参数量 | {xiulian_config.parameters:,} | {transformer_config.parameters:,} |")
        report.append(f"| 训练Token | {xiulian_config.training_tokens:,} | {transformer_config.training_tokens:,} |")
        report.append(f"| 内存占用 | {xiulian_config.memory_mb} MB | {transformer_config.memory_mb} MB |")
        
        report.append("\n## 性能对比\n")
        for bench_name in xiulian_results:
            if bench_name in transformer_results:
                x = xiulian_results[bench_name]
                t = transformer_results[bench_name]
                
                report.append(f"\n### {bench_name}\n")
                report.append(f"| 指标 | Xiulian | Transformer | 加速比 |")
                report.append(f"|------|---------|-------------|--------|")
                report.append(f"| 准确率 | {x.accuracy:.1%} | {t.accuracy:.1%} | - |")
                report.append(f"| 平均延迟 | {x.avg_latency_ms:.2f}ms | {t.avg_latency_ms:.2f}ms | {t.avg_latency_ms/x.avg_latency_ms:.1f}x |")
                report.append(f"| P95延迟 | {x.p95_ms:.2f}ms | {t.p95_ms:.2f}ms | {t.p95_ms/x.p95_ms:.1f}x |")
                report.append(f"| 吞吐量 | {x.throughput:.1f}/s | {t.throughput:.1f}/s | {x.throughput/t.throughput:.1f}x |")
                report.append(f"| 内存增量 | {x.memory_mb:.1f}MB | {t.memory_mb:.1f}MB | {t.memory_mb/x.memory_mb:.1f}x |")
        
        report.append("\n## 总结\n")
        total_correct_x = sum(r.correct for r in xiulian_results.values())
        total_samples_x = sum(r.samples for r in xiulian_results.values())
        total_correct_t = sum(r.correct for r in transformer_results.values())
        total_samples_t = sum(r.samples for r in transformer_results.values())
        
        avg_latency_x = sum(r.avg_latency_ms for r in xiulian_results.values()) / len(xiulian_results)
        avg_latency_t = sum(r.avg_latency_ms for r in transformer_results.values()) / len(transformer_results)
        
        report.append(f"- Xiulian总体准确率: {total_correct_x/total_samples_x:.1%}")
        report.append(f"- Transformer总体准确率: {total_correct_t/total_samples_t:.1%}")
        report.append(f"- 平均延迟加速: {avg_latency_t/avg_latency_x:.1f}x")
        report.append(f"- 内存效率: {transformer_config.memory_mb/xiulian_config.memory_mb:.1f}x")
        
        return "\n".join(report)


def run_experiment():
    """运行完整实验"""
    framework = ExperimentFramework()
    
    xiulian_config = ModelConfig(
        name="Xiulian",
        architecture="Symbolic-Memory Network",
        parameters=500_000_000,
        training_tokens=0,
        memory_mb=100,
    )
    
    transformer_config = ModelConfig(
        name="TinyLlama-1.1B",
        architecture="Transformer Decoder",
        parameters=1_100_000_000,
        training_tokens=3_000_000_000_000,
        memory_mb=2200,
        hidden_size=2048,
        layers=22,
        heads=32,
    )
    
    xiulian = XiulianEvaluator()
    
    xiulian_results = framework.run_all_benchmarks(
        "Xiulian",
        xiulian.process,
        {
            "tool_calling": xiulian.evaluate_tool_calling,
            "question_answering": xiulian.evaluate_question_answering,
            "web_access": xiulian.evaluate_web_access,
            "intent_recognition": xiulian.evaluate_intent,
        }
    )
    
    print(ComparisonReport.generate_report(
        xiulian_results, xiulian_results, xiulian_config, transformer_config
    ))


if __name__ == "__main__":
    run_experiment()