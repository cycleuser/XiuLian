#!/usr/bin/env python3
"""
实际性能测试脚本
收集真实的延迟、内存、吞吐量数据
"""

import sys
import time
import json
import tracemalloc
import statistics
from datetime import datetime

sys.path.insert(0, '/Users/fred/Documents/GitHub/cycleuser/XiuLian')

from xiulian.core import Engine

# 测试用例定义
TOOL_CALLING_CASES = [
    ("调用echo工具，消息为hello", "echo", True),
    ("使用calc计算", "calc", True),
    ("调用time", "time", True),
    ("用echo工具说你好", "echo", True),
    ("执行calc", "calc", True),
    ("调用echo msg=test", "echo", True),
    ("使用time工具", "time", True),
    ("用calc计算100*5", "calc", True),
    ("调用echo message=world", "echo", True),
    ("执行time", "time", True),
]

QUESTION_CASES = [
    ("什么是人工智能?", ["AI", "智能", "模拟"]),
    ("机器学习是什么?", ["学习", "数据", "算法"]),
    ("什么是深度学习?", ["深度", "神经网络"]),
    ("什么是Transformer?", ["注意力", "attention"]),
    ("什么是编程?", ["代码", "程序"]),
    ("Python是什么?", ["编程", "语言"]),
    ("什么是算法?", ["步骤", "计算"]),
    ("什么是数据结构?", ["存储", "组织"]),
    ("什么是操作系统?", ["系统", "软件"]),
    ("什么是网络?", ["连接", "通信"]),
]

INTENT_CASES = [
    ("搜索机器学习教程", "search"),
    ("查找Python文档", "search"),
    ("调用echo工具", "tool"),
    ("使用calc计算", "tool"),
    ("什么是AI?", "question"),
    ("如何学习编程?", "question"),
    ("打开 https://example.com", "web"),
    ("访问 http://test.org", "web"),
    ("run the echo tool", "tool"),
    ("search for AI papers", "search"),
]

def measure_performance(engine, test_cases, iterations=100):
    """测量性能"""
    latencies = []
    correct = 0
    total = 0
    
    for _ in range(iterations):
        for case in test_cases:
            start = time.perf_counter()
            result = engine.process(case[0])
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            total += 1
            
            if len(case) > 2 and case[2]:  # 检查是否预期成功
                if result.success:
                    correct += 1
            elif len(case) > 1:  # 检查意图/关键词
                if result.success:
                    if isinstance(case[1], str):  # 意图检查
                        if result.data and case[1] in str(result.data):
                            correct += 1
                    elif isinstance(case[1], list):  # 关键词检查
                        answer = str(result.data).lower() if result.data else ""
                        if any(kw.lower() in answer for kw in case[1]):
                            correct += 1
    
    latencies.sort()
    n = len(latencies)
    
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "std_ms": statistics.stdev(latencies) if n > 1 else 0,
    }

def measure_memory():
    """测量内存使用"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_full_benchmark():
    """运行完整基准测试"""
    print("=" * 60)
    print("XiuLian 实际性能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 初始化引擎
    tracemalloc.start()
    engine = Engine()
    
    # 加载知识库
    try:
        engine.load_knowledge('/Users/fred/Documents/GitHub/cycleuser/XiuLian/xiulian/data/knowledge.json')
        print("知识库加载成功")
    except Exception as e:
        print(f"知识库加载失败: {e}")
    
    # 测试各维度
    results = {}
    
    print("\n1. 工具调用测试...")
    results["tool_calling"] = measure_performance(engine, TOOL_CALLING_CASES, 50)
    print(f"   准确率: {results['tool_calling']['accuracy']:.1%}")
    print(f"   平均延迟: {results['tool_calling']['avg_latency_ms']:.3f}ms")
    
    print("\n2. 问题问答测试...")
    results["question_answering"] = measure_performance(engine, QUESTION_CASES, 50)
    print(f"   准确率: {results['question_answering']['accuracy']:.1%}")
    print(f"   平均延迟: {results['question_answering']['avg_latency_ms']:.3f}ms")
    
    print("\n3. 意图识别测试...")
    results["intent_recognition"] = measure_performance(engine, INTENT_CASES, 50)
    print(f"   准确率: {results['intent_recognition']['accuracy']:.1%}")
    print(f"   平均延迟: {results['intent_recognition']['avg_latency_ms']:.3f}ms")
    
    # 测量内存
    memory_mb = measure_memory()
    
    # 吞吐量测试
    print("\n4. 吞吐量测试...")
    start = time.time()
    count = 0
    for case in TOOL_CALLING_CASES[:5]:
        engine.process(case[0])
        count += 1
    for _ in range(99):
        for case in TOOL_CALLING_CASES[:5]:
            engine.process(case[0])
            count += 1
    elapsed = time.time() - start
    throughput = count / elapsed
    print(f"   吞吐量: {throughput:.1f} 请求/秒")
    
    # 复杂度测试
    print("\n5. 输入长度复杂度测试...")
    complexity_results = []
    test_inputs = [
        "测试" * 10,
        "测试" * 50,
        "测试" * 100,
        "测试" * 200,
        "测试" * 500,
    ]
    
    for inp in test_inputs:
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            engine.process(inp)
            latencies.append((time.perf_counter() - start) * 1000)
        avg_latency = statistics.mean(latencies)
        complexity_results.append({
            "length": len(inp),
            "latency_ms": avg_latency
        })
        print(f"   长度 {len(inp):4d}: {avg_latency:.3f}ms")
    
    # 汇总结果
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "platform": sys.platform,
            "python_version": sys.version,
        },
        "benchmarks": results,
        "memory_mb": memory_mb,
        "throughput_per_sec": throughput,
        "complexity_test": complexity_results,
        "summary": {
            "total_samples": sum(r["samples"] for r in results.values()),
            "overall_accuracy": sum(r["correct"] for r in results.values()) / sum(r["samples"] for r in results.values()),
            "avg_latency_ms": statistics.mean([r["avg_latency_ms"] for r in results.values()]),
            "p95_latency_ms": max(r["p95_ms"] for r in results.values()),
        }
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总样本数: {final_results['summary']['total_samples']}")
    print(f"总体准确率: {final_results['summary']['overall_accuracy']:.1%}")
    print(f"平均延迟: {final_results['summary']['avg_latency_ms']:.3f}ms")
    print(f"P95延迟: {final_results['summary']['p95_latency_ms']:.3f}ms")
    print(f"内存占用: {memory_mb:.1f}MB")
    print(f"吞吐量: {throughput:.1f} 请求/秒")
    
    # 保存结果
    with open('/Users/fred/Documents/GitHub/cycleuser/XiuLian/xiulian/data/benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 xiulian/data/benchmark_results.json")
    
    return final_results

if __name__ == "__main__":
    run_full_benchmark()